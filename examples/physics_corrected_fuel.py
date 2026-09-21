"""Compare OpenAP physics corrections with Acropole, OpenAP and BADA 4.

Requires external model assets; writes 150 dpi PNGs, CSVs and a JSON report.
Both corrections use only OpenAP. BADA 4 is a comparison/force model.
Transition Mach values are sensitivity assumptions, not measured training limits.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import warnings
from pathlib import Path

import casadi as ca
import matplotlib
import openap.casadi as oc
from openap.aero import ft, kts

import numpy as np
import opentop as top
import pandas as pd
from opentop._performance import build_performance_models
from opentop.neural import _evaluate

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {
    "openap": "OpenAP + wave drag",
    "bada4": "BADA 4",
    "acropole": "Acropole",
    "wave_increment": "Acropole + OpenAP wave increment",
    "speed_increment": "Acropole + OpenAP speed correction",
}
COLORS = {
    "openap": "#666666",
    "bada4": "#009E73",
    "acropole": "#D55E00",
    "wave_increment": "#CC79A7",
    "speed_increment": "#0072B2",
}


def make_optimizer(force, path, fuel):
    optimizer = top.Cruise(
        "A320-214" if force == "bada4" else "A320",
        "EHAM",
        "EDDF",
        m0=0.85,
        performance_model=force,
        bada_path=str(path) if force == "bada4" else None,
        fuel_model=fuel,
        h_min=30000 * ft,
        h_max=37000 * ft,
    )
    optimizer.setup(nodes=12, max_iter=800)
    optimizer.fix_cruise_altitude()
    optimizer.fix_mach_number()
    return optimizer


def solve(force, path, fuel, seed):
    optimizer = make_optimizer(force, path, fuel)
    opti = ca.Opti()
    trans = optimizer._add_formulation(opti, "fuel", minimize=True)
    mach, altitude = seed
    duration = float(optimizer.range) / float(oc.aero.mach2tas(mach, altitude * ft))
    opti.set_initial(optimizer.ts_final, duration)
    for u in trans.U:
        opti.set_initial(u[0], mach)
    for k, x in enumerate(trans.X):
        opti.set_initial(x[2], altitude * ft)
        opti.set_initial(x[4], duration * k / optimizer.nodes)
    for k, states in enumerate(trans.Xc):
        for tau, x in zip(trans.collocation_roots, states):
            opti.set_initial(x[2], altitude * ft)
            opti.set_initial(x[4], duration * (k + tau) / optimizer.nodes)
    df = optimizer._solve(trans.X, trans.U)
    result = optimizer._make_result(df)
    record = {
        "success": result.success,
        "status": result.status,
        "iterations": result.iters,
        "fuel_kg": result.fuel,
        "objective_kg": result.objective,
        "mach": float(df.mach.iloc[0]),
        "altitude_ft": float(df.altitude.iloc[0]),
        "duration_s": float(df.ts.iloc[-1]),
        "seed": list(seed),
    }
    if result.success:
        np.testing.assert_allclose(result.fuel, result.objective, rtol=0, atol=1e-3)
        np.testing.assert_allclose(df.fuel_cost.sum(), result.fuel, rtol=0, atol=1e-6)
        if fuel is not None:
            np.testing.assert_allclose(
                df.fuelflow,
                fuel.enroute(df.mass, df.tas, df.altitude, df.vertical_rate),
            )
    return record, df


def plots(models, records, destination, reference_mach, width):
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    mach = ca.MX.sym("mach")
    tas = oc.aero.mach2tas(mach, 32000 * ft) / kts
    points = np.linspace(0.70, 0.84, 561)
    table = {"mach": points}
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.4))
    fig.subplots_adjust(left=0.065, right=0.98, top=0.65, bottom=0.24, wspace=0.29)
    fig.text(
        0.065,
        0.94,
        "How physics correction changes the speed response",
        size=17,
        weight="bold",
    )
    fig.text(
        0.065,
        0.885,
        "A320 | 66,300 kg | 32,000 ft | Fixed-state diagnostic, "
        "including infeasible states",
    )
    handles = []
    for name, model in models.items():
        ff = model.enroute(66300, tas, 32000)
        cost = ff / (tas * kts) * 1000
        function = ca.Function("sweep", [mach], [ff, cost, ca.jacobian(cost, mach)])
        values = [np.asarray(v).ravel() for v in function(points)]
        for ax, key, value in zip(
            axes, ("fuel_flow_kg_s", "fuel_kg_km", "slope"), values
        ):
            (line,) = ax.plot(
                points,
                value,
                color=COLORS[name],
                label=LABELS[name],
                lw=2,
                ls="--" if name == "acropole" else "-",
            )
            table[f"{name}_{key}"] = value
        handles.append(line)
    for ax, title, ylabel in zip(
        axes,
        ("Fuel flow", "Fuel per distance", "Sensitivity to Mach"),
        ("kg/s", "kg/km", "d(kg/km) / dMach"),
    ):
        ax.set(title=title, ylabel=ylabel, xlabel="Mach", xlim=(0.70, 0.84))
        ax.axvspan(reference_mach, reference_mach + width, color="#0072B2", alpha=0.07)
        ax.axvline(0.82, color="#777777", ls=":", lw=1)
        ax.grid(axis="y", color="#dddddd", lw=0.5)
    axes[2].axhline(0, color="#777777", lw=0.8)
    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="upper left",
        bbox_to_anchor=(0.06, 0.84),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.065,
        0.10,
        f"Blue band: assumed transition M {reference_mach:.2f} "
        f"to {reference_mach + width:.2f}; dotted line: optimization Mach limit.\n"
        "Negative cost slope rewards faster flight. Corrections are uncalibrated; "
        "OpenAP consistency is not independent validation.",
        fontsize=9,
    )
    fig.savefig(destination / "physics_correction_sweep.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(table).to_csv(
        destination / "physics_correction_sweep.csv", index=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.65, bottom=0.22, wspace=0.18)
    fig.text(
        0.10,
        0.94,
        "Does the correction change the optimized cruise?",
        size=17,
        weight="bold",
    )
    fig.text(
        0.10,
        0.89,
        "EHAM to EDDF | A320 | Initial mass 66,300 kg | Constant Mach and altitude",
    )
    markers = {
        "openap": "P",
        "bada4": "s",
        "acropole": "o",
        "wave_increment": "^",
        "speed_increment": "D",
    }
    handles = []
    for ax, force in zip(axes, ("openap", "bada4")):
        for name in models:
            r = next(
                r
                for r in records
                if r["force_model"] == force
                and r["model"] == name
                and r["run"] == "main"
            )
            if not r["success"]:
                continue
            point = ax.scatter(
                r["mach"],
                r["altitude_ft"] / 1000,
                color=COLORS[name],
                marker=markers[name],
                s=65,
                edgecolors="white",
                label=LABELS[name],
                zorder=3,
            )
            if force == "openap":
                handles.append(point)
        for r in records:
            if (
                r["force_model"] == force
                and r["run"] == "transition_sensitivity"
                and r["success"]
            ):
                ax.scatter(
                    r["mach"],
                    r["altitude_ft"] / 1000,
                    marker="D",
                    facecolors="none",
                    edgecolors=COLORS["speed_increment"],
                    s=50,
                )
                ax.annotate(
                    f"start {r['reference_mach']:.2f}",
                    (r["mach"], r["altitude_ft"] / 1000),
                    xytext=(5, 8)
                    if r["reference_mach"] < reference_mach
                    else (-42, -20),
                    textcoords="offset points",
                    fontsize=8,
                )
        ax.set(
            title=f"{'OpenAP' if force == 'openap' else 'BADA 4'} force constraints",
            xlabel="Optimized Mach",
            xlim=(0.70, 0.83),
            ylim=(30, 37),
        )
        ax.grid(color="#dddddd", lw=0.5)
    axes[0].set_ylabel("Optimized altitude (1,000 ft)")
    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="upper left",
        bbox_to_anchor=(0.09, 0.84),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.10,
        0.10,
        "Hollow diamonds: alternative assumed transition starts (same width).\n"
        "Both physics corrections use OpenAP; BADA 4 is a comparison/force model. "
        "No claim of real fuel savings.",
        fontsize=9,
    )
    fig.savefig(destination / "physics_correction_optimization.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acropole-dir", type=Path, required=True)
    parser.add_argument("--bada-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-mach", type=float, default=0.76)
    parser.add_argument("--transition-width", type=float, default=0.02)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = args.acropole_dir / "src/acropole"
    neural = top.AcropoleFuelFlow(
        "A320",
        model_path=root / "models/acropole_fuel_model.onnx",
        aircraft_params_path=root / "data/aircraft_params.csv",
    )
    bada_reference = build_performance_models(
        "A320-214",
        engine=None,
        use_synonym=False,
        performance_model="bada4",
        bada_path=str(args.bada_path),
    ).fuelflow
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Wave drag is experimental.*")
        wave = oc.FuelFlow("A320", wave_drag=True)
        no_wave = oc.FuelFlow("A320", wave_drag=False)
    models = {
        "openap": wave,
        "bada4": bada_reference,
        "acropole": neural,
        "wave_increment": top.PhysicsCorrectedFuelFlow(
            neural, wave, method="wave_increment", wave_free_model=no_wave
        ),
        "speed_increment": top.PhysicsCorrectedFuelFlow(
            neural,
            wave,
            reference_mach=args.reference_mach,
            transition_width=args.transition_width,
        ),
    }

    # The optimizer matches the common aircraft type before wiring the override.
    # BADA's native adapter uses a variant code; retain its existing native API.
    class PhysicalFuelOverride:
        actype = "A320"

        def __init__(self, physical):
            mass, tas, alt, vs, dT = [
                ca.MX.sym(s) for s in ("mass", "tas", "alt", "vs", "dT")
            ]
            self.function = ca.Function(
                "physical_fuel",
                [mass, tas, alt, vs, dT],
                [physical.enroute(mass, tas, alt, vs, dT=dT)],
            )

        def enroute(self, mass, tas, alt, vs=0, *, dT=0):
            return _evaluate(self.function, mass, tas, alt, vs, dT)

    records = []
    report = {
        "model_sha256": neural.model_sha256,
        "params_sha256": neural.params_sha256,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("casadi", "openap", "opentop")
        },
        "configuration": {
            "reference_mach": args.reference_mach,
            "transition_width": args.transition_width,
            "wave_strength": 1,
            "correction_model": "openap",
            "correction_wave_drag": True,
            "comparison_models": ["openap", "bada4"],
            "force_models": ["openap", "bada4"],
            "nodes": 12,
            "polydeg": 3,
            "mass_kg": 66300,
            "altitude_bounds_ft": [30000, 37000],
            "hessian": "exact",
            "bada_path": str(args.bada_path),
        },
        "limitations": (
            "Uncalibrated transition assumptions; no training coverage "
            "or independent high-Mach validation. OpenAP speed correction includes "
            "all modeled speed effects, not only wave drag."
        ),
        "solves": records,
    }
    for force in ("openap", "bada4"):
        jobs = [
            (
                name,
                PhysicalFuelOverride(model) if name in {"openap", "bada4"} else model,
                "main",
                args.reference_mach,
                (0.74, 33000),
            )
            for name, model in models.items()
        ]
        jobs += [
            (
                "speed_increment",
                top.PhysicsCorrectedFuelFlow(
                    neural,
                    wave,
                    reference_mach=reference,
                    transition_width=args.transition_width,
                ),
                "transition_sensitivity",
                reference,
                (0.74, 33000),
            )
            for reference in (args.reference_mach - 0.02, args.reference_mach + 0.02)
        ]
        jobs.append(
            (
                "speed_increment",
                models["speed_increment"],
                "second_start",
                args.reference_mach,
                (0.81, 31000),
            )
        )
        for name, model, run, reference, seed in jobs:
            record, df = solve(force, args.bada_path, model, seed)
            record.update(
                force_model=force,
                model=name,
                run=run,
                reference_mach=reference if name == "speed_increment" else None,
                correction_source="openap"
                if name in {"wave_increment", "speed_increment"}
                else None,
            )
            records.append(record)
            df.to_csv(
                args.output_dir / f"{force}_{name}_{run}_{reference:.2f}.csv",
                index=False,
            )
            (args.output_dir / "report.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(json.dumps(record), flush=True)
    plots(models, records, args.output_dir, args.reference_mach, args.transition_width)
    if not all(r["success"] for r in records):
        raise SystemExit("One or more solves failed; inspect report.")


if __name__ == "__main__":
    main()
