"""Diagnose Acropole cruise speed preferences and performance constraints.

Uses external Acropole/BADA assets. Creates 150 dpi PNGs and numerical reports.
No model corrections or smoothing are applied.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from time import perf_counter

import casadi as ca
import matplotlib
import openap.casadi as oc
from openap.aero import ft, kts

import numpy as np
import opentop as top
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "openap": "#0072B2",
    "no_wave": "#56B4E9",
    "bada4": "#009E73",
    "acropole": "#D55E00",
}
LABELS = {
    "openap": "OpenAP + wave drag",
    "no_wave": "OpenAP, wave drag off",
    "bada4": "BADA 4",
    "acropole": "Acropole",
}


def optimizer(force_model, bada_path, neural=None):
    opt = top.Cruise(
        "A320-214" if force_model == "bada4" else "A320",
        "EHAM",
        "EDDF",
        performance_model=force_model,
        bada_path=str(bada_path) if force_model == "bada4" else None,
        fuel_model=neural,
        h_min=30000 * ft,
        h_max=37000 * ft,
    )
    opt.setup(nodes=12, max_iter=800)
    opt.fix_cruise_altitude()
    opt.fix_mach_number()
    return opt


def evaluators(bada_path, neural):
    mach, altitude, mass = [ca.MX.sym(s) for s in ("mach", "altitude", "mass")]
    speed = oc.aero.mach2tas(mach, altitude * ft)
    tas = speed / kts
    result = {}
    for name in ("openap", "bada4", "no_wave", "acropole"):
        if name == "acropole":
            ff = neural.enroute(mass, tas, altitude)
            outputs = [ff, ff / speed * 1000, ca.jacobian(ff / speed * 1000, mach)]
            names = ["ff", "kg_km", "cost_slope"]
        else:
            opt = optimizer("bada4" if name == "bada4" else "openap", bada_path)
            if name == "no_wave":
                opt.drag = oc.Drag("A320", wave_drag=False)
                opt.fuelflow = oc.FuelFlow("A320", opt.engtype, wave_drag=False)
            drag = opt.drag.clean(mass, tas, altitude)
            thrust = opt._thrust_climb(tas, altitude)
            lift_drag = opt._lift_margin_drag(mass, tas, altitude)
            ff = opt.fuelflow.enroute(mass, tas, altitude, 0)
            outputs = [
                ff,
                ff / speed * 1000,
                ca.jacobian(ff / speed * 1000, mach),
                drag,
                thrust,
                0.95 * thrust - drag,
                0.90 * thrust - lift_drag,
            ]
            names = [
                "ff",
                "kg_km",
                "cost_slope",
                "drag",
                "thrust",
                "drag_slack",
                "lift_slack",
            ]
        result[name] = ca.Function(
            f"evaluate_{name}",
            [mach, altitude, mass],
            outputs,
            ["mach", "altitude", "mass"],
            names,
        )
    return result


def evaluate(function, mach, altitude, mass):
    arrays = np.broadcast_arrays(mach, altitude, mass)
    return {
        key: np.asarray(value).reshape(arrays[0].shape)
        for key, value in function(
            **{
                key: np.asarray(a).reshape(1, -1)
                for key, a in zip(("mach", "altitude", "mass"), arrays)
            }
        ).items()
    }


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelcolor": "#243746",
            "text.color": "#243746",
            "axes.edgecolor": "#89939A",
            "axes.linewidth": 0.7,
        }
    )


def sweep_plot(functions, args):
    mach = np.linspace(0.66, 0.84, 361)
    values = {
        name: evaluate(fn, mach, args.altitude, args.mass)
        for name, fn in functions.items()
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.7), sharex=True)
    fig.subplots_adjust(
        top=0.77, bottom=0.15, left=0.09, right=0.96, hspace=0.4, wspace=0.26
    )
    fig.text(
        0.09,
        0.955,
        "Does high-speed drag reach the fuel objective?",
        size=17,
        weight="bold",
    )
    fig.text(
        0.09,
        0.915,
        f"A320 | {args.mass:,.0f} kg | {args.altitude:,.0f} ft | "
        "Level flight, still air",
        color="#536572",
    )
    handles = []
    for name, vals in values.items():
        line_style = "--" if name in ("no_wave", "acropole") else "-"
        options = dict(color=COLORS[name], ls=line_style, lw=2, label=LABELS[name])
        (line,) = axes[0, 1].plot(mach, vals["ff"], **options)
        handles.append(line)
        axes[1, 0].plot(mach, vals["kg_km"], **options)
        if "drag" in vals:
            axes[0, 0].plot(mach, vals["drag"] / 1000, **options)
        if name in ("openap", "bada4"):
            axes[1, 1].plot(
                mach,
                vals["drag_slack"] / 1000,
                color=COLORS[name],
                label=f"{LABELS[name]}: thrust minus drag",
            )
            axes[1, 1].plot(
                mach,
                vals["lift_slack"] / 1000,
                color=COLORS[name],
                ls="--",
                label=f"{LABELS[name]}: lift-margin check",
            )
    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="upper left",
        bbox_to_anchor=(0.08, 0.875),
        ncol=2,
        frameon=False,
    )
    titles = [
        "A   Drag / required level-flight thrust",
        "B   Predicted fuel flow",
        "C   Fuel per distance (lower is better)",
        "D   Performance constraint slack",
    ]
    labels = [
        "Force (kN)",
        "Fuel flow (kg/s)",
        "Fuel per distance (kg/km)",
        "Slack (kN); feasible above zero",
    ]
    for ax, title, label in zip(axes.flat, titles, labels):
        ax.set_title(title, loc="left", pad=9)
        ax.set_ylabel(label)
        ax.set_xlim(mach[0], mach[-1])
        ax.axvspan(0.82, 0.84, color="#999999", alpha=0.12)
        ax.axvline(0.82, color="#666666", ls=":", lw=1)
        ax.grid(axis="y", color="#DDE3E8", lw=0.6)
        ax.set_axisbelow(True)
    axes[1, 1].axhline(0, color="black", lw=0.8)
    axes[1, 1].legend(fontsize=7.5, frameon=False, loc="lower left")
    for ax in axes[1]:
        ax.set_xlabel("Mach number")
    fig.text(
        0.09,
        0.055,
        "Gray band: above M 0.82. Curves include infeasible states for diagnosis.\n"
        "Fixed-mass sensitivity check; not a trajectory solve.",
        size=9,
        color="#536572",
    )
    fig.savefig(args.output_dir / "mach_diagnostic.png", dpi=150)
    plt.close(fig)
    table = {"mach": mach}
    for name, data in values.items():
        table.update({f"{name}_{key}": value for key, value in data.items()})
    pd.DataFrame(table).to_csv(args.output_dir / "mach_sweep.csv", index=False)
    return values


def landscape(functions, args):
    mach = np.linspace(0.66, 0.82, 241)
    altitude = np.linspace(30000, 37000, 211)
    mm, hh = np.meshgrid(mach, altitude)
    neural = evaluate(functions["acropole"], mm, hh, args.mass)["kg_km"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.87, top=0.75, bottom=0.2, wspace=0.15)
    fig.text(
        0.08,
        0.945,
        "Same neural fuel surface, different feasible regions",
        size=17,
        weight="bold",
    )
    fig.text(
        0.08,
        0.90,
        f"Acropole fuel per distance at {args.mass:,.0f} kg; "
        "gray = violates a force constraint",
        size=10,
    )
    output = {}
    for ax, name in zip(axes, ("openap", "bada4")):
        vals = evaluate(functions[name], mm, hh, args.mass)
        slack = np.minimum(vals["drag_slack"], vals["lift_slack"])
        feasible = slack >= 0
        mesh = ax.pcolormesh(
            mm, hh / 1000, neural, cmap="viridis", shading="auto", rasterized=True
        )
        ax.contourf(
            mm,
            hh / 1000,
            (~feasible).astype(float),
            levels=[0.5, 1.5],
            colors=["#DDDDDD"],
            alpha=0.9,
        )
        ax.contour(mm, hh / 1000, slack, levels=[0], colors="black", linewidths=1)
        cost = np.where(feasible, neural, np.inf)
        idx = np.unravel_index(np.argmin(cost), cost.shape)
        ax.plot(
            mm[idx],
            hh[idx] / 1000,
            marker="*",
            color="white",
            mec="black",
            ms=14,
            label="Best feasible grid point",
        )
        source = (
            Path("examples/data/neural_fuel_smoke.json")
            if name == "openap"
            else Path("examples/data/neural_fuel_bada4_smoke.json")
        )
        report = json.loads(source.read_text())
        solve = (report["exact"] if name == "openap" else report)["solves"]["acropole"]
        ax.plot(
            solve["mach"],
            solve["altitude_ft"] / 1000,
            "o",
            color="#D55E00",
            mec="white",
            ms=7,
            label="Previous NLP solution",
        )
        ax.set_title(f"{LABELS[name]} constraints", loc="left")
        ax.set_xlabel("Mach number")
        ax.set_xlim(0.66, 0.822)
        ax.legend(loc="lower left", framealpha=0.9, fontsize=8)
        output[name] = {
            "grid_mach": float(mm[idx]),
            "grid_altitude_ft": float(hh[idx]),
            "grid_kg_km": float(neural[idx]),
        }
    axes[0].set_ylabel("Altitude (1,000 ft)")
    bar = fig.colorbar(mesh, cax=fig.add_axes([0.90, 0.2, 0.018, 0.55]))
    bar.set_label("Acropole fuel per distance (kg/km)")
    fig.text(
        0.08,
        0.055,
        "Grid search holds mass fixed; NLP trajectories lose mass. "
        "Optima may differ slightly.\n"
        "This identifies competing basins and constraint boundaries, "
        "not a proof of global optimality.",
        size=9,
        color="#536572",
    )
    fig.savefig(args.output_dir / "feasible_landscape.png", dpi=150)
    plt.close(fig)
    return output


def multistart(functions, neural, args):
    records = []
    seeds = [(0.70, 30500), (0.74, 33000), (0.78, 31500), (0.81, 32000)]
    for force in ("openap", "bada4"):
        for mach, altitude in seeds:
            opt = optimizer(force, args.bada_path, neural)
            opti = ca.Opti()
            start = perf_counter()
            trans = opt._add_formulation(opti, "fuel", minimize=True)
            speed = float(oc.aero.mach2tas(mach, altitude * ft))
            duration = float(opt.range) / speed
            opti.set_initial(opt.ts_final, duration)
            for control in trans.U:
                opti.set_initial(control[0], mach)
            for k, state in enumerate(trans.X):
                opti.set_initial(state[2], altitude * ft)
                opti.set_initial(state[4], duration * k / opt.nodes)
            for k, states in enumerate(trans.Xc):
                for tau, state in zip(trans.collocation_roots, states):
                    opti.set_initial(state[2], altitude * ft)
                    opti.set_initial(state[4], duration * (k + tau) / opt.nodes)
            df = opt._solve(trans.X, trans.U)
            result = opt._make_result(df)
            points = list(trans.path_points())
            states = np.array([opt._last_solution.value(x) for x, _ in points])
            controls = np.array([opt._last_solution.value(u) for _, u in points])
            diagnostics = evaluate(
                functions[force], controls[:, 0], states[:, 2] / ft, states[:, 3]
            )
            record = {
                "force_model": force,
                "seed_mach": mach,
                "seed_altitude_ft": altitude,
                "success": result.success,
                "status": result.status,
                "fuel_kg": result.fuel,
                "objective_kg": result.objective,
                "mach": float(controls[0, 0]),
                "altitude_ft": float(states[0, 2] / ft),
                "minimum_drag_slack_N": float(diagnostics["drag_slack"].min()),
                "minimum_lift_slack_N": float(diagnostics["lift_slack"].min()),
                "mach_limit_slack": float(opt.mach_max - controls[:, 0].max()),
                "iterations": result.iters,
                "seconds": perf_counter() - start,
            }
            if result.success:
                np.testing.assert_allclose(
                    result.objective, result.fuel, atol=1e-3, rtol=0
                )
            records.append(record)
            print(json.dumps(record), flush=True)
            (args.output_dir / "multistart.json").write_text(
                json.dumps(records, indent=2) + "\n"
            )
    plot_multistart(records, args)
    return records


def plot_multistart(records, args):
    seeds = [
        (r["seed_mach"], r["seed_altitude_ft"])
        for r in records
        if r["force_model"] == "openap"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    fig.subplots_adjust(top=0.75, bottom=0.26, left=0.10, right=0.96, wspace=0.34)
    fig.text(
        0.10,
        0.94,
        "Do different initial guesses change the answer?",
        size=16,
        weight="bold",
    )
    fig.text(
        0.10,
        0.885,
        "Acropole fuel | Four Mach/altitude seeds per force model | Exact Hessians",
        size=10,
    )
    for ax, force in zip(axes, ("openap", "bada4")):
        data = [r for r in records if r["force_model"] == force]
        for i, r in enumerate(data):
            ax.scatter(
                i,
                r["fuel_kg"],
                marker="o" if r["success"] else "x",
                s=45,
                color=COLORS[force],
            )
        mean = np.mean([r["fuel_kg"] for r in data])
        ax.set_ylim(mean - 5, mean + 5)
        ax.set_xticks(
            range(4), [f"M {m:.2f}\n{h:,.0f} ft" for m, h in seeds], fontsize=8
        )
        ax.set_ylabel("Optimized fuel (kg)")
        ax.set_xlabel("Initial guess")
        ax.set_title(f"{LABELS[force]} constraints", loc="left")
        ax.grid(axis="y", color="#DDE3E8")
    fig.text(
        0.10,
        0.055,
        "Agreement across these starts reduces concern about local trapping;\n"
        "it does not prove a global optimum. Crosses indicate failed solves.",
        size=9,
        color="#536572",
    )
    fig.savefig(args.output_dir / "multistart.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acropole-dir", required=True, type=Path)
    parser.add_argument("--bada-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mass", default=66300.0, type=float)
    parser.add_argument("--altitude", default=32000.0, type=float)
    parser.add_argument("--multistart", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = args.acropole_dir / "src/acropole"
    neural = top.AcropoleFuelFlow(
        "A320",
        model_path=root / "models/acropole_fuel_model.onnx",
        aircraft_params_path=root / "data/aircraft_params.csv",
    )
    sys.path.insert(0, str(args.acropole_dir / "src"))
    from acropole import AircraftFuelEstimator

    reference = AircraftFuelEstimator(
        "A320",
        model_path=str(root / "models/acropole_fuel_model.onnx"),
        aircraft_params_path=str(root / "data/aircraft_params.csv"),
    )
    parity = []
    for altitude in (32000.0, 33500.0, 35000.0):
        mach = np.linspace(0.78, 0.84, 121)
        tas = np.asarray(oc.aero.mach2tas(mach, altitude * ft)).ravel() / kts
        actual = neural.enroute(args.mass, tas, altitude)
        expected = reference.estimate(
            tas,
            np.full(len(mach), altitude),
            np.zeros(len(mach)),
            airspeed=tas,
            mass=np.full(len(mach), args.mass),
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        parity.append(
            {
                "altitude_ft": altitude,
                "samples": len(mach),
                "max_absolute_error_kg_s": float(np.max(np.abs(actual - expected))),
                "max_relative_error": float(np.max(np.abs(actual / expected - 1))),
            }
        )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Wave drag is experimental.*")
        functions = evaluators(args.bada_path, neural)
    style()
    sweep_plot(functions, args)
    report = {
        "mass_kg": args.mass,
        "altitude_ft": args.altitude,
        "model_sha256": neural.model_sha256,
        "high_mach_reference_parity": parity,
        "grid": landscape(functions, args),
        "point_values": {},
    }
    for mach in (0.74, 0.78, 0.80, 0.82):
        report["point_values"][str(mach)] = {
            name: {
                k: float(v)
                for k, v in evaluate(fn, mach, args.altitude, args.mass).items()
            }
            for name, fn in functions.items()
        }
    (args.output_dir / "diagnostic.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report["grid"], indent=2), flush=True)
    print(args.output_dir / "mach_diagnostic.png", flush=True)
    if args.multistart:
        multistart(functions, neural, args)


if __name__ == "__main__":
    main()
