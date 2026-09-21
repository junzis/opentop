"""Reproduce Acropole import validation and a still-air A320 cruise experiment.

See docs/design/neural_fuel.md for dependencies, assumptions, and commands.
No downloads occur in this script. Results are exploratory, not fuel savings.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import casadi as ca
from openap.aero import fpm, ft, kts

import numpy as np
import opentop as top


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acropole-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--nodes", type=int, default=12)
    parser.add_argument(
        "--performance-model", choices=["openap", "bada4"], default="openap"
    )
    parser.add_argument("--bada-path", type=Path)
    parser.add_argument(
        "--hessian", choices=["exact", "limited-memory"], default="exact"
    )
    args = parser.parse_args()
    if args.performance_model == "bada4" and args.bada_path is None:
        parser.error("--bada-path is required for BADA 4")
    actype = "A320-214" if args.performance_model == "bada4" else "A320"
    root = args.acropole_dir.resolve()
    sys.path.insert(0, str(root / "src"))
    from acropole import AircraftFuelEstimator

    paths = dict(
        model_path=root / "src/acropole/models/acropole_fuel_model.onnx",
        aircraft_params_path=root / "src/acropole/data/aircraft_params.csv",
    )
    start = perf_counter()
    neural = top.AcropoleFuelFlow("A320", **paths)
    import_seconds = perf_counter() - start
    reference = AircraftFuelEstimator("A320", **{k: str(v) for k, v in paths.items()})
    rng = np.random.default_rng(20260918)
    count = 1000
    mass = rng.uniform(50000, 70000, count)
    tas = rng.uniform(380, 470, count)
    altitude = rng.uniform(28000, 39000, count)
    vs = rng.uniform(-1500, 1500, count)
    gs = tas * np.cos(np.arctan2(vs * fpm, tas * kts))
    expected = reference.estimate(gs, altitude, vs, airspeed=tas, mass=mass)
    actual = neural.enroute(mass, tas, altitude, vs)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    # Derivative check in scaled coordinates, at one point away from kinks.
    z = ca.MX.sym("z", 4)
    center = np.array([62000, 440, 33000, 0])
    scale = np.array([10000, 100, 10000, 1000])
    inputs = center + ca.times(scale, z)
    y = neural.enroute(*ca.vertsplit(inputs))
    derivative = ca.Function("check", [z], [y, ca.gradient(y, z), ca.hessian(y, z)[0]])
    _, grad, hess = derivative(np.zeros(4))
    numeric_grad, numeric_hess = [], []
    step = 1e-5
    for direction in np.eye(4):
        yp, gp, _ = derivative(direction * step)
        ym, gm, _ = derivative(-direction * step)
        numeric_grad.append(float(yp - ym) / (2 * step))
        numeric_hess.append(np.asarray((gp - gm) / (2 * step)).ravel())
    np.testing.assert_allclose(
        np.asarray(grad).ravel(), numeric_grad, rtol=1e-5, atol=1e-7
    )
    np.testing.assert_allclose(
        np.asarray(hess), np.array(numeric_hess).T, rtol=1e-4, atol=1e-7
    )
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    report = {
        "acropole_commit": commit,
        "model_sha256": neural.model_sha256,
        "params_sha256": neural.params_sha256,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("casadi", "onnx", "onnxruntime", "openap", "opentop", "numpy")
        },
        "validation": {
            "samples": count,
            "seed": 20260918,
            "max_absolute_error_kg_s": float(np.max(np.abs(actual - expected))),
            "max_relative_error": float(
                np.max(np.abs(actual - expected) / np.abs(expected))
            ),
            "scaled_gradient_max_error": float(
                np.max(np.abs(np.asarray(grad).ravel() - numeric_grad))
            ),
            "scaled_hessian_max_error": float(
                np.max(np.abs(np.asarray(hess) - np.array(numeric_hess).T))
            ),
            "import_seconds": import_seconds,
        },
        "experiment": {
            "aircraft": actype,
            "force_model": args.performance_model,
            "bada_path": str(args.bada_path) if args.bada_path else None,
            "origin": "EHAM",
            "destination": "EDDF",
            "m0_fraction": 0.85,
            "altitude_bounds_ft": [30000, 37000],
            "constant_altitude": True,
            "constant_mach": True,
            "wind": False,
            "speed_derivatives": "zero",
            "nodes": args.nodes,
            "polydeg": 3,
            "hessian": args.hessian,
            "max_iter": 800,
        },
        "solves": {},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in ((args.performance_model, None), ("acropole", neural)):
        optimizer = top.Cruise(
            actype,
            "EHAM",
            "EDDF",
            m0=0.85,
            fuel_model=model,
            performance_model=args.performance_model,
            bada_path=str(args.bada_path) if args.bada_path else None,
            h_min=30000 * ft,
            h_max=37000 * ft,
        )
        optimizer.setup(
            nodes=args.nodes,
            max_iter=800,
            ipopt_kwargs={"hessian_approximation": args.hessian},
        )
        optimizer.fix_cruise_altitude()
        optimizer.fix_mach_number()
        start = perf_counter()
        result = optimizer.trajectory(objective="fuel", result_object=True)
        elapsed = perf_counter() - start
        result.df.to_csv(args.output_dir / f"{name}.csv", index=False)
        report["solves"][name] = {
            "success": result.success,
            "status": result.status,
            "iterations": result.iters,
            "build_and_solve_seconds": elapsed,
            "fuel_kg": result.fuel,
            "objective_kg": result.objective,
            "altitude_ft": float(result.df.altitude.iloc[0]),
            "mach": float(result.df.mach.iloc[0]),
            "duration_s": float(result.df.ts.iloc[-1]),
        }
        print(name, json.dumps(report["solves"][name]), flush=True)
    output = args.output_dir / "report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {output}")
    if not all(solve["success"] for solve in report["solves"].values()):
        raise SystemExit(
            "One or more solves failed; inspect report before using results."
        )


if __name__ == "__main__":
    main()
