#!/usr/bin/env python
"""Benchmark centralized roundabout conflict cases.

Usage:
    uv run python tests/benchmark_multi_aircraft.py --aircraft 2 4 6 8
"""

from __future__ import annotations

import argparse
import json
import time

import openap

import opentop


def _flights(count: int, nodes: int, fixed_altitude: bool):
    center_lat, center_lon = 51.0, 7.0
    radius_m = 200_000.0
    flights = []
    for index in range(count):
        bearing = 360.0 * index / count
        origin = openap.aero.latlon(center_lat, center_lon, radius_m, bearing)
        destination = openap.aero.latlon(
            center_lat, center_lon, radius_m, (bearing + 180.0) % 360.0
        )
        optimizer = opentop.Cruise(
            "A320",
            (float(origin[0]), float(origin[1])),
            (float(destination[0]), float(destination[1])),
            0.8,
        )
        optimizer.setup(nodes=nodes, max_iter=3000)
        if fixed_altitude:
            optimizer.fix_cruise_altitude()
        flights.append(opentop.FlightSpec(f"AC{index}", optimizer))
    return flights


def _run(count: int, nodes: int, fixed_altitude: bool, hessian: str) -> dict:
    started = time.perf_counter()
    result = opentop.MultiAircraft(
        _flights(count, nodes, fixed_altitude),
        max_iter=5000,
        ipopt_kwargs={"hessian_approximation": hessian},
    ).trajectory()
    return {
        "aircraft": count,
        "nodes": nodes,
        "fixed_altitude": fixed_altitude,
        "hessian": hessian,
        "success": result.success,
        "status": result.status,
        "elapsed_s": time.perf_counter() - started,
        "build_time_s": result.build_time_s,
        "solve_time_s": result.solve_time_s,
        "verification_time_s": result.verification_time_s,
        "iterations": result.stats.get("iter_count"),
        "nlp_variables": result.nlp_variables,
        "nlp_constraints": result.nlp_constraints,
        "separation_constraints": result.separation_constraints,
        "refinement_rounds": result.refinement_rounds,
        "minimum_metric": min(
            (report.minimum_metric for report in result.pair_reports),
            default=float("inf"),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aircraft", nargs="+", type=int, default=[2, 4, 6, 8])
    parser.add_argument("--nodes", type=int, default=20)
    parser.add_argument("--fixed-altitude", action="store_true")
    parser.add_argument(
        "--hessian", choices=("exact", "limited-memory"), default="limited-memory"
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    rows = []
    for count in args.aircraft:
        row = _run(count, args.nodes, args.fixed_altitude, args.hessian)
        rows.append(row)
        print(json.dumps(row), flush=True)
    output = json.dumps(rows, indent=2)
    print(output)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(output + "\n")


if __name__ == "__main__":
    main()
