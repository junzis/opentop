"""Waypoint-constrained cruise optimization example.

Run from the repository root with:

    uv run python examples/waypoints.py
"""

from __future__ import annotations

import openap

import opentop as top
import pandas as pd

FIX_NAMES = ("ARNEM", "KENUM", "DODEN", "BOMBI")


def openap_waypoints(fix_names: tuple[str, ...]) -> list[tuple[float, float]]:
    """Load named fixes from OpenAP's packaged navigation database."""
    waypoints = []
    for fix_name in fix_names:
        lat, lon, _ = openap.nav.fix(fix_name)
        waypoints.append((float(lat), float(lon)))
    return waypoints


def main() -> None:
    opt = top.Cruise("A320", "EHAM", "EDDF", 0.85)
    waypoints = openap_waypoints(FIX_NAMES)

    flight = opt.trajectory(
        objective="fuel",
        waypoints=waypoints,
        waypoint_tolerance_m=20_000,
    )
    assert isinstance(flight, pd.DataFrame)

    print("Route: EHAM -> " + " -> ".join(FIX_NAMES) + " -> EDDF")
    for fix_name, waypoint in zip(FIX_NAMES, waypoints):
        distances = [
            openap.aero.distance(lat, lon, waypoint[0], waypoint[1])
            for lat, lon in zip(flight.latitude, flight.longitude)
        ]
        print(f"{fix_name}: closest trajectory point {min(distances) / 1000:.1f} km")
    print(f"Fuel burn: {flight.mass.iloc[0] - flight.mass.iloc[-1]:.1f} kg")
    print(f"Elapsed time: {flight.ts.iloc[-1] / 60:.1f} min")


if __name__ == "__main__":
    main()
