"""Regression for a waypoint route departing away from the direct OD bearing."""

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import opentop as top
from opentop.routes import route_initial_guess, validate_waypoint_trajectory
from pyproj import Geod


def test_complete_flight_with_southerly_departure_converges():
    route = json.loads(
        (Path(__file__).parent / "fixtures/rad_departure_route.json").read_text()
    )
    geod = Geod(ellps="WGS84")
    dense = [route[0]]
    for a, b in pairwise(route):
        distance = geod.inv(a[1], a[0], b[1], b[0])[2]
        intervals = int(np.ceil(distance / 25000))
        if intervals > 1:
            dense.extend(
                (lat, lon)
                for lon, lat in geod.npts(a[1], a[0], b[1], b[0], intervals - 1)
            )
        dense.append(b)
    opt = top.CompleteFlight("A320", dense[0], dense[-1], m0=65205 / 78000)
    nodes = len(dense) - 1
    opt.setup(nodes=nodes, ipopt_kwargs={"max_cpu_time": 120})
    guess = route_initial_guess(
        top.RouteOption("departure turn", tuple(map(tuple, dense))),
        nodes=nodes,
        altitude_ft=33000,
        mass_kg=65205,
        complete_flight=True,
    )
    guess["latitude"] = [p[0] for p in dense]
    guess["longitude"] = [p[1] for p in dense]
    result = opt.trajectory(
        objective="fuel",
        initial_guess=guess,
        waypoints=dense[1:-1],
        waypoint_node_indices=list(range(1, nodes)),
        waypoint_tolerance_m=2000,
        dt_min=1,
        dt_max=600,
        result_object=True,
    )
    assert result.success, result.status
    assert validate_waypoint_trajectory(
        result.df, dense[1:-1], tolerance_m=2100, node_indices=list(range(1, nodes))
    )
    assert (
        np.isfinite(result.df[["mass", "h", "ts", "latitude", "longitude"]]).all().all()
    )
    assert result.df.ts.diff().dropna().gt(0).all()
    assert abs(result.df.mass.iloc[0] - 65205) < 1
    assert result.df.altitude.max() >= 15000 - 1
    assert 0 < result.fuel < 65205
