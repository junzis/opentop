"""Tests for waypoint-constrained trajectories."""

import openap
import pytest

import opentop as top
import pandas as pd


def _midpoint_waypoint(opt):
    xp_0, yp_0 = opt.proj(opt.lon1, opt.lat1)
    xp_f, yp_f = opt.proj(opt.lon2, opt.lat2)
    lon, lat = opt.proj((xp_0 + xp_f) / 2, (yp_0 + yp_f) / 2, inverse=True)
    return lat, lon


def test_cruise_waypoint_solution_passes_near_waypoint(aircraft_type, short_flight):
    opt = top.Cruise(
        aircraft_type,
        short_flight["origin"],
        short_flight["destination"],
        short_flight["m0"],
    )
    waypoint = _midpoint_waypoint(opt)

    df = opt.trajectory(
        objective="fuel",
        waypoints=[waypoint],
        waypoint_tolerance_m=10_000,
    )
    assert isinstance(df, pd.DataFrame)

    distances = [
        openap.aero.distance(lat, lon, waypoint[0], waypoint[1])
        for lat, lon in zip(df.latitude, df.longitude)
    ]
    assert min(distances) <= 10_500
    assert df.ts.diff().dropna().gt(0).all()


def test_waypoint_variable_timestep_supports_time_objective(
    monkeypatch, aircraft_type, short_flight
):
    opt = top.Cruise(
        aircraft_type,
        short_flight["origin"],
        short_flight["destination"],
        short_flight["m0"],
    )
    waypoint = _midpoint_waypoint(opt)

    class FakeSolution:
        def stats(self):
            return {"success": True}

    def fake_solve(X, U, **kwargs):
        opt._last_solution = FakeSolution()
        return pd.DataFrame(
            {
                "altitude": [30_000.0, 30_000.0],
                "mass": [opt.mass_init, opt.mass_init - 1.0],
            }
        )

    monkeypatch.setattr(opt, "_solve", fake_solve)

    df = opt.trajectory(objective="time", waypoints=[waypoint])

    assert df is not None
    assert opt._variable_timestep is True
    assert len(opt._interval_dts) == opt.nodes


def test_waypoint_latitude_is_validated(aircraft_type, short_flight):
    opt = top.Cruise(
        aircraft_type,
        short_flight["origin"],
        short_flight["destination"],
        short_flight["m0"],
    )

    with pytest.raises(ValueError, match="latitude"):
        opt.trajectory(objective="fuel", waypoints=[(95.0, 0.0)])
