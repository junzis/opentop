"""Tests for the Climb trajectory optimizer."""

from typing import cast

import pytest

import numpy as np
import opentop as top
import pandas as pd


@pytest.fixture(scope="module")
def climb_optimizer(aircraft_type, medium_flight):
    return top.Climb(
        aircraft_type,
        medium_flight["origin"],
        medium_flight["destination"],
        medium_flight["m0"],
    )


@pytest.fixture(scope="module")
def climb_clipped_df(climb_full_df):
    return climb_full_df.query("vertical_rate > 100")


@pytest.fixture(scope="module")
def climb_full_df(climb_optimizer):
    df = climb_optimizer.trajectory(objective="fuel", remove_cruise=False)
    assert climb_optimizer.success, climb_optimizer.stats
    return df


@pytest.fixture(scope="module")
def climb_alt_stop_df(climb_optimizer):
    df = climb_optimizer.trajectory(
        objective="fuel", alt_stop=30000, remove_cruise=False
    )
    assert climb_optimizer.success, climb_optimizer.stats
    return df


@pytest.fixture(scope="module")
def climb_alt_stop_low_df(climb_optimizer):
    df = climb_optimizer.trajectory(
        objective="fuel", alt_stop=25000, remove_cruise=False
    )
    assert climb_optimizer.success, climb_optimizer.stats
    return df


class TestClimb:
    def test_valid_trajectory(self, climb_clipped_df):
        df = climb_clipped_df
        assert df is not None
        assert len(df) > 0
        for col in ("altitude", "heading", "vertical_rate"):
            assert col in df.columns

    def test_altitude_increases(self, climb_clipped_df):
        assert climb_clipped_df.altitude.iloc[-1] > climb_clipped_df.altitude.iloc[0]

    def test_remove_cruise_false_includes_cruise(self, climb_full_df):
        assert (climb_full_df.vertical_rate.abs() < 100).any()

    def test_alt_stop(self, climb_alt_stop_df):
        assert abs(climb_alt_stop_df.altitude.max() - 30000) < 500

    def test_alt_stop_vs_default(self, climb_full_df, climb_alt_stop_low_df):
        assert climb_alt_stop_low_df.altitude.max() < climb_full_df.altitude.max()

    def test_turn_rate_within_limit(self, climb_clipped_df):
        heading = np.unwrap(np.deg2rad(climb_clipped_df.heading.to_numpy()))
        turn_rate = np.diff(heading) / np.diff(climb_clipped_df.ts.to_numpy())
        assert np.max(np.abs(turn_rate)) <= np.deg2rad(0.5) + 1e-6

    def test_mass_decreases(self, climb_clipped_df):
        df = climb_clipped_df
        assert df.mass.iloc[-1] < df.mass.iloc[0]

    def test_fuel_cost_column(self, climb_clipped_df):
        df = climb_clipped_df
        assert "fuel_cost" in df.columns
        assert (df["fuel_cost"].dropna() >= 0).all()


@pytest.mark.parametrize("remove_cruise", [False, True])
def test_remove_cruise_filters_threshold(monkeypatch, remove_cruise):
    opt = top.Climb("A320", "EHAM", "EDDF", 0.85)
    rates = [-101.0, -100.0, 0.0, 100.0, 101.0]
    solved = pd.DataFrame({"vertical_rate": rates})
    xp, yp = opt.proj(np.array([opt.lon1, opt.lon2]), np.array([opt.lat1, opt.lat2]))
    cruise = pd.DataFrame(
        {
            "x": xp,
            "y": yp,
            "h": [9000.0] * 2,
            "mach": [0.75] * 2,
            "mass": [65000.0] * 2,
        }
    )
    monkeypatch.setattr(opt, "_solve", lambda *args, **kwargs: solved)
    result = opt.trajectory(df_cruise=cruise, remove_cruise=remove_cruise)
    expected = solved.iloc[[4]] if remove_cruise else solved
    pd.testing.assert_frame_equal(result, expected)


def test_climb_north_south_route_converges():
    """Zero projected dx must work in the terminal collinearity constraint."""
    opt = top.Climb("A320", (48.0, 5.0), (55.0, 5.0), m0=0.85)
    xp, yp = opt.proj(np.array([5.0, 5.0]), np.array([48.0, 55.0]))
    assert abs(xp[1] - xp[0]) < 1e-8
    cruise = pd.DataFrame(
        {
            "x": xp,
            "y": yp,
            "h": [9000.0] * 2,
            "mach": [0.75] * 2,
        }
    )
    df = opt.trajectory(objective="fuel", df_cruise=cruise, remove_cruise=False)
    assert opt.success, opt.stats
    assert isinstance(df, pd.DataFrame)
    df = cast(pd.DataFrame, df)
    assert abs(df.x.iloc[-1] - xp[0]) < 1e-4
    assert df.y.iloc[-1] > df.y.iloc[0]
    assert df.h.iloc[-1] == pytest.approx(9000.0, abs=1e-3)
