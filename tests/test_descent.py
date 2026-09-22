"""Tests for the Descent trajectory optimizer."""

import casadi as ca
import pytest

import numpy as np
import opentop as top
import pandas as pd


@pytest.fixture(scope="module")
def descent_optimizer(aircraft_type, medium_flight):
    return top.Descent(
        aircraft_type,
        medium_flight["origin"],
        medium_flight["destination"],
        medium_flight["m0"],
    )


@pytest.fixture(scope="module")
def descent_clipped_df(descent_full_df):
    return descent_full_df.query("vertical_rate < -100")


@pytest.fixture(scope="module")
def descent_full_df(descent_optimizer):
    df = descent_optimizer.trajectory(objective="fuel", remove_cruise=False)
    assert descent_optimizer.success, descent_optimizer.stats
    return df


@pytest.fixture(scope="module")
def descent_alt_start_df(descent_optimizer):
    df = descent_optimizer.trajectory(
        objective="fuel", alt_start=30000, remove_cruise=False
    )
    assert descent_optimizer.success, descent_optimizer.stats
    return df


class TestDescent:
    def test_valid_trajectory(self, descent_clipped_df):
        df = descent_clipped_df
        assert df is not None
        assert len(df) > 0
        for col in ("altitude", "heading", "vertical_rate"):
            assert col in df.columns

    def test_remove_cruise_false_includes_cruise(self, descent_full_df):
        assert (descent_full_df.vertical_rate.abs() < 100).any()

    def test_ends_low(self, descent_full_df):
        assert descent_full_df.altitude.iloc[-1] < 1000

    def test_alt_start(self, descent_alt_start_df):
        assert abs(descent_alt_start_df.altitude.iloc[0] - 30000) < 500

    def test_turn_rate_within_limit(self, descent_full_df):
        heading = np.unwrap(np.deg2rad(descent_full_df.heading.to_numpy()))
        turn_rate = np.diff(heading) / np.diff(descent_full_df.ts.to_numpy())
        assert np.max(np.abs(turn_rate)) <= np.deg2rad(0.5) + 1e-6

    def test_mass_decreases(self, descent_full_df):
        df = descent_full_df
        assert df.mass.iloc[-1] < df.mass.iloc[0]

    def test_fuel_cost_column(self, descent_full_df):
        df = descent_full_df
        assert "fuel_cost" in df.columns
        assert (df["fuel_cost"].dropna() >= 0).all()

    def test_max_duration_must_be_positive(self, descent_optimizer, descent_full_df):
        with pytest.raises(ValueError, match="finite and positive"):
            descent_optimizer.init_conditions(descent_full_df, max_duration_s=0.0)

    def test_inbound_route_side_adds_one_constraint_per_interior_node(
        self, descent_optimizer
    ):
        descent_optimizer._route_xy = [(0.0, 0.0), (0.0, 10.0)]
        descent_optimizer._route_anchor_nodes = [0, 3, descent_optimizer.nodes]
        opti = ca.Opti()
        states = [opti.variable(5) for _ in range(descent_optimizer.nodes + 1)]

        descent_optimizer._constrain_inbound_route_side(
            opti, states, inbound_route_side="right"
        )

        assert opti.g.numel() == 2

    def test_inbound_route_side_rejects_unknown_side(self, descent_optimizer):
        opti = ca.Opti()
        with pytest.raises(ValueError, match="inbound_route_side"):
            descent_optimizer._constrain_inbound_route_side(
                opti, [], inbound_route_side="up"
            )


@pytest.mark.parametrize("remove_cruise", [False, True])
def test_remove_cruise_filters_threshold(monkeypatch, remove_cruise):
    opt = top.Descent("A320", "EHAM", "EDDF", 0.85)
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
    expected = solved.iloc[[0]] if remove_cruise else solved
    pd.testing.assert_frame_equal(result, expected)
