"""Tests for the Descent trajectory optimizer."""

import casadi as ca
import pytest

import opentop as top


@pytest.fixture(scope="module")
def descent_optimizer(aircraft_type, medium_flight):
    return top.Descent(
        aircraft_type,
        medium_flight["origin"],
        medium_flight["destination"],
        medium_flight["m0"],
    )


@pytest.fixture(scope="module")
def descent_clipped_df(descent_optimizer):
    return descent_optimizer.trajectory(objective="fuel")


@pytest.fixture(scope="module")
def descent_full_df(descent_optimizer):
    return descent_optimizer.trajectory(objective="fuel", remove_cruise=False)


@pytest.fixture(scope="module")
def descent_alt_start_df(descent_optimizer):
    return descent_optimizer.trajectory(
        objective="fuel", alt_start=30000, remove_cruise=False
    )


class TestDescent:
    def test_valid_trajectory(self, descent_clipped_df):
        df = descent_clipped_df
        assert df is not None
        assert len(df) > 0
        for col in ("altitude", "heading", "vertical_rate"):
            assert col in df.columns

    def test_remove_cruise_clips(self, descent_clipped_df, descent_full_df):
        assert len(descent_clipped_df) <= len(descent_full_df)
        assert (descent_clipped_df.vertical_rate < -100).all()

    def test_remove_cruise_false_includes_cruise(self, descent_full_df):
        assert (descent_full_df.vertical_rate.abs() < 100).any()

    def test_ends_low(self, descent_full_df):
        assert descent_full_df.altitude.iloc[-1] < 1000

    def test_alt_start(self, descent_alt_start_df):
        assert abs(descent_alt_start_df.altitude.iloc[0] - 30000) < 500

    def test_heading_reasonable(self, descent_full_df):
        df = descent_full_df
        assert df.heading.max() - df.heading.min() < 30

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
