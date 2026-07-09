"""Tests for CompleteFlight trajectory optimizer."""

import pytest

import opentop as top


@pytest.fixture(scope="module")
def complete_flight_df(aircraft_type, short_flight):
    optimizer = top.CompleteFlight(
        aircraft_type,
        short_flight["origin"],
        short_flight["destination"],
        short_flight["m0"],
    )
    return optimizer.trajectory(objective="fuel")


@pytest.fixture(scope="module")
def complete_flight_medium_df(aircraft_type, medium_flight):
    optimizer = top.CompleteFlight(
        aircraft_type,
        medium_flight["origin"],
        medium_flight["destination"],
        medium_flight["m0"],
    )
    return optimizer.trajectory(objective="fuel")


class TestCompleteFlight:
    def test_valid_trajectory(self, complete_flight_df):
        df = complete_flight_df
        assert df is not None
        assert len(df) > 0
        assert "altitude" in df.columns
        assert "heading" in df.columns

    def test_starts_and_ends_low(self, complete_flight_df):
        df = complete_flight_df
        assert df.altitude.iloc[0] < 1000
        assert df.altitude.iloc[-1] < 1000

    def test_climbs_to_cruise(self, complete_flight_df):
        assert complete_flight_df.altitude.max() > 20000

    def test_mass_decreases(self, complete_flight_df):
        df = complete_flight_df
        assert df.mass.iloc[-1] < df.mass.iloc[0]

    def test_heading_reasonable(self, complete_flight_df):
        df = complete_flight_df
        assert df.heading.max() - df.heading.min() < 30

    def test_fuel_cost_column(self, complete_flight_df):
        df = complete_flight_df
        assert "fuel_cost" in df.columns
        assert (df["fuel_cost"].dropna() >= 0).all()

    def test_medium_route(self, complete_flight_medium_df):
        df = complete_flight_medium_df
        assert df is not None
        assert len(df) > 0


def test_complete_flight_payload_makes_initial_mass_bounded(
    aircraft_type, short_flight
):
    payload = 10_000.0

    with pytest.warns(UserWarning, match="m0 is used only as the initial mass guess"):
        opt = top.CompleteFlight(
            aircraft_type,
            short_flight["origin"],
            short_flight["destination"],
            short_flight["m0"],
            payload=payload,
        )
    opt.init_conditions()

    expected_min_mass = opt.oew + payload
    expected_max_mass = min(opt.aircraft["mtow"], expected_min_mass + opt.fuel_max)

    assert opt.mass_min == expected_min_mass
    assert opt.x_0_lb[3] == expected_min_mass
    assert opt.x_0_ub[3] == expected_max_mass
    assert opt.x_f_lb[3] == expected_min_mass
    assert opt.x_lb[3] == expected_min_mass
    assert opt.x_ub[3] == expected_max_mass
    assert expected_min_mass <= opt.x_guess[0, 3] <= expected_max_mass


def test_complete_flight_without_payload_keeps_initial_mass_fixed(
    aircraft_type, short_flight
):
    opt = top.CompleteFlight(
        aircraft_type,
        short_flight["origin"],
        short_flight["destination"],
        short_flight["m0"],
    )
    opt.init_conditions()

    assert opt.x_0_lb[3] == opt.mass_init
    assert opt.x_0_ub[3] == opt.mass_init
    assert opt.x_f_lb[3] == opt.oew * 0.5


def test_complete_flight_explicit_phase_nodes_shape_altitude_guess():
    opt = top.CompleteFlight("A320", "EHAM", "LIRF", m0=0.85)
    opt.setup(nodes=30)

    opt.init_conditions(climb_nodes=9, descent_nodes=8)

    assert opt._phase_node_indices(climb_nodes=9, descent_nodes=8) == (9, 22)
    assert opt.x_guess[0, 2] < 1_000
    assert opt.x_guess[9, 2] == opt.aircraft["cruise"]["height"]
    assert opt.x_guess[22, 2] == opt.aircraft["cruise"]["height"]
    assert opt.x_guess[-1, 2] < 1_000


def test_complete_flight_auto_setup_and_phase_nodes_are_dense():
    opt = top.CompleteFlight("A320", "EHAM", "LIRF", m0=0.85)

    opt.setup()

    assert opt.nodes == 30
    assert opt._phase_node_indices() == (10, 20)


def test_complete_flight_phase_nodes_fit_dense_short_route_mesh():
    opt = top.CompleteFlight("A320", "EHAM", "EDDF", m0=0.85)
    opt.setup(nodes=41)

    assert opt._phase_node_indices() == (13, 28)


@pytest.mark.parametrize(
    ("phase_kwargs", "expected_indices"),
    [
        ({"climb_nodes": 5}, (5, 20)),
        ({"descent_nodes": 5}, (10, 25)),
    ],
)
def test_complete_flight_partial_phase_node_override(phase_kwargs, expected_indices):
    opt = top.CompleteFlight("A320", "EHAM", "EDDF", m0=0.85)

    assert opt._phase_node_indices(**phase_kwargs) == expected_indices


def test_complete_flight_phase_nodes_must_leave_cruise_interval():
    opt = top.CompleteFlight("A320", "EHAM", "LIRF", m0=0.85)
    opt.setup(nodes=15)

    with pytest.raises(ValueError, match="leave at least one cruise interval"):
        opt.init_conditions(climb_nodes=8, descent_nodes=7)


def test_complete_flight_callable_objective():
    """Verify objective=callable end-to-end, pinning the `(x, u, dt, **kwargs) -> ca.MX`
    contract. Before Phase 3 changes objective dispatch, this ensures user-supplied
    callables keep working."""
    import opentop as top

    opt = top.CompleteFlight("A320", "EHAM", "EDDF", m0=0.85)
    opt.setup(max_iter=1200)

    def fuel_twice(x, u, dt, **kwargs):
        # Trivial callable: 2x fuel. Optimum path should match pure-fuel; scale differs.
        return 2.0 * opt.obj_fuel(x, u, dt)

    df = opt.trajectory(objective=fuel_twice)
    assert df is not None
    assert opt.success


def test_complete_flight_return_failed_returns_df_on_tight_fuel_budget():
    """When max_fuel is impossibly tight, the mass-violation or infeasibility path
    would normally return None. With return_failed=True, the function must return
    the partial DataFrame instead."""
    import opentop as top

    opt = top.CompleteFlight("A320", "EHAM", "EDDF", m0=0.85)
    opt.setup(max_iter=200)

    with pytest.warns(UserWarning):
        df = opt.trajectory(
            objective="fuel",
            max_fuel=100.0,  # physically impossible for EHAM→EDDF
            return_failed=True,
        )
    # Regardless of whether the solver fails outright or returns a degenerate
    # trajectory that violates mass constraints, return_failed=True must hand
    # back a DataFrame (not None).
    assert df is not None
    # Verify we actually hit the failure path (not a miraculous convergence).
    assert not opt.success, (
        "max_fuel=100 should be infeasible; if this passes, something is wrong "
        "with the solver or route"
    )
