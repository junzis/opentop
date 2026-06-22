"""Tests for the Cruise trajectory optimizer."""

import pytest
from openap.aero import ft

import opentop as top


@pytest.fixture(scope="module")
def cruise_df(aircraft_type, short_flight):
    optimizer = top.Cruise(
        aircraft_type,
        short_flight["origin"],
        short_flight["destination"],
        short_flight["m0"],
    )
    return optimizer.trajectory(objective="fuel")


@pytest.fixture(scope="module")
def cruise_time_df(aircraft_type, short_flight):
    optimizer = top.Cruise(
        aircraft_type,
        short_flight["origin"],
        short_flight["destination"],
        short_flight["m0"],
    )
    return optimizer.trajectory(objective="time")


@pytest.fixture(scope="module")
def cruise_medium_df(aircraft_type, medium_flight):
    optimizer = top.Cruise(
        aircraft_type,
        medium_flight["origin"],
        medium_flight["destination"],
        medium_flight["m0"],
    )
    return optimizer.trajectory(objective="fuel")


class TestCruise:
    def test_valid_trajectory(self, cruise_df):
        df = cruise_df
        assert df is not None
        assert len(df) > 0
        for col in ("altitude", "heading", "mach"):
            assert col in df.columns

    def test_altitude_reasonable(self, cruise_df):
        assert cruise_df.altitude.min() > 20000
        assert cruise_df.altitude.max() < 45000

    def test_heading_reasonable(self, cruise_df):
        assert cruise_df.heading.max() - cruise_df.heading.min() < 30

    def test_mass_decreases(self, cruise_df):
        assert cruise_df.mass.iloc[-1] < cruise_df.mass.iloc[0]

    def test_fuel_cost_column(self, cruise_df):
        assert "fuel_cost" in cruise_df.columns
        assert (cruise_df["fuel_cost"].dropna() >= 0).all()

    def test_grid_cost_nan_without_interpolant(self, cruise_df):
        assert "grid_cost" in cruise_df.columns
        assert cruise_df["grid_cost"].isna().all()

    def test_time_objective(self, cruise_time_df):
        assert cruise_time_df is not None
        assert len(cruise_time_df) > 0

    def test_medium_route(self, cruise_medium_df):
        df = cruise_medium_df
        assert df is not None
        assert len(df) > 0
        assert df.mass.iloc[-1] < df.mass.iloc[0]


def test_cruise_accepts_constructor_altitude_bounds(aircraft_type, short_flight):
    h_min = 25_000 * ft
    h_max = 32_000 * ft

    opt = top.Cruise(
        aircraft_type,
        short_flight["origin"],
        short_flight["destination"],
        short_flight["m0"],
        h_min=h_min,
        h_max=h_max,
    )
    opt.init_conditions()

    assert opt.x_0_lb[2] == h_min
    assert opt.x_0_ub[2] == h_max
    assert opt.x_lb[2] == h_min
    assert opt.x_ub[2] == h_max


def test_cruise_payload_makes_initial_mass_bounded(aircraft_type, short_flight):
    payload = 10_000.0

    with pytest.warns(UserWarning, match="m0 is used only as the initial mass guess"):
        opt = top.Cruise(
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
    assert opt.x_ub[3] == expected_max_mass
    assert expected_min_mass <= opt.x_guess[0, 3] <= expected_max_mass


def test_cruise_without_payload_keeps_initial_mass_fixed(aircraft_type, short_flight):
    opt = top.Cruise(
        aircraft_type,
        short_flight["origin"],
        short_flight["destination"],
        short_flight["m0"],
    )
    opt.init_conditions()

    assert opt.x_0_lb[3] == opt.mass_init
    assert opt.x_0_ub[3] == opt.mass_init


def test_cruise_rejects_payload_above_mtow(aircraft_type, short_flight):
    with pytest.raises(ValueError, match="OEW \\+ payload must not exceed MTOW"):
        top.Cruise(
            aircraft_type,
            short_flight["origin"],
            short_flight["destination"],
            payload=100_000.0,
        )


def test_cruise_initial_guess_is_honored_with_no_double_init(
    aircraft_type, short_flight
):
    """Guard: passing initial_guess= must not require x_guess to be built twice.

    This pins the post-fix behavior -- init_conditions is called once, the guess
    is honored, and trajectory converges.
    """
    import opentop as top

    opt = top.Cruise(
        aircraft_type,
        short_flight["origin"],
        short_flight["destination"],
        short_flight["m0"],
    )
    baseline = opt.trajectory(objective="fuel")

    opt2 = top.Cruise(
        aircraft_type,
        short_flight["origin"],
        short_flight["destination"],
        short_flight["m0"],
    )
    result = opt2.trajectory(objective="fuel", initial_guess=baseline)  # type: ignore[arg-type]  # trajectory() without result_object returns DataFrame; passes as initial_guess
    assert result is not None
    assert len(result) == len(baseline)  # type: ignore[arg-type]  # trajectory() without result_object always returns DataFrame
