"""Tests for the Cruise trajectory optimizer."""

import casadi as ca
import pytest
from openap.aero import ft

import numpy as np
import opentop as top
import pandas as pd


@pytest.fixture(scope="module")
def cruise_df(aircraft_type, short_flight):
    optimizer = top.Cruise(
        aircraft_type,
        short_flight["origin"],
        short_flight["destination"],
        short_flight["m0"],
    )
    df = optimizer.trajectory(objective="fuel")
    assert optimizer.success, optimizer.stats
    return df


@pytest.fixture(scope="module")
def cruise_time_df(aircraft_type, short_flight):
    optimizer = top.Cruise(
        aircraft_type,
        short_flight["origin"],
        short_flight["destination"],
        short_flight["m0"],
    )
    df = optimizer.trajectory(objective="time")
    assert optimizer.success, optimizer.stats
    return df


@pytest.fixture(scope="module")
def cruise_medium_df(aircraft_type, medium_flight):
    optimizer = top.Cruise(
        aircraft_type,
        medium_flight["origin"],
        medium_flight["destination"],
        medium_flight["m0"],
    )
    df = optimizer.trajectory(objective="fuel")
    assert optimizer.success, optimizer.stats
    return df


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

    def test_turn_rate_within_limit(self, cruise_df):
        heading = np.unwrap(np.deg2rad(cruise_df.heading.to_numpy()))
        turn_rate = np.diff(heading) / np.diff(cruise_df.ts.to_numpy())
        assert np.max(np.abs(turn_rate)) <= np.deg2rad(0.5) + 1e-6

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

    def test_vertical_rate_change_is_smooth(self, cruise_df):
        vertical_acceleration = (
            cruise_df.vertical_rate.diff() / cruise_df.ts.diff()
        ).dropna()
        assert vertical_acceleration.abs().max() <= 5.1


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


def test_cruise_terminal_performance_uses_shared_thrust_helper(
    monkeypatch, aircraft_type, short_flight
):
    opt = top.Cruise(
        aircraft_type,
        short_flight["origin"],
        short_flight["destination"],
        short_flight["m0"],
    )

    thrust_calls = []
    performance_calls = []
    original_thrust_climb = opt._thrust_climb
    original_constrain_clean_performance = opt._constrain_clean_performance

    def spy_thrust_climb(tas, alt):
        thrust_calls.append((tas, alt))
        return original_thrust_climb(tas, alt)

    def spy_constrain_clean_performance(opti, mass, tas, alt, thrust_max, **kwargs):
        performance_calls.append((mass, tas, alt, thrust_max))
        return original_constrain_clean_performance(
            opti, mass, tas, alt, thrust_max, **kwargs
        )

    class FakeSolution:
        def stats(self):
            return {"success": True}

    built = {}

    def fake_solve(X, U, **kwargs):
        built.update(X=X, U=U)
        opt._last_solution = FakeSolution()
        return pd.DataFrame(
            {
                "altitude": [30_000.0, 30_000.0],
                "mass": [opt.mass_init, opt.mass_init - 1.0],
            }
        )

    monkeypatch.setattr(opt, "_thrust_climb", spy_thrust_climb)
    monkeypatch.setattr(
        opt, "_constrain_clean_performance", spy_constrain_clean_performance
    )
    monkeypatch.setattr(opt, "_solve", fake_solve)

    opt.trajectory(objective="fuel")

    import casadi as ca

    expected = opt.nodes + 1 + opt.nodes * opt.polydeg
    assert len(thrust_calls) == len(performance_calls) == expected
    # The true terminal control is checked, as are interpolated interior controls.
    assert ca.depends_on(performance_calls[opt.nodes][1], built["U"][-1])
    assert ca.depends_on(performance_calls[opt.nodes][0], built["X"][-1])
    for k in range(opt.nodes):
        for j in range(opt.polydeg):
            tas = performance_calls[opt.nodes + 1 + k * opt.polydeg + j][1]
            assert ca.depends_on(tas, built["U"][k])
            assert ca.depends_on(tas, built["U"][k + 1])


def test_fuel_cost_sum_matches_mass_difference(cruise_df):
    mass_difference = cruise_df.mass.iloc[0] - cruise_df.mass.iloc[-1]
    assert cruise_df.fuel_cost.sum() == pytest.approx(mass_difference, abs=1e-8)


def test_cruise_formulation_preserves_supplied_initial_guess():
    opt = top.Cruise("A320", "EHAM", "EDDF", 0.85)
    opt.setup(nodes=2)
    guess = pd.DataFrame(
        {
            "longitude": [opt.lon1, (opt.lon1 + opt.lon2) / 2, opt.lon2],
            "latitude": [opt.lat1, (opt.lat1 + opt.lat2) / 2, opt.lat2],
            "altitude": [30000.0, 31000.0, 32000.0],
            "mass": [65000.0, 64800.0, 64500.0],
            "ts": [0.0, 800.0, 1900.0],
        }
    )
    problem = ca.Opti()
    tr = opt._add_formulation(problem, initial_guess=guess)
    x, y = opt.proj(guess.longitude.to_numpy(), guess.latitude.to_numpy())
    expected = np.column_stack([x, y, guess.altitude * ft, guess.mass, guess.ts])
    for state, row in zip(tr.X, expected):
        np.testing.assert_allclose(problem.debug.value(state, problem.initial()), row)
