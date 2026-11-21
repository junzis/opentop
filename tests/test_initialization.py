"""Analytic checks for seed geometry, mass integration and endpoint controls."""

from types import SimpleNamespace

import casadi as ca
import pytest
from openap.aero import fpm, ft

import numpy as np
import pandas as pd
from opentop import _dynamics, tools
from opentop._initialization import (
    _wind_function,
    build_coherent_pulse_seed,
    build_full_flight_seed,
)


class ConstantWind:
    def __init__(self, u=0.0, v=0.0):
        self.u, self.v = u, v

    def calc_u(self, x, y, h, ts):
        return 0 * x + self.u

    def calc_v(self, x, y, h, ts):
        return 0 * y + self.v


class AnalyticFuel:
    def enroute(self, mass, tas, altitude, vertical_rate, dT=0):
        return 0.7 + 1e-5 * mass


def optimizer(distance=2_000_000.0, wind=None, nodes=40, ceiling=12_000.0):
    bearing = 0.7
    lower = [0.1, -2500 * fpm, bearing - np.pi / 2]
    upper = [0.82, 2500 * fpm, bearing + np.pi / 2]
    opt = SimpleNamespace(
        nodes=nodes,
        aircraft={"cruise": {"height": 10_000.0, "mach": 0.78}},
        x_0_lb=[0, 0, 100 * ft, 70_000, 0],
        x_0_ub=[0, 0, 100 * ft, 70_000, 0],
        x_f_lb=[
            distance * np.sin(bearing),
            distance * np.cos(bearing),
            100 * ft,
            35_000,
            0,
        ],
        x_f_ub=[
            distance * np.sin(bearing),
            distance * np.cos(bearing),
            100 * ft,
            65_000,
            36_000,
        ],
        x_lb=[-3_000_000, -3_000_000, 100 * ft, 30_000, 0],
        x_ub=[3_000_000, 3_000_000, ceiling, 70_000, 36_000],
        u_lb=lower,
        u_ub=upper,
        u_0_lb=[0.1, 500 * fpm, bearing],
        u_0_ub=[0.3, 2500 * fpm, bearing],
        u_f_lb=[0.1, -1500 * fpm, bearing],
        u_f_ub=[0.3, -300 * fpm, bearing],
        u_guess=[0.6, 1000 * fpm, bearing],
        mass_init=70_000.0,
        dT=0.0,
        wind=wind,
        fuelflow=AnalyticFuel(),
        MAX_VERTICAL_ACCELERATION=5 * fpm,
        _cruise_mach_min=lambda: None,
    )
    x, u, dt = ca.MX.sym("x", 5), ca.MX.sym("u", 3), ca.MX.sym("dt")
    opt.func_dynamics = ca.Function(
        "seed_test_f",
        [x, u, dt],
        [
            _dynamics.xdot(x, u, fuelflow=opt.fuelflow, dT=0.0, wind=wind),
            dt * 0,
        ],
    )
    return opt


ROOTS = tuple(ca.collocation_points(3, "legendre"))


def seed(opt, **kwargs):
    return build_full_flight_seed(opt, (13, 27), collocation_roots=ROOTS, **kwargs)


def test_default_seed_phase_controls_and_exact_vertical_interpolant():
    opt = optimizer()
    before = (list(opt.x_0_lb), list(opt.u_guess))
    result = seed(opt)
    assert result.dts.sum() != pytest.approx(7200.0)
    assert np.all(result.U[1, :13] > 0)
    assert np.all(result.U[1, 13:28] == 0)
    assert np.all(result.U[1, 28:] < 0)
    assert 0.1 <= result.U[0, 0] <= 0.3
    assert 0.1 <= result.U[0, -1] <= 0.3
    assert result.U[1, 0] >= 500 * fpm
    assert -1500 * fpm <= result.U[1, -1] <= -300 * fpm
    np.testing.assert_allclose(result.X[:3, 0], opt.x_0_lb[:3])
    np.testing.assert_allclose(result.X[:3, -1], opt.x_f_lb[:3], atol=1e-8)
    for k, dt in enumerate(result.dts):
        h0, v0, v1 = result.X[2, k], result.U[1, k], result.U[1, k + 1]
        np.testing.assert_allclose(
            result.X[2, k + 1], h0 + dt * (v0 + v1) / 2, atol=1e-8
        )
        for j, tau in enumerate(ROOTS):
            np.testing.assert_allclose(
                result.Xc[k, 2, j], h0 + dt * (v0 * tau + (v1 - v0) * tau**2 / 2)
            )
            np.testing.assert_allclose(result.Xc[k, 4, j], result.X[4, k] + dt * tau)
    assert before == (opt.x_0_lb, opt.u_guess)
    assert result.diagnostics["seed_only"]
    assert result.diagnostics["new_constraints"] is False


def test_model_mass_integration_matches_analytic_solution_without_landing_clip():
    opt = optimizer()
    opt.x_f_ub[3] = 40_000
    result = seed(opt)
    expected = (70_000 + 70_000) * np.exp(-1e-5 * result.X[4]) - 70_000
    np.testing.assert_allclose(result.X[3], expected, atol=2e-7)
    xc_expected = 140_000 * np.exp(-1e-5 * result.Xc[:, 4]) - 70_000
    np.testing.assert_allclose(result.Xc[:, 3], xc_expected, atol=2e-7)
    assert np.all(np.diff(result.X[3]) < 0)
    assert result.X[3, -1] > opt.x_f_ub[3]
    assert result.diagnostics["landing_mass_bound_residual_kg"] > 0


def test_duration_tracks_distance_and_along_route_wind():
    bearing = 0.7
    along = np.array([np.sin(bearing), np.cos(bearing)])
    still = seed(optimizer())
    shorter = seed(optimizer(distance=1_500_000))
    tail = seed(optimizer(wind=ConstantWind(*(30 * along))))
    head = seed(optimizer(wind=ConstantWind(*(-30 * along))))
    assert shorter.dts.sum() < still.dts.sum()
    assert tail.dts.sum() < still.dts.sum() < head.dts.sum()
    assert tail.diagnostics["wind_used"]


def test_crosswind_changes_interior_heading_and_preserves_fixed_endpoint_heading():
    opt = optimizer(wind=ConstantWind(u=30))
    result = seed(opt)
    assert result.U[2, 0] == opt.u_0_lb[2]
    assert result.U[2, -1] == opt.u_f_lb[2]
    assert result.U[2, 20] != pytest.approx(opt.u_guess[2])
    assert np.all(result.U[2] >= opt.u_lb[2])
    assert np.all(result.U[2] <= opt.u_ub[2])


def test_initial_absolute_time_preserved():
    opt = optimizer(wind=ConstantWind(10))
    opt.x_0_lb[4] = opt.x_0_ub[4] = 3600.0
    result = seed(opt)
    assert result.X[4, 0] == 3600
    np.testing.assert_allclose(np.diff(result.X[4]), result.dts)


def test_real_polywind_batch_matches_scalar_api_and_uses_absolute_time():
    rows = [
        dict(longitude=lon, latitude=lat, h=h, ts=ts, u=10 + 0.0002 * ts, v=0.0)
        for lon in (0.0, 1.0, 2.0)
        for lat in (0.0, 1.0, 2.0)
        for h in (0.0, 6000.0, 12000.0)
        for ts in (0.0, 18000.0, 36000.0)
    ]
    wind = tools.PolyWind(
        pd.DataFrame(rows), lambda lon, lat: (lon * 1e6, lat * 1e6), 0, 0, 2, 2
    )
    points = np.array(
        [[1e5, 6e5, 1.2e6], [2e5, 8e5, 1.5e6], [100, 4000, 10000], [3600, 7200, 20000]],
        dtype=float,
    )
    expected = np.array(
        [[wind.calc_u(*point), wind.calc_v(*point)] for point in points.T]
    ).T
    np.testing.assert_allclose(
        np.asarray(_wind_function(wind)(points)), expected, rtol=1e-12, atol=1e-12
    )
    early = seed(optimizer(wind=wind))
    opt = optimizer(wind=wind)
    opt.x_0_lb[4] = opt.x_0_ub[4] = 3600
    late = seed(opt)
    assert late.X[4, 0] == 3600
    assert late.dts.sum() < early.dts.sum()
    assert late.diagnostics["wind_used"]
    assert np.isfinite(late.X).all()


def test_duration_and_altitude_bounds_are_used_for_variable_mesh_seed():
    opt = optimizer(ceiling=6_000)
    result = seed(opt, variable_timestep=True, dt_min=250, dt_max=300)
    assert np.all((result.dts >= 250) & (result.dts <= 300))
    assert result.X[2].max() <= 6_000 + 1e-8
    with pytest.raises(ValueError, match="duration bounds"):
        seed(opt, variable_timestep=True, dt_min=400, dt_max=300)


def test_invalid_route_or_wind_rejected():
    with pytest.raises(ValueError, match="route length"):
        seed(optimizer(distance=0))
    with pytest.raises(ValueError, match="Wind is nonfinite"):
        seed(optimizer(wind=ConstantWind(u=float("nan"))))


def test_extreme_crosswind_guard_is_reported():
    result = seed(optimizer(wind=ConstantWind(u=500)))
    assert result.diagnostics["wind_speed_guard_used"]
    assert np.isfinite(result.X).all()


def one_pulse(first, last, capacity=None):
    if not capacity:
        return [list(range(first, last))], []
    middle = (first + last - 3) // 2
    return [list(range(first, middle)), list(range(middle + 3, last))], [
        (middle, middle + 1, middle + 2)
    ]


def test_pulse_seed_respects_rate_choice_and_does_not_disable_actual_wind():
    opt = optimizer(wind=ConstantWind(u=10))
    transcription = SimpleNamespace(
        interval_dts=[1] * opt.nodes, collocation_roots=ROOTS
    )
    result = build_coherent_pulse_seed(
        opt,
        transcription,
        13,
        27,
        one_pulse,
        lambda first, last, duration, minimum_ramp_s: (1, {}),
        maximum_capacity=1,
        seed_amplitude_fpm=250,
    )
    states, controls, xc, dts, info = result
    assert opt.wind.u == 10
    assert info["actual_optimizer_wind_enabled"]
    assert not info["wind_used_in_seed"]
    assert info["seed_amplitude_fpm"] == 250
    assert np.max(controls[1, 13:28]) / fpm == pytest.approx(250)
    np.testing.assert_allclose(states[:3, -1], opt.x_f_lb[:3], atol=1e-3)
    assert np.all((dts >= 30) & (dts <= 900))
    assert np.isfinite(xc).all()


def test_still_air_pulse_policy_has_same_seed_with_wind_enabled():
    arrays = []
    for wind in (None, ConstantWind(u=10)):
        opt = optimizer(wind=wind)
        transcription = SimpleNamespace(
            interval_dts=[1] * opt.nodes, collocation_roots=ROOTS
        )
        arrays.append(
            build_coherent_pulse_seed(
                opt,
                transcription,
                13,
                27,
                one_pulse,
                lambda first, last, duration, minimum_ramp_s: (1, {}),
                maximum_capacity=1,
            )
        )
    for dry, windy in zip(arrays[0][:4], arrays[1][:4]):
        np.testing.assert_array_equal(dry, windy)


def test_zero_capacity_pulse_seed_preserves_mass_guess_and_clock():
    opt = optimizer()
    opt.x_0_lb[3] = 60_000
    opt.x_0_lb[4] = opt.x_0_ub[4] = 3600
    transcription = SimpleNamespace(
        interval_dts=[1] * opt.nodes, collocation_roots=ROOTS
    )
    states, controls, _, _, info = build_coherent_pulse_seed(
        opt,
        transcription,
        13,
        27,
        one_pulse,
        lambda first, last, duration, minimum_ramp_s: (0, {}),
        maximum_capacity=0,
    )
    assert info["capacity"] == 0
    assert states[3, 0] == opt.mass_init
    assert states[4, 0] == 3600
    assert np.all(controls[1, 13:28] == 0)
    np.testing.assert_allclose(np.ptp(states[2, 13:28]), 0, atol=1e-8)


def test_pulse_seed_honors_aircraft_mach_and_altitude_limits():
    opt = optimizer(distance=1_500_000, ceiling=5000)
    opt.u_0_lb[0] = opt.u_f_lb[0] = 0.3
    opt.u_ub[0] = 0.7
    opt._cruise_mach_min = lambda: 0.65
    opt._mach_change_limit = lambda: 0.08
    transcription = SimpleNamespace(
        interval_dts=[1] * opt.nodes, collocation_roots=ROOTS
    )
    states, controls, _, _, _ = build_coherent_pulse_seed(
        opt,
        transcription,
        13,
        27,
        one_pulse,
        lambda first, last, duration, minimum_ramp_s: (1, {}),
        maximum_capacity=1,
    )
    assert np.max(states[2]) <= 5000 + 1e-8
    assert controls[0, 0] == 0.3
    assert controls[0, -1] == 0.3
    assert np.all(controls[0, 13:28] >= 0.65)
    assert np.max(controls[0]) <= 0.7
    assert np.max(np.abs(np.diff(controls[0]))) <= 0.08
