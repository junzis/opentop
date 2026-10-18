"""Saved warm starts must seed time and all optional controls consistently."""

import pytest
from openap.aero import fpm

import numpy as np
import opentop as top
import pandas as pd


def build_guess(times):
    opt = top.CompleteFlight("A320", "EHAM", "EDDF", 0.85)
    opt.setup(nodes=4)
    opt.init_conditions()
    heading = np.rad2deg(opt.u_guess[2])
    guess = pd.DataFrame(
        {
            "latitude": np.linspace(opt.lat1, opt.lat2, 5),
            "longitude": np.linspace(opt.lon1, opt.lon2, 5),
            "altitude": [100, 12000, 30000, 15000, 100],
            "mass": np.linspace(opt.mass_init, opt.mass_init - 1000, 5),
            "ts": times,
            "mach": [0.3, 0.6, 0.7, 0.6, 0.3],
            "vertical_rate": [1000, 0, -600, -1000, -900],
            "heading": (heading + np.array([0, 10, 20, 5, 0])) % 360,
        }
    )
    opt.x_guess = opt.initial_guess(guess)
    return opt, guess


def test_warm_start_controls_and_bounded_interval_times():
    opt, guess = build_guess([0, 10, 80, 160, 300])
    _, controls = opt._build_opti(
        "fuel", 7200, initial_guess=guess, variable_timestep=True, dt_min=20, dt_max=100
    )
    problem = opt._opti

    def evaluate(value):
        return problem.debug.value(value, problem.initial())

    np.testing.assert_allclose(
        [evaluate(dt) for dt in opt._interval_dts], [20, 70, 80, 100]
    )
    assert evaluate(opt.ts_final) == pytest.approx(270)
    expected_heading = np.deg2rad(guess.heading)
    psi = opt.u_guess[2]
    expected_heading = psi + (expected_heading - psi + np.pi) % (2 * np.pi) - np.pi
    for k, control in enumerate(controls):
        lower, upper = (
            (opt.u_0_lb, opt.u_0_ub)
            if k == 0
            else ((opt.u_f_lb, opt.u_f_ub) if k == 4 else (opt.u_lb, opt.u_ub))
        )
        expected = np.clip(
            [
                guess.mach.iloc[k],
                guess.vertical_rate.iloc[k] * fpm,
                expected_heading.iloc[k],
            ],
            lower,
            upper,
        )
        np.testing.assert_allclose(evaluate(control), expected)


@pytest.mark.parametrize("column", ["mach", "vertical_rate", "heading"])
def test_nonfinite_optional_control_is_rejected(column):
    opt, guess = build_guess([0, 10, 80, 160, 300])
    guess.loc[2, column] = np.nan
    with pytest.raises(ValueError, match=column):
        opt._build_opti("fuel", 7200, initial_guess=guess)


def test_nonincreasing_supplied_time_is_rejected():
    opt, guess = build_guess([0, 80, 80, 160, 300])
    with pytest.raises(ValueError, match="increasing"):
        opt._build_opti("fuel", 7200, initial_guess=guess)
