"""Automatic expansion stays within the measured compatible wind/fuel scope."""

import casadi as ca
import pytest

import numpy as np
import opentop as top
import pandas as pd
from opentop.tools import PolyWind


def optimizer(*, wind=True, expand=None):
    opt = top.Cruise("A320", "EHAM", "EDDF", m0=60000)
    if wind:
        frame = pd.DataFrame(
            [
                dict(longitude=lon, latitude=lat, h=h, ts=t, u=10.0, v=0.0)
                for lon in (0.0, 10.0)
                for lat in (48.0, 55.0)
                for h in (0.0, 12000.0)
                for t in (0.0, 18000.0)
            ]
        )
        opt.enable_wind(frame)
    opt.setup(nodes=3, expand=expand)
    opt.init_conditions()
    return opt


def build(opt, objective="fuel", **kwargs):
    opt._build_opti(objective, ts_final_guess=3600, **kwargs)
    return opt._effective_solver_options()


def test_automatic_wind_fuel_expansion_does_not_mutate_requested_options():
    opt = optimizer()
    requested = opt.solver_options.copy()
    assert build(opt)["expand"] is True
    assert opt.solver_options == requested
    assert "expand" not in opt.solver_options


@pytest.mark.parametrize("explicit", [False, True])
def test_setup_and_existing_dictionary_override_win(explicit):
    opt = optimizer(expand=explicit)
    assert build(opt)["expand"] is explicit
    opt.solver_options["expand"] = not explicit
    assert build(opt)["expand"] is not explicit


def test_complete_flight_forwards_explicit_expansion():
    opt = top.CompleteFlight("A320", "EHAM", "LGAV", m0=70000)
    opt.setup(nodes=4, expand=False)
    assert opt.solver_options["expand"] is False


def test_still_air_keeps_existing_solver_options():
    opt = optimizer(wind=False)
    requested = opt.solver_options.copy()
    assert build(opt) == requested


def test_rebuilding_with_custom_time_grid_or_no_wind_drops_automatic_default():
    opt = optimizer()
    assert build(opt)["expand"] is True
    assert "expand" not in build(opt, lambda x, u, dt, **kw: dt)
    assert "expand" not in build(opt, "time")
    grid = ca.interpolant("option_grid", "linear", [[0, 1]] * 3, np.zeros(8))
    assert "expand" not in build(opt, interpolant=grid, n_dim=3, time_dependent=False)
    assert build(opt)["expand"] is True
    opt.wind = None
    assert "expand" not in build(opt)


def test_follow_track_splines_are_excluded():
    opt = optimizer()
    opt.follow_track(
        [opt.lat1, 51.6, 50.8, opt.lat2],
        [opt.lon1, 6.0, 7.2, opt.lon2],
    )
    opt.init_conditions()
    assert opt.track_ref is not None
    assert "expand" not in build(opt)


def test_unproven_wind_subclasses_and_models_are_excluded(monkeypatch):
    class OtherWind(PolyWind):
        pass

    opt = optimizer()
    opt.wind.__class__ = OtherWind
    assert "expand" not in build(opt)
    opt = optimizer()
    opt.performance_model = "bada3"

    class StopBeforeModelBuild(Exception):
        pass

    def stop(*args, **kwargs):
        raise StopBeforeModelBuild

    # Test the eligibility gate without requiring licensed BADA tables.
    monkeypatch.setattr(opt, "init_model", stop)
    with pytest.raises(StopBeforeModelBuild):
        build(opt)
    assert "expand" not in opt._effective_solver_options()
