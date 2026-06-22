"""Tests for optional BADA performance model integration."""

import os

import casadi as ca
import pytest
from openap.aero import fpm, ft

import numpy as np
import opentop as top
from opentop._performance import build_numeric_fuelflow, build_performance_models

BADA3_PATH = os.environ.get("OPENTOP_BADA3_PATH", "/home/junzi/arc/data/bada_312")
BADA4_PATH = os.environ.get(
    "OPENTOP_BADA4_PATH", "/home/junzi/arc/data/bada_4.2/tables"
)


pytestmark = pytest.mark.skipif(
    not os.path.exists(BADA3_PATH) or not os.path.exists(BADA4_PATH),
    reason="BADA data paths are not available",
)


def test_build_native_performance_models():
    bundle = build_performance_models(
        "A320",
        engine=None,
        use_synonym=False,
        performance_model="openap",
        bada_path=None,
    )

    assert bundle.name == "openap"
    assert bundle.aircraft["aircraft"].startswith("Airbus A320")
    assert hasattr(bundle.fuelflow, "enroute")


def test_build_bada3_performance_models():
    bundle = build_performance_models(
        "A320",
        engine=None,
        use_synonym=False,
        performance_model="bada3",
        bada_path=BADA3_PATH,
    )

    assert bundle.name == "bada3"
    assert hasattr(bundle.thrust, "climb")
    assert hasattr(bundle.drag, "clean_drag_polar_params")
    assert hasattr(bundle.fuelflow, "enroute")


def test_build_bada4_performance_models():
    bundle = build_performance_models(
        "A320-214",
        engine=None,
        use_synonym=False,
        performance_model="bada4",
        bada_path=BADA4_PATH,
    )

    assert bundle.name == "bada4"
    assert hasattr(bundle.thrust, "takeoff")
    assert hasattr(bundle.drag, "clean_drag_polar_params")
    assert hasattr(bundle.fuelflow, "enroute")


def test_bada4_domain_guard_constrains_positive_fuel_flow():
    opt = top.Cruise(
        "A320-214",
        "EHAM",
        "EDDF",
        0.85,
        performance_model="bada4",
        bada_path=BADA4_PATH,
    )
    opt._opti = ca.Opti()
    opt._constrain_performance_model_domain(opt._opti.variable(5))

    assert opt._opti.g.shape == (1, 1)


def test_bada4_complete_flight_uses_clean_mach_domain():
    opt = top.CompleteFlight(
        "A320-214",
        (0.0, 0.0),
        (0.0, 1.0),
        0.85,
        performance_model="bada4",
        bada_path=BADA4_PATH,
    )
    opt.init_conditions()

    assert opt.u_0_lb[0] == 0.3
    assert opt.u_f_lb[0] == 0.3
    assert opt.u_lb[0] == 0.3


def test_bada4_complete_flight_tightens_cruise_vertical_rate_domain():
    bada4_opt = top.CompleteFlight(
        "A320-214",
        (0.0, 0.0),
        (0.0, 1.0),
        0.85,
        performance_model="bada4",
        bada_path=BADA4_PATH,
    )
    openap_opt = top.CompleteFlight("A320", (0.0, 0.0), (0.0, 1.0), 0.85)

    assert bada4_opt._cruise_vertical_rate_limit() == pytest.approx(100 * fpm)
    assert openap_opt._cruise_vertical_rate_limit() == pytest.approx(500 * fpm)


def test_bada4_complete_flight_uses_procedural_cruise_speed_domain():
    bada4_opt = top.CompleteFlight(
        "A320-214",
        (0.0, 0.0),
        (0.0, 1.0),
        0.85,
        performance_model="bada4",
        bada_path=BADA4_PATH,
    )
    openap_opt = top.CompleteFlight("A320", (0.0, 0.0), (0.0, 1.0), 0.85)

    assert bada4_opt._cruise_mach_min() == 0.72
    assert bada4_opt._mach_change_limit() == 0.08
    assert openap_opt._cruise_mach_min() is None
    assert openap_opt._mach_change_limit() == 0.2


def test_openap_domain_guard_does_not_add_constraints():
    opt = top.Cruise("A320", "EHAM", "EDDF", 0.85)
    opt._opti = ca.Opti()
    opt._constrain_performance_model_domain(opt._opti.variable(5))

    assert opt._opti.g.shape == (0, 1)


def test_bada_model_requires_bada_path():
    with pytest.raises(ValueError, match="bada_path"):
        build_performance_models(
            "A320",
            engine=None,
            use_synonym=False,
            performance_model="bada3",
            bada_path=None,
        )


def test_cruise_constructor_accepts_bada3_model():
    opt = top.Cruise(
        "A320",
        "EHAM",
        "EDDF",
        0.85,
        performance_model="bada3",
        bada_path=BADA3_PATH,
    )

    assert opt.performance_model == "bada3"
    assert hasattr(opt.fuelflow, "enroute")


@pytest.mark.parametrize(
    "phase_cls", [top.Cruise, top.CompleteFlight, top.Climb, top.Descent]
)
def test_phase_constructors_accept_bada3_model(phase_cls):
    opt = phase_cls(
        "A320",
        "EHAM",
        "EDDF",
        0.85,
        performance_model="bada3",
        bada_path=BADA3_PATH,
    )

    assert opt.performance_model == "bada3"
    if hasattr(opt, "cruise"):
        assert opt.cruise.performance_model == "bada3"


@pytest.mark.parametrize(
    ("actype", "performance_model", "bada_path"),
    [
        ("A320", "bada3", BADA3_PATH),
        ("A320-214", "bada4", BADA4_PATH),
    ],
)
def test_cruise_fuel_optimization_with_bada_model(actype, performance_model, bada_path):
    opt = top.Cruise(
        actype,
        "EHAM",
        "EDDF",
        0.85,
        performance_model=performance_model,
        bada_path=bada_path,
        h_min=25000 * ft,
        h_max=39000 * ft,
    )
    opt.nodes = 8
    opt.polydeg = 3

    df = opt.trajectory(objective="fuel", return_failed=True)
    assert df is not None

    expected_fuelflow = build_numeric_fuelflow(
        actype,
        engtype=opt.engtype,
        use_synonym=opt.use_synonym,
        performance_model=performance_model,
        bada_path=bada_path,
    ).enroute(
        mass=df.mass,
        tas=df.tas,
        alt=df.altitude,
        vs=df.vertical_rate,
        dT=opt.dT,
    )

    assert len(df) == opt.nodes + 1
    assert df.mass.iloc[-1] < df.mass.iloc[0]
    assert df.fuel_cost.dropna().sum() > 0
    assert df.fuelflow.dropna().gt(0).all()
    assert df.fuelflow.to_numpy().reshape(-1) == pytest.approx(
        np.asarray(expected_fuelflow).reshape(-1)
    )
