"""Offline tests using an original tiny network, not Acropole's model weights."""

import csv

import casadi as ca
import pytest
from openap.aero import fpm, ft, kts

import numpy as np
import opentop as top
import pandas as pd

onnx = pytest.importorskip("onnx")
pytestmark = pytest.mark.skipif(
    not hasattr(ca, "GraphBuilder"), reason="requires CasADi 3.8.1 ONNX support"
)


@pytest.fixture(scope="module")
def model_files(tmp_path_factory):
    directory = tmp_path_factory.mktemp("neural-fuel")
    params = dict(
        ACFT_ICAO_TYPE="A320",
        ENGINE_TYPE=0,
        SURFACE=122.6,
        MAX_OPE_ALTI=41100,
        MAX_OPE_SPEED=471.5,
        OPE_EMPTY_WEIGHT=43700,
        MAX_TO_WEIGHT=78000,
        FUEL_FLOW_TO=1.132,
        ENGINE_NUM=2,
    )
    params_path = directory / "aircraft.csv"
    with params_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(params))
        writer.writeheader()
        writer.writerow(params)
    # All twelve features affect this synthetic network. Its affine output
    # stays away from the ReLU kink at our derivative test points.
    weights = np.linspace(0.01, 0.12, 12, dtype=np.float32).reshape(12, 1)
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("MatMul", ["features", "weights"], ["linear"]),
            onnx.helper.make_node("Relu", ["linear"], ["positive"]),
            onnx.helper.make_node("Sigmoid", ["positive"], ["flow"]),
        ],
        "test_fuel",
        [
            onnx.helper.make_tensor_value_info(
                "features", onnx.TensorProto.FLOAT, ["N", 12]
            )
        ],
        [onnx.helper.make_tensor_value_info("flow", onnx.TensorProto.FLOAT, ["N", 1])],
        [onnx.numpy_helper.from_array(weights, "weights")],
    )
    path = directory / "fuel.onnx"
    onnx.save(
        onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)]),
        path,
    )
    return path, params_path


@pytest.fixture(scope="module")
def fuel(model_files):
    path, params = model_files
    return top.AcropoleFuelFlow("a320", model_path=path, aircraft_params_path=params)


def test_numeric_preprocessing_and_broadcast(fuel):
    mass = np.array([60000, 65000])
    tas = np.array([420, 450])
    alt = np.array([30000, 35000])
    vs = np.array([1200, -1000])
    p = fuel.params
    raw = np.array(
        [
            [
                p["ENGINE_TYPE"],
                v / 60,
                0,
                0,
                p["SURFACE"],
                p["MAX_OPE_ALTI"],
                p["MAX_OPE_SPEED"],
                h,
                t * np.cos(np.arctan2(v * fpm, t * kts)),
                t,
                v,
                (m - 43700) / (78000 - 43700),
            ]
            for m, t, h, v in zip(mass, tas, alt, vs)
        ]
    )
    low = np.array([0, -5000, -50, -50, 0, 0, 200, 0, 200, 200, -5000, 0])
    high = np.array([1, 5000, 50, 50, 600, 50000, 800, 50000, 800, 800, 5000, 1])
    z = ((raw - low) / (high - low)) @ np.linspace(0.01, 0.12, 12, dtype=np.float32)
    expected = 2 * 1.132 / (1 + np.exp(-np.maximum(z, 0)))
    np.testing.assert_allclose(fuel.enroute(mass, tas, alt, vs), expected, rtol=1e-12)
    assert isinstance(fuel.enroute(60000, 420, 30000), float)
    assert fuel.enroute(mass[:, None], tas, 30000).shape == (2, 2)
    assert fuel.enroute([], [], []).shape == (0,)


def test_symbolic_gradient_and_hessian(fuel):
    x = ca.MX.sym("x", 4)
    y = fuel.enroute(*ca.vertsplit(x))
    derivatives = ca.Function(
        "derivatives", [x], [ca.gradient(y, x), ca.hessian(y, x)[0]]
    )
    point = np.array([62000, 440, 32000, 500])
    steps = np.array([1, 0.01, 1, 0.1])
    gradient, hessian = [np.asarray(v) for v in derivatives(point)]
    numeric_gradient = []
    numeric_hessian = []
    for i, step in enumerate(steps):
        delta = np.eye(4)[i] * step
        numeric_gradient.append(
            (fuel.enroute(*(point + delta)) - fuel.enroute(*(point - delta)))
            / (2 * step)
        )
        gp, _ = derivatives(point + delta)
        gm, _ = derivatives(point - delta)
        numeric_hessian.append(np.asarray((gp - gm) / (2 * step)).ravel())
    np.testing.assert_allclose(
        gradient.ravel(), numeric_gradient, rtol=1e-6, atol=1e-12
    )
    np.testing.assert_allclose(
        hessian, np.array(numeric_hessian).T, rtol=1e-5, atol=1e-13
    )


def test_unknown_aircraft_and_invalid_mass_range(model_files, tmp_path):
    path, params = model_files
    with pytest.raises(ValueError, match="parameter row"):
        top.AcropoleFuelFlow("ZZZZ", model_path=path, aircraft_params_path=params)
    bad = tmp_path / "bad.csv"
    bad.write_text(params.read_text().replace("78000", "43700"))
    with pytest.raises(ValueError, match="must exceed"):
        top.AcropoleFuelFlow("A320", model_path=path, aircraft_params_path=bad)


def test_reject_aircraft_mismatch_and_wind(fuel):
    with pytest.raises(ValueError, match="aircraft type"):
        top.Cruise("B738", "EHAM", "EDDF", fuel_model=fuel)
    opt = top.Cruise("A320", "EHAM", "EDDF", fuel_model=fuel)
    with pytest.raises(ValueError, match="still air"):
        opt.enable_wind(pd.DataFrame())


def test_fuel_model_accepts_aircraft_variant(fuel, monkeypatch):
    # Variant performance data need not be installed to test typecode matching.
    from opentop._performance import build_performance_models

    models = build_performance_models("A320", engine=None, use_synonym=False)
    monkeypatch.setattr("opentop.base.build_performance_models", lambda *a, **k: models)
    opt = top.Cruise("A320-214", "EHAM", "EDDF", fuel_model=fuel)
    assert opt.fuelflow is fuel
    assert opt.actype == "A320-214"


def test_cruise_mass_objective_and_report_use_same_model(fuel):
    opt = top.Cruise(
        "A320", "EHAM", "EDDF", fuel_model=fuel, h_min=30000 * ft, h_max=35000 * ft
    )
    opt.setup(nodes=6, max_iter=500)
    opt.fix_cruise_altitude()
    opt.fix_mach_number()
    result = opt.trajectory(objective="fuel", result_object=True)
    assert result.success, result.status
    df = result.df
    assert result.fuel > 0
    assert result.objective == pytest.approx(result.fuel, abs=1e-4)
    assert df.fuel_cost.sum() == pytest.approx(result.fuel)
    expected = fuel.enroute(df.mass, df.tas, df.altitude, df.vertical_rate)
    np.testing.assert_allclose(df.fuelflow, expected)
    x, u = opt.X[:, 0], opt.U[:, 0]
    rate = -float(opt.xdot(x, u)[3])
    assert float(opt.obj_fuel(x, u, 1, symbolic=False)) == pytest.approx(rate)
    co2 = opt._calc_emission(x, u, symbolic=False)[0]
    assert float(co2) == pytest.approx(float(opt.emission.co2(rate)))


class SyntheticPhysics:
    """Known high-speed curvature to test correction derivatives independently."""

    def __init__(self, wave=True):
        self.wave = wave

    def enroute(self, mass, tas, alt, vs=0, *, dT=0):
        from openap.casadi import aero

        mach = aero.tas2mach(tas * kts, alt * ft, dT=dT)
        base = 0.5 + mass * 1e-6 + alt * 1e-6 + vs * 1e-5 + 0.2 * mach
        return base + (20 * (mach - 0.7) ** 2 if self.wave else 0)


def test_speed_physics_correction_limits_temperature_and_broadcast(fuel):
    from openap.casadi import aero

    physical = SyntheticPhysics()
    corrected = top.PhysicsCorrectedFuelFlow(fuel, physical, reference_mach=0.76)
    mach = np.array([0.70, 0.76, 0.78, 0.82])
    # Use symbolic altitude/temperature as in the optimizer: OpenAP uses a
    # smoothed atmosphere on that path, unlike a direct numeric evaluation.
    mm, hh, dt = [ca.MX.sym(s) for s in ("mach", "alt", "dT")]
    speed = ca.Function(
        "speed", [mm, hh, dt], [aero.mach2tas(mm, hh * ft, dT=dt) / kts]
    )
    for temperature in (0, 15):
        tas = np.asarray(speed(mach, 32000, temperature)).ravel()
        original = fuel.enroute(62000, tas, 32000)
        actual = corrected.enroute(62000, tas, 32000, dT=temperature)
        np.testing.assert_allclose(actual[:2], original[:2], atol=1e-12)
        ref_tas = float(speed(0.76, 32000, temperature))
        ref_fuel = fuel.enroute(62000, ref_tas, 32000)
        expected = (
            ref_fuel
            + 0.2 * (mach[2:] - 0.76)
            + 20 * ((mach[2:] - 0.7) ** 2 - (0.76 - 0.7) ** 2)
        )
        np.testing.assert_allclose(actual[2:], expected, rtol=1e-12, atol=1e-12)
    assert corrected.enroute([], [], []).shape == (0,)
    assert corrected.enroute(np.array([[60000], [65000]]), [400, 450], 32000).shape == (
        2,
        2,
    )


def test_physics_correction_derivatives_and_transition(fuel):
    from openap.casadi import aero

    corrected = top.PhysicsCorrectedFuelFlow(
        fuel, SyntheticPhysics(), reference_mach=0.76
    )
    mach = ca.MX.sym("mach")
    tas = aero.mach2tas(mach, 32000 * ft) / kts
    value = corrected.enroute(62000, tas, 32000)
    first = ca.jacobian(value, mach)
    derivative = ca.Function(
        "corrected_derivatives", [mach], [value, first, ca.jacobian(first, mach)]
    )
    for point in (0.74, 0.765, 0.77, 0.775, 0.80):
        y, g, h = [float(v) for v in derivative(point)]
        yp, gp, _ = [float(v) for v in derivative(point + 1e-6)]
        ym, gm, _ = [float(v) for v in derivative(point - 1e-6)]
        assert g == pytest.approx((yp - ym) / 2e-6, rel=1e-5, abs=1e-7)
        assert h == pytest.approx((gp - gm) / 2e-6, rel=1e-5, abs=1e-5)
        assert y > 0
    # The blend adds no value/gradient/Hessian discontinuity at either end.
    for edge in (0.76, 0.78):
        left = np.array(derivative(edge - 1e-9)).ravel()
        right = np.array(derivative(edge + 1e-9)).ravel()
        np.testing.assert_allclose(left, right, rtol=1e-4, atol=1e-3)
    assert float(derivative(0.80)[1]) == pytest.approx(
        0.2 + 40 * (0.80 - 0.7), rel=2e-6
    )


def test_wave_increment_and_zero_strength(fuel):
    wave, no_wave = SyntheticPhysics(), SyntheticPhysics(wave=False)
    for strength in (0, 0.5, 1):
        corrected = top.PhysicsCorrectedFuelFlow(
            fuel,
            wave,
            method="wave_increment",
            wave_free_model=no_wave,
            strength=strength,
        )
        original = fuel.enroute(62000, 450, 32000)
        delta = float(
            wave.enroute(62000, 450, 32000) - no_wave.enroute(62000, 450, 32000)
        )
        assert corrected.enroute(62000, 450, 32000) == pytest.approx(
            original + strength * delta
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"reference_mach": float("nan")},
        {"reference_mach": 0.76, "transition_width": 0},
        {"reference_mach": 0.99},
        {"reference_mach": 0.76, "strength": 0.5},
        {"method": "unknown"},
        {"method": "wave_increment"},
        {
            "method": "wave_increment",
            "wave_free_model": SyntheticPhysics(False),
            "strength": -1,
        },
    ],
)
def test_physics_correction_invalid_configuration(fuel, kwargs):
    with pytest.raises(ValueError):
        top.PhysicsCorrectedFuelFlow(fuel, SyntheticPhysics(), **kwargs)


@pytest.mark.parametrize("method", ["speed_increment", "wave_increment"])
def test_corrected_cruise_accounting(fuel, method):
    options = (
        {"reference_mach": 0.72}
        if method == "speed_increment"
        else {"wave_free_model": SyntheticPhysics(False)}
    )
    corrected = top.PhysicsCorrectedFuelFlow(
        fuel, SyntheticPhysics(), method=method, **options
    )
    test_cruise_mass_objective_and_report_use_same_model(corrected)
    test_reject_aircraft_mismatch_and_wind(corrected)


@pytest.mark.parametrize("epsilon", [0.0, 0.001, 0.1])
def test_activation_smoothing_value_and_derivatives(model_files, epsilon):
    path, params = model_files
    model = top.AcropoleFuelFlow(
        "A320",
        model_path=path,
        aircraft_params_path=params,
        activation_smoothing=epsilon,
    )
    weights = np.linspace(0.01, 0.12, 12, dtype=np.float32)
    # Move across a hidden activation boundary, including exactly on it.
    z = ca.MX.sym("z")
    output = model.network(ca.repmat(z / float(weights.sum(dtype=np.float64)), 1, 12))
    evaluate = ca.Function(
        "smooth_check", [z], [output, ca.jacobian(output, z), ca.hessian(output, z)[0]]
    )
    for point in [-0.1, 0.0, 0.1]:
        value, gradient, hessian = (float(v) for v in evaluate(point))
        positive = (
            max(point, 0)
            if epsilon == 0
            else (point + np.sqrt(point**2 + float(np.float32(epsilon**2)))) / 2
        )
        assert value == pytest.approx(1 / (1 + np.exp(-positive)), abs=1e-12)
        if epsilon > 0:
            step = epsilon * 1e-3
            upper = evaluate(point + step)
            lower = evaluate(point - step)
            assert gradient == pytest.approx(
                float((upper[0] - lower[0]) / (2 * step)), rel=2e-5, abs=1e-8
            )
            assert hessian == pytest.approx(
                float((upper[1] - lower[1]) / (2 * step)), rel=2e-5, abs=1e-8
            )


@pytest.mark.parametrize("epsilon", [-1, float("nan"), float("inf")])
def test_invalid_activation_smoothing(model_files, epsilon):
    path, params = model_files
    with pytest.raises(ValueError, match="finite and nonnegative"):
        top.AcropoleFuelFlow(
            "A320",
            model_path=path,
            aircraft_params_path=params,
            activation_smoothing=epsilon,
        )


def test_complete_flight_neural_mass_accounting(fuel):
    opt = top.CompleteFlight("A320", "LFPG", "EDDM", fuel_model=fuel)
    opt.setup(nodes=12, max_iter=800)
    result = opt.trajectory(objective="fuel", result_object=True)
    assert result.success, result.status
    assert result.objective == pytest.approx(result.fuel, abs=1e-3)
    assert result.df.fuel_cost.sum() == pytest.approx(result.fuel)
    df = result.df
    np.testing.assert_allclose(
        df.fuelflow, fuel.enroute(df.mass, df.tas, df.altitude, df.vertical_rate)
    )
    assert df.vertical_rate.max() > 0
    assert df.vertical_rate.min() < 0


@pytest.mark.parametrize("performance_model", ["openap", "bada3", "bada4"])
@pytest.mark.parametrize("temperature", [0, 15])
def test_override_temperature_consistent_across_consumers(
    fuel, monkeypatch, performance_model, temperature
):
    from dataclasses import replace

    from opentop._performance import build_performance_models

    # Exercise dispatch for each baseline without requiring licensed BADA files.
    models = build_performance_models("A320", engine=None, use_synonym=False)
    monkeypatch.setattr(
        "opentop.base.build_performance_models",
        lambda *a, **k: replace(models, name=performance_model),
    )
    corrected = top.PhysicsCorrectedFuelFlow(
        fuel, SyntheticPhysics(), reference_mach=0.76
    )
    opt = top.Cruise(
        "A320",
        "EHAM",
        "EDDF",
        dT=temperature,
        performance_model=performance_model,
        fuel_model=corrected,
    )
    x = np.array([0, 0, 32000 * ft, 62000, 0])
    u = np.array([0.82, 0, 0])
    rate = -float(opt.xdot(x, u)[3])
    assert float(opt.obj_fuel(x, u, 1)) == pytest.approx(rate, rel=1e-10)
    assert float(opt.obj_fuel(x, u, 1, symbolic=False)) == pytest.approx(rate, rel=1e-5)
    co2 = opt._calc_emission(x, u, symbolic=False)[0]
    assert float(co2) == pytest.approx(float(opt.emission.co2(rate)), rel=1e-5)
    opt.setup(nodes=2)
    states = np.repeat(x[:, None], 3, axis=1)
    states[4] = [0, 30, 60]
    states[3] -= np.array([0, 30, 60]) * rate
    controls = np.repeat(u[:, None], 3, axis=1)
    df = opt.to_trajectory(60, states, controls)
    assert df.fuelflow.iloc[0] == pytest.approx(rate, rel=1e-5)


def test_reject_incompatible_correction_models(fuel):
    import openap.casadi as oc

    wave = oc.FuelFlow("A320", wave_drag=True)
    wrong_aircraft = oc.FuelFlow("B738", wave_drag=False)
    with pytest.raises(ValueError, match="aircraft"):
        top.PhysicsCorrectedFuelFlow(fuel, wrong_aircraft, reference_mach=0.76)
    with pytest.raises(ValueError, match="aircraft"):
        top.PhysicsCorrectedFuelFlow(
            fuel, wave, method="wave_increment", wave_free_model=wrong_aircraft
        )
    wrong_engine = oc.FuelFlow("A320", "V2500-A1", wave_drag=False)
    with pytest.raises(ValueError, match="same engine"):
        top.PhysicsCorrectedFuelFlow(
            fuel, wave, method="wave_increment", wave_free_model=wrong_engine
        )
    with pytest.raises(ValueError, match="wave drag"):
        top.PhysicsCorrectedFuelFlow(
            fuel, wave, method="wave_increment", wave_free_model=wave
        )
    corrected = top.PhysicsCorrectedFuelFlow(
        fuel,
        wave,
        method="wave_increment",
        wave_free_model=oc.FuelFlow("A320", wave_drag=False),
    )
    assert corrected.enroute(62000, 450, 32000) > 0


@pytest.mark.parametrize("phase", [top.Climb, top.Descent])
def test_neural_phase_and_preliminary_cruise_use_same_model(fuel, phase):
    opt = phase("A320", "LFPG", "EDDM", fuel_model=fuel)
    assert opt.cruise.fuel_model is fuel
    with pytest.raises(ValueError, match="still air"):
        opt.enable_wind(pd.DataFrame())
    opt.setup(nodes=12, max_iter=1000)
    opt.cruise.setup(nodes=6, max_iter=800)
    result = opt.trajectory(objective="fuel", remove_cruise=False, result_object=True)
    assert result.success, result.status
    assert result.objective == pytest.approx(result.fuel, abs=1e-3)
    df = result.df
    np.testing.assert_allclose(
        df.fuelflow, fuel.enroute(df.mass, df.tas, df.altitude, df.vertical_rate)
    )
