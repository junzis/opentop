import warnings

import pytest
from openap.aero import ft, kts

import numpy as np
import opentop as top
from opentop._performance import build_numeric_fuelflow, build_performance_models


def test_openap_model_build_suppresses_wave_drag_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_performance_models(
            "A320",
            engine=None,
            use_synonym=False,
            performance_model="openap",
        )

    assert not any("Wave drag is experimental" in str(item.message) for item in caught)


@pytest.mark.parametrize("mach", [0.70, 0.78, 0.82])
def test_numeric_fuel_and_emissions_match_dynamics(mach):
    """Reported fuel/climate costs must retain the dynamics' wave-drag penalty."""
    import openap

    opt = top.Cruise("A320", "EHAM", "EDDF")
    mass, altitude = 66300.0, 32000.0
    x = np.array([0, 0, altitude * ft, mass, 0])
    u = np.array([mach, 0, 0])
    tas = openap.aero.mach2tas(mach, altitude * ft) / kts
    expected = -float(opt.xdot(x, u)[3])
    numeric = build_numeric_fuelflow(
        "A320",
        engtype=opt.engtype,
        use_synonym=False,
        performance_model="openap",
        bada_path=None,
    )
    actual = np.asarray(numeric.enroute(mass, tas, altitude)).item()
    # Allow small NumPy/CasADi evaluation differences; omitting wave drag
    # instead causes a percent-level discrepancy at Mach 0.82.
    assert actual == pytest.approx(expected, rel=1e-5)
    assert np.asarray(opt.obj_fuel(x, u, 1, symbolic=False)).item() == pytest.approx(
        expected, rel=1e-5
    )
    co2 = np.asarray(opt._calc_emission(x, u, symbolic=False)[0]).item()
    assert co2 == pytest.approx(float(opt.emission.co2(expected)), rel=1e-5)


@pytest.mark.parametrize("symbolic", [False, True])
def test_bada3_descent_idle_floor_uses_kg_per_second(symbolic):
    import casadi as ca
    from openap.backends import CasadiBackend, NumpyBackend

    from opentop._performance import _Bada3EnrouteFuelFlow

    class FuelModel:
        smooth = symbolic
        backend = CasadiBackend() if symbolic else NumpyBackend()

        def enroute(self, mass, tas, alt, vs, *, acc=None, dT=0):
            # Mimic the upstream double conversion of the idle floor.
            return self.backend.maximum(vs, 0.12 / 60)

        def idle(self, mass, tas, alt, vs):
            return 0.12

    model = _Bada3EnrouteFuelFlow(FuelModel())
    if symbolic:
        vs = ca.MX.sym("vs")
        expression = model.enroute(60000, 250, 10000, vs)
        evaluate = ca.Function("idle_floor", [vs], [expression])
        actual = np.array(evaluate([-1, 0.5])).ravel()
    else:
        actual = model.enroute(60000, 250, 10000, np.array([-1, 0.5]))
    np.testing.assert_allclose(actual, [0.12, 0.5], atol=2e-5, rtol=0)
