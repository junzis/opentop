"""Initial mass units apply consistently to all flight phases."""

import pytest

import opentop as top


@pytest.mark.parametrize(
    "phase", [top.CompleteFlight, top.Climb, top.Cruise, top.Descent]
)
@pytest.mark.parametrize("fraction", [0.87, 1.0])
def test_optimizer_accepts_kg_and_mtow_fraction(phase, fraction):
    fractional = phase("A320", "EHAM", "LIRF", m0=fraction)
    kilograms = phase("A320", "EHAM", "LIRF", m0=fraction * fractional.aircraft["mtow"])
    assert kilograms.mass_init == fractional.mass_init
    assert kilograms.mass_init_lb == fractional.mass_init_lb
    assert kilograms.mass_init_ub == fractional.mass_init_ub


@pytest.mark.parametrize(
    "phase", [top.CompleteFlight, top.Climb, top.Cruise, top.Descent]
)
@pytest.mark.parametrize("mass", [78001.0, 0, -1, float("nan"), float("inf")])
def test_optimizer_rejects_invalid_mass(phase, mass):
    with pytest.raises(ValueError, match=r"m0|mass"):
        phase("A320", "EHAM", "LIRF", m0=mass)


def test_payload_guess_still_rejects_mass_above_mtow():
    with pytest.raises(ValueError, match="MTOW"):
        top.CompleteFlight("A320", "EHAM", "LIRF", m0=78001, payload=10000)


@pytest.mark.parametrize(
    "performance_model, actype, bada_path",
    [
        ("openap", "A320", None),
        ("bada3", "A320", "/home/junzi/arc/data/bada_312"),
        ("bada4", "A320-214", "/home/junzi/arc/data/bada_4.2/tables"),
    ],
)
def test_mass_units_across_performance_models(performance_model, actype, bada_path):
    from pathlib import Path

    if bada_path is not None and not Path(bada_path).exists():
        pytest.skip("BADA data unavailable")
    kwargs = {"performance_model": performance_model, "bada_path": bada_path}
    fractional = top.Cruise(actype, "EHAM", "LIRF", m0=0.87, **kwargs)
    kilograms = top.Cruise(actype, "EHAM", "LIRF", m0=fractional.mass_init, **kwargs)
    assert kilograms.mass_init == fractional.mass_init
    with pytest.raises(ValueError, match="MTOW"):
        top.Cruise(actype, "EHAM", "LIRF", m0=fractional.aircraft["mtow"] + 1, **kwargs)


def test_full_mtow_dynamics_are_finite():
    import numpy as np

    opt = top.Cruise("A320", "EHAM", "EDDF", m0=1)
    state = np.array([0.0, 0.0, 10000.0, opt.aircraft["mtow"], 0.0])
    rates = np.asarray(opt.xdot(state, np.array([0.78, 0.0, 0.0])), dtype=float)
    assert np.isfinite(rates).all()
    assert rates[3] < 0
