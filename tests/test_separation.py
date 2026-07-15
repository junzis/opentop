import casadi as ca
import pytest

import numpy as np
from opentop import _dynamics
from opentop._separation import (
    SeparationConfig,
    collocation_roots,
    interpolate_collocation_state,
    numeric_collocation_state,
    separation_metric,
)


def test_default_separation_config_matches_reference_formulation():
    config = SeparationConfig()

    assert config.horizontal_m == 5 * 1852
    assert config.vertical_m == 1000 * 0.3048
    assert config.vertical_power == 8
    assert config.minimum_metric == 1.3


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"horizontal_m": 0}, "horizontal_m"),
        ({"vertical_m": -1}, "vertical_m"),
        ({"vertical_power": 3}, "vertical_power"),
        ({"vertical_power": 1}, "vertical_power"),
        ({"minimum_metric": float("nan")}, "minimum_metric"),
        ({"max_refinements": -1}, "max_refinements"),
        ({"max_refinements": 1.5}, "max_refinements"),
        ({"max_refinements": True}, "max_refinements"),
    ],
)
def test_separation_config_validates_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SeparationConfig(**kwargs)


def test_high_order_metric_is_even_in_vertical_difference():
    config = SeparationConfig()
    positive = separation_metric(0.0, 0.0, config.vertical_m, config)
    negative = separation_metric(0.0, 0.0, -config.vertical_m, config)

    assert positive == pytest.approx(1.0)
    assert negative == pytest.approx(positive)


def test_high_order_metric_supports_symbolic_inputs():
    dz = ca.MX.sym("dz")  # type: ignore[arg-type]
    expression = separation_metric(0.0, 0.0, dz, SeparationConfig())
    function = ca.Function("metric", [dz], [expression])
    metric = function(1000 * 0.3048)

    assert isinstance(metric, ca.DM)
    assert float(metric) == pytest.approx(1.0)


def test_collocation_interpolation_recovers_all_roots():
    x_start = np.array([1.0, 2.0])
    x_collocation = np.array([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    expected = [x_start, *list(x_collocation)]

    for tau, value in zip(collocation_roots(3), expected):
        actual = numeric_collocation_state(x_start, x_collocation, float(tau))
        np.testing.assert_allclose(actual, value, atol=1e-12)


def test_symbolic_and_numeric_collocation_interpolation_match():
    x_start = np.array([1.0, 2.0])
    x_collocation = np.array([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    tau = ca.MX.sym("tau")  # type: ignore[arg-type]
    expression = interpolate_collocation_state(
        ca.DM(x_start), [ca.DM(value) for value in x_collocation], tau
    )
    function = ca.Function("dense_state", [tau], [expression])

    for value in [0.0, 0.13, 0.57, 1.0]:
        expected = numeric_collocation_state(x_start, x_collocation, value)
        np.testing.assert_allclose(np.asarray(function(value)).ravel(), expected)


def test_collocation_endpoint_uses_continuity_polynomial():
    x_start = np.array([1.0, 2.0])
    x_collocation = np.array([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    _, continuity, _ = _dynamics.collocation_coeff(3)
    expected = continuity[0] * x_start
    for coefficient, value in zip(continuity[1:], x_collocation):
        expected += coefficient * value

    actual = numeric_collocation_state(x_start, x_collocation, 1.0)
    np.testing.assert_allclose(actual, expected)
