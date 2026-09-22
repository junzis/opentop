"""Analytical integration catches stepped controls in dynamics or quadrature."""

import casadi as ca
import pytest

import numpy as np
from opentop.base import Base


class Integrator(Base):
    def __init__(self):
        self.nodes = 2
        self.polydeg = 3
        self.performance_model = "openap"
        self.solver_options = {}
        self.x_lb = [-100.0] * 5
        self.x_ub = [100.0] * 5
        self.x_0_lb = self.x_0_ub = [0.0] * 5
        self.x_f_lb, self.x_f_ub = self.x_lb, self.x_ub
        self.u_lb = self.u_0_lb = self.u_f_lb = [-10.0] * 3
        self.u_ub = self.u_0_ub = self.u_f_ub = [10.0] * 3
        self.x_guess = np.zeros((3, 5))
        self.u_guess = [0.0] * 3

    def init_model(self, objective, *, function_name="f", **kwargs):
        self.x = ca.MX.sym("x", 5)  # type: ignore[arg-type]  # CasADi stubs reject valid symbolic constructors
        self.u = ca.MX.sym("u", 3)  # type: ignore[arg-type]
        dt = ca.MX.sym("dt")  # type: ignore[arg-type]
        self.func_dynamics = ca.Function(
            function_name,
            [self.x, self.u, dt],
            [ca.vertcat(self.u[0], 0, 0, 0, 1), dt * self.u[0] ** 2],
        )

        self.func_grid_cost = ca.Function(
            "grid_cost", [self.x, self.u, dt], [dt * self.x[4] ** 2]
        )


@pytest.mark.parametrize("variable", [False, True])
def test_linear_controls_integrate_dynamics_and_cost(variable):
    opt = Integrator()
    problem = ca.Opti()
    durations = [2.0, 3.0] if variable else [2.5, 2.5]
    t = opt._add_transcription(
        problem, "fuel", 5.0, variable_timestep=variable, dt_min=1, dt_max=4
    )
    assert len(t.U) == len(t.X) == 3
    values = [0.0, 2.0, 1.0]
    for control, value in zip(t.U, values):
        problem.set_initial(control, [value, 0, 0])
    elapsed, position, cost = 0.0, 0.0, 0.0
    for k, dt in enumerate(durations):
        if variable:
            problem.set_initial(t.interval_dts[k], dt)
        problem.set_initial(t.X[k], [position, 0, 0, 0, elapsed])
        u0, u1 = values[k : k + 2]
        for tau, x in zip(t.collocation_roots, t.Xc[k]):
            xp = position + dt * (u0 * tau + (u1 - u0) * tau**2 / 2)
            problem.set_initial(x, [xp, 0, 0, 0, elapsed + tau * dt])
        position += dt * (u0 + u1) / 2
        cost += dt * (u0 * u0 + u0 * u1 + u1 * u1) / 3
        elapsed += dt
        np.testing.assert_allclose(
            problem.debug.value(t.control_at(k, 1), problem.initial()),
            problem.debug.value(t.U[k + 1], problem.initial()),
        )
    problem.set_initial(t.X[-1], [position, 0, 0, 0, elapsed])

    def evaluate(x):
        return np.asarray(problem.debug.value(x, problem.initial())).ravel()

    residual, lower, upper = map(evaluate, [problem.g, problem.lbg, problem.ubg])
    assert np.max(np.maximum(lower - residual, residual - upper)) < 1e-10
    np.testing.assert_allclose(evaluate(t.objective_raw), cost, atol=1e-12)
    # Integral of time squared on [0, 5] is 125/3. Boundary rectangle
    # costs differ deterministically, independent of any solver's optimum.
    np.testing.assert_allclose(evaluate(t.grid_cost_raw), 125 / 3, atol=1e-12)
    rectangle = sum(dt * sum(durations[:k]) ** 2 for k, dt in enumerate(durations))
    assert rectangle != pytest.approx(125 / 3)
    assert len(list(t.path_points())) == 9
    extra = [(0, 0.37), (1, 0.63)]
    points = list(t.path_points(extra))
    assert len(points) == 11
    for (k, tau), (state, control) in zip(extra, points[-2:]):
        u0, u1 = values[k : k + 2]
        start_position = sum(
            dt * (values[j] + values[j + 1]) / 2 for j, dt in enumerate(durations[:k])
        )
        xp = start_position + durations[k] * (u0 * tau + (u1 - u0) * tau**2 / 2)
        np.testing.assert_allclose(
            evaluate(state),
            [xp, 0, 0, 0, sum(durations[:k]) + tau * durations[k]],
            atol=1e-12,
        )
        np.testing.assert_allclose(evaluate(control), [u0 + (u1 - u0) * tau, 0, 0])
    np.testing.assert_allclose(evaluate(t.state_at(1, 1)), evaluate(t.X[-1]))
    # The terminal control participates in the final interval's physics and cost.
    problem.set_initial(t.U[-1], [3, 0, 0])
    changed = evaluate(problem.g)
    assert np.max(np.maximum(lower - changed, changed - upper)) > 0.1
    assert abs(evaluate(t.objective_raw)[0] - cost) > 1


@pytest.mark.parametrize("supplied_guess", [False, True])
def test_state_initialization_matches_mesh_and_preserves_supplied_times(supplied_guess):
    opt = Integrator()
    opt.x_guess = np.array(
        [[0, 0, 0, 50, 0], [10, 4, 6, 45, 8], [20, 8, 12, 40, 20]],
        dtype=float,
    )
    original = opt.x_guess.copy()
    problem = ca.Opti()
    tr = opt._add_transcription(
        problem, "fuel", 5.0, initial_guess=object() if supplied_guess else None
    )
    expected = original.copy()
    if not supplied_guess:
        expected[:, 4] = [0, 2.5, 5]
    for state, value in zip(tr.X, expected):
        np.testing.assert_allclose(problem.debug.value(state, problem.initial()), value)
    for k, states in enumerate(tr.Xc):
        for tau, state in zip(tr.collocation_roots, states):
            np.testing.assert_allclose(
                problem.debug.value(state, problem.initial()),
                (1 - tau) * expected[k] + tau * expected[k + 1],
            )
    np.testing.assert_array_equal(opt.x_guess, original)


@pytest.mark.parametrize(
    "interval,tau",
    [(-1, 0.5), (2, 0.5), (0, -0.1), (0, 1.1), (0, float("nan")), (0, float("inf"))],
)
def test_extra_path_points_reject_invalid_locations(interval, tau):
    opt = Integrator()
    tr = opt._add_transcription(ca.Opti(), "fuel", 5.0)
    with pytest.raises(ValueError, match="path constraint"):
        list(tr.path_points([(interval, tau)]))
