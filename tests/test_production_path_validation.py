"""Configured physics, interval contexts, and strict bounded acceptance."""

from types import SimpleNamespace

import casadi as ca
import openap.casadi as oc
import pytest
from openap.aero import ft, kts

import numpy as np
import opentop as top
import pandas as pd
from opentop import _path_validation as pv
from opentop._continuous_step import pulse_pattern, step_timestep_defaults


def test_phase_contract_uses_only_configured_rows_and_descent_thrust():
    optimizer = top.Cruise("A320", "EHAM", "LGAV", m0=70000)
    state = ca.DM([0, 0, 10000, 65000, 0])
    control, context = ca.DM([0.7, 0, 0]), ca.DM.zeros(2)
    ordinary = pv.configured_rows(optimizer, state, control, context, "cruise")
    full = pv.configured_rows(optimizer, state, control, context, "full")
    descent = pv.configured_rows(optimizer, state, control, context, "descent")
    assert [r.name for r in ordinary] == ["thrust95_drag", "thrust90_lift_drag"]
    assert full[-1].name == descent[-1].name == "energy"
    speed = oc.aero.mach2tas(control[0], state[2], dT=optimizer.dT)
    expected = optimizer.thrust.cruise(speed / kts, state[2] / ft, dT=optimizer.dT)
    assert float(descent[0].lhs) == pytest.approx(float(expected * 0.95))


def test_climb_context_preserves_native_interval_average_energy():
    optimizer = top.Cruise("A320", "EHAM", "LGAV", m0=70000)
    x0, x1 = ca.DM([0, 0, 5000, 65000, 0]), ca.DM([0, 0, 5600, 64900, 100])
    u0, u1 = ca.DM([0.5, 6, 0]), ca.DM([0.55, 6, 0])
    tr = SimpleNamespace(
        optimizer=optimizer,
        physical_contract="climb",
        X=[x0, x1],
        U=[u0, u1],
        interval_dts=[100],
    )
    context = pv.interval_contexts(tr)[0]
    row = pv.configured_rows(optimizer, x0, u0, context, "climb")[0]
    v0 = oc.aero.mach2tas(u0[0], x0[2], dT=optimizer.dT)
    v1 = oc.aero.mach2tas(u1[0], x1[2], dT=optimizer.dT)
    thrust = optimizer._thrust_climb(v0 / kts, x0[2] / ft)
    drag = optimizer.drag.clean(x0[3], v0 / kts, x0[2] / ft, dT=optimizer.dT)
    old = (
        (thrust - drag) / x0[3]
        - oc.aero.g0 / v0 * (x1[2] - x0[2]) / 100
        - (v1 - v0) / 100
    )
    assert float(row.lhs) == float(old)
    assert row.name == "interval_average_energy"
    assert pv.point_key(0, 1, interval_dependent=True) != pv.point_key(
        1, 0, interval_dependent=True
    )
    assert pv.point_key(0, 1) == pv.point_key(1, 0)
    assert pv.point_key(0, 1 - 1e-13) != pv.point_key(0, 1)


def test_interpolation_keeps_endpoint_override_and_nonuniform_intervals():
    x, u = np.arange(15.0).reshape(5, 3), np.arange(9.0).reshape(3, 3)
    xc = np.arange(30.0).reshape(2, 5, 3)
    result = pv.interpolate_arrays(
        x, xc, u, np.array([31, 187]), tuple(ca.collocation_points(3, "legendre"))
    )
    coarse = pv.interpolate_arrays(
        x, xc, u, np.array([31, 187]), tuple(ca.collocation_points(3, "legendre")), 81
    )
    subset = np.arange(322).reshape(2, 161)[:, ::2].ravel()
    for key in result:
        np.testing.assert_array_equal(result[key][..., subset], coarse[key])
    np.testing.assert_array_equal(result["X"][:, 160], x[:, 1])
    np.testing.assert_array_equal(result["X"][:, 161], x[:, 1])
    np.testing.assert_array_equal(result["X"][:, -1], x[:, -1])


def fake_controller(monkeypatch, outcomes, *, solver_success=True, finite=True):
    calls, cuts = [], []
    opti = SimpleNamespace(ng=10, x="primal", set_initial=lambda x, v: None)
    tr = SimpleNamespace(opti=opti, physical_contract="full")
    optimizer = SimpleNamespace(_last_transcription=tr)

    class Audit:
        def __init__(self, tr, density):
            pass

        def audit(self, solution):
            return dict(
                success=outcomes[min(len(calls) - 1, len(outcomes) - 1)], finite=finite
            ), None

        def select_points(self, data):
            return {("interval", 0, len(calls) / 4): (0, len(calls) / 4)}

        def constrain(self, k, tau, buffer):
            cuts.append((k, tau))
            opti.ng += 3

    monkeypatch.setattr(pv, "PhysicalAudit", Audit)

    def run():
        calls.append(True)
        optimizer._last_solution = SimpleNamespace(
            stats=lambda: dict(
                success=solver_success, return_status="test", iter_count=2
            ),
            value=lambda x: np.array([1.0]),
        )
        return pd.DataFrame({"mass": [70000, 69000]})

    return optimizer, run, calls, cuts


def test_first_pass_acceptance_uses_a_single_solve(monkeypatch):
    optimizer, run, calls, cuts = fake_controller(monkeypatch, [True])
    pv.solve_validated(optimizer, run, top.PhysicalValidationOptions())
    assert len(calls) == 1 and not cuts
    assert optimizer.physical_validation_diagnostics["success"]


def test_repeated_repair_points_stop_without_another_solve(monkeypatch):
    optimizer, run, calls, cuts = fake_controller(monkeypatch, [False])
    monkeypatch.setattr(
        pv.PhysicalAudit, "select_points", lambda self, data: {("node", 1): (0, 1.0)}
    )
    pv.solve_validated(optimizer, run, top.PhysicalValidationOptions())
    assert len(calls) == 2 and cuts == [(0, 1.0)]
    assert optimizer.physical_validation_diagnostics["status"] == "no_new_repair_points"
    assert not optimizer._last_solution.stats()["success"]


def test_bounded_controller_repairs_twice_then_accepts(monkeypatch):
    optimizer, run, calls, cuts = fake_controller(monkeypatch, [False, False, True])
    frame = pv.solve_validated(optimizer, run, top.PhysicalValidationOptions())
    assert len(frame) == 2 and len(calls) == 3 and len(cuts) == 2
    report = optimizer.physical_validation_diagnostics
    assert report["success"] and report["solves"] == 3
    assert report["total_iterations"] == 6


@pytest.mark.parametrize(
    "solver_success,finite,status",
    [
        (False, True, "solver_failed"),
        (True, False, "nonfinite_physical_audit"),
        (True, True, "repair_budget_exhausted"),
    ],
)
def test_rejection_cannot_be_a_successful_result(
    monkeypatch, solver_success, finite, status
):
    optimizer, run, calls, cuts = fake_controller(
        monkeypatch, [False], solver_success=solver_success, finite=finite
    )
    pv.solve_validated(optimizer, run, top.PhysicalValidationOptions(max_repairs=0))
    assert len(calls) == 1 and not cuts
    assert not optimizer._last_solution.stats()["success"]
    assert optimizer.physical_validation_diagnostics["status"] == status


def test_step_mesh_defaults_and_level_only_basis():
    dense_short = SimpleNamespace(range=10000, nodes=120)
    lower, upper = step_timestep_defaults(dense_short, None, None)
    assert 0 < lower < 30 and upper == 900
    coarse_long = SimpleNamespace(range=10_000_000, nodes=10)
    assert step_timestep_defaults(coarse_long, None, None) == (30, 10000)
    assert step_timestep_defaults(coarse_long, 17, 222) == (17, 222)
    assert pulse_pattern(2, 5) == ([[2, 3, 4]], [])


def test_continuous_options_and_explicit_legacy_contract():
    assert top.StepClimbOptions().min_climb_rate_fpm == 0
    assert top.StepClimbOptions().penalty_weight == 0
    with pytest.raises(ValueError, match="legacy"):
        top.StepClimbOptions(min_climb_rate_fpm=500)
    old = top.StepClimbOptions(
        method="legacy", min_climb_rate_fpm=500, penalty_weight=0.01
    )
    assert old.method == "legacy"
    optimizer = top.Cruise("A320", "EHAM", "EDDF", m0=60000)
    with pytest.raises(ValueError, match="variable_timestep"):
        optimizer.trajectory(step_climb=True, variable_timestep=False)


@pytest.mark.parametrize("optimizer_class", [top.Cruise, top.CompleteFlight])
@pytest.mark.parametrize("failure", ["altitude", "mass"])
def test_posthoc_rejection_remains_failed_with_diagnostic_frame(
    monkeypatch, optimizer_class, failure
):
    optimizer = optimizer_class("A320", "EHAM", "EDDF", m0=60000)
    optimizer.setup(nodes=8)

    def fake_solve(*args, **kwargs):
        optimizer._last_solution = SimpleNamespace(
            stats=lambda: {"success": True, "return_status": "Solve_Succeeded"}
        )
        return pd.DataFrame(
            {
                "altitude": [1000, 1000] if failure == "altitude" else [10000, 10000],
                "mass": [
                    60000,
                    optimizer.mass_min - 100 if failure == "mass" else 59000,
                ],
                "ts": [0, 100],
            }
        )

    monkeypatch.setattr(optimizer, "_solve", fake_solve)
    with pytest.warns(UserWarning):
        result = optimizer.trajectory(return_failed=True, result_object=True)
    assert not result.success and not optimizer.success
    assert not result.df.empty
    assert "Check_Failed" in result.status


def test_ramp_bounds_can_select_zero_capacity_without_overriding_limits():
    from opentop._continuous_step import select_capacity

    options = top.StepClimbOptions(max_climb_rate_fpm=25)
    count, info = select_capacity(0, 20, 12000, options=options, dt_min=400, dt_max=900)
    assert count == 0 and info["mesh_capacity"] == 0


def test_repeated_interval_gets_full_audit_grid_with_all_rows():
    validator = object.__new__(pv.PhysicalAudit)
    validator.tr = SimpleNamespace(interval_dts=[100, 100], collocation_roots=())
    validator.density = 161
    validator.interval_dependent = False
    validator.rows = [{"tolerance": 1e-7}] * 4
    validator._previous_violating_intervals = set()
    data = {
        "tau": np.tile(np.linspace(0, 1, 161), 2),
        "margins": np.ones((4, 322)),
    }
    data["margins"][3, 35] = -1
    first = validator.select_points(data)
    assert len(first) < 30
    data["margins"][3, 35] = 1
    data["margins"][3, 42] = -1
    second = validator.select_points(data)
    assert validator.last_recurrent_intervals == [0]
    assert all(pv.point_key(0, float(t)) in second for t in np.linspace(0, 1, 161))
    assert len(second) < 180


@pytest.mark.parametrize("aircraft", ["A320", "B789"])
@pytest.mark.parametrize("initialization", [None, "auto"])
def test_default_seed_policy_and_optional_control_bounds(
    monkeypatch, aircraft, initialization
):
    optimizer = top.CompleteFlight(aircraft, "EHAM", "LGAV", m0=0.85)
    optimizer.setup(nodes=12)

    class CheckedSeed(Exception):
        pass

    def inspect_seed(*args, **kwargs):
        tr = optimizer._last_transcription
        controls = np.asarray(tr.opti.debug.value(ca.horzcat(*tr.U), tr.opti.initial()))
        for k, values in enumerate(controls.T):
            lower, upper = (
                (optimizer.u_0_lb, optimizer.u_0_ub)
                if k == 0
                else (optimizer.u_f_lb, optimizer.u_f_ub)
                if k == optimizer.nodes
                else (optimizer.u_lb, optimizer.u_ub)
            )
            if initialization == "auto":
                assert np.all(values >= lower) and np.all(values <= upper)
            else:
                np.testing.assert_array_equal(values, optimizer.u_guess)
        if initialization is None:
            assert optimizer.initialization_diagnostics["method"] == "legacy"
        else:
            assert optimizer.initialization_diagnostics["seed_only"]
        raise CheckedSeed

    monkeypatch.setattr(optimizer, "_solve", inspect_seed)
    with pytest.raises(CheckedSeed):
        optimizer.trajectory(initialization=initialization)


@pytest.mark.parametrize(
    "target,value,reason",
    [
        ("X", np.nan, "X"),
        ("dt", 0, "nonpositive_durations"),
        ("dt", -1, "nonpositive_durations"),
        ("dt", np.nan, "dts"),
    ],
)
def test_mutable_numeric_solution_is_not_cached_and_invalid_inputs_reject(
    target, value, reason
):
    optimizer = top.Cruise("A320", "EHAM", "EDDF", m0=60000)
    state = ca.DM([0, 0, 9000, 60000, 0])
    tr = SimpleNamespace(
        optimizer=optimizer,
        physical_contract="cruise",
        X=[state, state],
        U=[ca.DM([0.7, 0, 0])] * 2,
        Xc=[[state] * 3],
        interval_dts=[100.0],
        collocation_roots=tuple(ca.collocation_points(3, "legendre")),
    )
    validator = pv.PhysicalAudit(tr)
    solution = SimpleNamespace(value=lambda expression: np.asarray(expression))
    assert validator.audit(solution)[0]["finite"]
    if target == "X":
        tr.X[0] = ca.DM([value, 0, 9000, 60000, 0])
    else:
        tr.interval_dts[0] = value
    audit, _ = validator.audit(solution)
    assert not audit["success"] and not audit["finite"]
    assert reason in audit["invalid_inputs"]
    assert validator._cache is None


def test_all_repair_rows_preserve_fuel_units_and_zero_fuel_buffer():
    validator = object.__new__(pv.PhysicalAudit)
    opti = ca.Opti()
    state = opti.variable(5)
    x, u, c = ca.MX.sym("x", 5), ca.MX.sym("u", 3), ca.MX.sym("c", 2)
    validator.evaluate = ca.Function("four_rows", [x, u, c], [x[:4]])
    validator.rows = [{"audit_unit": unit} for unit in ["1", "1", "1", "kg/s"]]
    validator.contexts = [ca.DM.zeros(2)]
    validator.tr = SimpleNamespace(
        opti=opti,
        state_at=lambda k, tau: state,
        control_at=lambda k, tau: ca.DM.zeros(3),
    )
    validator.constrain(0, 0.3, 1e-4)
    assert opti.ng == 4
    np.testing.assert_array_equal(opti.debug.value(opti.lbg), [1e-4, 1e-4, 1e-4, 0])


def test_fixed_altitude_step_uses_zero_capacity_without_duplicate_equalities(
    monkeypatch,
):
    from opentop._continuous_step import ContinuousStepClimb

    optimizer = top.Cruise("A320", "EHAM", "EDDF", m0=60000)
    optimizer.setup(nodes=8)
    optimizer.fix_cruise_altitude()
    original = ContinuousStepClimb.add

    class CheckedBasis(Exception):
        pass

    def inspect(tr, options, first, last):
        constraints = tr.opti.ng
        basis = original(tr, options, first, last)
        assert not basis.pulses
        assert tr.opti.ng == constraints
        raise CheckedBasis

    monkeypatch.setattr(ContinuousStepClimb, "add", inspect)
    with pytest.raises(CheckedBasis):
        optimizer.trajectory(step_climb=True)


def test_capacity_requires_each_level_block_to_fit_its_dwell():
    from opentop._continuous_step import select_capacity

    options = top.StepClimbOptions(min_level_duration_s=150)
    count, _ = select_capacity(0, 6, 530, options=options, dt_min=30, dt_max=100)
    assert count == 0
