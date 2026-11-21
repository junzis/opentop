"""Step-climb search and preservation of physical trajectory results."""

import os
from typing import cast

import casadi as ca
import pytest
from openap.aero import fpm

import numpy as np
import opentop as top
import pandas as pd
from opentop import _step_climb
from opentop._step_climb import SparseAscent, activation_candidates, resolve_step_climb
from opentop._transcription import AircraftTranscription


@pytest.mark.parametrize(
    "kwargs",
    [
        {"penalty_weight": -1},
        {"penalty_weight": float("nan")},
        {"max_climb_rate_fpm": -1},
        {"activation_rate_fpm": float("inf")},
        {"method": "legacy", "min_climb_rate_fpm": 500, "activation_rate_fpm": 1500},
        {"min_climb_rate_fpm": 1600},
        {"min_level_duration_s": 0},
        {"max_transition_duration_s": -1},
    ],
)
def test_step_climb_options_reject_invalid_values(kwargs):
    with pytest.raises(ValueError, match="step-climb"):
        top.StepClimbOptions(**kwargs)


def test_step_climb_flag_requires_explicit_boolean_or_options():
    assert resolve_step_climb(False) is None
    assert resolve_step_climb(True) == top.StepClimbOptions()
    with pytest.raises(TypeError, match="step_climb"):
        resolve_step_climb("true")  # type: ignore[arg-type]


def test_sparse_ascent_integrates_duration_and_prefers_concentrated_climb():
    opti = ca.Opti()
    controls = [opti.variable(3) for _ in range(5)]
    interval_dt = opti.parameter()
    opti.set_value(interval_dt, 100)
    transcription = AircraftTranscription(
        optimizer=None,
        opti=opti,
        X=[],
        Xc=[],
        U=controls,
        ts_final=4 * interval_dt,
        interval_dts=[interval_dt] * 4,
        objective_raw=ca.MX(1),
        objective_scaled=ca.MX(1),
        objective_scale=1,
        objective_kwargs={},
        collocation_roots=(),
    )
    preference = SparseAscent.add(
        transcription,
        top.StepClimbOptions(
            method="legacy", min_climb_rate_fpm=500, penalty_weight=0.01
        ),
        0,
        4,
    )
    evaluate = ca.Function("active_time", [opti.x, opti.p], [preference.active_time])
    parameter_values = opti.debug.value(opti.p, opti.initial())

    def active_time(rates):
        values = np.zeros((5, 3))
        values[:, 1] = np.array(rates) * fpm
        return float(evaluate(values.ravel(), parameter_values))

    # Both profiles gain the same altitude. Concentrating that climb into one
    # pulse pays less active-time cost than spending every interval ascending.
    distributed = active_time([0, 100, 100, 100, 0])
    concentrated = active_time([0, 0, 300, 0, 0])
    assert 0 < concentrated < distributed < 400
    opti.set_value(interval_dt, 200)
    doubled = opti.debug.value(opti.p, opti.initial())
    pulse = np.zeros((5, 3))
    pulse[2, 1] = 300 * fpm
    np.testing.assert_allclose(
        evaluate(pulse.ravel(), doubled), 2 * concentrated, atol=1e-12
    )


def test_short_cruise_can_choose_no_steps_and_reports_physical_fuel():
    optimizer = top.Cruise("A320", "EHAM", "EDDF", m0=60_000)
    optimizer.setup(nodes=20, max_iter=700)
    solver_options = optimizer.solver_options.copy()
    frame = cast(pd.DataFrame, optimizer.trajectory(step_climb=True))

    assert optimizer.success, optimizer.stats
    assert 1 <= len(optimizer.step_climb_stats) <= 3
    assert all(stage["solver_success"] for stage in optimizer.step_climb_stats)
    assert frame.vertical_rate.abs().max() < 1
    assert frame.altitude.max() - frame.altitude.min() < 10
    assert optimizer.step_climb_penalty is not None
    assert optimizer.step_climb_penalty < 0.1
    assert optimizer.objective_value == pytest.approx(
        frame.mass.iloc[0] - frame.mass.iloc[-1], rel=1e-7
    )
    assert optimizer.solver_options == solver_options

    optimizer.trajectory()
    assert optimizer.success, optimizer.stats
    assert optimizer.step_climb_penalty is None
    assert optimizer.step_climb_stats == []
    assert optimizer.step_climb_diagnostics is None


def test_step_climb_conflicts_with_cruise_descent():
    optimizer = top.Cruise("A320", "EHAM", "EDDF", m0=60_000)
    optimizer.allow_cruise_descent()
    with pytest.raises(ValueError, match="allow_cruise_descent"):
        optimizer.trajectory(step_climb=True)


@pytest.mark.parametrize("optimizer_class", [top.Cruise, top.CompleteFlight])
def test_step_climb_requires_builtin_fuel_objective(optimizer_class):
    optimizer = optimizer_class("A320", "EHAM", "EDDF", m0=60_000)
    with pytest.raises(ValueError, match='objective="fuel"'):
        optimizer.trajectory(objective="time", step_climb=True)


@pytest.mark.skipif(
    os.environ.get("OPENTOP_RUN_STEP_CLIMB_INTEGRATION") != "1",
    reason="set OPENTOP_RUN_STEP_CLIMB_INTEGRATION=1 for the multi-minute search",
)
def test_long_cruise_discovers_sustained_climbs_between_level_dwells():
    optimizer = top.Cruise("A320", "EHAM", "LGAV", m0=70_000)
    optimizer.setup(nodes=30, max_iter=1500, ipopt_kwargs={"max_cpu_time": 45})
    frame = cast(pd.DataFrame, optimizer.trajectory(step_climb=True))
    assert optimizer.success, optimizer.stats
    diagnostics = optimizer.step_climb_diagnostics
    assert diagnostics["climb_seconds"] > 0
    assert len(diagnostics["level_durations_s"]) >= 2
    assert min(diagnostics["level_durations_s"]) >= 300 - 0.01
    assert diagnostics["path_audit"]["success"]
    controls = optimizer.U[1] / fpm
    active = diagnostics["activation"] > 0.5
    assert controls[active].min() > 0
    assert np.abs(controls[~active]).max() <= 0.01
    assert optimizer.objective_value == pytest.approx(
        frame.mass.iloc[0] - frame.mass.iloc[-1], rel=1e-7
    )


def test_candidate_search_includes_zero_and_multiple_discovered_climbs():
    rates = np.array([0, 0, 200, 10, 0, 0, 100, 10, 0, 0], dtype=float)
    candidates = activation_candidates(rates)
    assert len(candidates) == 3
    assert not candidates[0].any()
    assert candidates[1].sum() == 2
    assert candidates[2].sum() == 4
    assert all(not mode[:2].any() and not mode[-2:].any() for mode in candidates)


def test_rejected_path_audit_propagates_to_public_result(monkeypatch):
    monkeypatch.setattr(
        _step_climb,
        "refine_performance",
        lambda optimizer, solution, **kwargs: (solution, {"success": False}),
    )
    optimizer = top.Cruise("A320", "EHAM", "EDDF", m0=60_000)
    optimizer.setup(nodes=8, max_iter=700)
    optimizer.fix_cruise_altitude()
    with pytest.warns(UserWarning, match="flight might be infeasible"):
        result = optimizer.trajectory(
            step_climb=top.StepClimbOptions(
                method="legacy", min_climb_rate_fpm=500, penalty_weight=0.01
            ),
            result_object=True,
            return_failed=True,
        )
    assert not optimizer.success
    assert not result.success
    assert result.status == "Step_Climb_Path_Audit_Failed"
    assert not result.df.empty
    assert result.objective == pytest.approx(
        result.df.mass.iloc[0] - result.df.mass.iloc[-1], rel=1e-7
    )
    assert optimizer.step_climb_diagnostics["success"] is False


@pytest.mark.parametrize("reject_audit", [False, True])
@pytest.mark.parametrize("fallback", ["none", "recover", "fail_exact"])
def test_candidate_selection_restores_winner_and_reports_rejected_audit(
    monkeypatch, reject_audit, fallback
):
    """Solve a small real mode NLP with the winning candidate deliberately first."""
    opti = ca.Opti()
    duration = opti.variable()
    opti.subject_to(duration == 500)
    opti.set_initial(duration, 500)
    controls = [opti.variable(3) for _ in range(6)]
    for control in controls:
        opti.subject_to(control[0] == 0.7)
        opti.subject_to(control[2] == 0)
    physical = 1 + sum((control[1] / (500 * fpm)) ** 2 for control in controls)
    opti.minimize(physical)
    opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": False})
    transcription = AircraftTranscription(
        optimizer=None,
        opti=opti,
        X=[ca.DM([0, 0, 10000, 60000, k * 100]) for k in range(6)],
        Xc=[],
        U=controls,
        ts_final=duration,
        interval_dts=[duration / 5] * 5,
        objective_raw=physical,
        objective_scaled=physical,
        objective_scale=1,
        objective_kwargs={},
        collocation_roots=(),
    )

    preference = SparseAscent.add(
        transcription,
        top.StepClimbOptions(
            method="legacy",
            min_climb_rate_fpm=500,
            penalty_weight=0.01,
            min_level_duration_s=100,
        ),
        0,
        5,
    )
    initial = opti.solve()
    candidate_calls = []
    audited_softness = []

    class ToyOptimizer:
        MAX_VERTICAL_ACCELERATION = 5 * fpm

        def __init__(self):
            self.step_climb_stats = []

        def _solve_once(self, X, U, **kwargs):
            self._last_solution = opti.solve()
            mode = np.asarray(opti.debug.value(preference.mode, opti.initial()))
            softness = float(opti.debug.value(preference.softness, opti.initial()))
            if mode.any():
                achieved = (
                    np.asarray(self._last_solution.value(ca.horzcat(*U)))[1] / fpm
                )
                if softness == 0:
                    assert achieved[mode.ravel() > 0.5].min() >= 500 - 0.01
                    assert np.abs(achieved[mode.ravel() < 0.5]).max() <= 0.01
                candidate_calls.append(softness)
                if fallback != "none" and (
                    len(candidate_calls) == 1
                    or (fallback == "fail_exact" and len(candidate_calls) == 3)
                ):
                    self._last_solution = _step_climb.RejectedSolution(
                        self._last_solution, "Forced_Exact_Failure"
                    )
            self.objective_value = float(self._last_solution.value(physical))
            return self.to_trajectory(
                500,
                np.asarray(self._last_solution.value(ca.horzcat(*X))),
                np.asarray(self._last_solution.value(ca.horzcat(*U))),
            )

        def to_trajectory(self, duration, states, controls, **kwargs):
            self.X, self.U, self.dt = states, controls, duration / 5
            return pd.DataFrame({"vertical_rate": controls[1] / fpm})

    monkeypatch.setattr(
        _step_climb,
        "activation_candidates",
        lambda _: [np.zeros(6), np.array([0, 0, 1, 1, 0, 0])],
    )

    def audit(optimizer, solution, **kwargs):
        audited_softness.append(
            float(opti.debug.value(preference.softness, opti.initial()))
        )
        return solution, {"success": not reject_audit}

    monkeypatch.setattr(_step_climb, "refine_performance", audit)
    optimizer = ToyOptimizer()
    frame = preference._select_modes(optimizer, initial, 0.001)
    np.testing.assert_allclose(
        optimizer.U,
        optimizer._last_solution.value(ca.horzcat(*controls)),
    )
    assert optimizer.objective_value == pytest.approx(
        optimizer._last_solution.value(physical)
    )
    assert audited_softness and all(value == 0 for value in audited_softness)
    assert candidate_calls == ([0] if fallback == "none" else [0, 0.1, 0])
    candidate_stats = [
        stage for stage in optimizer.step_climb_stats if stage["candidate"] == 1
    ]
    if fallback != "none":
        assert not candidate_stats[0]["success"]
        assert candidate_stats[1]["stage"] == "mode_fallback"
        assert "candidate_success" not in candidate_stats[1]
    if fallback == "fail_exact":
        assert not candidate_stats[-1]["success"]
        assert "candidate_success" not in candidate_stats[-1]
    elif not reject_audit:
        assert candidate_stats[-1]["candidate_success"]
        assert candidate_stats[-1]["integrality_error"] == 0
    if reject_audit:
        assert not optimizer._last_solution.stats()["success"]
        audited_stats = [
            stage for stage in optimizer.step_climb_stats if "path_audit" in stage
        ]
        assert all(stage["candidate_success"] is False for stage in audited_stats)
        expected_status = (
            "Forced_Exact_Failure"
            if fallback == "fail_exact"
            else "Step_Climb_Path_Audit_Failed"
        )
        assert optimizer._last_solution.stats()["return_status"] == expected_status
    else:
        assert optimizer.step_climb_diagnostics["selected_candidate"] == 0
        assert optimizer.step_climb_penalty == pytest.approx(0, abs=1e-12)
        winner_stats = optimizer.step_climb_stats[0]
        assert winner_stats["candidate"] == 0
        assert winner_stats["candidate_success"]
        assert winner_stats["penalty"] == pytest.approx(optimizer.step_climb_penalty)
        assert winner_stats["objective"] == pytest.approx(optimizer.objective_value)
        if fallback != "fail_exact":
            assert candidate_stats[-1]["penalty"] > optimizer.step_climb_penalty
        assert optimizer.step_climb_stats[-1]["candidate"] == 1
        assert frame.vertical_rate.abs().max() < 0.01
