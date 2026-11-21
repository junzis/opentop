"""Local search for optimizer-selected level cruise and step climbs."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from time import perf_counter
from typing import TYPE_CHECKING, Any

import casadi as ca
import openap.casadi as oc
from openap.aero import fpm, ft, kts

import numpy as np

if TYPE_CHECKING:
    from ._transcription import AircraftTranscription


@dataclass(frozen=True, slots=True)
class StepClimbOptions:
    """Optional cruise-climb pulses with optimized rates and switching times.

    The default continuous method minimizes fuel with up to ``max_steps``
    optional pulses (currently capped at three by the basis). Each pulse has
    a constant-rate hold and acceleration-limited entry/exit ramps. Its rate
    may be zero; there is no minimum climb rate or preference penalty.
    Level blocks last at least ``min_level_duration_s``. Inactive slots still
    reserve their ramp and dwell intervals. This is a local smooth NLP, not
    a global search over step counts or mesh allocations.

    ``method="legacy"`` retains sparse-ascent discovery and candidate search.
    It requires an explicit positive ``min_climb_rate_fpm``; for the former
    policy also set ``penalty_weight=0.01``. ``activation_rate_fpm`` applies
    only to that method. Both methods support the built-in fuel objective.
    """

    penalty_weight: float = 0.0
    min_climb_rate_fpm: float = 0.0
    max_climb_rate_fpm: float = 1500.0
    activation_rate_fpm: float = 30.0
    min_level_duration_s: float = 300.0
    max_transition_duration_s: float = 300.0
    method: str = "continuous"
    max_steps: int = 3

    def __post_init__(self) -> None:
        if self.method not in {"continuous", "legacy"}:
            raise ValueError("step-climb method must be continuous or legacy")
        if (
            isinstance(self.max_steps, bool)
            or not isinstance(self.max_steps, int)
            or self.max_steps < 0
        ):
            raise ValueError("step-climb max_steps must be a nonnegative integer")
        for name in (
            "penalty_weight",
            "max_climb_rate_fpm",
            "min_climb_rate_fpm",
            "activation_rate_fpm",
            "min_level_duration_s",
            "max_transition_duration_s",
        ):
            value = getattr(self, name)
            lower_ok = (
                value >= 0
                if name in {"penalty_weight", "min_climb_rate_fpm"}
                else value > 0
            )
            if not isfinite(value) or not lower_ok:
                raise ValueError(f"step-climb {name} must be finite and positive")
        if self.method == "continuous" and (
            self.min_climb_rate_fpm or self.penalty_weight
        ):
            raise ValueError(
                "continuous step-climb uses no minimum rate or penalty; "
                "use method='legacy' for those options"
            )
        if self.method == "legacy" and self.min_climb_rate_fpm <= 0:
            raise ValueError(
                "legacy step-climb requires an explicit positive "
                "min_climb_rate_fpm (for example 500)"
            )
        if (
            self.method == "legacy"
            and self.activation_rate_fpm >= self.max_climb_rate_fpm
        ):
            raise ValueError(
                "step-climb activation_rate_fpm must be below max_climb_rate_fpm"
            )
        if self.min_climb_rate_fpm > self.max_climb_rate_fpm:
            raise ValueError(
                "step-climb min_climb_rate_fpm must not exceed max_climb_rate_fpm"
            )


def resolve_step_climb(value: bool | StepClimbOptions) -> StepClimbOptions | None:
    if isinstance(value, StepClimbOptions):
        return value
    if value is True:
        return StepClimbOptions()
    if value is False:
        return None
    raise TypeError("step_climb must be a bool or StepClimbOptions")


def validate_step_climb_objective(options: StepClimbOptions | None, objective: Any):
    if options is not None and (not isinstance(objective, str) or objective != "fuel"):
        raise ValueError('step_climb currently requires objective="fuel"')


@dataclass(slots=True)
class SparseAscent:
    """Build-specific objective parameters and continuation solve."""

    transcription: AircraftTranscription
    options: StepClimbOptions
    coefficient: Any
    activation: Any
    active_time: Any
    mode: Any
    softness: Any
    first: int
    last: int

    @classmethod
    def add(
        cls,
        transcription: AircraftTranscription,
        options: StepClimbOptions,
        first: int,
        last: int,
    ) -> SparseAscent:
        opti = transcription.opti
        optimizer = transcription.optimizer
        if optimizer is not None and options.max_transition_duration_s < (
            options.min_climb_rate_fpm * fpm / optimizer.MAX_VERTICAL_ACCELERATION
        ):
            raise ValueError(
                "step-climb max_transition_duration_s is shorter than the "
                "minimum-rate acceleration ramp"
            )
        coefficient, activation = opti.parameter(), opti.parameter()
        opti.set_value(coefficient, 0.0)
        opti.set_value(activation, max(100.0, options.activation_rate_fpm))
        mode = opti.parameter(last - first + 1)
        softness = opti.parameter()
        opti.set_value(mode, np.zeros(last - first + 1))
        opti.set_value(softness, 1)
        endpoints = {first, min(first + 1, last), max(first, last - 1), last}
        for index in range(first, last + 1):
            rate = transcription.U[index][1]
            if index in endpoints:
                opti.subject_to(rate == 0)
            else:
                active = mode[index - first]
                opti.subject_to(
                    opti.bounded(
                        (1 - softness) * options.min_climb_rate_fpm * fpm * active,
                        rate,
                        options.max_climb_rate_fpm
                        * fpm
                        * (active + softness * (1 - active)),
                    )
                )
        # Integrate the continuous linear controls, rather than counting active
        # nodes. The duration factor matters when interval times are optimized.
        roots, weights = np.polynomial.legendre.leggauss(5)
        active_time = 0
        for k in range(first, last):
            for root, weight in zip(roots, weights):
                rate = transcription.control_at(k, float((root + 1) / 2))[1] / fpm
                active_time += (
                    transcription.interval_dts[k]
                    * weight
                    / 2
                    * rate
                    / (rate + activation)
                )
        opti.minimize(
            (transcription.objective_raw + coefficient * active_time)
            / transcription.objective_scale
        )
        return cls(
            transcription,
            options,
            coefficient,
            activation,
            active_time,
            mode,
            softness,
            first,
            last,
        )

    def solve(self, optimizer: Any, **kwargs: Any) -> Any:
        """Warm-start increasing regularization from a feasible physical solve."""
        hessian_key = "ipopt.hessian_approximation"
        previous = optimizer.solver_options.get(hessian_key)
        optimizer.solver_options[hessian_key] = "exact"
        started = perf_counter()
        try:
            frame = self._solve_stages(optimizer, **kwargs)
            from ._path_validation import (
                PhysicalAudit,
                RejectedPhysicalSolution,
                resolve_validation,
            )

            validation = resolve_validation(kwargs.get("physical_validation", True))
            report = dict(
                enabled=validation.enabled,
                method="legacy_final_audit",
                success=None,
                solves=len(optimizer.step_climb_stats),
            )
            if optimizer.success and validation.enabled:
                audit, _ = PhysicalAudit(
                    self.transcription, validation.points_per_interval
                ).audit(optimizer._last_solution)
                report.update(audit)
                if not audit["success"]:
                    optimizer._last_solution = RejectedPhysicalSolution(
                        optimizer._last_solution,
                        "Physical_Validation_legacy_final_audit_failed",
                    )
            optimizer.physical_validation_diagnostics = report
            if optimizer.step_climb_diagnostics is not None:
                optimizer.step_climb_diagnostics["success"] = optimizer.success
                optimizer.step_climb_diagnostics["final_physical_validation"] = report
            return frame
        finally:
            if optimizer.step_climb_diagnostics is None:
                optimizer.step_climb_diagnostics = {"success": False}
            optimizer.step_climb_diagnostics["search_seconds"] = (
                perf_counter() - started
            )
            optimizer.step_climb_diagnostics["solves"] = len(optimizer.step_climb_stats)
            if previous is None:
                optimizer.solver_options.pop(hessian_key, None)
            else:
                optimizer.solver_options[hessian_key] = previous

    def _solve_stages(self, optimizer: Any, **kwargs: Any) -> Any:
        transcription = self.transcription
        opti = transcription.opti
        optimizer.step_climb_stats = []
        started = perf_counter()
        frame = optimizer._solve_once(transcription.X, transcription.U, **kwargs)
        solution = optimizer._last_solution
        optimizer.step_climb_stats.append(
            dict(solution.stats(), stage="reference", seconds=perf_counter() - started)
        )
        if not solution.stats()["success"]:
            return frame
        reference = abs(float(solution.value(transcription.objective_raw)))
        duration = float(solution.value(transcription.ts_final))
        if not isfinite(reference) or reference <= 1e-30 or duration <= 0:
            raise ValueError(
                "step-climb reference objective and duration must be nonzero"
            )
        coefficient = self.options.penalty_weight * reference / duration
        broad_activation = max(100.0, self.options.activation_rate_fpm)
        for fraction, activation in (
            (0.1, broad_activation),
            (1.0, broad_activation),
            (1.0, self.options.activation_rate_fpm),
        ):
            opti.set_initial(opti.x, solution.value(opti.x))
            opti.set_value(self.coefficient, coefficient * fraction)
            opti.set_value(self.activation, activation)
            previous_solution = solution
            started = perf_counter()
            frame = optimizer._solve_once(transcription.X, transcription.U, **kwargs)
            solution = optimizer._last_solution
            optimizer.step_climb_stats.append(
                dict(solution.stats(), stage="sparse", seconds=perf_counter() - started)
            )
            if not solution.stats()["success"]:
                solution = previous_solution
                break
        return self._select_modes(optimizer, solution, coefficient, **kwargs)

    def _select_modes(
        self, optimizer: Any, initial_solution: Any, coefficient: float, **kwargs: Any
    ) -> Any:
        transcription, options = self.transcription, self.options
        opti = transcription.opti
        previous_variables = opti.x
        previous_values = initial_solution.value(previous_variables)
        rates = np.array(
            [
                float(initial_solution.value(control[1])) / fpm
                for control in transcription.U
            ]
        )[self.first : self.last + 1]
        candidates = activation_candidates(rates)
        level_flags = []
        for candidate in candidates:
            flag = opti.parameter()
            level_flags.append(flag)
            opti.set_value(flag, 0)
            if candidate.any():
                for start, stop in level_runs(candidate > 0.5):
                    duration = sum(
                        transcription.interval_dts[self.first + k]
                        for k in range(start, stop)
                    )
                    opti.subject_to(duration >= options.min_level_duration_s * flag)
                for k in np.flatnonzero(candidate[:-1] != candidate[1:]):
                    # The outer search selects transitions; their location in
                    # time and their duration remain optimization variables.
                    opti.subject_to(
                        transcription.interval_dts[self.first + k] * flag
                        <= options.max_transition_duration_s
                    )
        activation = self.mode
        active_time = sum(
            transcription.interval_dts[k]
            * (activation[k - self.first] + activation[k - self.first + 1])
            / 2
            for k in range(self.first, self.last)
        )
        changes = sum(
            ca.sqrt((activation[i + 1] - activation[i]) ** 2 + 1e-6) - 0.001
            for i in range(len(rates) - 1)
        )
        # Charge each mode change the equivalent of one minimum-rate ramp.
        ramp_time = (
            options.min_climb_rate_fpm * fpm / optimizer.MAX_VERTICAL_ACCELERATION
        )
        penalty = coefficient * (active_time + ramp_time * changes)
        opti.minimize(
            (transcription.objective_raw + penalty) / transcription.objective_scale
        )
        best = None
        last_audit = None
        for candidate_index, candidate in enumerate(candidates):
            for index, flag in enumerate(level_flags):
                opti.set_value(flag, float(index == candidate_index))
            opti.set_initial(previous_variables, previous_values)
            opti.set_value(activation, candidate)
            for i, control in enumerate(transcription.U[self.first : self.last + 1]):
                opti.set_initial(
                    control[1], options.min_climb_rate_fpm * fpm * candidate[i]
                )
            # Most discovered masks can be imposed directly. A failed exact
            # solve gets one softened warm start, followed by exact bounds again.
            stages = [0.0]
            for stage_index, softness in enumerate(stages):
                opti.set_value(self.softness, softness)
                started = perf_counter()
                optimizer._solve_once(transcription.X, transcription.U, **kwargs)
                solution = optimizer._last_solution
                stats = dict(solution.stats())
                stats.update(
                    candidate=candidate_index,
                    stage="mode" if stage_index == 0 else "mode_fallback",
                    relaxation=softness,
                    seconds=perf_counter() - started,
                )
                optimizer.step_climb_stats.append(stats)
                if not stats["success"]:
                    if stage_index == 0:
                        opti.set_initial(previous_variables, previous_values)
                        stages.extend((0.1, 0.0))
                        continue
                    break
                opti.set_initial(opti.x, solution.value(opti.x))
            if not solution.stats()["success"]:
                continue
            solution, audit = refine_performance(optimizer, solution, **kwargs)
            last_audit = audit
            optimizer.step_climb_stats[-1].update(
                candidate=candidate_index, path_audit=audit
            )
            if not audit["success"]:
                optimizer.step_climb_stats[-1].update(
                    candidate_success=False,
                    rejection_reason="Step_Climb_Path_Audit_Failed",
                )
                optimizer._last_solution = RejectedSolution(
                    solution, "Step_Climb_Path_Audit_Failed"
                )
                continue
            mode = candidate.copy()
            achieved_rates = np.array(
                [
                    float(solution.value(control[1])) / fpm
                    for control in transcription.U[self.first : self.last + 1]
                ]
            )
            error = np.max(np.minimum(mode, 1 - mode))
            separated = np.all(
                (achieved_rates <= 0.01)
                | (achieved_rates >= options.min_climb_rate_fpm - 0.01)
            )
            dts = np.array(
                [
                    float(solution.value(dt))
                    for dt in transcription.interval_dts[self.first : self.last]
                ]
            )
            active = mode > 0.5
            durations = [
                float(dts[start:stop].sum()) for start, stop in level_runs(active)
            ]
            dwell = not active.any() or (
                bool(durations)
                and min(durations) >= options.min_level_duration_s - 0.01
                and np.any(active[:-1] & active[1:])
            )
            bounded = (
                mode.min() >= -2e-6
                and mode.max() <= 1 + 2e-6
                and achieved_rates.min() >= -0.01
                and achieved_rates.max() <= options.max_climb_rate_fpm + 0.01
            )
            acceleration = np.max(np.abs(np.diff(achieved_rates) / dts))
            transitions = dts[active[:-1] != active[1:]]
            valid = bool(
                error <= 2e-6
                and separated
                and dwell
                and bounded
                and acceleration <= optimizer.MAX_VERTICAL_ACCELERATION / fpm + 0.001
                and (
                    not len(transitions)
                    or transitions.max() <= options.max_transition_duration_s + 0.01
                )
            )
            optimizer.step_climb_stats[-1].update(
                mode_success=valid,
                candidate_success=valid,
                objective=float(solution.value(transcription.objective_raw)),
                penalty=float(solution.value(penalty)),
                integrality_error=float(error),
            )
            if not valid:
                optimizer.step_climb_stats[-1]["rejection_reason"] = (
                    "Step_Climb_Mode_Validation_Failed"
                )
                optimizer._last_solution = RejectedSolution(
                    solution, "Step_Climb_Mode_Validation_Failed"
                )
                continue
            score = float(solution.value(transcription.objective_raw + penalty))
            if best is None or score < best[0]:
                best = (
                    score,
                    candidate_index,
                    solution,
                    float(solution.value(penalty)),
                    audit,
                )
        if best is None:
            if optimizer._last_solution.stats()["success"]:
                optimizer._last_solution = RejectedSolution(
                    optimizer._last_solution, "No_Valid_Step_Climb_Candidate"
                )
            failed = optimizer._last_solution
            optimizer.objective_value = float(failed.value(transcription.objective_raw))
            optimizer.step_climb_penalty = float(failed.value(penalty))
            optimizer.step_climb_diagnostics = {
                "success": False,
                "candidates": len(candidates),
                "path_audit": last_audit,
            }
            return optimizer.to_trajectory(
                float(failed.value(transcription.ts_final)),
                np.asarray(failed.value(ca.horzcat(*transcription.X))),
                np.asarray(failed.value(ca.horzcat(*transcription.U))),
                **kwargs,
            )
        _, selected_candidate, solution, regularization, audit = best
        optimizer._last_solution = solution
        optimizer.objective_value = float(solution.value(transcription.objective_raw))
        optimizer.step_climb_penalty = regularization
        if transcription.grid_cost_raw is not None:
            optimizer.grid_cost_value = float(
                solution.value(transcription.grid_cost_raw)
            )
        optimizer.step_climb_diagnostics = {
            "success": True,
            "activation": candidates[selected_candidate].copy(),
            "cruise_indices": (self.first, self.last),
            "candidates": len(candidates),
            "selected_candidate": selected_candidate,
            "path_audit": audit,
            "integrality_error": 0.0,
        }
        mode = candidates[selected_candidate] > 0.5
        dts = np.array(
            [
                float(solution.value(dt))
                for dt in transcription.interval_dts[self.first : self.last]
            ]
        )
        optimizer.step_climb_diagnostics.update(
            level_seconds=float(dts[~(mode[:-1] | mode[1:])].sum()),
            climb_seconds=float(dts[mode[:-1] & mode[1:]].sum()),
            transition_seconds=float(dts[mode[:-1] != mode[1:]].sum()),
            level_durations_s=[
                float(dts[start:stop].sum()) for start, stop in level_runs(mode)
            ],
        )
        # Restore caches as well as the frame when the winner is not last.
        return optimizer.to_trajectory(
            float(solution.value(transcription.ts_final)),
            np.asarray(solution.value(ca.horzcat(*transcription.X))),
            np.asarray(solution.value(ca.horzcat(*transcription.U))),
            **kwargs,
        )


def activation_candidates(rates: np.ndarray) -> list[np.ndarray]:
    """Generate a bounded local search from optimizer-discovered ascent peaks."""
    count = len(rates)
    candidates = [np.zeros(count)]
    peaks = [
        i
        for i in range(2, count - 2)
        if rates[i] > 1 and rates[i] >= rates[i - 1] and rates[i] > rates[i + 1]
    ]
    peaks.sort(key=lambda i: rates[i], reverse=True)
    for subset in (peaks[:1], peaks[:2], peaks[:3], peaks):
        seed = np.zeros(count)
        for peak in subset:
            neighbor = peak + 1 if peak + 1 < count - 2 else peak - 1
            if neighbor >= 2:
                seed[[peak, neighbor]] = 1
        for i in range(1, count - 1):
            if seed[i - 1] == seed[i + 1] == 1:
                seed[i] = 1
        if not any(np.array_equal(seed, existing) for existing in candidates):
            candidates.append(seed)
    return candidates


def level_runs(mode: np.ndarray) -> list[tuple[int, int]]:
    """Half-open interval ranges occupied entirely by level controls."""
    level = ~(mode[:-1] | mode[1:])
    edges = np.diff(np.r_[False, level, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


@dataclass(slots=True)
class RejectedSolution:
    """Keep the failed iterate inspectable without claiming IPOPT feasibility."""

    solution: Any
    status: str

    def stats(self) -> dict[str, Any]:
        return dict(self.solution.stats(), success=False, return_status=self.status)

    def value(self, *args: Any) -> Any:
        return self.solution.value(*args)


def refine_performance(optimizer: Any, solution: Any, **kwargs: Any):
    """Audit a common 81-point grid and constrain missed force/energy peaks.

    A fixed common audit grid means every accepted candidate satisfies every
    extra constraint added while polishing another candidate. This remains a
    sampled check rather than a continuous-path certificate.
    """
    transcription = optimizer._last_transcription
    opti = transcription.opti
    state, control = ca.MX.sym("audit_state", 5), ca.MX.sym("audit_control", 3)
    speed = oc.aero.mach2tas(control[0], state[2], dT=optimizer.dT)
    tas, altitude = speed / kts, state[2] / ft
    thrust = optimizer._thrust_climb(tas, altitude)
    drag = optimizer.drag.clean(state[3], tas, altitude, dT=optimizer.dT)
    lift_drag = optimizer._lift_margin_drag(state[3], tas, altitude)
    margin_list = [
        0.95 * thrust - drag,
        0.9 * thrust - lift_drag,
        (thrust - drag) - state[3] * oc.aero.g0 * control[1] / speed,
    ]
    margins = ca.vertcat(*margin_list) / ca.fmax(1, thrust)
    if optimizer.performance_model == "bada4":
        derivative, _ = optimizer.func_dynamics(state, control, 1.0)
        margins = ca.vertcat(
            margins, -derivative[3] - optimizer.BADA4_MIN_FUELFLOW_KG_S
        )
    evaluate = ca.Function("step_path_audit", [state, control], [margins])
    points = [
        (k, float(tau))
        for k in range(len(transcription.interval_dts))
        for tau in np.linspace(0, 1, 81)
    ]
    states = ca.horzcat(*(transcription.state_at(k, tau) for k, tau in points))
    controls = ca.horzcat(*(transcription.control_at(k, tau) for k, tau in points))
    added = 0
    for iteration in range(5):
        values = np.asarray(
            evaluate(solution.value(states), solution.value(controls)), dtype=float
        )
        minimum = float(values.min())
        report = {
            "success": bool(np.isfinite(values).all() and minimum >= -1e-7),
            "min_normalized_margin": minimum,
            "points_per_interval": 81,
            "refinements": iteration,
            "added_points": added,
        }
        if report["success"] or iteration == 4 or not np.isfinite(values).all():
            return solution, report
        violating = set()
        for k in range(len(transcription.interval_dts)):
            for margin in range(values.shape[0]):
                offset = int(np.argmin(values[margin, k * 81 : (k + 1) * 81]))
                if values[margin, k * 81 + offset] < -1e-7:
                    # Neighboring points constrain small peak shifts after a solve.
                    for neighbor in range(max(0, offset - 1), min(80, offset + 1) + 1):
                        violating.add(points[k * 81 + neighbor])
        for k, tau in sorted(violating):
            x, u = transcription.state_at(k, tau), transcription.control_at(k, tau)
            speed = oc.aero.mach2tas(u[0], x[2], dT=optimizer.dT)
            tas, altitude = speed / kts, x[2] / ft
            thrust = optimizer._thrust_climb(tas, altitude)
            drag = optimizer._constrain_clean_performance(
                opti, x[3], tas, altitude, thrust
            )
            opti.subject_to((thrust - drag) * speed - x[3] * oc.aero.g0 * u[1] >= 0)
            if optimizer.performance_model == "bada4":
                derivative, _ = optimizer.func_dynamics(
                    x, u, transcription.interval_dts[k]
                )
                optimizer._constrain_performance_model_domain(derivative)
        added += len(violating)
        opti.set_initial(opti.x, solution.value(opti.x))
        started = perf_counter()
        optimizer._solve_once(transcription.X, transcription.U, **kwargs)
        solution = optimizer._last_solution
        optimizer.step_climb_stats.append(
            dict(
                solution.stats(),
                stage="path_refinement",
                seconds=perf_counter() - started,
                added_points=len(violating),
            )
        )
        if not solution.stats()["success"]:
            return solution, {**report, "success": False}
    raise AssertionError("unreachable")
