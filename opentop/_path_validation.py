"""Batched checks and bounded repair of each phase's physical constraints.

The audit checks a finite grid; it is not a continuous-path certificate. Climb
retains its interval-average energy model, including its acceleration term.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from operator import index
from time import perf_counter
from typing import Any

import casadi as ca
import openap.casadi as oc
from openap.aero import ft, kts

import numpy as np

from ._physical_constraints import ConstraintRow, fuel_row, physical_rows


@dataclass(frozen=True, slots=True)
class PhysicalValidationOptions:
    """Audit configured physical inequalities after each single-aircraft solve.

    At most ``max_repairs`` additional solves follow the initial solve. Force
    residuals are divided by available thrust; the BADA fuel-domain residual
    retains kg/s units. Their acceptance tolerance is 1e-7 in the reported unit.
    Repair adds a 1e-4 dimensionless force buffer, never a kg/s fuel buffer.
    """

    enabled: bool = True
    max_repairs: int = 2
    points_per_interval: int = 161
    force_buffer: float = 1e-4

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise TypeError("physical validation enabled must be a bool")
        if isinstance(self.max_repairs, bool) or index(self.max_repairs) < 0:
            raise ValueError("physical validation max_repairs must be nonnegative")
        n = index(self.points_per_interval)
        if n < 161 or n % 2 != 1:
            raise ValueError(
                "physical validation points_per_interval must be odd and >=161"
            )
        if not np.isfinite(self.force_buffer) or self.force_buffer < 0:
            raise ValueError(
                "physical validation force_buffer must be finite and nonnegative"
            )


def resolve_validation(value):
    if isinstance(value, PhysicalValidationOptions):
        return value
    if isinstance(value, bool):
        return PhysicalValidationOptions(enabled=value)
    raise TypeError("physical_validation must be a bool or PhysicalValidationOptions")


@dataclass(slots=True)
class RejectedPhysicalSolution:
    solution: Any
    status: str

    def stats(self):
        return dict(self.solution.stats(), success=False, return_status=self.status)

    def value(self, *args):
        return self.solution.value(*args)


@lru_cache(maxsize=32)
def interpolation_weights(roots, density):
    tau = np.linspace(0, 1, density)
    nodes = (0.0, *roots)
    result = np.ones((density, len(nodes)))
    for j in range(len(nodes)):
        for i, node in enumerate(nodes):
            if i != j:
                result[:, j] *= (tau - node) / (nodes[j] - node)
    result.setflags(write=False)
    return result


def interpolate_arrays(X, Xc, U, dts, roots, density=161):
    weights = interpolation_weights(tuple(roots), density)
    values = np.concatenate((X[:, :-1].T[:, :, None], Xc), axis=2)
    states = np.einsum("nsc,tc->nst", values, weights)
    # The transcription uses the next node, not the polynomial endpoint limit.
    states[:, :, -1] = X[:, 1:].T
    tau = np.linspace(0, 1, density)
    controls = (1 - tau)[None, None, :] * U[:, :-1].T[:, :, None] + tau[
        None, None, :
    ] * U[:, 1:].T[:, :, None]
    return dict(
        X=states.transpose(1, 0, 2).reshape(X.shape[0], -1),
        U=controls.transpose(1, 0, 2).reshape(U.shape[0], -1),
        interval=np.repeat(np.arange(len(dts)), density),
        tau=np.tile(tau, len(dts)),
    )


def point_key(interval, tau, *, interval_dependent=False):
    if not interval_dependent:
        if tau == 0.0:
            return ("node", interval)
        if tau == 1.0:
            return ("node", interval + 1)
    return ("interval", interval, float(tau))


def interval_contexts(tr):
    if tr.physical_contract != "climb":
        return [ca.DM.zeros(2) for _ in tr.interval_dts]
    optimizer = tr.optimizer
    contexts = []
    for k, dt in enumerate(tr.interval_dts):
        left = oc.aero.mach2tas(tr.U[k][0], tr.X[k][2], dT=optimizer.dT)
        right = oc.aero.mach2tas(tr.U[k + 1][0], tr.X[k + 1][2], dT=optimizer.dT)
        contexts.append(
            ca.vertcat((tr.X[k + 1][2] - tr.X[k][2]) / dt, (right - left) / dt)
        )
    return contexts


def configured_rows(optimizer, state, control, context, contract):
    if contract == "climb":
        speed = oc.aero.mach2tas(control[0], state[2], dT=optimizer.dT)
        thrust = optimizer._thrust_climb(speed / kts, state[2] / ft)
        drag = optimizer.drag.clean(
            state[3], speed / kts, state[2] / ft, dT=optimizer.dT
        )
        residual = (
            (thrust - drag) / state[3] - oc.aero.g0 / speed * context[0] - context[1]
        )
        scale = ca.fmax(1, thrust) / state[3]
        rows = [
            ConstraintRow(
                "interval_average_energy",
                "m/s^2",
                "1",
                residual,
                0,
                scale,
                residual / scale,
            )
        ]
        if optimizer.performance_model == "bada4":
            derivative, _ = optimizer.func_dynamics(state, control, 1.0)
            rows.append(fuel_row(optimizer, derivative))
        return rows
    return physical_rows(
        optimizer,
        state,
        control,
        energy=contract != "cruise",
        thrust_kind="cruise" if contract == "descent" else "climb",
    )


class PhysicalAudit:
    """Build-specific evaluator; caches only the last immutable solution."""

    def __init__(self, tr, density=161):
        self.tr = tr
        self.density = density
        self.interval_dependent = tr.physical_contract == "climb"
        self.contexts = interval_contexts(tr)
        state, control, context = (
            ca.MX.sym("state", 5),
            ca.MX.sym("control", 3),
            ca.MX.sym("context", 2),
        )
        rows = configured_rows(
            tr.optimizer, state, control, context, tr.physical_contract
        )
        self.rows = [
            dict(
                name=r.name,
                native_unit=r.raw_unit,
                audit_unit=r.audit_unit,
                tolerance=r.audit_atol,
            )
            for r in rows
        ]
        self.evaluate = ca.Function(
            "physical_path_audit",
            [state, control, context],
            [ca.vertcat(*(r.audit_residual for r in rows))],
        )
        self._cache = None
        self._previous_violating_intervals = set()
        self.last_recurrent_intervals = []

    def sample(self, solution):
        if self._cache is not None and self._cache[0] is solution:
            return self._cache[1]
        tr = self.tr
        X = np.asarray(solution.value(ca.horzcat(*tr.X)), dtype=float)
        U = np.asarray(solution.value(ca.horzcat(*tr.U)), dtype=float)
        Xc = np.stack(
            [
                np.asarray(solution.value(ca.horzcat(*states)), dtype=float)
                for states in tr.Xc
            ]
        )
        dts = np.array([float(solution.value(dt)) for dt in tr.interval_dts])
        invalid = [
            name
            for name, value in dict(X=X, Xc=Xc, U=U, dts=dts).items()
            if not np.isfinite(value).all()
        ]
        if not (dts > 0).all():
            invalid.append("nonpositive_durations")
        data = interpolate_arrays(X, Xc, U, dts, tr.collocation_roots, self.density)
        contexts = np.asarray(
            solution.value(ca.horzcat(*self.contexts)), dtype=float
        ).reshape(2, -1)
        context = np.repeat(contexts, self.density, axis=1)
        if not np.isfinite(context).all():
            invalid.append("interval_context")
        if not np.isfinite(data["X"]).all() or not np.isfinite(data["U"]).all():
            invalid.append("interpolated_state_or_control")
        data["margins"] = (
            np.full((len(self.rows), len(data["tau"])), np.nan)
            if invalid
            else np.asarray(self.evaluate(data["X"], data["U"], context), dtype=float)
        )
        data["invalid_inputs"] = invalid
        if isinstance(solution, ca.OptiSol):
            self._cache = (solution, data)
        return data

    def audit(self, solution):
        started = perf_counter()
        data = self.sample(solution)
        values = data["margins"]
        finite = not data["invalid_inputs"] and bool(np.isfinite(values).all())
        rows = []
        for j, row in enumerate(self.rows):
            finite_row = bool(np.isfinite(values[j]).all())
            worst = int(np.argmin(values[j])) if finite_row else 0
            minimum = float(values[j, worst]) if finite_row else float("nan")
            rows.append(
                dict(
                    **row,
                    minimum=minimum,
                    interval=int(data["interval"][worst]),
                    tau=float(data["tau"][worst]),
                    success=finite_row and minimum >= -row["tolerance"],
                )
            )
        return dict(
            success=finite and all(row["success"] for row in rows),
            finite=finite,
            invalid_inputs=data["invalid_inputs"],
            constraints=rows,
            points_per_interval=self.density,
            sampled_points=len(data["tau"]),
            scope=self.tr.physical_contract,
            model="interval-average acceleration energy"
            if self.interval_dependent
            else "configured pointwise force/fuel constraints",
            audit_seconds=perf_counter() - started,
        ), data

    def select_points(self, data):
        values = data["margins"]
        points = {}
        violating = set()
        n = self.density
        for k in range(len(self.tr.interval_dts)):
            local = values[:, k * n : (k + 1) * n]
            for j, row in enumerate(self.rows):
                bad = local[j] < -row["tolerance"]
                edges = np.diff(np.r_[False, bad, False].astype(int))
                for start, stop in zip(
                    np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
                ):
                    violating.add(k)
                    worst = int(start + np.argmin(local[j, start : stop + 1]))
                    for i in (
                        max(0, worst - 1),
                        worst,
                        min(n - 1, worst + 1),
                        max(0, int(start) - 1),
                        min(n - 1, int(stop) + 1),
                    ):
                        tau = float(data["tau"][k * n + i])
                        points.setdefault(
                            point_key(
                                k, tau, interval_dependent=self.interval_dependent
                            ),
                            (k, tau),
                        )
        recurrent = violating & self._previous_violating_intervals
        self._previous_violating_intervals.update(violating)
        self.last_recurrent_intervals = sorted(recurrent)
        # A moving peak can escape a narrow targeted neighborhood. On a repeat
        # violation constrain the complete audit grid in that interval only.
        for k in sorted(recurrent):
            for tau in data["tau"][k * n : (k + 1) * n]:
                tau = float(tau)
                points.setdefault(
                    point_key(k, tau, interval_dependent=self.interval_dependent),
                    (k, tau),
                )
        neighbors = {
            q
            for k in violating
            for q in (k - 1, k + 1)
            if 0 <= q < len(self.tr.interval_dts)
        }
        fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
        for root in self.tr.collocation_roots:
            if not any(abs(root - tau) < 1e-12 for tau in fractions):
                fractions.append(root)
        for k in sorted(violating | neighbors):
            for tau in sorted(fractions):
                points.setdefault(
                    point_key(k, tau, interval_dependent=self.interval_dependent),
                    (k, tau),
                )
        return points

    def constrain(self, interval, tau, buffer):
        tr = self.tr
        margins = self.evaluate(
            tr.state_at(interval, tau),
            tr.control_at(interval, tau),
            self.contexts[interval],
        )
        for j, row in enumerate(self.rows):
            tr.opti.subject_to(
                margins[j] >= (buffer if row["audit_unit"] == "1" else 0.0)
            )


def solve_validated(optimizer, run, options):
    """Run one solve and only the repairs needed for its configured contract."""
    tr = optimizer._last_transcription
    report = dict(
        enabled=options.enabled,
        scope=tr.physical_contract,
        max_repairs=options.max_repairs,
        success=False,
        attempts=[],
    )
    report["preliminary_solver_calls"] = getattr(
        optimizer, "preliminary_solver_calls", 0
    )
    optimizer.physical_validation_diagnostics = report
    if not options.enabled or tr.physical_contract is None:
        frame = run()
        report.update(
            status="disabled" if not options.enabled else "unregistered_formulation",
            success=None,
            solves=1,
            total_solver_calls=1 + report["preliminary_solver_calls"],
        )
        return frame
    validator = PhysicalAudit(tr, options.points_per_interval)
    added = set()
    started = perf_counter()
    reason = "repair_budget_exhausted"
    frame = None
    for attempt in range(options.max_repairs + 1):
        began = perf_counter()
        frame = run()
        solution = optimizer._last_solution
        solver = dict(solution.stats())
        record = dict(
            attempt=attempt,
            solver_status=solver.get("return_status"),
            solver_success=bool(solver.get("success")),
            iterations=int(solver.get("iter_count", 0)),
            solve_seconds=perf_counter() - began,
            constraints=int(tr.opti.ng),
        )
        report["attempts"].append(record)
        if not record["solver_success"]:
            reason = "solver_failed"
            break
        audit, data = validator.audit(solution)
        record["audit"] = audit
        if not audit["finite"]:
            reason = "nonfinite_physical_audit"
            break
        if audit["success"]:
            reason = "passed"
            break
        if attempt == options.max_repairs:
            break
        candidates = validator.select_points(data)
        fresh = [(key, point) for key, point in candidates.items() if key not in added]
        if not fresh:
            reason = "no_new_repair_points"
            break
        primal = np.asarray(solution.value(tr.opti.x), dtype=float)
        for key, (k, tau) in fresh:
            validator.constrain(k, tau, options.force_buffer)
            added.add(key)
        tr.opti.set_initial(tr.opti.x, primal)
        record["added_points"] = [point for _, point in fresh]
        record["added_rows"] = len(fresh) * len(getattr(validator, "rows", []))
        record["recurrent_intervals"] = getattr(
            validator, "last_recurrent_intervals", []
        )
        record["force_buffer"] = options.force_buffer
    report.update(
        status=reason,
        success=reason == "passed",
        solves=len(report["attempts"]),
        elapsed_seconds=perf_counter() - started,
        added_points=len(added),
        total_iterations=sum(r["iterations"] for r in report["attempts"]),
        total_solver_seconds=sum(r["solve_seconds"] for r in report["attempts"]),
    )
    report["total_solver_calls"] = report["solves"] + report["preliminary_solver_calls"]
    if reason != "passed":
        optimizer._last_solution = RejectedPhysicalSolution(
            optimizer._last_solution, "Physical_Validation_" + reason
        )
    return frame
