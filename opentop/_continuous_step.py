"""Optional continuous cruise-climb pulses with optimized switching times."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import casadi as ca
from openap.aero import fpm, ft

import numpy as np

from ._path_validation import RejectedPhysicalSolution


def step_timestep_defaults(optimizer, dt_min, dt_max, *, method="continuous"):
    """Broad route/mesh defaults; neither bound fixes the initial duration."""
    if method == "legacy":
        return 30.0 if dt_min is None else dt_min, 900.0 if dt_max is None else dt_max
    average = max(1.0, float(optimizer.range) / 200.0) / optimizer.nodes
    return (
        max(1e-3, min(30.0, 0.25 * average)) if dt_min is None else dt_min,
        max(900.0, 2 * average) if dt_max is None else dt_max,
    )


def pulse_pattern(first, last, capacity=None):
    intervals = last - first
    maximum = min(3, max(0, (intervals - 1) // 4))
    capacity = maximum if capacity is None else capacity
    if not 0 <= capacity <= maximum:
        raise ValueError("step-climb pulse capacity exceeds the cruise mesh")
    level_count = intervals - 3 * capacity
    quotient, remainder = divmod(level_count, capacity + 1)
    levels, pulses, cursor = [], [], first
    for j in range(capacity + 1):
        count = quotient + (j < remainder)
        levels.append(list(range(cursor, cursor + count)))
        cursor += count
        if j < capacity:
            pulses.append((cursor, cursor + 1, cursor + 2))
            cursor += 3
    return levels, pulses


def allocate_levels(levels, remaining, minimum_dt, maximum_dt, dwell):
    counts = np.array([len(block) for block in levels])
    lower = np.maximum(minimum_dt * counts, dwell)
    upper = maximum_dt * counts
    if (
        (lower > upper).any()
        or remaining < lower.sum() - 1e-6
        or remaining > upper.sum() + 1e-6
    ):
        raise ValueError("step-climb initialization does not fit the interval bounds")
    low, high = 0.0, maximum_dt
    for _ in range(80):
        middle = (low + high) / 2
        if np.clip(middle * counts, lower, upper).sum() < remaining:
            low = middle
        else:
            high = middle
    return np.clip((low + high) / 2 * counts, lower, upper) / counts


def select_capacity(
    first, last, duration, minimum_ramp_s=100.0, *, options, dt_min, dt_max
):
    mesh = min(options.max_steps, 3, max(0, (last - first - 1) // 4))
    if dt_min > options.max_transition_duration_s:
        mesh = 0
    for count in range(mesh, -1, -1):
        levels, _ = pulse_pattern(first, last, count)
        sizes = np.array([len(block) for block in levels])
        remaining = duration - count * (2 * minimum_ramp_s + dt_min)
        lower = np.maximum(dt_min * sizes, options.min_level_duration_s if count else 0)
        upper = dt_max * sizes
        if (
            lower <= upper
        ).all() and lower.sum() - 1e-6 <= remaining <= upper.sum() + 1e-6:
            return count, dict(
                mesh_capacity=mesh, capacity=count, seed_duration_s=float(duration)
            )
    # A short cruise must remain a valid level-only problem. Its duration seed
    # can be clipped to the interval bounds without imposing a step.
    return 0, dict(
        mesh_capacity=mesh,
        capacity=0,
        seed_duration_s=float(duration),
        reason="no pulse fits the initial duration",
    )


@dataclass
class ContinuousStepClimb:
    transcription: Any
    options: Any
    first: int
    last: int
    amplitudes: Any
    levels: list[list[int]]
    pulses: list[tuple[int, int, int]]
    initialization: dict

    @classmethod
    def add(cls, tr, options, first, last):
        optimizer, opti = tr.optimizer, tr.opti
        if not optimizer._variable_timestep:
            raise ValueError(
                "continuous step_climb requires variable_timestep=True; "
                "use method='legacy' for fixed interval times"
            )
        initial = opti.initial()

        def value(expression):
            return np.asarray(opti.debug.value(expression, initial), dtype=float)

        X = value(ca.horzcat(*tr.X))
        U = value(ca.horzcat(*tr.U))
        Xc = np.stack([value(ca.horzcat(*states)) for states in tr.Xc])
        dts = np.array([float(value(dt)) for dt in tr.interval_dts])
        dt_min, dt_max, _ = optimizer._variable_timestep_bounds(
            float(dts.sum()),
            dt_min=tr.objective_kwargs.get("dt_min"),
            dt_max=tr.objective_kwargs.get("dt_max"),
        )
        amplitude = min(
            500.0,
            options.max_climb_rate_fpm,
            min(dt_max, options.max_transition_duration_s)
            * optimizer.MAX_VERTICAL_ACCELERATION
            / fpm,
        )
        ramp = max(dt_min, amplitude * fpm / optimizer.MAX_VERTICAL_ACCELERATION)
        if ramp > options.max_transition_duration_s:
            amplitude = (
                options.max_transition_duration_s
                * optimizer.MAX_VERTICAL_ACCELERATION
                / fpm
            )
            ramp = max(dt_min, options.max_transition_duration_s)

        def capacity(a, b, duration, minimum_ramp_s=ramp):
            return select_capacity(
                a,
                b,
                duration,
                minimum_ramp_s,
                options=options,
                dt_min=dt_min,
                dt_max=dt_max,
            )

        count, info = capacity(first, last, float(dts[first:last].sum()))
        fixed_altitude = bool(getattr(optimizer, "fix_alt", False))
        if fixed_altitude:
            count = 0
            info["reason"] = "fixed cruise altitude selects the level-only basis"
        supplied = tr.objective_kwargs.get("initial_guess") is not None
        coherent = False
        if (
            first > 0
            and last < len(tr.interval_dts)
            and not supplied
            and tr.objective_kwargs.get("initialization") != "legacy"
        ):
            from ._initialization import build_coherent_pulse_seed

            try:
                X, U, Xc, dts, info = build_coherent_pulse_seed(
                    optimizer,
                    tr,
                    first,
                    last,
                    pulse_pattern,
                    capacity,
                    maximum_capacity=min(options.max_steps, 3)
                    if dt_min <= options.max_transition_duration_s
                    else 0,
                    seed_amplitude_fpm=amplitude,
                    dt_min=dt_min,
                    dt_max=dt_max,
                    cruise_floor_m=15000 * ft
                    if optimizer.x_ub[2] >= 15000 * ft
                    else optimizer.x_lb[2],
                )
                count = info["capacity"]
                coherent = True
            except ValueError as exc:
                info["coherent_seed_fallback"] = str(exc)
        levels, pulses = pulse_pattern(first, last, count)
        amplitudes = opti.variable(count) if count else None
        if count:
            opti.subject_to(opti.bounded(0, amplitudes, options.max_climb_rate_fpm))
            opti.set_initial(amplitudes, amplitude)
        active = {
            node: j for j, (_, hold, down) in enumerate(pulses) for node in (hold, down)
        }
        for k in range(first, last + 1):
            if not fixed_altitude:
                opti.subject_to(
                    tr.U[k][1] == (amplitudes[active[k]] * fpm if k in active else 0)
                )
        if count:
            for block in levels:
                opti.subject_to(
                    sum(tr.interval_dts[k] for k in block)
                    >= options.min_level_duration_s
                )
        for up, _, down in pulses:
            opti.subject_to(tr.interval_dts[up] <= options.max_transition_duration_s)
            opti.subject_to(tr.interval_dts[down] <= options.max_transition_duration_s)

        if not coherent and not supplied:
            old_time = np.r_[0, np.cumsum(dts)]
            duration = float(dts[first:last].sum())
            for up, hold, down in pulses:
                dts[[up, hold, down]] = [ramp, dt_min, ramp]
            remaining = duration - sum(dts[k] for pulse in pulses for k in pulse)
            if not count:
                remaining = float(
                    np.clip(remaining, dt_min * (last - first), dt_max * (last - first))
                )
            level_dts = allocate_levels(
                levels,
                remaining,
                dt_min,
                dt_max,
                options.min_level_duration_s if count else 0,
            )
            for block, dt in zip(levels, level_dts):
                dts[block] = dt
            U[1, first : last + 1] = 0
            for _, hold, down in pulses:
                U[1, hold : down + 1] = amplitude * fpm
            gain = np.r_[
                0,
                np.cumsum(
                    0.5
                    * (U[1, first:last] + U[1, first + 1 : last + 1])
                    * dts[first:last]
                ),
            ]
            h0 = np.clip(
                X[2, last] - gain[-1],
                optimizer.x_lb[2],
                max(optimizer.x_lb[2], optimizer.x_ub[2] - gain[-1]),
            )
            X[2, first : last + 1] = h0 + gain
            times = np.r_[0, np.cumsum(dts)] + X[4, 0]
            for row in (0, 1, 3):
                X[row] = np.interp(times, old_time + X[4, 0], X[row])
            X[4] = times
            for k, dt in enumerate(dts):
                for j, tau in enumerate(tr.collocation_roots):
                    Xc[k, :, j] = (1 - tau) * X[:, k] + tau * X[:, k + 1]
                    if first <= k < last:
                        Xc[k, 2, j] = X[2, k] + dt * (
                            U[1, k] * tau + (U[1, k + 1] - U[1, k]) * tau * tau / 2
                        )
        if not supplied:
            for k, state in enumerate(tr.X):
                opti.set_initial(state, X[:, k])
                opti.set_initial(tr.U[k], U[:, k])
            for k, dt in enumerate(dts):
                opti.set_initial(tr.interval_dts[k], dt)
                for j, state in enumerate(tr.Xc[k]):
                    opti.set_initial(state, Xc[k, :, j])
            opti.set_initial(tr.ts_final, float(dts.sum()))
        else:
            if count:
                opti.set_initial(
                    amplitudes,
                    [
                        np.clip(U[1, hold] / fpm, 0, options.max_climb_rate_fpm)
                        for _, hold, _ in pulses
                    ],
                )
            info["supplied_initial_guess_preserved"] = True
        info.update(
            capacity=count,
            coherent=coherent,
            phase=[first, last],
            dt_min=dt_min,
            dt_max=dt_max,
        )
        return cls(tr, options, first, last, amplitudes, levels, pulses, info)

    def solve(self, optimizer, **kwargs):
        started = perf_counter()
        frame = optimizer._solve(self.transcription.X, self.transcription.U, **kwargs)
        sol = optimizer._last_solution
        dts = np.array([float(sol.value(dt)) for dt in self.transcription.interval_dts])
        rates = np.asarray(sol.value(ca.horzcat(*self.transcription.U)))[1] / fpm
        values = (
            np.asarray(sol.value(self.amplitudes)).reshape(-1)
            if self.pulses
            else np.empty(0)
        )
        active = np.zeros(self.last - self.first + 1, dtype=bool)
        for a, (_, hold, down) in zip(values, self.pulses):
            if a > 0.01:
                active[hold - self.first : down - self.first + 1] = True
        checks = dict(
            level_nodes=all(
                abs(rates[k]) <= 0.01
                for k in range(self.first, self.last + 1)
                if k not in {n for _, hold, down in self.pulses for n in (hold, down)}
            ),
            plateau_equalities=all(
                max(abs(rates[hold] - a), abs(rates[down] - a)) <= 0.01
                for a, (_, hold, down) in zip(values, self.pulses)
            ),
            plateau_duration=all(
                dts[hold] >= self.initialization["dt_min"] - 1e-6
                for _, hold, _ in self.pulses
            ),
            nonnegative_rates=bool(np.min(rates[self.first : self.last + 1]) >= -0.01),
            rate_ceiling=bool(
                np.max(rates[self.first : self.last + 1])
                <= self.options.max_climb_rate_fpm + 0.01
            ),
            level_dwell=all(
                float(dts[block].sum()) >= self.options.min_level_duration_s - 0.01
                for block in self.levels
            )
            if self.pulses
            else True,
            ramp_duration=all(
                max(dts[up], dts[down]) <= self.options.max_transition_duration_s + 0.01
                for up, _, down in self.pulses
            ),
            vertical_acceleration=bool(
                np.max(
                    np.abs(
                        np.diff(rates[self.first : self.last + 1])
                        / dts[self.first : self.last]
                    )
                )
                <= optimizer.MAX_VERTICAL_ACCELERATION / fpm + 0.001
            ),
        )
        if not all(checks.values()):
            optimizer._last_solution = RejectedPhysicalSolution(
                sol, "Step_Climb_Profile_Failed"
            )
        optimizer.step_climb_penalty = 0.0
        validation = optimizer.physical_validation_diagnostics or {}
        optimizer.step_climb_stats = validation.get("attempts", [])
        optimizer.step_climb_diagnostics = dict(
            success=optimizer.success,
            method="continuous",
            capacity=len(self.pulses),
            step_count=int(np.count_nonzero(values > 0.01)),
            amplitudes_fpm=values.tolist(),
            activation=active,
            profile_checks=checks,
            initialization=self.initialization,
            level_durations_s=[float(dts[block].sum()) for block in self.levels],
            climb_seconds=float(
                sum(
                    dts[hold]
                    for a, (_, hold, _) in zip(values, self.pulses)
                    if a > 0.01
                )
            ),
            path_audit=validation,
            solves=validation.get("solves", 1),
            search_seconds=perf_counter() - started,
            limitation=(
                "Finite pulse basis; inactive slots retain mesh/time allocation. "
                "No global step-count guarantee."
            ),
        )
        return frame
