"""Numeric full-flight initial guesses; these functions add no constraints."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from math import atan2
from typing import Any

import casadi as ca
import openap.aero as aero
from openap.aero import fpm, ft
from scipy.integrate import solve_ivp
from scipy.optimize import root

import numpy as np

from . import _dynamics


@dataclass(frozen=True)
class InitialSeed:
    """States and continuous controls at the transcription's sample locations."""

    X: np.ndarray
    U: np.ndarray
    Xc: np.ndarray
    dts: np.ndarray
    diagnostics: dict[str, Any]


def _rhs_function(optimizer, *, still_air=False):
    """Build only the physical derivatives, without evaluating an objective."""
    x, u = ca.MX.sym("seed_x", 5), ca.MX.sym("seed_u", 3)
    derivative = _dynamics.xdot(
        x,
        u,
        fuelflow=optimizer.fuelflow,
        dT=optimizer.dT,
        wind=None if still_air else optimizer.wind,
    )
    function = ca.Function("seed_derivatives", [x, u], [derivative])

    def evaluate(state, control):
        return np.asarray(function(state, control), dtype=float)

    return evaluate


def _control_bounds(optimizer):
    count = optimizer.nodes + 1
    lower = np.repeat(np.asarray(optimizer.u_lb, dtype=float)[:, None], count, axis=1)
    upper = np.repeat(np.asarray(optimizer.u_ub, dtype=float)[:, None], count, axis=1)
    lower[:, 0], upper[:, 0] = optimizer.u_0_lb, optimizer.u_0_ub
    lower[:, -1], upper[:, -1] = optimizer.u_f_lb, optimizer.u_f_ub
    return lower, upper


def _wind_function(wind):
    """Use the symbolic wind API for batches; PolyWind's numeric API is scalar."""
    point = ca.MX.sym("seed_wind_point", 4)
    inputs = [point[index] for index in range(4)]
    return ca.Function(
        "seed_wind",
        [point],
        [ca.vertcat(wind.calc_u(*inputs), wind.calc_v(*inputs))],
    )


def _phase_rates(duration, first, last, nodes, start_h, end_h, target_h, lower, upper):
    """Bounded phase rates with a common, exactly integrated cruise altitude."""
    dt = duration / nodes
    climb_shape = np.minimum(1.0, (1 - np.linspace(0, 1, first + 1)) / 0.3)
    descent_shape = np.minimum(np.linspace(0, 1, nodes - last + 1) / 0.3, 1.0)
    descent_shape[-1] *= 2 / 3

    def phase(shape, direction, amplitude, start):
        result = direction * amplitude * shape
        return np.clip(
            result,
            lower[1, start : start + len(shape)],
            upper[1, start : start + len(shape)],
        )

    def gain(rates):
        return dt * float(np.sum((rates[:-1] + rates[1:]) / 2))

    climb_min = gain(phase(climb_shape, 1, 0.0, 0))
    climb_max = gain(phase(climb_shape, 1, 2500 * fpm, 0))
    descent_min = -gain(phase(descent_shape, -1, 0.0, last))
    descent_max = -gain(phase(descent_shape, -1, 2500 * fpm, last))
    lowest = max(start_h + climb_min, end_h + descent_min)
    highest = min(start_h + climb_max, end_h + descent_max)
    if lowest > highest:
        raise ValueError("Default initial phase durations cannot fit endpoint rates")
    cruise_h = float(np.clip(target_h, lowest, highest))

    def fit(shape, direction, requested, start):
        lo, hi = 0.0, 2500 * fpm
        for _ in range(52):
            middle = (lo + hi) / 2
            if direction * gain(phase(shape, direction, middle, start)) < requested:
                lo = middle
            else:
                hi = middle
        return phase(shape, direction, (lo + hi) / 2, start)

    rates = np.zeros(nodes + 1)
    rates[: first + 1] = fit(climb_shape, 1, cruise_h - start_h, 0)
    rates[last:] = fit(descent_shape, -1, cruise_h - end_h, last)
    heights = start_h + np.r_[0, np.cumsum(dt * (rates[:-1] + rates[1:]) / 2)]
    return rates, heights, cruise_h


def build_full_flight_seed(
    optimizer,
    phase_indices,
    *,
    collocation_roots,
    variable_timestep=False,
    dt_min=None,
    dt_max=None,
):
    """Construct a phase-aware default guess without a preliminary NLP solve.

    The projected route and total duration use sampled wind. Vertical motion is
    the exact integral of the piecewise-linear rate controls. Mass is integrated
    along that prescribed route using the actual performance model. Horizontal
    dynamics are approximate: the endpoints remain on the requested route.
    Callers must bypass this helper when the user supplies an initial trajectory.
    """
    nodes = optimizer.nodes
    first, last = phase_indices
    if not 0 < first < last < nodes:
        raise ValueError("Full-flight seed needs nonempty climb, cruise and descent")
    start = np.asarray(optimizer.x_0_lb, dtype=float).copy()
    end = np.asarray(optimizer.x_f_lb, dtype=float)
    start[3] = optimizer.mass_init
    target = end[:2] - start[:2]
    distance = float(np.linalg.norm(target))
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("Full-flight seed needs a finite positive route length")
    roots = np.asarray(collocation_roots, dtype=float)
    lower, upper = _control_bounds(optimizer)
    lower[1, : first + 1] = np.maximum(lower[1, : first + 1], 0)
    upper[1, last:] = np.minimum(upper[1, last:], 0)
    lower[1, first : last + 1] = 0
    upper[1, first : last + 1] = 0
    direction = target / distance
    normal = np.array([direction[1], -direction[0]])
    bearing = atan2(direction[0], direction[1])
    cruise_mach = float(
        np.clip(
            optimizer.aircraft["cruise"].get("mach", 0.78),
            lower[0, first],
            upper[0, first],
        )
    )
    start_mach = float(np.clip(0.28, lower[0, 0], upper[0, 0]))
    end_mach = float(np.clip(0.28, lower[0, -1], upper[0, -1]))
    mach_change = getattr(optimizer, "_mach_change_limit", lambda: np.inf)()
    cruise_mach = min(
        cruise_mach,
        start_mach + first * mach_change,
        end_mach + (nodes - last) * mach_change,
    )
    cruise_min = optimizer._cruise_mach_min()
    if cruise_min is not None and optimizer.x_ub[2] >= 15000 * ft:
        cruise_mach = max(cruise_mach, cruise_min)
    mach = np.r_[
        np.linspace(start_mach, cruise_mach, first + 1),
        np.full(last - first, cruise_mach),
        np.linspace(cruise_mach, end_mach, nodes - last + 1)[1:],
    ]
    target_h = min(
        float(optimizer.aircraft["cruise"]["height"]), float(optimizer.x_ub[2])
    )
    progress = np.linspace(0, 1, nodes + 1)
    xy = start[:2, None] + target[:, None] * progress
    duration = distance / max(
        100.0, float(aero.mach2tas(cruise_mach, target_h, dT=optimizer.dT)) * 0.8
    )
    duration_min = max(
        1.0,
        nodes * (float(dt_min) if variable_timestep and dt_min is not None else 0.0),
    )
    duration_max = float(optimizer.x_f_ub[4]) - start[4]
    if variable_timestep and dt_max is not None:
        duration_max = min(duration_max, nodes * float(dt_max))
    if duration_max < duration_min:
        raise ValueError("Default initial duration bounds are incompatible")
    duration = float(np.clip(duration, duration_min, duration_max))
    wind_limited = False
    headings = np.full(nodes + 1, bearing)
    rhs = _rhs_function(optimizer)
    wind_function = None if optimizer.wind is None else _wind_function(optimizer.wind)

    for _ in range(8):
        rates, heights, cruise_h = _phase_rates(
            duration, first, last, nodes, start[2], end[2], target_h, lower, upper
        )
        times = start[4] + progress * duration
        tas = np.asarray(aero.mach2tas(mach, heights, dT=optimizer.dT), dtype=float)
        horizontal_tas = tas / np.sqrt(1 + (rates / tas) ** 2)
        wind = np.zeros((2, nodes + 1))
        if wind_function is not None:
            wind = np.asarray(
                wind_function(np.vstack((xy, heights, times))), dtype=float
            )
        if not np.isfinite(wind).all():
            raise ValueError("Wind is nonfinite at the default initial route")
        crosswind = normal @ wind
        ratio = crosswind / horizontal_tas
        wind_limited |= bool(np.any(np.abs(ratio) >= 0.95))
        headings = bearing - np.arcsin(np.clip(ratio, -0.95, 0.95))
        headings += 2 * np.pi * round((optimizer.u_guess[2] - bearing) / (2 * np.pi))
        headings = np.clip(headings, lower[2], upper[2])
        along = horizontal_tas * np.cos(headings - bearing) + direction @ wind
        wind_limited |= bool(np.any(along < 20.0))
        along = np.maximum(along, 20.0)
        mean_speed = float(np.sum((along[:-1] + along[1:]) / 2) / nodes)
        requested_duration = float(
            np.clip(distance / mean_speed, duration_min, duration_max)
        )
        # The next wind sample follows the nonuniform progress implied by speed.
        cumulative = np.r_[0, np.cumsum((along[:-1] + along[1:]) / 2)]
        xy = start[:2, None] + target[:, None] * (cumulative / cumulative[-1])
        if abs(requested_duration - duration) < 0.01:
            duration = requested_duration
            break
        duration = (duration + requested_duration) / 2

    rates, heights, cruise_h = _phase_rates(
        duration, first, last, nodes, start[2], end[2], target_h, lower, upper
    )
    dts = np.full(nodes, duration / nodes)
    times = start[4] + np.r_[0, np.cumsum(dts)]
    controls = np.clip(np.vstack((mach, rates, headings)), lower, upper)
    states = np.vstack((xy, heights, np.full(nodes + 1, start[3]), times))
    collocation = np.empty((nodes, 5, len(roots)))
    nonpositive_flow = False

    def sample(k, tau, mass):
        duration_k = dts[k]
        state = (1 - tau) * states[:, k] + tau * states[:, k + 1]
        state[2] = heights[k] + duration_k * (
            rates[k] * tau + (rates[k + 1] - rates[k]) * tau**2 / 2
        )
        state[3] = mass
        return state

    for k, duration_k in enumerate(dts):
        mass = states[3, k]
        fractions = np.unique(np.r_[0.0, roots, 1.0])
        samples = {}
        for left, right in pairwise(fractions):
            # Bound the numerical quadrature step without changing the NLP mesh.
            subdivisions = max(1, int(np.ceil((right - left) * duration_k / 60)))
            points = np.linspace(left, right, subdivisions + 1)
            for a, b in pairwise(points):
                elapsed = (b - a) * duration_k

                def mass_rate(tau, value):
                    control = controls[:, k] + tau * (
                        controls[:, k + 1] - controls[:, k]
                    )
                    return float(rhs(sample(k, tau, value), control).reshape(-1)[3])

                q1 = mass_rate(a, mass)
                q2 = mass_rate((a + b) / 2, mass + elapsed * q1 / 2)
                q3 = mass_rate((a + b) / 2, mass + elapsed * q2 / 2)
                q4 = mass_rate(b, mass + elapsed * q3)
                nonpositive_flow |= max(q1, q2, q3, q4) >= 0
                mass += elapsed * (q1 + 2 * q2 + 2 * q3 + q4) / 6
                if not np.isfinite(mass) or mass <= 0:
                    raise ValueError("Default initial mass integration is invalid")
            samples[right] = mass
        states[3, k + 1] = mass
        for j, tau in enumerate(roots):
            collocation[k, :, j] = sample(k, tau, samples[tau])

    if not all(np.isfinite(value).all() for value in (states, controls, collocation)):
        raise ValueError("Default full-flight initialization is nonfinite")
    return InitialSeed(
        states,
        controls,
        collocation,
        dts,
        {
            "method": "phase-aware route and wind estimate with model mass integration",
            "seed_only": True,
            "new_constraints": False,
            "total_duration_s": float(duration),
            "cruise_height_m": float(cruise_h),
            "wind_used": optimizer.wind is not None,
            "wind_speed_guard_used": wind_limited,
            "mass_kg": [float(states[3, 0]), float(states[3, -1])],
            "mass_monotone": bool(np.all(np.diff(states[3]) <= 0)),
            "nonpositive_fuel_sample": nonpositive_flow,
            "landing_mass_bound_residual_kg": max(
                0.0, float(states[3, -1] - optimizer.x_f_ub[3])
            ),
            "horizontal_dynamics": "approximate; prescribed endpoint route",
        },
    )


def build_coherent_pulse_seed(
    optimizer,
    transcription,
    first,
    last,
    pattern,
    capacity,
    *,
    maximum_capacity=3,
    seed_amplitude_fpm=500.0,
    dt_min=30.0,
    dt_max=900.0,
    cruise_floor_m=None,
):
    """Port of the tested continuous-pulse seed, with a still-air seed policy.

    Actual wind remains enabled in the optimizer. Ignoring it here preserves the
    known initial guess for the wind benchmark; this is recorded in diagnostics.
    """
    nodes = len(transcription.interval_dts)
    initial_state = np.array(optimizer.x_0_lb, dtype=float)
    initial_state[3] = optimizer.mass_init
    destination = np.array(optimizer.x_f_lb, dtype=float)
    displacement_target = destination[:2] - initial_state[:2]
    distance = float(np.linalg.norm(displacement_target))
    projected_heading = atan2(displacement_target[0], displacement_target[1])
    endpoint_headings = (optimizer.u_0_lb[2], optimizer.u_f_lb[2])
    roots, weights = np.polynomial.legendre.leggauss(20)
    tau, weights = (roots + 1) / 2, weights / 2
    collocation = np.array(transcription.collocation_roots)
    climb_s = np.linspace(0, 1, first + 1)
    descent_s = np.linspace(0, 1, nodes - last + 1)
    climb_rates = 1500 * fpm * np.minimum(1, (1 - climb_s) / 0.3)
    descent_rates = -1500 * fpm * np.minimum(descent_s / 0.3, 1)
    descent_rates[-1] = -1000 * fpm
    capacity_value = min(maximum_capacity, max(0, (last - first - 1) // 4))
    root_reports = []

    lower_controls, upper_controls = _control_bounds(optimizer)
    start_mach = float(np.clip(0.28, lower_controls[0, 0], upper_controls[0, 0]))
    end_mach = float(np.clip(0.28, lower_controls[0, -1], upper_controls[0, -1]))
    mach_change = getattr(optimizer, "_mach_change_limit", lambda: np.inf)()
    cruise_mach = min(
        0.78,
        float(np.min(upper_controls[0, first : last + 1])),
        start_mach + first * mach_change,
        end_mach + (nodes - last) * mach_change,
    )
    required_mach = optimizer._cruise_mach_min()
    if optimizer.x_ub[2] >= 15000 * ft and required_mach is not None:
        if required_mach > cruise_mach:
            raise ValueError(
                "Coherent seed cannot fit cruise Mach and transition bounds"
            )
        cruise_mach = max(cruise_mach, required_mach)
    mach = np.r_[
        np.linspace(start_mach, cruise_mach, first + 1),
        np.full(last - first, cruise_mach),
        np.linspace(cruise_mach, end_mach, nodes - last + 1)[1:],
    ]
    mach = np.clip(mach, lower_controls[0], upper_controls[0])
    climb_rates = np.clip(
        climb_rates, lower_controls[1, : first + 1], upper_controls[1, : first + 1]
    )
    descent_rates = np.clip(
        descent_rates, lower_controls[1, last:], upper_controls[1, last:]
    )
    minimum_ramp_s = max(
        dt_min, seed_amplitude_fpm / (optimizer.MAX_VERTICAL_ACCELERATION / fpm)
    )
    pulse_seconds = 2 * minimum_ramp_s + dt_min
    peak_height = min(10500.0, float(optimizer.x_ub[2]))
    floor = (
        (15000 * ft if optimizer.x_ub[2] >= 15000 * ft else 100 * ft)
        if cruise_floor_m is None
        else cruise_floor_m
    )
    peak_height = min(peak_height, max(floor, distance * 0.012))
    while (
        capacity_value
        and peak_height
        - capacity_value * seed_amplitude_fpm * fpm * (minimum_ramp_s + dt_min)
        < floor
    ):
        capacity_value -= 1
    still_air_rhs = (
        _rhs_function(optimizer, still_air=True) if optimizer.wind is not None else None
    )

    def derivative(state, control):
        if still_air_rhs is not None:
            return still_air_rhs(state, control)
        result, _ = optimizer.func_dynamics(state, control, 1.0)
        return np.asarray(result)

    for _ in range(4):
        levels, pulses = pattern(first, last, capacity_value)
        level_count = sum(len(block) for block in levels)
        toc_h = peak_height - capacity_value * seed_amplitude_fpm * fpm * (
            minimum_ramp_s + dt_min
        )
        climb_T = (toc_h - initial_state[2]) / np.trapezoid(climb_rates, climb_s)
        descent_T = (peak_height - destination[2]) / (
            -np.trapezoid(descent_rates, descent_s)
        )
        vs = np.zeros(nodes + 1)
        vs[: first + 1], vs[last:] = climb_rates, descent_rates
        for _, hold, down in pulses:
            vs[hold : down + 1] = seed_amplitude_fpm * fpm

        def arrays(level_dt, heading):
            dt = np.r_[
                np.full(first, climb_T / first),
                np.zeros(last - first),
                np.full(nodes - last, descent_T / (nodes - last)),
            ]
            for block in levels:
                dt[block] = level_dt
            for up, hold, down in pulses:
                dt[[up, hold, down]] = [minimum_ramp_s, dt_min, minimum_ramp_s]
            h = initial_state[2] + np.r_[0, np.cumsum(0.5 * (vs[:-1] + vs[1:]) * dt)]
            headings = np.full(nodes + 1, heading)
            headings[0], headings[-1] = endpoint_headings
            controls = np.vstack((mach, vs, headings))
            return dt, h, controls

        def displacement(level_dt, heading):
            dt, h, controls = arrays(level_dt, heading)
            result = np.zeros(2)
            for k, duration in enumerate(dt):
                heights = (
                    h[k]
                    + vs[k] * duration * tau
                    + 0.5 * (vs[k + 1] - vs[k]) * duration * tau**2
                )
                states = np.vstack(
                    (
                        np.zeros((2, len(tau))),
                        heights,
                        np.full(len(tau), initial_state[3]),
                        np.zeros(len(tau)),
                    )
                )
                control = controls[:, k, None] + tau * (
                    controls[:, k + 1, None] - controls[:, k, None]
                )
                result += duration * np.sum(
                    weights * derivative(states, control)[:2], axis=1
                )
            return result

        initial_level_dt = max(
            dt_min,
            (7200.0 - climb_T - descent_T - pulse_seconds * capacity_value)
            / level_count,
        )
        solved = root(
            lambda z: (
                (displacement(z[0] * 1000, z[1]) - displacement_target) / distance
            ),
            [initial_level_dt / 1000, projected_heading],
            method="hybr",
            options={"xtol": 1e-10, "maxfev": 100},
        )
        level_dt, heading = float(solved.x[0] * 1000), float(solved.x[1])
        residual = displacement(level_dt, heading) - displacement_target
        if not solved.success or np.max(np.abs(residual)) > 1e-3:
            raise ValueError(
                f"Coherent endpoint root failed: {solved.message}; {residual}"
            )
        dt, h, controls = arrays(level_dt, heading)
        selected, capacity_details = capacity(
            first, last, float(dt[first:last].sum()), minimum_ramp_s
        )
        root_reports.append(
            {
                "capacity": capacity_value,
                "selected_capacity": selected,
                "success": bool(solved.success),
                "message": str(solved.message),
                "nfev": int(solved.nfev),
                "level_dt_s": level_dt,
                "interior_heading_rad": heading,
                "residual_m": residual.tolist(),
                "capacity_details": capacity_details,
            }
        )
        if selected == capacity_value:
            break
        capacity_value = selected
    else:
        raise ValueError("Coherent seed capacity did not stabilize")
    if not np.all((dt >= dt_min - 1e-8) & (dt <= dt_max + 1e-8)):
        raise ValueError("Coherent initial interval durations violate their bounds")
    times = initial_state[4] + np.r_[0, np.cumsum(dt)]
    states = np.zeros((5, nodes + 1))
    states[:, 0] = initial_state
    xc = np.zeros((nodes, 5, len(collocation)))
    integration_ok = True
    for k, duration in enumerate(dt):

        def rhs(elapsed, state):
            fraction = elapsed / duration
            control = controls[:, k] + fraction * (controls[:, k + 1] - controls[:, k])
            return derivative(state, control).reshape(-1)

        solution = solve_ivp(
            rhs,
            [0, float(duration)],
            states[:, k],
            rtol=1e-10,
            atol=1e-9,
            max_step=30,
            dense_output=True,
        )
        integration_ok &= bool(solution.success)
        states[:, k + 1] = solution.y[:, -1]
        xc[k] = solution.sol(collocation * duration)
    if not integration_ok:
        raise ValueError("Coherent initial numerical integration failed")
    position_error = float(np.linalg.norm(states[:2, -1] - destination[:2]))
    h_error = float(np.max(np.abs(states[2] - h)))
    t_error = float(np.max(np.abs(states[4] - times)))
    if position_error > 1e-3 or h_error > 1e-6 or t_error > 1e-6:
        raise ValueError("Coherent integration endpoint/height/time check failed")
    # Use exact vertical/time expressions rather than ODE roundoff in those rows.
    states[2], states[4] = h, times
    for k, duration in enumerate(dt):
        elapsed = collocation * duration
        xc[k, 2] = (
            h[k] + vs[k] * elapsed + 0.5 * (vs[k + 1] - vs[k]) * elapsed**2 / duration
        )
        xc[k, 4] = times[k] + elapsed
    if not np.all(np.diff(states[3]) < 0):
        raise ValueError("Coherent initial mass is not decreasing")
    for k in range(nodes + 1):
        lower, upper = (
            (optimizer.u_0_lb, optimizer.u_0_ub)
            if k == 0
            else (optimizer.u_f_lb, optimizer.u_f_ub)
            if k == nodes
            else (optimizer.u_lb, optimizer.u_ub)
        )
        if np.any(controls[:, k] < np.array(lower) - 1e-9) or np.any(
            controls[:, k] > np.array(upper) + 1e-9
        ):
            raise ValueError(f"Coherent initial control {k} exceeds existing bounds")
    flow = -derivative(states[:, :-1], controls[:, :-1])[3]
    valid = np.isfinite(flow) & (flow > 0)
    if not np.all(valid):
        raise ValueError("Coherent initial fuel flow is not finite/positive everywhere")
    mean_flow = float(np.average(flow, weights=dt))
    info = {
        "seed_only": True,
        "total_duration_s": float(dt.sum()),
        "phase_durations_s": [
            float(dt[:first].sum()),
            float(dt[first:last].sum()),
            float(dt[last:].sum()),
        ],
        "new_constraints": False,
        "tod_seed_m": peak_height,
        "wind_used_in_seed": False,
        "actual_optimizer_wind_enabled": optimizer.wind is not None,
        "seed_amplitude_fpm": seed_amplitude_fpm,
        "climb_plateau_seed_fpm": float(np.max(climb_rates) / fpm),
        "phase_intervals": [first, last - first, nodes - last],
        "capacity": capacity_value,
        "root_reports": root_reports,
        "root_method": "hybr; scaled(level_dt/1000,heading); xtol1e-10; maxfev100",
        "position_error_m": position_error,
        "height_integration_error_m": h_error,
        "time_integration_error_s": t_error,
        "monotonic_mass": True,
        "endpoint_controls_within_existing_bounds": True,
        "dt_bounds_pass": True,
        "initial_mass_kg": float(states[3, 0]),
        "final_mass_kg": float(states[3, -1]),
        "nodal_weighted_flow_kg_s": mean_flow,
        "coefficient_at_weight03_kg_s": 0.03 * mean_flow,
        "capacity_qualification": (
            "Capacity reductions mean this seed cannot fit its timing bounds; "
            "they do not prove that larger capacities are infeasible or suboptimal"
        ),
        "heading_endpoint_rad": list(endpoint_headings),
        "heading_rates_deg_s": [
            float((controls[2, 1] - controls[2, 0]) / dt[0] * 180 / np.pi),
            float((controls[2, -1] - controls[2, -2]) / dt[-1] * 180 / np.pi),
        ],
    }
    return states, controls, xc, dt, info
