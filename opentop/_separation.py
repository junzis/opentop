"""Separation metrics and direct-collocation dense-output helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isfinite
from typing import Any

import casadi as ca

import numpy as np


@dataclass(frozen=True, slots=True)
class SeparationConfig:
    """Physical separation model and encounter-refinement settings.

    The defaults reproduce the high-order conflict-region approximation used
    by the original multi-aircraft prototype.
    """

    horizontal_m: float = 5.0 * 1852.0
    vertical_m: float = 1000.0 * 0.3048
    vertical_power: int = 8
    minimum_metric: float = 1.3
    watch_factor: float = 1.5
    detect_dt_s: float = 15.0
    constraint_dt_s: float = 10.0
    encounter_buffer_s: float = 35.0
    verification_dt_s: float = 5.0
    max_refinements: int = 2
    verification_tolerance: float = 1e-3
    constraint_buffer: float = 1e-2

    def __post_init__(self) -> None:
        positive_finite = {
            "horizontal_m": self.horizontal_m,
            "vertical_m": self.vertical_m,
            "minimum_metric": self.minimum_metric,
            "watch_factor": self.watch_factor,
            "detect_dt_s": self.detect_dt_s,
            "constraint_dt_s": self.constraint_dt_s,
            "encounter_buffer_s": self.encounter_buffer_s,
            "verification_dt_s": self.verification_dt_s,
        }
        for name, value in positive_finite.items():
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(self.vertical_power, bool)
            or not isinstance(self.vertical_power, int)
            or self.vertical_power < 2
            or self.vertical_power % 2
        ):
            raise ValueError("vertical_power must be an even integer >= 2")
        if (
            isinstance(self.max_refinements, bool)
            or not isinstance(self.max_refinements, int)
            or self.max_refinements < 0
        ):
            raise ValueError("max_refinements must be a non-negative integer")
        if not isfinite(self.verification_tolerance) or self.verification_tolerance < 0:
            raise ValueError("verification_tolerance must be finite and non-negative")
        if not isfinite(self.constraint_buffer) or self.constraint_buffer < 0:
            raise ValueError("constraint_buffer must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class PairSeparationReport:
    """Worst verified separation for one aircraft pair."""

    first: str
    second: str
    overlaps: bool
    minimum_metric: float
    required_metric: float
    margin: float
    worst_time: float
    horizontal_m: float
    vertical_m: float


def separation_metric(
    dx: Any,
    dy: Any,
    dz: Any,
    config: SeparationConfig,
) -> Any:
    """Return the configured high-order pairwise separation metric."""
    return (
        (dx / config.horizontal_m) ** 2
        + (dy / config.horizontal_m) ** 2
        + (dz / config.vertical_m) ** config.vertical_power
    )


@lru_cache(maxsize=None)
def _collocation_roots(polydeg: int) -> tuple[float, ...]:
    return (
        0.0,
        *(float(value) for value in ca.collocation_points(polydeg, "legendre")),
    )


def collocation_roots(polydeg: int) -> np.ndarray:
    """Legendre roots including the interval start at zero."""
    return np.asarray(_collocation_roots(polydeg), dtype=float)


def interpolate_collocation_state(
    x_start: Any,
    x_collocation: list[Any],
    tau: Any,
) -> Any:
    """Evaluate a direct-collocation state polynomial at ``tau`` in [0, 1]."""
    polydeg = len(x_collocation)
    roots = collocation_roots(polydeg)
    values = [x_start, *x_collocation]
    state = 0
    for j, value in enumerate(values):
        basis = 1
        for r in range(polydeg + 1):
            if r != j:
                basis = basis * (tau - roots[r]) / (roots[j] - roots[r])
        state = state + basis * value
    return state


def numeric_collocation_state(
    x_start: np.ndarray,
    x_collocation: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Numeric equivalent of :func:`interpolate_collocation_state`."""
    values = [x_start, *list(x_collocation)]
    roots = collocation_roots(len(x_collocation))
    state = np.zeros_like(np.asarray(x_start, dtype=float))
    for j, value in enumerate(values):
        basis = 1.0
        for r in range(len(values)):
            if r != j:
                basis *= (tau - roots[r]) / (roots[j] - roots[r])
        state += basis * np.asarray(value, dtype=float)
    return state
