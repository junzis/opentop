"""Internal representation of one aircraft in a CasADi Opti problem."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import comb, prod
from operator import index
from typing import Any


@dataclass(slots=True)
class AircraftTranscription:
    """Symbolic variables and expressions for one discretized trajectory.

    The object deliberately owns the build-specific expressions that used to
    be available only through mutable ``Base`` attributes.  A single-aircraft
    solve and a shared multi-aircraft solve can therefore use the same
    transcription code.
    """

    optimizer: Any
    opti: Any
    X: list[Any]
    Xc: list[list[Any]]
    U: list[Any]
    ts_final: Any
    interval_dts: list[Any]
    objective_raw: Any
    objective_scaled: Any
    objective_scale: float
    objective_kwargs: dict[str, Any]
    collocation_roots: tuple[float, ...]
    projection_center: tuple[float, float] | None = None
    grid_cost_raw: Any = None

    def control_at(self, interval: int, tau: float) -> Any:
        """Continuous, piecewise-linear control at local interval time tau."""
        return (1 - tau) * self.U[interval] + tau * self.U[interval + 1]

    def state_at(self, interval: int, tau: float) -> Any:
        """Evaluate the collocation state polynomial inside one interval."""
        interval = index(interval)
        if not 0 <= interval < len(self.Xc):
            raise ValueError("path constraint interval is outside the mesh")
        if not 0 <= tau <= 1:
            raise ValueError("path constraint tau must be finite and in [0, 1]")
        if tau == 1:
            return self.X[interval + 1]
        roots = (0.0, *self.collocation_roots)
        states = (self.X[interval], *self.Xc[interval])
        return sum(
            prod((tau - r) / (roots[j] - r) for i, r in enumerate(roots) if i != j)
            * state
            for j, state in enumerate(states)
        )

    def path_points(self, extra_points: Iterable[tuple[int, float]] = ()):
        """State/control pairs at boundaries, collocation, and extra points.

        Extra points are (zero-based interval, local time fraction) pairs.
        States use the collocation polynomial; controls interpolate linearly.
        """
        yield from zip(self.X, self.U)
        for k, states in enumerate(self.Xc):
            for tau, state in zip(self.collocation_roots, states):
                yield state, self.control_at(k, tau)

        for interval, tau in extra_points:
            yield self.state_at(interval, tau), self.control_at(interval, tau)

    def constrain_altitude_ceiling(self, ceiling_m: float) -> None:
        """Bound each altitude polynomial using its Bernstein coefficients."""
        import casadi as ca

        import numpy as np

        roots = np.array((0.0, *self.collocation_roots))
        degree = len(roots) - 1
        basis = np.column_stack(
            [
                comb(degree, j) * roots**j * (1 - roots) ** (degree - j)
                for j in range(degree + 1)
            ]
        )
        transform = np.linalg.inv(basis)
        for k, states in enumerate(self.Xc):
            heights = ca.vertcat(self.X[k][2], *(state[2] for state in states))
            coefficients = ca.mtimes(transform, heights)
            self.opti.subject_to(coefficients <= ceiling_m)
