"""Cheap route-ranking costs used before nonlinear optimization."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ._models import RouteEdge

WindComponent = Callable[[RouteEdge], float]


@dataclass(frozen=True, slots=True)
class DistanceCost:
    """Rank routes by geodesic edge distance."""

    def __call__(self, edge: RouteEdge) -> float:
        return edge.distance_m


@dataclass(frozen=True, slots=True)
class NominalFuelCost:
    """Wind-adjusted nominal cruise fuel surrogate.

    This intentionally avoids evaluating OpenAP on every graph expansion. With a
    constant nominal fuel flow it primarily ranks wind-adjusted travel time; the
    final OpenTOP solve remains the authoritative fuel comparison.
    """

    true_airspeed_mps: float
    fuel_flow_kg_s: float
    wind_component_mps: WindComponent = lambda _edge: 0.0
    minimum_groundspeed_mps: float = 80.0
    penalty: Callable[[RouteEdge], float] = lambda _edge: 0.0

    def __post_init__(self) -> None:
        if self.true_airspeed_mps <= 0:
            raise ValueError("true airspeed must be positive")
        if self.fuel_flow_kg_s <= 0:
            raise ValueError("fuel flow must be positive")
        if self.minimum_groundspeed_mps <= 0:
            raise ValueError("minimum groundspeed must be positive")

    def __call__(self, edge: RouteEdge) -> float:
        groundspeed = max(
            self.minimum_groundspeed_mps,
            self.true_airspeed_mps + float(self.wind_component_mps(edge)),
        )
        return edge.distance_m / groundspeed * self.fuel_flow_kg_s + float(
            self.penalty(edge)
        )
