"""Helpers that convert discrete RAD routes to source-neutral route options."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ..routes import (
    RouteOption,
    simplify_waypoints,
    validate_waypoint_trajectory,
)
from ..routes import (
    route_initial_guess as _generic_initial_guess,
)
from ._models import RoutePath
from .graph import DirectedMultiGraph


def route_waypoints(
    graph: DirectedMultiGraph,
    route: RoutePath,
    *,
    include_endpoints: bool = False,
) -> list[tuple[float, float]]:
    """Convert a route path to ordered ``(latitude, longitude)`` pairs."""

    node_ids = route.nodes if include_endpoints else route.nodes[1:-1]
    return [
        (graph.nodes[node_id].latitude, graph.nodes[node_id].longitude)
        for node_id in node_ids
    ]


def route_option(
    graph: DirectedMultiGraph,
    route: RoutePath,
    *,
    name: str = "RAD route",
) -> RouteOption:
    """Convert a graph route into a generic :class:`RouteOption`."""

    return RouteOption(
        name,
        tuple(
            (graph.nodes[node_id].latitude, graph.nodes[node_id].longitude)
            for node_id in route.nodes
        ),
        metadata={"edge_ids": route.edge_ids, "node_ids": route.nodes},
    )


def simplify_route_waypoints(
    graph: DirectedMultiGraph,
    route: RoutePath,
    *,
    tolerance_m: float,
    include_endpoints: bool = False,
) -> list[tuple[float, float]]:
    """Remove geometrically redundant RAD points via the generic helper."""

    return simplify_waypoints(
        route_option(graph, route).waypoints,
        tolerance_m=tolerance_m,
        include_endpoints=include_endpoints,
    )


def route_initial_guess(
    graph: DirectedMultiGraph,
    route: RoutePath,
    *,
    nodes: int,
    altitude_ft: float,
    mass_kg: float | None = None,
    nominal_groundspeed_mps: float = 220.0,
    complete_flight: bool = False,
) -> pd.DataFrame:
    """Build a route-shaped initial guess through the generic route layer."""

    return _generic_initial_guess(
        route_option(graph, route),
        nodes=nodes,
        altitude_ft=altitude_ft,
        mass_kg=mass_kg,
        nominal_groundspeed_mps=nominal_groundspeed_mps,
        complete_flight=complete_flight,
    )


def validate_route_trajectory(
    trajectory: pd.DataFrame,
    waypoints: Sequence[tuple[float, float]],
    *,
    tolerance_m: float,
    node_indices: Sequence[int] | None = None,
) -> bool:
    """Check waypoint order through the generic route layer."""

    return validate_waypoint_trajectory(
        trajectory,
        waypoints,
        tolerance_m=tolerance_m,
        node_indices=node_indices,
    )
