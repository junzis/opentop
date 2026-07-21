"""Generic route-choice and trajectory-optimization helpers.

This module has no dependency on RAD. A route source only needs to provide a
name and an ordered polyline from origin to destination.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from time import perf_counter
from types import MappingProxyType
from typing import Any

from openap.aero import ft

import numpy as np
import pandas as pd
from pyproj import CRS, Geod, Transformer

from .routing import (
    DirectedMultiGraph,
    EdgeCost,
    Heuristic,
    RouteEdge,
    RouteNode,
    RoutePath,
    RoutePlanner,
    RouteSelectionConfig,
    distance_heuristic,
    geodesic_distance_m,
    zero_heuristic,
)

Waypoint = tuple[float, float]

_GEOD = Geod(ellps="WGS84")


def _distance_cost(edge: RouteEdge) -> float:
    return edge.distance_m


def _normalize_waypoint(waypoint: Waypoint) -> Waypoint:
    latitude, longitude = float(waypoint[0]), float(waypoint[1])
    if not (
        math.isfinite(latitude)
        and math.isfinite(longitude)
        and -90 <= latitude <= 90
        and -180 <= longitude <= 180
    ):
        raise ValueError("route coordinates are invalid")
    return latitude, longitude


@dataclass(frozen=True, slots=True)
class RouteOption:
    """One named route polyline, including origin and destination."""

    name: str
    waypoints: tuple[Waypoint, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("route name must not be empty")
        normalized = tuple(_normalize_waypoint(waypoint) for waypoint in self.waypoints)
        if len(normalized) < 2:
            raise ValueError("a route needs at least origin and destination")
        object.__setattr__(self, "waypoints", normalized)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def interior_waypoints(self) -> tuple[Waypoint, ...]:
        """Return waypoint constraints without origin and destination."""

        return self.waypoints[1:-1]

    @property
    def distance_m(self) -> float:
        """Return WGS84 polyline distance in meters."""

        return sum(
            float(_GEOD.inv(lon_a, lat_a, lon_b, lat_b)[2])
            for (lat_a, lon_a), (lat_b, lon_b) in pairwise(self.waypoints)
        )


@dataclass(frozen=True, slots=True)
class OptimizedRouteOption:
    """One route option and its OpenTOP optimization result."""

    route: RouteOption
    trajectory: pd.DataFrame | None
    success: bool
    fuel_kg: float
    status: str


@dataclass(frozen=True, slots=True)
class RouteChoiceResult:
    """All route-option solves and the associated wall-clock timings."""

    optimized: tuple[OptimizedRouteOption, ...]
    solve_seconds: tuple[float, ...]

    @property
    def successful(self) -> tuple[OptimizedRouteOption, ...]:
        """Return route options with valid optimized trajectories."""

        return tuple(result for result in self.optimized if result.success)

    @property
    def best(self) -> OptimizedRouteOption | None:
        """Return the lowest-fuel successful option, if one exists."""

        return min(self.successful, key=lambda result: result.fuel_kg, default=None)

    @property
    def optimal_route(self) -> RouteOption | None:
        """Return the selected route option, if optimization succeeded."""

        return None if self.best is None else self.best.route

    @property
    def trajectory(self) -> pd.DataFrame | None:
        """Return the selected trajectory, if optimization succeeded."""

        return None if self.best is None else self.best.trajectory


@dataclass(frozen=True, slots=True)
class RouteOptimizationConfig:
    """Continuous optimization settings shared by every route source."""

    objective: str = "fuel"
    waypoint_tolerance_m: float = 2_000.0
    minimum_nodes: int = 20
    nodes_per_leg: int = 2
    waypoint_simplification_tolerance_m: float | None = None
    trajectory_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.objective:
            raise ValueError("objective must not be empty")
        if self.waypoint_tolerance_m <= 0:
            raise ValueError("waypoint tolerance must be positive")
        if self.minimum_nodes < 1 or self.nodes_per_leg < 1:
            raise ValueError("node settings must be positive")
        tolerance = self.waypoint_simplification_tolerance_m
        if tolerance is not None and tolerance <= 0:
            raise ValueError("waypoint simplification tolerance must be positive")
        object.__setattr__(
            self,
            "trajectory_kwargs",
            MappingProxyType(dict(self.trajectory_kwargs)),
        )


@dataclass(frozen=True, slots=True)
class RouteNetworkSelection:
    """Candidate paths and source-neutral options selected from a route network."""

    paths: tuple[RoutePath, ...]
    options: tuple[RouteOption, ...]


@dataclass(frozen=True, slots=True)
class RouteNetwork:
    """A generic network of named waypoints and permitted directed legs."""

    graph: DirectedMultiGraph

    @classmethod
    def from_connections(
        cls,
        waypoints: Mapping[str, Waypoint],
        connections: Iterable[tuple[str, str]],
        *,
        bidirectional: bool = False,
    ) -> RouteNetwork:
        """Build a network from named points and permitted connections."""

        graph = DirectedMultiGraph()
        for waypoint_id, waypoint in waypoints.items():
            latitude, longitude = _normalize_waypoint(waypoint)
            graph.add_node(
                RouteNode(
                    str(waypoint_id),
                    latitude,
                    longitude,
                    "COMPANY",
                )
            )
        for index, (source, target) in enumerate(connections):
            if source == target:
                raise ValueError("route-network connections cannot be self loops")
            if source not in graph.nodes or target not in graph.nodes:
                raise KeyError("connection endpoints must exist in waypoints")
            directions = ((source, target),)
            if bidirectional:
                directions += ((target, source),)
            for direction_index, (leg_source, leg_target) in enumerate(directions):
                source_node = graph.nodes[leg_source]
                target_node = graph.nodes[leg_target]
                graph.add_edge(
                    RouteEdge(
                        edge_id=(
                            f"company:{index}:{direction_index}:"
                            f"{leg_source}:{leg_target}"
                        ),
                        source=leg_source,
                        target=leg_target,
                        distance_m=geodesic_distance_m(
                            source_node.latitude,
                            source_node.longitude,
                            target_node.latitude,
                            target_node.longitude,
                        ),
                        layer="company",
                    )
                )
        return cls(graph)

    @property
    def waypoints(self) -> Mapping[str, Waypoint]:
        """Return network waypoint coordinates keyed by identifier."""

        return MappingProxyType(
            {
                node_id: (node.latitude, node.longitude)
                for node_id, node in self.graph.nodes.items()
            }
        )

    def select_routes(
        self,
        source: str,
        target: str,
        *,
        config: RouteSelectionConfig = RouteSelectionConfig(),
        edge_cost: EdgeCost | None = None,
        heuristic: Heuristic | None = None,
    ) -> RouteNetworkSelection:
        """Generate diverse route options through the permitted network legs."""

        if edge_cost is None:
            effective_cost = _distance_cost
            effective_heuristic = (
                distance_heuristic(self.graph) if heuristic is None else heuristic
            )
        else:
            effective_cost = edge_cost
            effective_heuristic = zero_heuristic if heuristic is None else heuristic
        paths = tuple(
            RoutePlanner(
                self.graph,
                edge_cost=effective_cost,
                heuristic=effective_heuristic,
            ).candidates(source, target, config=config)
        )
        options = tuple(
            RouteOption(
                f"Network route {index + 1}",
                tuple(
                    (
                        self.graph.nodes[node_id].latitude,
                        self.graph.nodes[node_id].longitude,
                    )
                    for node_id in path.nodes
                ),
                metadata={
                    "source": "waypoint_network",
                    "node_ids": path.nodes,
                    "edge_ids": path.edge_ids,
                    "search_cost": path.cost,
                },
            )
            for index, path in enumerate(paths)
        )
        return RouteNetworkSelection(paths, options)


def simplify_waypoints(
    waypoints: Sequence[Waypoint],
    *,
    tolerance_m: float,
    include_endpoints: bool = True,
) -> list[Waypoint]:
    """Remove geometrically redundant points with RDP simplification."""

    if tolerance_m <= 0:
        raise ValueError("simplification tolerance must be positive")
    coordinates = tuple(waypoints)
    if len(coordinates) < 2:
        raise ValueError("a route needs at least two coordinates")
    if len(coordinates) == 2:
        return list(coordinates) if include_endpoints else []

    center_latitude = sum(latitude for latitude, _ in coordinates) / len(coordinates)
    center_longitude = sum(longitude for _, longitude in coordinates) / len(coordinates)
    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={center_latitude} +lon_0={center_longitude} "
        "+datum=WGS84 +units=m"
    )
    transformer = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
    longitudes = [longitude for _, longitude in coordinates]
    latitudes = [latitude for latitude, _ in coordinates]
    x_values, y_values = transformer.transform(longitudes, latitudes)
    projected = np.column_stack((x_values, y_values))

    def retain(start: int, end: int) -> list[int]:
        if end <= start + 1:
            return [start, end]
        segment = projected[end] - projected[start]
        segment_length = float(np.linalg.norm(segment))
        offsets = projected[start + 1 : end] - projected[start]
        if segment_length == 0:
            distances = np.linalg.norm(offsets, axis=1)
        else:
            cross_products = segment[0] * offsets[:, 1] - segment[1] * offsets[:, 0]
            distances = np.abs(cross_products) / segment_length
        relative_index = int(np.argmax(distances))
        if float(distances[relative_index]) <= tolerance_m:
            return [start, end]
        split = start + relative_index + 1
        return retain(start, split)[:-1] + retain(split, end)

    indices = retain(0, len(coordinates) - 1)
    simplified = [coordinates[index] for index in indices]
    return simplified if include_endpoints else simplified[1:-1]


def _resample_polyline(
    coordinates: Sequence[Waypoint], count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if count < 2:
        raise ValueError("at least two output points are required")
    if len(coordinates) < 2:
        raise ValueError("route must contain at least two coordinates")
    segment_distances: list[float] = []
    segment_bearings: list[float] = []
    for (lat_a, lon_a), (lat_b, lon_b) in pairwise(coordinates):
        bearing, _, distance = _GEOD.inv(lon_a, lat_a, lon_b, lat_b)
        segment_bearings.append(float(bearing))
        segment_distances.append(float(distance))
    cumulative = np.concatenate(([0.0], np.cumsum(segment_distances)))
    if cumulative[-1] <= 0:
        raise ValueError("route length must be positive")
    samples = np.linspace(0.0, cumulative[-1], count)
    latitudes = np.empty(count)
    longitudes = np.empty(count)
    for output_index, distance_along in enumerate(samples):
        segment_index = min(
            int(np.searchsorted(cumulative, distance_along, side="right") - 1),
            len(segment_distances) - 1,
        )
        offset = distance_along - cumulative[segment_index]
        lat_a, lon_a = coordinates[segment_index]
        lon, lat, _ = _GEOD.fwd(
            lon_a,
            lat_a,
            segment_bearings[segment_index],
            offset,
        )
        latitudes[output_index] = lat
        longitudes[output_index] = lon
    latitudes[0], longitudes[0] = coordinates[0]
    latitudes[-1], longitudes[-1] = coordinates[-1]
    return latitudes, longitudes, samples


def route_initial_guess(
    route: RouteOption,
    *,
    nodes: int,
    altitude_ft: float,
    mass_kg: float | None = None,
    nominal_groundspeed_mps: float = 220.0,
    complete_flight: bool = False,
) -> pd.DataFrame:
    """Build a distance-spaced initial guess along a route option."""

    if nodes < 1:
        raise ValueError("nodes must be positive")
    if altitude_ft <= 0:
        raise ValueError("altitude must be positive")
    if nominal_groundspeed_mps <= 0:
        raise ValueError("nominal groundspeed must be positive")
    latitudes, longitudes, distances = _resample_polyline(
        route.waypoints,
        nodes + 1,
    )
    if complete_flight:
        progress = distances / distances[-1]
        climb = np.clip(progress / 0.2, 0.0, 1.0)
        descent = np.clip((1.0 - progress) / 0.15, 0.0, 1.0)
        altitudes = altitude_ft * np.minimum(climb, descent)
        altitudes[[0, -1]] = 100.0
    else:
        altitudes = np.full(nodes + 1, altitude_ft)
    data: dict[str, Any] = {
        "latitude": latitudes,
        "longitude": longitudes,
        "altitude": altitudes,
        "ts": distances / nominal_groundspeed_mps,
    }
    if mass_kg is not None:
        data["mass"] = np.full(nodes + 1, mass_kg)
    return pd.DataFrame(data)


def validate_waypoint_trajectory(
    trajectory: pd.DataFrame,
    waypoints: Sequence[Waypoint],
    *,
    tolerance_m: float,
    node_indices: Sequence[int] | None = None,
) -> bool:
    """Check that a trajectory visits waypoint constraints in order."""

    if tolerance_m <= 0:
        raise ValueError("tolerance must be positive")
    if not {"latitude", "longitude"}.issubset(trajectory.columns):
        raise ValueError("trajectory needs latitude and longitude columns")
    if node_indices is not None and len(node_indices) != len(waypoints):
        raise ValueError("node indices must match waypoints")

    start_index = 0
    for waypoint_index, (latitude, longitude) in enumerate(waypoints):
        if node_indices is None:
            indices = range(start_index, len(trajectory))
        else:
            index = int(node_indices[waypoint_index])
            if index < start_index or index >= len(trajectory):
                return False
            indices = (index,)
        best_index: int | None = None
        best_distance = float("inf")
        for index in indices:
            _, _, distance = _GEOD.inv(
                longitude,
                latitude,
                float(trajectory.longitude.iloc[index]),
                float(trajectory.latitude.iloc[index]),
            )
            if distance < best_distance:
                best_distance = float(distance)
                best_index = index
        if best_index is None or best_distance > tolerance_m:
            return False
        start_index = best_index + 1
    return True


def _optimize_route_options(
    routes: Sequence[RouteOption],
    optimizer_factory: Any,
    *,
    objective: str = "fuel",
    waypoint_tolerance_m: float = 2_000.0,
    minimum_nodes: int = 20,
    nodes_per_leg: int = 2,
    waypoint_simplification_tolerance_m: float | None = None,
    trajectory_kwargs: dict[str, Any] | None = None,
) -> RouteChoiceResult:
    """Implement independent continuous solves for route options."""

    from .full import CompleteFlight

    if not routes:
        raise ValueError("at least one route option is required")
    if minimum_nodes < 1 or nodes_per_leg < 1:
        raise ValueError("node settings must be positive")
    extra_kwargs = dict(trajectory_kwargs or {})
    results: list[OptimizedRouteOption] = []
    solve_seconds: list[float] = []

    for route in routes:
        started = perf_counter()
        try:
            optimizer = optimizer_factory()
            waypoints = (
                list(route.interior_waypoints)
                if waypoint_simplification_tolerance_m is None
                else simplify_waypoints(
                    route.waypoints,
                    tolerance_m=waypoint_simplification_tolerance_m,
                    include_endpoints=False,
                )
            )
            leg_count = (
                len(route.waypoints) - 1
                if waypoint_simplification_tolerance_m is None
                else len(waypoints) + 1
            )
            nodes = max(minimum_nodes, nodes_per_leg * max(1, leg_count))
            nodes = max(nodes, len(waypoints) + 1)
            optimizer.setup(nodes=nodes)
            guess = route_initial_guess(
                route,
                nodes=nodes,
                altitude_ft=float(optimizer.aircraft["cruise"]["height"]) / ft,
                mass_kg=float(optimizer.mass_init),
                complete_flight=isinstance(optimizer, CompleteFlight),
            )
            trajectory = optimizer.trajectory(
                objective=objective,
                initial_guess=guess,
                waypoints=waypoints,
                waypoint_tolerance_m=waypoint_tolerance_m,
                result_object=True,
                **extra_kwargs,
            )
            df = trajectory.df
            conforms = trajectory.success and validate_waypoint_trajectory(
                df,
                waypoints,
                tolerance_m=waypoint_tolerance_m * 1.05,
            )
            results.append(
                OptimizedRouteOption(
                    route,
                    df,
                    conforms,
                    trajectory.fuel,
                    trajectory.status
                    if conforms
                    else f"{trajectory.status}; route validation failed",
                )
            )
        except Exception as error:
            results.append(
                OptimizedRouteOption(
                    route,
                    None,
                    False,
                    float("nan"),
                    f"{type(error).__name__}: {error}",
                )
            )
        solve_seconds.append(perf_counter() - started)

    return RouteChoiceResult(tuple(results), tuple(solve_seconds))


def optimize_routes(
    routes: Sequence[RouteOption],
    optimizer_factory: Any,
    *,
    config: RouteOptimizationConfig = RouteOptimizationConfig(),
) -> RouteChoiceResult:
    """Optimize options from any route source with one consistent API."""

    return _optimize_route_options(
        routes,
        optimizer_factory,
        objective=config.objective,
        waypoint_tolerance_m=config.waypoint_tolerance_m,
        minimum_nodes=config.minimum_nodes,
        nodes_per_leg=config.nodes_per_leg,
        waypoint_simplification_tolerance_m=(
            config.waypoint_simplification_tolerance_m
        ),
        trajectory_kwargs=dict(config.trajectory_kwargs),
    )
