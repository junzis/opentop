"""Source-neutral graph search for route candidate generation."""

from __future__ import annotations

import heapq
import itertools
import math
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from pyproj import Geod

_GEOD = Geod(ellps="WGS84")


def geodesic_distance_m(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
) -> float:
    """Return WGS84 geodesic distance in meters."""

    _, _, distance = _GEOD.inv(
        longitude_1,
        latitude_1,
        longitude_2,
        latitude_2,
    )
    return float(distance)


@dataclass(frozen=True, slots=True)
class RouteNode:
    """One named node in a route graph."""

    node_id: str
    latitude: float
    longitude: float
    point_type: str = ""


@dataclass(frozen=True, slots=True)
class RouteEdge:
    """One directed, identity-preserving route-graph edge."""

    edge_id: str
    source: str
    target: str
    distance_m: float
    layer: str = "base"
    min_flight_level: int | None = None
    max_flight_level: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provenance: Any = None

    def __post_init__(self) -> None:
        if self.distance_m < 0:
            raise ValueError("edge distance must be non-negative")
        if (
            self.min_flight_level is not None
            and self.max_flight_level is not None
            and self.min_flight_level > self.max_flight_level
        ):
            raise ValueError("minimum flight level cannot exceed maximum")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class RoutePath:
    """A loopless path returned by graph search."""

    nodes: tuple[str, ...]
    edges: tuple[RouteEdge, ...]
    cost: float

    @property
    def distance_m(self) -> float:
        """Return the summed physical edge distance."""

        return sum(edge.distance_m for edge in self.edges)

    @property
    def edge_ids(self) -> tuple[str, ...]:
        """Return stable edge identities in traversal order."""

        return tuple(edge.edge_id for edge in self.edges)


class DirectedMultiGraph:
    """A small directed multigraph that keeps parallel edge identities."""

    def __init__(self) -> None:
        self._nodes: dict[str, RouteNode] = {}
        self._edges: dict[str, RouteEdge] = {}
        self._outgoing: dict[str, list[str]] = defaultdict(list)

    @property
    def nodes(self) -> Mapping[str, RouteNode]:
        return self._nodes

    @property
    def edges(self) -> Mapping[str, RouteEdge]:
        return self._edges

    def add_node(self, node: RouteNode) -> None:
        existing = self._nodes.get(node.node_id)
        if existing is not None and existing != node:
            raise ValueError(f"conflicting definitions for node {node.node_id!r}")
        self._nodes[node.node_id] = node

    def add_edge(self, edge: RouteEdge) -> None:
        if edge.edge_id in self._edges:
            raise ValueError(f"duplicate edge identifier {edge.edge_id!r}")
        if edge.source not in self._nodes or edge.target not in self._nodes:
            raise ValueError("edge endpoints must be added before the edge")
        self._edges[edge.edge_id] = edge
        self._outgoing[edge.source].append(edge.edge_id)

    def outgoing(self, node_id: str) -> Iterator[RouteEdge]:
        for edge_id in self._outgoing.get(node_id, ()):
            yield self._edges[edge_id]

    def copy(self) -> DirectedMultiGraph:
        copied = DirectedMultiGraph()
        for node in self._nodes.values():
            copied.add_node(node)
        for edge in self._edges.values():
            copied.add_edge(edge)
        return copied


EdgeCost = Callable[[RouteEdge], float]
EdgePredicate = Callable[[RouteEdge], bool]
Heuristic = Callable[[str, str], float]


class RouteNotFound(RuntimeError):
    """Raised when no route satisfies the search constraints."""


class SearchBudgetExceeded(RuntimeError):
    """Raised when a route search reaches its configured work budget."""


@dataclass(frozen=True, slots=True)
class SearchBudget:
    """Hard limits that keep graph search latency predictable."""

    max_expansions: int = 250_000
    max_cost: float | None = None

    def __post_init__(self) -> None:
        if self.max_expansions <= 0:
            raise ValueError("max_expansions must be positive")
        if self.max_cost is not None and self.max_cost < 0:
            raise ValueError("max_cost must be non-negative")


def zero_heuristic(_node: str, _target: str) -> float:
    return 0.0


def distance_heuristic(graph: DirectedMultiGraph) -> Heuristic:
    """Create an admissible straight-line-distance heuristic."""

    def estimate(node_id: str, target_id: str) -> float:
        node, target = graph.nodes[node_id], graph.nodes[target_id]
        return geodesic_distance_m(
            node.latitude,
            node.longitude,
            target.latitude,
            target.longitude,
        )

    return estimate


def shortest_path(
    graph: DirectedMultiGraph,
    source: str,
    target: str,
    *,
    edge_cost: EdgeCost = lambda edge: edge.distance_m,
    heuristic: Heuristic = zero_heuristic,
    edge_predicate: EdgePredicate = lambda _edge: True,
    budget: SearchBudget = SearchBudget(),
    excluded_nodes: frozenset[str] = frozenset(),
    excluded_edges: frozenset[str] = frozenset(),
) -> RoutePath:
    """Find the lowest-cost path with A* and node reopening."""

    if source not in graph.nodes or target not in graph.nodes:
        raise KeyError("source and target must exist in the graph")
    if source in excluded_nodes or target in excluded_nodes:
        raise RouteNotFound(f"no route from {source!r} to {target!r}")
    if source == target:
        return RoutePath((source,), (), 0.0)

    counter = itertools.count()
    queue: list[tuple[float, float, str, int]] = []
    initial_h = float(heuristic(source, target))
    if not math.isfinite(initial_h) or initial_h < 0:
        raise ValueError("heuristic values must be finite and non-negative")
    heapq.heappush(queue, (initial_h, 0.0, source, next(counter)))
    best_cost = {source: 0.0}
    predecessor: dict[str, tuple[str, RouteEdge]] = {}
    expansions = 0

    while queue:
        _, cost_so_far, node_id, _ = heapq.heappop(queue)
        if cost_so_far != best_cost.get(node_id):
            continue
        if node_id == target:
            edges: list[RouteEdge] = []
            nodes = [target]
            current = target
            while current != source:
                previous, edge = predecessor[current]
                edges.append(edge)
                nodes.append(previous)
                current = previous
            edges.reverse()
            nodes.reverse()
            return RoutePath(tuple(nodes), tuple(edges), cost_so_far)

        expansions += 1
        if expansions > budget.max_expansions:
            raise SearchBudgetExceeded(
                f"route search exceeded {budget.max_expansions} expansions"
            )
        for edge in graph.outgoing(node_id):
            if (
                edge.edge_id in excluded_edges
                or edge.target in excluded_nodes
                or not edge_predicate(edge)
            ):
                continue
            weight = float(edge_cost(edge))
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("edge costs must be finite and non-negative")
            candidate_cost = cost_so_far + weight
            if budget.max_cost is not None and candidate_cost > budget.max_cost:
                continue
            if candidate_cost >= best_cost.get(edge.target, math.inf):
                continue
            estimate = float(heuristic(edge.target, target))
            if not math.isfinite(estimate) or estimate < 0:
                raise ValueError("heuristic values must be finite and non-negative")
            best_cost[edge.target] = candidate_cost
            predecessor[edge.target] = node_id, edge
            heapq.heappush(
                queue,
                (
                    candidate_cost + estimate,
                    candidate_cost,
                    edge.target,
                    next(counter),
                ),
            )
    raise RouteNotFound(f"no route from {source!r} to {target!r}")


def iter_k_shortest_paths(
    graph: DirectedMultiGraph,
    source: str,
    target: str,
    *,
    edge_cost: EdgeCost = lambda edge: edge.distance_m,
    heuristic: Heuristic = zero_heuristic,
    edge_predicate: EdgePredicate = lambda _edge: True,
    budget: SearchBudget = SearchBudget(),
) -> Iterator[RoutePath]:
    """Yield loopless alternatives using edge-ID-aware Yen search."""

    accepted = [
        shortest_path(
            graph,
            source,
            target,
            edge_cost=edge_cost,
            heuristic=heuristic,
            edge_predicate=edge_predicate,
            budget=budget,
        )
    ]
    yield accepted[0]
    candidates: list[tuple[float, tuple[str, ...], RoutePath]] = []
    queued_edge_sequences: set[tuple[str, ...]] = set()

    while True:
        previous_path = accepted[-1]
        for spur_index in range(len(previous_path.nodes) - 1):
            root_nodes = previous_path.nodes[: spur_index + 1]
            root_edges = previous_path.edges[:spur_index]
            excluded_edges: set[str] = set()
            for path in accepted:
                if (
                    path.nodes[: spur_index + 1] == root_nodes
                    and len(path.edges) > spur_index
                ):
                    excluded_edges.add(path.edges[spur_index].edge_id)
            try:
                spur_path = shortest_path(
                    graph,
                    root_nodes[-1],
                    target,
                    edge_cost=edge_cost,
                    heuristic=heuristic,
                    edge_predicate=edge_predicate,
                    budget=budget,
                    excluded_nodes=frozenset(root_nodes[:-1]),
                    excluded_edges=frozenset(excluded_edges),
                )
            except RouteNotFound:
                continue
            edges = root_edges + spur_path.edges
            nodes = root_nodes[:-1] + spur_path.nodes
            edge_ids = tuple(edge.edge_id for edge in edges)
            if edge_ids in queued_edge_sequences:
                continue
            queued_edge_sequences.add(edge_ids)
            cost = sum(float(edge_cost(edge)) for edge in edges)
            candidate = RoutePath(nodes, edges, cost)
            heapq.heappush(candidates, (cost, edge_ids, candidate))

        if not candidates:
            return
        _, edge_ids, next_path = heapq.heappop(candidates)
        queued_edge_sequences.discard(edge_ids)
        accepted.append(next_path)
        yield next_path


def k_shortest_paths(
    graph: DirectedMultiGraph,
    source: str,
    target: str,
    k: int,
    **kwargs: object,
) -> list[RoutePath]:
    """Return at most ``k`` loopless paths."""

    if k < 1:
        raise ValueError("k must be positive")
    paths = iter_k_shortest_paths(graph, source, target, **kwargs)  # type: ignore[arg-type]
    return list(itertools.islice(paths, k))


@dataclass(frozen=True, slots=True)
class RouteSelectionConfig:
    """Limits and diversity controls for discrete route candidates."""

    candidates: int = 5
    search_candidates: int = 30
    max_cost_ratio: float = 1.35
    max_distance_ratio: float = 1.5
    maximum_shared_edge_fraction: float = 0.9
    budget: SearchBudget = field(default_factory=SearchBudget)

    def __post_init__(self) -> None:
        if self.candidates < 1:
            raise ValueError("candidates must be positive")
        if self.search_candidates < self.candidates:
            raise ValueError("search_candidates cannot be smaller than candidates")
        if self.max_cost_ratio < 1 or self.max_distance_ratio < 1:
            raise ValueError("route ratios must be at least one")
        if not 0 <= self.maximum_shared_edge_fraction <= 1:
            raise ValueError("shared-edge fraction must be between zero and one")


def _shared_edge_fraction(route: RoutePath, accepted: RoutePath) -> float:
    if not route.edges:
        return 1.0
    shared = set(route.edge_ids).intersection(accepted.edge_ids)
    return len(shared) / len(route.edges)


class RoutePlanner:
    """Generate a small, diverse route set for subsequent OpenTOP solves."""

    def __init__(
        self,
        graph: DirectedMultiGraph,
        *,
        edge_cost: EdgeCost = lambda edge: edge.distance_m,
        heuristic: Heuristic = zero_heuristic,
        edge_predicate: EdgePredicate = lambda _edge: True,
    ) -> None:
        self.graph = graph
        self.edge_cost = edge_cost
        self.heuristic = heuristic
        self.edge_predicate = edge_predicate

    def candidates(
        self,
        source: str,
        target: str,
        *,
        config: RouteSelectionConfig = RouteSelectionConfig(),
    ) -> list[RoutePath]:
        """Return cost/detour-pruned candidates with bounded edge overlap."""

        accepted: list[RoutePath] = []
        best_cost: float | None = None
        best_distance: float | None = None
        paths = iter_k_shortest_paths(
            self.graph,
            source,
            target,
            edge_cost=self.edge_cost,
            heuristic=self.heuristic,
            edge_predicate=self.edge_predicate,
            budget=config.budget,
        )
        for path_index, route in enumerate(paths):
            if path_index >= config.search_candidates:
                break
            if best_cost is None:
                best_cost = route.cost
                best_distance = route.distance_m
            assert best_distance is not None
            cost_limit = (
                best_cost * config.max_cost_ratio
                if best_cost > 0
                else config.max_cost_ratio - 1
            )
            distance_limit = (
                best_distance * config.max_distance_ratio
                if best_distance > 0
                else config.max_distance_ratio - 1
            )
            if route.cost > cost_limit or route.distance_m > distance_limit:
                continue
            if accepted and any(
                _shared_edge_fraction(route, previous)
                > config.maximum_shared_edge_fraction
                for previous in accepted
            ):
                continue
            accepted.append(route)
            if len(accepted) == config.candidates:
                break
        return accepted
