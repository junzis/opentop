"""High-level construction and candidate selection for per-flight RAD graphs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from ..routing import (
    EdgeCost,
    Heuristic,
    RoutePlanner,
    RouteSelectionConfig,
    zero_heuristic,
)
from ._models import Airport, FlightContext, RouteEdge, RouteNode, RoutePath
from .graph import DirectedMultiGraph, edge_available, geodesic_distance_m


def flight_graph(
    graph: DirectedMultiGraph, context: FlightContext
) -> DirectedMultiGraph:
    """Materialize the edge subset applicable to a static flight context."""

    filtered = DirectedMultiGraph()
    for node in graph.nodes.values():
        filtered.add_node(node)
    for edge in graph.edges.values():
        if edge_available(edge, context):
            filtered.add_edge(edge)
    return filtered


def add_airport_connectors(
    graph: DirectedMultiGraph,
    departure: Airport,
    arrival: Airport,
    *,
    connector_count: int = 5,
    maximum_distance_m: float = 150_000.0,
    connector_cost_factor: float = 1.0,
) -> tuple[DirectedMultiGraph, str, str]:
    """Attach departure/arrival airport nodes to their nearest network points."""

    if connector_count < 1:
        raise ValueError("connector_count must be positive")
    if maximum_distance_m <= 0 or connector_cost_factor <= 0:
        raise ValueError("connector distance and cost factor must be positive")
    connected = graph.copy()
    departure_id = f"airport:{departure.icao}:departure"
    arrival_id = f"airport:{arrival.icao}:arrival"
    connected.add_node(
        RouteNode(departure_id, departure.latitude, departure.longitude, "AIRPORT")
    )
    connected.add_node(
        RouteNode(arrival_id, arrival.latitude, arrival.longitude, "AIRPORT")
    )

    network_nodes = [
        node for node in graph.nodes.values() if not node.node_id.startswith("airport:")
    ]

    def nearest(latitude: float, longitude: float) -> list[tuple[float, RouteNode]]:
        distances = [
            (
                geodesic_distance_m(latitude, longitude, node.latitude, node.longitude),
                node,
            )
            for node in network_nodes
        ]
        distances.sort(key=lambda item: (item[0], item[1].node_id))
        return [
            item
            for item in distances[:connector_count]
            if item[0] <= maximum_distance_m
        ]

    departure_connections = nearest(departure.latitude, departure.longitude)
    arrival_connections = nearest(arrival.latitude, arrival.longitude)
    if not departure_connections:
        raise ValueError(f"no network point near departure {departure.icao}")
    if not arrival_connections:
        raise ValueError(f"no network point near arrival {arrival.icao}")

    for index, (distance, node) in enumerate(departure_connections):
        connected.add_edge(
            RouteEdge(
                edge_id=f"connector:{departure_id}:{index}:{node.node_id}",
                source=departure_id,
                target=node.node_id,
                distance_m=distance * connector_cost_factor,
                layer="connector",
                metadata={"physical_distance_m": distance},
            )
        )
    for index, (distance, node) in enumerate(arrival_connections):
        connected.add_edge(
            RouteEdge(
                edge_id=f"connector:{node.node_id}:{index}:{arrival_id}",
                source=node.node_id,
                target=arrival_id,
                distance_m=distance * connector_cost_factor,
                layer="connector",
                metadata={"physical_distance_m": distance},
            )
        )
    return connected, departure_id, arrival_id


def airport_index(airports: Iterable[Airport]) -> Mapping[str, Airport]:
    """Build an ICAO airport index with duplicate protection."""

    result: dict[str, Airport] = {}
    for airport in airports:
        if airport.icao in result:
            raise ValueError(f"duplicate airport {airport.icao!r}")
        result[airport.icao] = airport
    return result


def select_flight_routes(
    graph: DirectedMultiGraph,
    context: FlightContext,
    airports: Sequence[Airport] | Mapping[str, Airport],
    *,
    edge_cost: EdgeCost = lambda edge: edge.distance_m,
    heuristic: Heuristic = zero_heuristic,
    config: RouteSelectionConfig = RouteSelectionConfig(),
    connector_count: int = 5,
    maximum_connector_distance_m: float = 150_000.0,
) -> tuple[DirectedMultiGraph, list[RoutePath]]:
    """Build a flight graph, connect airports, and select route candidates."""

    index = airports if isinstance(airports, Mapping) else airport_index(airports)
    if context.departure not in index or context.arrival not in index:
        raise KeyError("departure and arrival must exist in the airport index")
    applicable = flight_graph(graph, context)
    connected, source, target = add_airport_connectors(
        applicable,
        index[context.departure],
        index[context.arrival],
        connector_count=connector_count,
        maximum_distance_m=maximum_connector_distance_m,
    )
    planner = RoutePlanner(connected, edge_cost=edge_cost, heuristic=heuristic)
    return connected, planner.candidates(source, target, config=config)
