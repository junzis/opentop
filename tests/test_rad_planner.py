"""Tests for high-level per-flight route planning."""

from pathlib import Path

from opentop import rad

FIXTURES = Path(__file__).parent / "fixtures" / "rad"


def _network():
    navpoints = rad.read_nnpt(FIXTURES / "network.nnpt").records
    routes = rad.read_routes(FIXTURES / "routes.rts").records
    return rad.graph_from_routes(routes, navpoints)


def test_airport_connectors_have_operational_direction():
    graph = _network()
    airports = rad.read_arp(FIXTURES / "airports.arp").records

    connected, source, target = rad.add_airport_connectors(
        graph, airports[0], airports[1], connector_count=1
    )

    assert source == "airport:AAAA:departure"
    assert target == "airport:DDDD:arrival"
    assert next(connected.outgoing(source)).target == "AAA"
    assert not list(connected.outgoing(target))


def test_high_level_flight_route_selection():
    graph = _network()
    airports = rad.read_arp(FIXTURES / "airports.arp").records
    context = rad.FlightContext("AAAA", "DDDD")

    connected, routes = rad.select_flight_routes(
        graph,
        context,
        airports,
        connector_count=1,
        config=rad.RouteSelectionConfig(candidates=1),
    )

    assert len(routes) == 1
    assert routes[0].nodes == (
        "airport:AAAA:departure",
        "AAA",
        "A_B",
        "C",
        "DDD",
        "airport:DDDD:arrival",
    )
    assert len(connected.edges) == 5


def test_candidate_overlap_filter_keeps_diverse_routes():
    graph = rad.DirectedMultiGraph()
    for node_id in "ABCDE":
        graph.add_node(rad.RouteNode(node_id, 0.0, float(ord(node_id))))
    for edge_id, source, target, cost in (
        ("ab", "A", "B", 1.0),
        ("bd", "B", "D", 1.0),
        ("ac", "A", "C", 1.1),
        ("cd", "C", "D", 1.1),
        ("be", "B", "E", 1.0),
        ("ed", "E", "D", 1.1),
    ):
        graph.add_edge(rad.RouteEdge(edge_id, source, target, cost))

    routes = rad.RoutePlanner(graph).candidates(
        "A",
        "D",
        config=rad.RouteSelectionConfig(
            candidates=2,
            search_candidates=5,
            maximum_shared_edge_fraction=0.25,
        ),
    )

    assert [route.nodes for route in routes] == [("A", "B", "D"), ("A", "C", "D")]
