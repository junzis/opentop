"""Tests for RAD multigraph search and cheap route costs."""

import pytest

from opentop import rad


def _graph():
    graph = rad.DirectedMultiGraph()
    for node_id, latitude, longitude in (
        ("A", 50.0, 2.0),
        ("B", 50.5, 3.0),
        ("C", 49.5, 3.0),
        ("D", 50.0, 4.0),
    ):
        graph.add_node(rad.RouteNode(node_id, latitude, longitude))
    for edge_id, source, target, distance in (
        ("ab-fast", "A", "B", 4.0),
        ("ab-slow", "A", "B", 6.0),
        ("bd", "B", "D", 4.0),
        ("ac", "A", "C", 5.0),
        ("cd", "C", "D", 5.0),
        ("bc", "B", "C", 1.0),
    ):
        graph.add_edge(rad.RouteEdge(edge_id, source, target, distance))
    return graph


def test_a_star_reopens_nodes_and_preserves_parallel_edge_identity():
    graph = _graph()

    path = rad.shortest_path(graph, "A", "D")

    assert path.nodes == ("A", "B", "D")
    assert path.edge_ids == ("ab-fast", "bd")
    assert path.cost == pytest.approx(8.0)


def test_yen_returns_loopless_parallel_and_geometric_alternatives():
    graph = _graph()

    paths = rad.k_shortest_paths(graph, "A", "D", 4)

    assert [path.cost for path in paths] == pytest.approx([8.0, 10.0, 10.0, 10.0])
    assert len({path.edge_ids for path in paths}) == 4
    assert all(len(path.nodes) == len(set(path.nodes)) for path in paths)


def test_search_layer_and_flight_level_filter():
    graph = rad.DirectedMultiGraph()
    graph.add_node(rad.RouteNode("A", 0.0, 0.0))
    graph.add_node(rad.RouteNode("B", 0.0, 1.0))
    graph.add_edge(
        rad.RouteEdge("edge", "A", "B", 1.0, layer="night", min_flight_level=300)
    )
    day = rad.FlightContext("AAAA", "BBBB", 350)
    night = rad.FlightContext("AAAA", "BBBB", 350, enabled_layers=frozenset({"night"}))

    with pytest.raises(rad.RouteNotFound):
        rad.shortest_path(
            graph, "A", "B", edge_predicate=lambda edge: rad.edge_available(edge, day)
        )
    assert rad.shortest_path(
        graph, "A", "B", edge_predicate=lambda edge: rad.edge_available(edge, night)
    ).edge_ids == ("edge",)


def test_nominal_fuel_cost_uses_along_track_wind():
    edge = rad.RouteEdge("edge", "A", "B", 100_000.0)
    still = rad.NominalFuelCost(200.0, 1.0)
    tailwind = rad.NominalFuelCost(200.0, 1.0, lambda _edge: 50.0)

    assert still(edge) == pytest.approx(500.0)
    assert tailwind(edge) == pytest.approx(400.0)


def test_search_budget_stops_pathological_search():
    graph = _graph()

    with pytest.raises(rad.SearchBudgetExceeded):
        rad.shortest_path(graph, "A", "D", budget=rad.SearchBudget(max_expansions=1))
