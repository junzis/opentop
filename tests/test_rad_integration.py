"""Tests for converting selected routes into OpenTOP inputs."""

from pathlib import Path

import pandas as pd
from opentop import RouteOptimizationConfig, optimize_routes, rad

FIXTURES = Path(__file__).parent / "fixtures" / "rad"


def _route():
    graph = rad.DirectedMultiGraph()
    graph.add_node(rad.RouteNode("A", 50.0, 2.0))
    graph.add_node(rad.RouteNode("B", 50.5, 3.0))
    graph.add_node(rad.RouteNode("C", 51.0, 4.0))
    edges = (
        rad.RouteEdge("ab", "A", "B", 90_000.0),
        rad.RouteEdge("bc", "B", "C", 90_000.0),
    )
    for edge in edges:
        graph.add_edge(edge)
    return graph, rad.RoutePath(("A", "B", "C"), edges, 180_000.0)


def test_route_initial_guess_follows_polyline_and_has_optimizer_shape():
    graph, route = _route()

    guess = rad.route_initial_guess(
        graph, route, nodes=10, altitude_ft=30_000.0, mass_kg=60_000.0
    )

    assert len(guess) == 11
    assert guess.latitude.iloc[0] == 50.0
    assert guess.longitude.iloc[-1] == 4.0
    assert guess.ts.is_monotonic_increasing
    assert guess.altitude.eq(30_000.0).all()


def test_route_waypoint_simplification_removes_nearly_collinear_point():
    graph, route = _route()

    simplified = rad.simplify_route_waypoints(
        graph,
        route,
        tolerance_m=20_000.0,
        include_endpoints=True,
    )

    assert simplified == [(50.0, 2.0), (51.0, 4.0)]


def test_route_waypoint_simplification_rejects_invalid_tolerance():
    graph, route = _route()

    try:
        rad.simplify_route_waypoints(graph, route, tolerance_m=0.0)
    except ValueError as error:
        assert "tolerance" in str(error)
    else:
        raise AssertionError("zero simplification tolerance must fail")


def test_complete_flight_guess_has_low_terminal_altitudes():
    graph, route = _route()

    guess = rad.route_initial_guess(
        graph, route, nodes=20, altitude_ft=35_000.0, complete_flight=True
    )

    assert guess.altitude.iloc[0] == 100.0
    assert guess.altitude.iloc[-1] == 100.0
    assert guess.altitude.max() == 35_000.0


def test_route_validation_checks_waypoint_order_and_tolerance():
    trajectory = pd.DataFrame(
        {
            "latitude": [50.0, 50.5, 51.0],
            "longitude": [2.0, 3.0, 4.0],
        }
    )

    assert rad.validate_route_trajectory(trajectory, [(50.5, 3.0)], tolerance_m=100.0)
    assert not rad.validate_route_trajectory(
        trajectory, [(50.5, 3.0), (50.0, 2.0)], tolerance_m=100.0
    )


def test_high_level_dataset_api_selects_and_optimizes_route():
    graph, _ = _route()
    provenance = rad.Provenance(Path(__file__), 1, "")
    airports = {
        "AAAA": rad.Airport("AAAA", 50.0, 2.0, None, provenance),
        "CCCC": rad.Airport("CCCC", 51.0, 4.0, None, provenance),
    }
    dataset = rad.RadDataset(graph, airports)
    context = rad.FlightContext("AAAA", "CCCC")

    class Result:
        def __init__(self, df):
            self.df = df
            self.success = True
            self.fuel = 123.0
            self.status = "ok"

    class Optimizer:
        def __init__(self):
            self.aircraft = {"cruise": {"height": 10_000.0}}
            self.mass_init = 60_000.0

        def setup(self, *, nodes):
            self.nodes = nodes

        def trajectory(self, **kwargs):
            guess = kwargs["initial_guess"]
            waypoints = kwargs["waypoints"]
            for index, waypoint in enumerate(waypoints, start=1):
                row = round(index * (len(guess) - 1) / (len(waypoints) + 1))
                guess.loc[row, ["latitude", "longitude"]] = waypoint
            return Result(guess)

    selection = dataset.select_routes(
        context,
        config=rad.RouteSelectionConfig(
            candidates=1,
            search_candidates=1,
        ),
        connector_count=1,
        maximum_connector_distance_m=10_000.0,
    )
    result = optimize_routes(
        selection.options,
        Optimizer,
        config=RouteOptimizationConfig(
            waypoint_tolerance_m=100.0,
            minimum_nodes=10,
        ),
    )

    assert result.best is not None
    assert result.best.fuel_kg == 123.0
    assert result.optimal_route is result.best.route
    assert result.trajectory is not None
    assert len(result.solve_seconds) == 1
    assert len(selection.options) == 1
    assert selection.options[0].name == "RAD route 1"


def test_high_level_dataset_exposes_generic_route_selection():
    graph, _ = _route()
    provenance = rad.Provenance(Path(__file__), 1, "")
    dataset = rad.RadDataset(
        graph,
        {
            "AAAA": rad.Airport("AAAA", 50.0, 2.0, None, provenance),
            "CCCC": rad.Airport("CCCC", 51.0, 4.0, None, provenance),
        },
    )

    selection = dataset.select_routes(
        rad.FlightContext("AAAA", "CCCC"),
        config=rad.RouteSelectionConfig(
            candidates=1,
            search_candidates=1,
        ),
        connector_count=1,
        maximum_connector_distance_m=10_000.0,
    )

    assert len(selection.candidates) == 1
    assert len(selection.options) == 1
    assert selection.options[0].waypoints[0] == (50.0, 2.0)
    assert selection.options[0].waypoints[-1] == (51.0, 4.0)


def test_high_level_dataset_api_loads_ase_files():
    schema = rad.AseCodeSchema(
        "fixture",
        "1",
        {(0, 1, 10): rad.AseSemantics(rad.EdgeDirection.FORWARD)},
    )

    dataset = rad.RadDataset.from_ase_files(
        FIXTURES / "network.nnpt",
        FIXTURES / "segments.ase",
        FIXTURES / "airports.arp",
        schema=schema,
        layer="vst",
        strict=False,
    )

    assert len(dataset.graph.edges) == 2
    assert set(dataset.airports) == {"AAAA", "DDDD"}
    assert set(dataset.parse_results) == {"navpoints", "segments", "airports"}
