"""Tests for source-neutral route choice and optimization."""

import opentop as top
import pandas as pd


def test_route_option_normalizes_coordinates_and_reports_distance():
    route = top.RouteOption(
        "Company route A",
        ((52, 4), (50.0, 8.0), (42, 12)),
        metadata={"operator": "example"},
    )

    assert route.waypoints[0] == (52.0, 4.0)
    assert route.interior_waypoints == ((50.0, 8.0),)
    assert route.distance_m > 1_000_000.0
    assert route.metadata["operator"] == "example"


def test_simplify_waypoints_removes_nearly_collinear_point():
    simplified = top.routes.simplify_waypoints(
        ((50.0, 2.0), (50.5, 3.0), (51.0, 4.0)),
        tolerance_m=20_000.0,
    )

    assert simplified == [(50.0, 2.0), (51.0, 4.0)]


def test_complete_flight_initial_guess_has_climb_and_descent():
    route = top.RouteOption("A", ((52.0, 4.0), (42.0, 12.0)))

    guess = top.routes.route_initial_guess(
        route,
        nodes=20,
        altitude_ft=35_000.0,
        complete_flight=True,
    )

    assert len(guess) == 21
    assert guess.altitude.iloc[0] == 100.0
    assert guess.altitude.iloc[-1] == 100.0
    assert guess.altitude.max() == 35_000.0


def test_optimize_routes_selects_lowest_fuel_with_fresh_optimizers():
    routes = (
        top.RouteOption("A", ((50.0, 2.0), (50.5, 3.0), (51.0, 4.0))),
        top.RouteOption("B", ((50.0, 2.0), (50.3, 3.0), (51.0, 4.0))),
    )
    fuels = iter((200.0, 150.0))
    created = 0

    class Result:
        def __init__(self, df: pd.DataFrame, fuel: float):
            self.df = df
            self.success = True
            self.fuel = fuel
            self.status = "ok"

    class Optimizer:
        def __init__(self, fuel: float):
            self.aircraft = {"cruise": {"height": 10_000.0}}
            self.mass_init = 60_000.0
            self.fuel = fuel

        def setup(self, *, nodes: int) -> None:
            self.nodes = nodes

        def trajectory(self, **kwargs):
            guess = kwargs["initial_guess"]
            waypoints = kwargs["waypoints"]
            for index, waypoint in enumerate(waypoints, start=1):
                row = round(index * (len(guess) - 1) / (len(waypoints) + 1))
                guess.loc[row, ["latitude", "longitude"]] = waypoint
            return Result(guess, self.fuel)

    def factory() -> Optimizer:
        nonlocal created
        created += 1
        return Optimizer(next(fuels))

    result = top.optimize_routes(
        routes,
        factory,
        config=top.RouteOptimizationConfig(
            minimum_nodes=10,
            waypoint_tolerance_m=100.0,
        ),
    )

    assert created == 2
    assert len(result.successful) == 2
    assert result.best is not None
    assert result.best.route.name == "B"
    assert result.best.fuel_kg == 150.0
    assert result.optimal_route == routes[1]
    assert result.trajectory is not None
    assert len(result.solve_seconds) == 2


def test_optimize_routes_can_rank_an_optimizer_specific_objective():
    routes = (
        top.RouteOption("lower fuel", ((50.0, 2.0), (51.0, 4.0))),
        top.RouteOption("lower climate cost", ((50.0, 2.0), (51.0, 4.0))),
    )
    metrics = iter(((100.0, 4.0, 3.0), (120.0, 2.0, 1.0)))
    objective_owners = []

    class Result:
        def __init__(
            self,
            df: pd.DataFrame,
            fuel: float,
            objective: float,
            grid_cost: float,
        ):
            self.df = df
            self.success = True
            self.fuel = fuel
            self.objective = objective
            self.grid_cost = grid_cost
            self.status = "ok"

    class Optimizer:
        def __init__(self, metrics: tuple[float, float, float]):
            self.aircraft = {"cruise": {"height": 10_000.0}}
            self.mass_init = 60_000.0
            self.metrics = metrics

        def setup(self, *, nodes: int) -> None:
            self.nodes = nodes

        def trajectory(self, **kwargs):
            assert kwargs["objective"]() is self
            return Result(kwargs["initial_guess"], *self.metrics)

    def factory() -> Optimizer:
        return Optimizer(next(metrics))

    def objective_factory(optimizer):
        objective_owners.append(optimizer)
        return lambda: optimizer

    result = top.optimize_routes(
        routes,
        factory,
        config=top.RouteOptimizationConfig(
            objective_factory=objective_factory,
            ranking_metric="objective",
            minimum_nodes=5,
        ),
    )

    assert len(objective_owners) == 2
    assert result.ranking_metric == "objective"
    assert result.best is not None
    assert result.best.route.name == "lower climate cost"
    assert result.best.fuel_kg == 120.0
    assert result.best.objective_value == 2.0
    assert result.best.grid_cost == 1.0


def test_route_network_selects_source_neutral_options():
    network = top.RouteNetwork.from_connections(
        {
            "A": (50.0, 2.0),
            "B": (50.5, 2.8),
            "C": (49.8, 3.0),
            "D": (51.0, 4.0),
        },
        (
            ("A", "B"),
            ("B", "D"),
            ("A", "C"),
            ("C", "D"),
        ),
    )

    selection = network.select_routes(
        "A",
        "D",
        config=top.RouteSelectionConfig(
            candidates=2,
            search_candidates=2,
            max_cost_ratio=2.0,
            max_distance_ratio=2.0,
        ),
    )

    assert len(selection.paths) == 2
    assert len(selection.options) == 2
    assert all(isinstance(option, top.RouteOption) for option in selection.options)
    assert selection.options[0].waypoints[0] == (50.0, 2.0)
    assert selection.options[0].waypoints[-1] == (51.0, 4.0)
    assert selection.options[0].metadata["source"] == "waypoint_network"
    assert selection.options[0].metadata["node_ids"] == selection.paths[0].nodes


def test_route_network_validates_connections():
    try:
        top.RouteNetwork.from_connections(
            {"A": (50.0, 2.0)},
            (("A", "missing"),),
        )
    except KeyError as error:
        assert "endpoints" in str(error)
    else:
        raise AssertionError("unknown connection endpoint must fail")


def test_optimize_routes_isolates_candidate_failure():
    routes = (
        top.RouteOption("broken", ((50.0, 2.0), (51.0, 4.0))),
        top.RouteOption("valid", ((50.0, 2.0), (51.0, 4.0))),
    )
    calls = 0

    class Result:
        def __init__(self, df: pd.DataFrame):
            self.df = df
            self.success = True
            self.fuel = 100.0
            self.status = "ok"

    class Optimizer:
        def __init__(self):
            self.aircraft = {"cruise": {"height": 10_000.0}}
            self.mass_init = 60_000.0

        def setup(self, *, nodes: int) -> None:
            self.nodes = nodes

        def trajectory(self, **kwargs):
            return Result(kwargs["initial_guess"])

    def factory() -> Optimizer:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("candidate failed")
        return Optimizer()

    result = top.optimize_routes(
        routes,
        factory,
        config=top.RouteOptimizationConfig(minimum_nodes=5),
    )

    assert not result.optimized[0].success
    assert "candidate failed" in result.optimized[0].status
    assert result.optimized[1].success
    assert result.best == result.optimized[1]
