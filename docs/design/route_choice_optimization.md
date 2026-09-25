# Route choice and trajectory optimization

## Purpose

OpenTOP separates the source of route alternatives from the continuous flight
optimization. This supports company routes, dispatch-system candidates, RAD
search results, and future route providers without making the trajectory
optimizer understand every source format.

The shared pipeline is:

```text
route source                 common boundary             OpenTOP
-------------                ---------------             -------
company route catalogue  ->  RouteOption[]  ─┐
company waypoint network ->  RouteOption[]  ─┼-> optimize_routes()
RAD graph search          ->  RouteOption[]  ─┘        |
                                                     one NLP per route
                                                        |
                                                  RouteChoiceResult
                                                        |
                                             fuel or configured objective
```

`RouteOption` is intentionally small. It contains a name, an ordered polyline
including origin and destination, and optional source metadata. It does not
contain RAD rules, graph edges, aircraft data, or optimization settings.

## Generic API

For a company-route catalogue, callers construct options directly:

```python
import opentop as top

routes = [
    top.RouteOption(
        "Company route A",
        (
            (52.3086, 4.7639),
            (49.2, 8.0),
            (44.5, 11.0),
            (41.8003, 12.2389),
        ),
        metadata={"operator": "Example Air"},
    ),
]


def optimizer_factory():
    return top.CompleteFlight(
        "A320",
        routes[0].waypoints[0],
        routes[0].waypoints[-1],
    )


optimization = top.RouteOptimizationConfig(objective="fuel")
result = top.optimize_routes(routes, optimizer_factory, config=optimization)
trajectory = result.trajectory
```

Each option is optimized independently with a fresh optimizer. OpenTOP uses the
interior polyline points as ordered waypoint constraints, builds a route-shaped
initial guess, and selects the lowest-fuel successful trajectory by default. Set
`ranking_metric="objective"` to rank by the selected objective instead. Failures are
retained per option and do not stop the remaining candidates.

The factory currently assumes every option has the same origin, destination,
aircraft, and performance configuration. This matches an airport-pair company
route catalogue and ensures the resulting fuel values are comparable.

## Company waypoint networks

When an airline supplies reusable waypoints rather than complete routes, the
caller must also define the permitted connections between them. Coordinates
alone do not say which waypoint pairs form legal operational legs.

```python
network = top.RouteNetwork.from_connections(
    {
        "DEP": (52.3086, 4.7639),
        "FIX_A": (49.2, 8.0),
        "FIX_B": (47.0, 10.8),
        "ARR": (41.8003, 12.2389),
    },
    (("DEP", "FIX_A"), ("FIX_A", "FIX_B"), ("FIX_B", "ARR")),
)
selection = network.select_routes(
    "DEP",
    "ARR",
    config=top.RouteSelectionConfig(candidates=5),
)
result = top.optimize_routes(
    selection.options,
    optimizer_factory,
    config=optimization,
)
```

`RouteNetwork` and RAD use the same source-neutral directed multigraph, A*/Yen
search, candidate limits, and edge-overlap filtering. RAD adds parsing and
availability semantics before this shared search; the company network does not.

## RAD adapter

RAD remains responsible for parsing its source formats, evaluating availability,
building the applicable graph, and generating candidate paths. The handoff is
explicit:

```python
selection = dataset.select_routes(context)
selection.options  # tuple[RouteOption, ...]
result = top.optimize_routes(
    selection.options,
    optimizer_factory,
    config=optimization,
)
```

RAD-specific paths and conformance metadata remain available in
`RadRouteSelection`; they are not leaked into the generic `RouteChoiceResult`.

## Recommended API surface

The composable public workflow has one selection verb and one optimization verb:

```python
selection = route_source.select_routes(...)
result = top.optimize_routes(selection.options, optimizer_factory, config=...)
```

A predefined catalogue already contains complete `RouteOption` objects and
therefore skips the selection call. There is intentionally no RAD-specific
continuous optimization method.

`opentop.rad.integration` contains route-conversion and validation helpers, not
another optimization entry point. Continuous route optimization always uses
`optimize_routes()` plus `RouteOptimizationConfig`.

## Continuous optimization behavior

Route selection fixes the ordered horizontal corridor, within waypoint
tolerance. It does not fix speed or altitude. Those remain OpenTOP decision
variables and depend on the chosen phase optimizer:

- `CompleteFlight` optimizes climb, cruise, descent, speed, and fuel over the
  complete route.
- `Cruise` optimizes the cruise segment only. A fuel-optimal cruise result may
  stay nearly level even though altitude is not fixed.

The graph ranking cost is only a screening measure. The final route is selected
using fuel by default, or the configured objective when
`ranking_metric="objective"`. It does not use route distance or nominal graph
cost as the final ranking. `objective_factory` can bind an objective to each
fresh optimizer; `trajectory_kwargs` forwards trajectory options.

## Scope and future extensions

The common boundary can later accept route identifiers, validity intervals, or
operational tags through `metadata` without changing the optimizer. Source-side
adapters can reject invalid options before the expensive NLP stage. Larger
catalogues should use cheap geometric/nominal-fuel screening and coarse-to-fine
optimization before solving all candidates at the final mesh resolution.
