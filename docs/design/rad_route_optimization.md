# RAD route selection and trajectory optimization

## Purpose

OpenTOP's continuous trajectory optimizer is not intended to choose among a
continent-scale set of discrete air-route edges directly. The production design
therefore uses a hybrid process:

1. Read and normalize the supported RAD network and rule data.
2. Construct the edge subset applicable to a flight's static planning context.
3. Generate a small set of loopless route candidates with graph search.
4. Rank candidates with distance or a cheap wind-adjusted nominal fuel surrogate.
5. Pass the surviving ordered waypoints to independent OpenTOP NLP solves.
6. Compare actual optimized fuel and revalidate route and RAD conformance.

This keeps the existing `CompleteFlight` and phase APIs unchanged. RAD-specific
processing is available under `opentop.rad`; its selected paths cross the common
`RouteOption` boundary described in `route_choice_optimization.md` before using
the existing waypoint interface.

## High-level API

`RadDataset.from_ase_files` parses NNPT, ASE, and ARP inputs and builds a graph
with an explicit `AseCodeSchema`. `RadDataset.select_routes` applies a
`FlightContext`, adds airport connectors, selects diverse candidates, and
converts them to source-neutral `RouteOption` objects. Callers then pass those
options to the same top-level optimization function used by every other route
source:

```python
selection = dataset.select_routes(context, config=selection_config)
result = top.optimize_routes(
    selection.options,
    optimizer_factory,
    config=optimization_config,
)
```

`RadRouteSelection` retains the connected graph and discrete paths, while the
source-neutral `RouteChoiceResult` contains continuous results, solve timings,
and convenience accessors for the best route and trajectory.

The lower-level readers, graph builders, search functions, and integration
helpers remain public for dataset integration, validation, and tuning. The
high-level API composes those functions; it does not introduce a second routing
or optimization implementation.

## Source formats and attribution

The normalized NNPT, ARP, and RTS readers were informed by the private
`alazarovski/rddr` R package, with permission from the repository owner. They are
Python adaptations with stricter provenance and conformance handling. The ASE
reader is an OpenTOP implementation because rddr does not parse ASE files.

The supplied AIRAC 2406 sample establishes the following ASE structure:

- three integer metadata fields;
- source and target latitude/longitude in arc-minutes; and
- one endpoint token that resolves to two NNPT identifiers.

All 55,013 VST records resolve to the supplied NNPT file with matching
coordinates. The meanings of the first three integer fields are not inferred.
`AseCodeSchema` requires a named, versioned mapping from raw codes to verified
direction and usability semantics. Unknown codes fail closed by default.

VST, DCT, night, and weekend files are separate network layers. They must not be
blindly unioned. A `FlightContext` activates applicable layers and flight-level
bounds before search.

Legacy RAD text may be either UTF-8 or Windows-1252. Readers try UTF-8 first and
then Windows-1252 deterministically.

## Conformance contract

Parsing and rule evaluation use three outcomes:

- `valid`: the construct is understood and passes the supported checks;
- `invalid`: the source record is malformed; and
- `indeterminate`: the record is structurally readable but its meaning or an
  evaluation input is unsupported or unavailable.

Applicable unsupported rules are never treated as satisfied. Every normalized
record retains file, line number, and original text provenance.

The initial AWK/FLC2 rule parser supports boolean combinations of
regular-expression, string equality, and numeric comparison predicates. It
retains unsupported rules as indeterminate, preserves raw `REGLE` values, and
extracts `FL_CONT` assignments without inferring whether the surrounding RAD
prose describes a cap or another operational interpretation.

## Graph and search model

`DirectedMultiGraph` assigns every edge a stable identity. Parallel edges between
the same waypoint pair remain distinct, which is necessary for route names,
availability layers, and Yen alternatives.

The first path is found with A* (or Dijkstra when the heuristic is zero). The
implementation supports node reopening, deterministic ties, excluded edge IDs,
and explicit expansion/cost budgets. Further loopless alternatives are generated
lazily with an edge-ID-aware Yen algorithm.

Candidate generation applies:

- maximum cost and physical-distance ratios relative to the best route;
- a search-candidate budget; and
- a maximum shared-edge fraction to avoid sending nearly identical candidates to
  expensive NLP solves.

Airport connectors are directional: departure airport to nearby network points,
and nearby network points to the arrival airport. Connector count and maximum
distance are explicit.

## Cost model

Distance is the simplest safe graph cost. `NominalFuelCost` adds little overhead:

```text
nominal fuel = distance / max(minimum groundspeed, TAS + along-track wind)
               * nominal fuel flow
               + edge penalty
```

With constant nominal fuel flow this is primarily a wind-adjusted travel-time
ranking. It is deliberately not presented as the final fuel answer. OpenTOP's
aircraft model and optimized vertical profile determine actual candidate fuel.

Straight-line distance is an admissible A* heuristic only for distance costs.
Nominal-fuel searches default to a zero heuristic unless a correctly scaled lower
bound is supplied.

## OpenTOP handoff

Each candidate becomes ordered `(latitude, longitude)` waypoints. A geodesically
resampled route-shaped DataFrame provides the initial horizontal state, time is
initialized from nominal groundspeed, and altitude is initialized either as level
cruise or a complete-flight climb/cruise/descent profile.

Dense operational networks can contain many nearly collinear intermediate
points. `simplify_waypoints` applies Ramer-Douglas-Peucker simplification
after projecting a route to a local azimuthal-equidistant CRS, so its tolerance
is expressed in meters. `RouteOptimizationConfig` accepts the optional
simplification tolerance. The initial guess continues to follow the complete
graph polyline; only redundant NLP waypoint constraints are removed. The default
remains unsimplified.

Every candidate receives a fresh optimizer and is solved serially by default.
The route is post-validated by checking waypoint tolerance and order. Failed
candidates retain an explicit status rather than terminating evaluation of the
remaining candidates.

For large route sets, the graph stage should screen aggressively before the NLP
stage. A future coarse-to-fine or process-parallel implementation should remain
behind `top.optimize_routes` rather than adding another public optimization API.

RAD does not prescribe a constant optimized altitude in this interface. The RAD
example uses `Cruise`, so it optimizes only a cruise segment and its fuel optimum
settles at a nearly constant level. The requested flight level in
`FlightContext` filters and orients available graph edges; it is not an equality
constraint on the OpenTOP altitude state. Use `CompleteFlight` when climb and
descent should be part of the optimized trajectory.

## Current boundary and next increments

The implemented foundation includes NNPT, ARP, RTS, lossless ASE, initial AWK and
FLC2 parsing, layered flight graphs, A*/Yen search, candidate pruning, nominal
fuel ranking, route-shaped initial guesses, serial/coarse-to-fine OpenTOP handoff,
and post-solve waypoint validation.

Before an ASE dataset is route-usable, its authoritative field-code mapping must
be supplied as an `AseCodeSchema`. Additional AWK/FLC2 action semantics,
calendar/time predicates, departure/arrival procedure semantics, and optimized
profile rule revalidation will be added as independently tested conformance
increments. A later performance stage can use spawn-based worker processes with
serializable optimization requests; CasADi optimizer objects must not be shared
between processes.
