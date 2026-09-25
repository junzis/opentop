# Scalable Fleet Optimization: Design Direction

Status: deferred design note. This document records a possible path for
scaling separation-constrained optimization to hundreds of arriving flights.
It is not an implementation specification or a commitment to a particular
scheduling solver.

## Motivation

`MultiAircraft` currently constructs one joint CasADi Opti problem containing
all participating aircraft. This is appropriate for small coordinated groups,
but it should not be extended into a monolithic problem containing hundreds of
flights. Even with sparse encounter constraints, every aircraft contributes a
full direct-collocation transcription, while the number of possible pairs
grows quadratically.

The expected operational use case is more favorable: hundreds of arrivals are
distributed across several hours. Only a much smaller subset is airborne in
the same terminal region at the same time. The scalable architecture should
therefore depend primarily on peak concurrent traffic and local encounter
density, not on the total number of flights in the planning period.

The existing eighth-order separation approximation remains the lower-level
constraint:

```text
(dx / Rxy)^2 + (dy / Rxy)^2 + (dh / Rz)^8 >= minimum_metric
```

This formulation is not replaced by the decomposition described here.

## Proposed architecture

```text
Independent trajectories
        |
Temporal and spatial screening
        |
Conflict graph
        |
Connected components or rolling windows
        |
Parallel MultiAircraft solves
        |
Global dense verification
```

The current `MultiAircraft` class remains the solver for one small coordinated
group. A future fleet-level orchestration layer would decide which aircraft
belong in each joint problem.

## Independent baseline trajectories

Generate an independent trajectory for every flight before building joint
problems. These trajectories provide:

- predicted entry, arrival, and exit times;
- four-dimensional route envelopes for conflict screening;
- warm starts for the joint component solves;
- a baseline objective for measuring coordination penalties.

Independent solves can run in separate processes. Aircraft-model setup and
other immutable performance data should be cached where practical.

## Temporal and spatial screening

Do not compare every possible aircraft pair. Sort flights by the start of
their active time interval and use a sweep-line or interval-tree algorithm to
find overlapping intervals. Add a time buffer large enough to cover expected
trajectory retiming and encounter refinement.

For temporally overlapping flights, apply a second spatial test using route
envelopes, sector occupancy, merge streams, or projected bounding boxes. A
pair becomes a candidate only if it can approach the separation region.

This changes candidate discovery from approximately `O(N^2)` to
`O(N log N + E)`, where `E` is the number of plausible encounters. The current
candidate-time detection and adaptive constraint refinement then operate only
on those edges.

## Conflict graph decomposition

Represent each flight as a graph vertex and each plausible encounter as an
edge. Disconnected graph components have no separation constraints between
them. With an additive fleet objective, they can be solved independently
without changing the mathematical result.

Each connected component can therefore be passed to `MultiAircraft`, and
different components can be solved in parallel. This is the preferred first
scaling step because it preserves the current joint optimization semantics
inside every interacting group.

A long sequence of arrivals may nevertheless form one large component through
local neighbor-to-neighbor interactions. Graph decomposition alone is not
sufficient for that case.

## Rolling-horizon optimization

Use overlapping time windows when a dense arrival stream forms a large
connected component. Illustrative starting values are:

- optimization horizon: 45 to 60 minutes;
- committed interval: 10 to 15 minutes;
- overlap or look-ahead buffer: 15 to 20 minutes;
- target active problem size: approximately 10 to 30 aircraft.

These values must be calibrated to route duration, terminal geometry, traffic
density, and allowed delay. At each step:

1. Include active flights and flights entering during the look-ahead period.
2. Warm-start from the previous window and independent trajectories.
3. Freeze decisions that have already been committed.
4. Optimize the active window.
5. Commit only its leading interval and advance the horizon.

Flights crossing a window boundary remain in the next window with their
committed state and timing carried forward. The overlap prevents a flight near
the boundary from being optimized without knowledge of an imminent encounter.

## Hierarchical arrival scheduling

For same-airport arrivals, runway order and landing slots are discrete
scheduling decisions. Asking one non-convex trajectory NLP to discover both
the sequence and all continuous trajectories is unlikely to scale reliably.

A two-level approach is preferable:

1. An upper scheduling layer selects runway sequence, target arrival times,
   merge order, and minimum runway or wake-separation gaps. MILP or CP-SAT are
   possible technologies for this layer.
2. The lower `MultiAircraft` layer optimizes continuous paths, speeds, and
   altitudes while meeting the assigned timing targets and the high-order
   pairwise separation constraints.

The layers may iterate if the continuous problem reports that a proposed
schedule is infeasible or excessively costly. The first implementation should
avoid a tightly coupled distributed algorithm until the simpler sequential
hierarchy has been evaluated.

## Component-level numerical strategy

Within each component or rolling window, retain the current numerical
techniques:

- independent warm starts;
- lateral or vertical symmetry-breaking perturbations;
- candidate encounter windows instead of blanket pair-time constraints;
- single-interval collocation interpolation;
- limited-memory IPOPT Hessian approximation;
- objective normalization;
- dense post-solve separation verification;
- selective constraint refinement for detected violations.

The dense verifier must use a global absolute-time axis and must also check
pairs assigned to neighboring windows. A solution is accepted only after this
global verification succeeds.

## Parallel execution

Natural parallel units are:

- independent single-aircraft baseline solves;
- disconnected conflict-graph components;
- scenario or robustness evaluations.

Separate processes are preferable to threads initially because CasADi/IPOPT
objects contain native state and the optimizer classes are mutable. Parallel
execution must limit aggregate memory use; solving too many medium-sized NLPs
at once may be slower than a smaller worker pool.

Rolling windows from the same connected stream are sequential because each
window consumes committed decisions from its predecessor.

## Possible public API boundary

A future class tentatively named `FleetOptimizer` could sit above
`MultiAircraft` and own:

- flight time indexing and spatial screening;
- conflict-graph construction;
- connected-component decomposition;
- rolling-window state and commitment;
- optional scheduling-layer integration;
- worker-process orchestration;
- global verification and aggregated results.

`MultiAircraft` should remain usable directly for small problems. This avoids
breaking its current API and keeps fleet orchestration separate from the
direct-collocation formulation.

## Benchmark plan

Evaluate synthetic and representative arrival sets containing 100, 300, and
1,000 flights distributed across several hours. Report at least:

- total possible pairs and screened candidate pairs;
- number and size distribution of graph components;
- maximum aircraft count in a rolling window;
- NLP variables, constraints, and separation constraints per solve;
- build, solve, and verification time;
- peak memory and worker utilization;
- IPOPT iterations and refinement rounds;
- globally verified minimum separation metric;
- fuel, time, and delay penalties relative to independent trajectories;
- failures and any required schedule revisions.

For small cases, compare decomposed and monolithic solutions to quantify any
objective difference introduced by rolling-horizon commitment. No numerical
speedup claim should be made until both implementations are benchmarked under
the same hardware, mesh, traffic, and solver settings.

## Open questions

- What time and spatial buffers guarantee that screening does not omit a pair
  that becomes relevant after trajectory retiming?
- What window duration is sufficient for different terminal geometries?
- Should runway scheduling use fixed assignments, bounded target-time windows,
  or penalties for missing targets?
- How should infeasibility be communicated from trajectory optimization back
  to the scheduling layer?
- Which decisions can be frozen safely at a rolling-horizon boundary?
- How should wake category, multiple runways, holding, and missed approaches be
  represented?
- What is the acceptable optimality loss relative to a monolithic solve on
  cases small enough to solve both ways?

## Suggested implementation order

1. Add fleet-level temporal screening and conflict-graph construction without
   changing `MultiAircraft`.
2. Solve disconnected components independently and perform global dense
   verification.
3. Add process-level parallelism and resource limits.
4. Implement overlapping rolling windows for large connected components.
5. Introduce an arrival scheduling layer after the decomposition behavior is
   measured on realistic traffic.

This ordering delivers useful scaling incrementally while keeping the current
small-fleet optimizer and separation formulation stable.
