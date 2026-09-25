# Python API guide

This guide describes the current working tree. Import the public API with:

```python
import opentop as top
```

## Performance and fuel selection

`performance_model="openap"` is the default. `"bada3"` and `"bada4"` require
`bada_path=` and the corresponding external data. The baseline supplies thrust,
drag and default fuel flow. Operating limits and mass limits still use OpenAP
aircraft data; selecting BADA does not replace every aircraft parameter.

`fuel_model=` overrides fuel flow in mass dynamics, objectives, emissions fuel
input, and trajectory reporting. It is accepted by `CompleteFlight`, `Cruise`,
`Climb`, and `Descent`. Climb/descent forward it to their preliminary cruise solve.

```python
fuel = top.AcropoleFuelFlow(
    "A320",
    model_path="/path/to/acropole_fuel_model.onnx",
    aircraft_params_path="/path/to/aircraft_params.csv",
    activation_smoothing=0.01,
)
optimizer = top.CompleteFlight(
    "A320", "LFPG", "EDDM", m0=0.85,
    performance_model="openap", fuel_model=fuel,
)
result = optimizer.trajectory(objective="fuel", result_object=True)
```

Install the `neural` extra for ONNX import. Smoothing defaults to **0**, preserving
ReLU; 0.01 explicitly changes the network to a smooth approximation. Neural
optimization currently requires still air and sets speed-derivative features to
zero. `engine=` does not recalibrate Acropole's CSV scale or reconfigure supplied
correction models. See the [neural notes](design/neural_fuel.md#model-configuration-contract)
for physical corrections and compatibility checks. The CLI does not expose the
neural override; configure it through Python.

## Loading and initialization

`m0` is a fraction of OpenAP MTOW, not kilograms. Without `payload`, it fixes the
initial mass. `CompleteFlight` and `Cruise` also accept `payload` in kilograms:
this makes initial mass a decision variable and uses `m0` only as its starting
guess. Bounds use OEW plus payload, fuel capacity and MTOW. `Climb` and `Descent`
do not expose payload optimization.

Use `setup(nodes=..., max_iter=...)` to configure discretization and IPOPT.
`initial_guess=` supplies a trajectory DataFrame where supported; Mach values
initialize controls and are clipped to the node bounds. Supply `nodes + 1` rows, including finite Mach values when initializing speed.
Standalone `Climb` currently uses the supplied Mach profile but constructs its
own state guess; lateral/altitude multi-start perturbations do not initialize
its states.

## Cruise constraints

```python
cruise = top.Cruise("A320", "LFPG", "EDDM", mach_value=0.78)
cruise.fix_cruise_altitude()
```

`mach_value=0.78` pins the speed to that value. In contrast,
`fix_mach_number()` lets the optimizer choose one constant Mach.
`fix_cruise_altitude()` similarly chooses one constant altitude; it does not set
a specified flight level. Constructor `h_min` and `h_max` use **meters**.

`follow_track(lat, lon, tolerance_m=1000)` accepts matching coordinate arrays
with at least four points. Boundary nodes lie on a spline through the track;
collocation and quarter-interval checks enforce the tolerance between nodes.
This is a sampled constraint, not a continuous guarantee. It cannot be combined
with `fix_track_angle()`, which constrains heading rather than following a
recorded curved track. Turn-rate limits still apply.

## Results and costs

`trajectory()` normally returns a DataFrame. `result_object=True` returns a
`TrajectoryResult` with `df`, `success`, `status`, `objective`, `iters`, `fuel`,
`grid_cost`, and `stats`. Check success before treating a returned iterate as an
optimized solution; failure handling differs between phase methods.

- `fuel` is mass loss in kg; the DataFrame's `fuel_cost` sums to that loss.
- `fuelflow` is instantaneous total aircraft fuel flow in kg/s.
- `objective` is the selected objective in its physical scale, even when internal
  objective rescaling is enabled.
- `result.grid_cost` and `optimizer.grid_cost_value` use solver collocation
  quadrature when available. Summing DataFrame `grid_cost` gives a node-based
  approximation and can differ. No interpolant gives a NaN result grid cost.

## Multi-start

```python
cruise = top.Cruise("A320", "LFPG", "EDDM")
best, candidates = cruise.multi_start_trajectory(
    objective="fuel", n_starts=5, seed=42,
    lateral_jitter_km=0, altitude_jitter_ft=1500, mach_jitter=0.04,
)
```

`mach_jitter` defaults to zero. A supplied initial guess must include a `mach`
column to enable speed perturbations. Candidates are ranked by solver success,
then objective; this is not a global-optimality guarantee. See
[multi-start details](design/multi_start_trajectory.md).

## Route alternatives and multiple aircraft

[Route choice](design/route_choice_optimization.md) uses `RouteOption` objects
and `optimize_routes()`. Each route receives a fresh optimizer.
`RouteOptimizationConfig.ranking_metric` defaults to `"fuel"`; use `"objective"`
to choose by the configured optimization objective instead. Graph screening and
continuous trajectory optimization are distinct stages.

`MultiAircraft` combines `FlightSpec` entries with `Cruise` or `Descent`
optimizers in a joint separation-constrained solve. It does not accept
`CompleteFlight` or `Climb`. Each spec carries an already configured optimizer,
so a fuel override is configured on that optimizer. Fleet wind, variable
timesteps and time-dependent grid objectives are currently unsupported.
`common_altitude=True` requires fixed-altitude cruises;
`minimum_arrival_gap_s` is for descents. Check both solver and separation results
in `MultiAircraftResult`. Large-scale decomposition remains a
[proposal](design/scalable_fleet_optimization.md).
