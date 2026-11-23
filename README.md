# OpenTOP: Open Trajectory Optimizer

`opentop` is **v2 of [`openap-top`](https://pypi.org/project/openap-top/)**. The package has been renamed and restructured: what used to be installed as `openap-top` and imported as `openap.top` now installs as `opentop` and imports as `import opentop`. The three headline changes in v2.0:

- **New Opti-stack backend** — NLP construction has moved to CasADi's Opti stack, removing ~400 lines of boilerplate and fixing several long-standing bugs.
- **Standalone package** — `opentop` is no longer a namespace extension of `openap`; it installs as a top-level package and imports as `import opentop`.
- **Command-line interface** — `opentop optimize` and `opentop gengrid` expose the optimizer and grid-cache builder from the shell.

See the [documentation index](docs/README.md) for the current API and architecture.

See [What's New in 2.0](#whats-new-in-20) below for the full migration table.

Flight trajectory optimizer based on the [OpenAP](https://github.com/junzis/openap) aircraft performance model.

`opentop` uses non-linear optimal control via direct collocation (CasADi + IPOPT) to generate optimal flight trajectories. It provides simple interfaces for:

- Complete flight trajectories (takeoff → cruise → landing)
- Individual phases: climb, cruise, descent
- Fuel-optimal, time-optimal, cost-index, and climate-optimal objectives
- Wind integration
- Custom 4D grid cost functions (contrails, weather, airspace)
- User-defined objective functions and constraints

## 🕮 User Guide

Detailed guide and examples: <https://openap.dev/optimize>.

## Install

From PyPI:

```sh
pip install --upgrade opentop
```

From the development branch:

```sh
pip install --upgrade git+https://github.com/junzis/openap
pip install --upgrade git+https://github.com/junzis/opentop
```

`opentop` is a standalone package. Prior to v2.0 it shipped as `openap.top`, a namespace extension of `openap`; v2.0 drops that and installs as a top-level `opentop` package instead.

## Control representation

OpenTOP 2.6 uses continuous, piecewise-linear Mach, vertical-speed
and heading controls. With `nodes=N` intervals it optimizes `N+1` boundary
controls, including an independent arrival control. Dynamics and objective
quadrature evaluate the interpolated control at every collocation point.
Phase rate limits cover every interval using its actual duration, including
variable timesteps. Cruise performance constraints also cover internal
collocation points; nonlinear feasibility still requires between-node checks.
For `CompleteFlight`, `trajectory(path_constraint_points=[(interval, tau), ...])`
adds force and energy constraints at selected points inside intervals. Interval
indices start at zero; `tau` is the local time fraction from 0 to 1. States use
the collocation polynomial and controls use linear interpolation. Select these
points from a path audit for the specific route, mesh, and objective, then audit
the new solution again. The optimizer does not select them automatically, and
these additional samples do not guarantee feasibility everywhere between them.

For complete flights with ordered waypoints, the departure and arrival headings
follow the first and last nonzero route legs in the optimizer projection. Without
waypoints, the direct origin-to-destination heading remains the endpoint default.
Waypoint disks use dimensionless residuals while `waypoint_tolerance_m` remains
in metres. On dense or uneven route meshes, set `dt_min` and `dt_max` explicitly;
the automatic time bounds assume similarly sized intervals and can restrict
short segments too much.

All phase optimizers accept `trajectory(waypoint_min_spacing_m=20_000, ...)`.
Before initialization and problem construction, this option retains the first
waypoint at least 20 km from the previous retained point, starting at departure.
It also skips trailing waypoints within 20 km of arrival. Distances use the
optimizer projection. Departure and arrival remain fixed. The mesh and retained
explicit node assignments stay unchanged. The default is zero, which retains all
waypoints. `active_waypoints`, `active_waypoint_node_indices`, `skipped_waypoints`,
and `retained_waypoint_positions` expose the filter result; positions refer to
zero-based entries in the supplied waypoint list. Automatic node assignments
are still computed during initialization when explicit indices are absent.
Thinning relaxes route constraints and can move the solution outside the original
route corridor. Use it only for points that can be skipped, then check the route
against any required corridor or RAD restrictions.

This replaces the previous piecewise-constant control transcription. Saved
trajectory DataFrames retain their columns and `N+1` rows. The numeric export
helper still accepts historical `N`-column control arrays for replaying old data.
Continuity does not imply continuous control derivatives, a certified continuous
path envelope, or a complete acceleration-energy model.

## Quick Start

### A simple optimal flight

```python
import opentop

optimizer = opentop.CompleteFlight("A320", "EHAM", "LGAV", m0=0.85)
flight = optimizer.trajectory(objective="fuel")
```

`m0` accepts kilograms or an MTOW fraction in `(0, 1]`. For example,
`m0=60000` supplies 60,000 kg, and `m0=1` supplies full MTOW. Inputs above
MTOW raise `ValueError`.

`flight` is a Pandas DataFrame with columns for position, altitude, mass,
Mach, TAS, vertical rate, heading, and per-segment `fuel_cost`.

![example_optimal_flight](./docs/_static/flight_dataframe.png)

### Other built-in objectives

```python
optimizer.trajectory(objective="time")      # minimum time
optimizer.trajectory(objective="ci:30")     # cost index 30
optimizer.trajectory(objective="gwp100")    # 100-yr global warming potential
optimizer.trajectory(objective="gtp100")    # 100-yr global temperature potential
```

The supported climate metrics are `gwp20`, `gwp50`, `gwp100`, `gtp20`, `gtp50`, `gtp100`.

### Choosing a different engine

```python
optimizer = opentop.CompleteFlight(
    "A320", "EHAM", "LGAV", m0=0.85, engine="CFM56-5B4"
)
```

### Flight phase optimizers

```python
cruise    = opentop.Cruise("A320", "EHAM", "LGAV", m0=0.85).trajectory()
climb     = opentop.Climb("A320", "EHAM", "LGAV", m0=0.85).trajectory()
descent   = opentop.Descent("A320", "EHAM", "LGAV", m0=0.85).trajectory()
```

`Cruise` also supports constant-altitude, constant-Mach, and fixed-track modes:

```python
opt = opentop.Cruise("A320", "EHAM", "LGAV", m0=0.85)
opt.fix_cruise_altitude()
opt.fix_mach_number()
opt.fix_track_angle()
flight = opt.trajectory()
```

To hold the cruise at a specific Mach number instead, pass `mach_value`:

```python
flight = opentop.Cruise("A320", "EHAM", "LGAV", m0=0.85, mach_value=0.78).trajectory()
```

To follow a recorded cruise ground track while optimizing altitude and speed:

```python
opt = opentop.Cruise("A320", "EHAM", "EDDF", m0=0.85, mach_value=0.78)
opt.follow_track(recorded.latitude, recorded.longitude, tolerance_m=1000)
flight = opt.trajectory()
```

The reference is a cubic spline through at least four finite latitude/longitude
points, with no consecutive duplicates. Its endpoints must be within 1 km of the
flight endpoints and are snapped to them. Boundary nodes lie on the spline;
interior collocation and quarter-interval samples stay within `tolerance_m`
(default 1 km). This is a sampled constraint, not a continuous-path guarantee.
Sharper bends or tighter tolerances may require more nodes via `opt.setup(nodes=...)`.
Turn-rate limits remain active, so an infeasible recorded turn can prevent convergence.
Check `opt.success` after solving. `follow_track` cannot be combined with
`fix_track_angle`; fixed Mach, wind, and fleet optimization remain supported
(subject to the existing fleet restrictions on wind and time-dependent grids).

### Optimizer-selected step climbs

`Cruise` and `CompleteFlight` support continuous cruise-climb pulses for the
built-in fuel objective:

```python
opt = opentop.CompleteFlight("A320", "EHAM", "LGAV", m0=70_000)
opt.setup(nodes=40)
result = opt.trajectory(step_climb=True, result_object=True)
if not result.success:
    raise RuntimeError(result.status)
flight = result.df
```

The [step-climb notebook](examples/step_climb.ipynb) compares altitude and
vertical rate with step climbs enabled and disabled for both a full flight
and a cruise-only flight. It reports the runtime of each optimization.

The optimizer chooses pulse rates, interval durations, and continuous altitudes
in one smooth NLP. Rates range from zero to 1,500 ft/min: there is no mandatory
500 ft/min rate, minimum step height, or climb penalty. Zero amplitude removes
a pulse. The mesh and initialization choose a basis of up to three optional
pulses; this is not a global search over every step count or node allocation.
Inactive slots still reserve ramp and dwell intervals. Flight levels are not
rounded to ATC allocations. Initial climb and final descent keep their original
phase constraints.

`StepClimbOptions` controls the maximum rate, maximum number of slots (the basis
currently supports at most three), level dwell (default 300 s), and ramp duration
(default at most 300 s). Acceleration remains limited to 5 ft/min/s. If no pulse
fits, the basis is level-only. Continuous steps require variable timesteps;
explicit `variable_timestep=False` raises an error. Default interval bounds adapt
to route length and mesh density: `dt_min=min(30, 0.25*nominal_duration/nodes)`
with a small positive floor, and `dt_max=max(900, 2*nominal_duration/nodes)`.
Explicit user bounds are retained. These are initialization heuristics, not
prescribed switching times.

`objective_value` reports physical fuel; `step_climb_penalty` is zero for the
continuous method. `step_climb_stats` records solve attempts, and
`step_climb_diagnostics` reports amplitudes, level/ramp durations, shape checks,
physical audits, and total solve count. `opt.stats` retains the final solve's
iteration count; validation diagnostics separately aggregate all attempts.

The previous sparse-ascent candidate search remains available explicitly:

```python
legacy = opentop.StepClimbOptions(
    method="legacy", min_climb_rate_fpm=500, penalty_weight=0.01
)
```

Legacy search can take many solves and retains its candidate-specific repair
process. Its selected result receives a final strict audit; it is not repaired
under a different candidate's mode parameters. Historical 8--16 minute search
results refer to this older method, not the continuous default.

### Physical validation and initialization

Single-aircraft `CompleteFlight`, `Cruise`, `Climb`, and `Descent` validate their
configured physical inequalities on 161 samples per interval. Missed peaks add
constraints followed by at most two repair solves. The default therefore permits
one initial solve and at most two repairs. Check `result.success` or `opt.success`:
an exhausted repair budget reports failure even if IPOPT itself converged.
`return_failed=True` returns a diagnostic trajectory without changing that status.

```python
checks = opentop.PhysicalValidationOptions(max_repairs=2)
result = opt.trajectory(physical_validation=checks, result_object=True)
```

Force residuals are normalized by available thrust. BADA4 fuel-domain residuals
retain kg/s units. Acceptance tolerance is `1e-7` in each reported unit; repair
adds a `1e-4` dimensionless force buffer, never a kg/s fuel buffer. Climb preserves
its interval-average acceleration/energy model. Ordinary Cruise retains its
existing force guards; it does not silently acquire the step-cruise energy guard.
Finite sampling is not a continuous-path certificate. The coupled `MultiAircraft`
solver is outside this single-aircraft audit controller. Explicit
`physical_validation=False` disables validation and is recorded as disabled.
Climb/Descent preliminary cruise solves only prepare their target/seed and are
counted separately from the requested phase's validation attempts.

Ordinary complete flights retain their previous default seed. Explicit
`initialization="auto"` opts into a bounded-control, phase-based
altitude/rate profile with model fuel integration. A320 comparisons were similar
or faster, but a B789 wind case entered a worse local basin, so this broader
heuristic is not the ordinary default. It does not guarantee a feasible seed.
Continuous step climbs use their matching pulse seed by default, without an
extra optimization. `initialization="legacy"` forces the older seed; supplied
`initial_guess` data takes precedence over either policy.

### Multiple-aircraft separation optimization

`MultiAircraft` formulates several cruise trajectories in one CasADi Opti
problem and applies the default 5 NM / 1,000 ft high-order separation model:

```python
import opentop

flights = [
    opentop.FlightSpec(
        "AC1", opentop.Cruise("A320", (52.3, 9.8), (49.5, 2.5)), start_time=0
    ),
    opentop.FlightSpec(
        "AC2", opentop.Cruise("B738", (52.9, 4.6), (49.5, 9.3)), start_time=140
    ),
]

result = opentop.MultiAircraft(flights).trajectory()
df_ac1 = result.trajectories["AC1"]
```

The default constraint is
`(dx / Rxy)^2 + (dy / Rxy)^2 + (dh / Rz)^8 >= 1.3`. Pass a
`SeparationConfig` only to customize it. Fleet DataFrames preserve relative
`ts` and add `absolute_ts` using each flight's configured start time.

To study horizontal conflict avoidance at one shared flight level, first call
`fix_cruise_altitude()` on every child optimizer and then pass
`common_altitude=True`. The common altitude remains an optimization variable:

```python
for flight in flights:
    flight.optimizer.fix_cruise_altitude()

result = opentop.MultiAircraft(flights, common_altitude=True).trajectory()
```

`Descent` optimizers can also be coordinated. List flights in the desired
landing order and use `minimum_arrival_gap_s` to enforce a runway sequence.
Shared final-approach points can be supplied with ordered waypoint constraints.
For structured arrivals, `route_heading_tolerance_deg` keeps each route leg
near its nominal bearing, while `max_duration_s` can impose an operational
descent-time bound without adding time to the objective.
See the
[coordinated-arrivals notebook](examples/multi_aircraft_arrivals.ipynb) for a
three-aircraft runway-arrival example.

The implementation supports `Cruise` and `Descent` optimization, including
ordered waypoint constraints. Timesteps are uniform by default; set
`"variable_timestep": True` (optionally with `dt_min`/`dt_max`) in a flight's
options to optimize its interval durations in both the warm start and the
joint problem. Separation constraints then use each interval's actual start
time and duration. Wind, time-dependent 4D grid objectives, and other phases
are rejected explicitly until their fleet timing semantics are implemented.

### Wind integration

Download ERA5 (or similar) meteorological data in GRIB format, then:

```python
import opentop

windfield = opentop.tools.read_grids("wind.grib")

optimizer = opentop.CompleteFlight("A320", "EHAM", "LGAV", m0=0.85)
optimizer.enable_wind(windfield)

flight = optimizer.trajectory(objective="fuel")
opentop.vis.trajectory(flight, windfield=windfield, barb_steps=15)
```

OpenAP fuel optimization with the built-in wind model automatically expands the
CasADi graph before solving. This reduces derivative-evaluation cost without
changing the equations. The automatic setting excludes custom objectives, grid
costs, reference tracks, external fuel models and other performance models.
Use `optimizer.setup(expand=False)` to disable expansion, or `expand=True` to
request it explicitly. Other optimizations retain their existing default.

![example_optimal_flight](./docs/_static/optimal_flight_complete_example.png)

### Custom grid cost (contrails, weather, airspace)

Build a CasADi interpolant from a DataFrame with columns `longitude`, `latitude`, `height` (m), `cost`, and optionally `ts`:

```python
interpolant = opentop.tools.interpolant_from_dataframe(df_cost)

def contrail_objective(x, u, dt, **kwargs):
    grid_cost = optimizer.obj_grid_cost(
        x, u, dt, interpolant=kwargs["interpolant"], n_dim=3
    )
    fuel_cost = optimizer.obj_fuel(x, u, dt)
    return grid_cost + fuel_cost

flight = optimizer.trajectory(
    objective=contrail_objective,
    interpolant=interpolant,
    n_dim=3,
)
```

See <https://openap.dev/optimize/contrails.html> for a full contrail + CO₂ example.

### Custom objective functions

Any callable with signature `(x, u, dt, **kwargs) -> ca.MX` can be used:

```python
def my_objective(x, u, dt, **kwargs):
    # x: state [xp, yp, h, mass, ts]
    # u: control [mach, vs, psi]
    return your_cost_expression

flight = optimizer.trajectory(objective=my_objective)
```

### Multi-start optimization for non-convex objectives

Trajectory optimization with grid costs, blended objectives, or tight constraints can have multiple local minima — a single solve lands in whichever basin is closest to the initial guess. For problems where robustness matters, `multi_start_trajectory` runs N solves from different randomized initial guesses and returns the best:

```python
trajectory, candidates = optimizer.multi_start_trajectory(
    objective=contrail_objective,
    interpolant=interp,
    max_fuel=6500,
    n_starts=5,
    lateral_jitter_km=100.0,
    altitude_jitter_ft=3000.0,
    seed=0,
)
```

Start 0 uses your `initial_guess=` if provided, otherwise the default great-circle guess. Subsequent starts perturb your supplied guess, or start 0's returned trajectory if no guess was supplied: sinusoidal lateral bulges (endpoints preserved), constant altitude offsets, and optional additive Mach offsets. Set `mach_jitter=0.04`, for example, to draw offsets uniformly within ±0.04 Mach. It defaults to zero. A supplied guess needs a `mach` column when speed jitter is enabled; the optimizer clips the Mach guesses to each node's control bounds. These perturbations change only initialization, not the constraints or fuel model.

Returned is a `(trajectory, candidates)` tuple. `trajectory` is the winning DataFrame (feasibility-first, then lowest objective). `candidates` is a best-first ordered list of dicts with per-start metadata (`objective`, `fuel`, `grid_cost`, `success`, `iters`, `perturbation`, `wall_time_s`, `trajectory`).

`n_starts=1` is identical to calling `trajectory()` directly, so adding `multi_start_trajectory` to existing code is additive.

### Precomputed grid caches (recommended for contrail + CO₂)

Linear interpolation over a 4D contrail-cost grid has discontinuous derivatives at every grid cell boundary, which can cause IPOPT's line search to oscillate on non-convex blended objectives. The fix is to use a cubic B-spline interpolant, which has continuous derivatives. But building a bspline over a large grid can take several minutes, so we expose a cache utility:

```python
from opentop.tools import cached_interpolant_from_dataframe

interpolant = cached_interpolant_from_dataframe(
    df_cost, "cache/contrail.casadi", shape="bspline"
)
```

First call builds the bspline and writes it to disk (~1-3 minutes for a 60k-point slice); subsequent calls load the cache in under a second.

If your grid only covers the altitude band where contrails actually form (typically FL200-FL440), extend it with zero-cost levels outside that band before building the interpolant — otherwise `opentop.CompleteFlight` trajectories that start and end on the ground will query the interpolant outside its data range. The `opentop` CLI has a helper for this:

```sh
opentop gengrid --in raw_grid.parquet --out grid.casadi \
    --bbox 35:57,-9:7 --time 2022-02-20T10:00,2022-02-20T14:00 \
    --pad-altitudes
```

`--pad-altitudes` is on by default and adds zero-cost rows at altitudes from 0 to FL480 so the interpolant returns 0 (physically correct — no contrails below ~FL200 or above ~FL440) outside the data band.

## Command-line interface

Installing `opentop` also installs the `opentop` executable, which exposes two subcommands: `optimize` (run a trajectory optimization) and `gengrid` (precompute a grid-cost interpolant).

### `opentop optimize`

Run a trajectory optimization without writing any Python:

```sh
opentop optimize EHAM EDDF -a A320 --phase cruise --obj fuel
```

A concise solver summary (status, iterations, objective, fuel burn, max altitude, flight time) is printed to stdout. Pass `-o flight.parquet` to also save the full trajectory DataFrame.

Supported objectives: `fuel`, `time`, `ci:N` (cost index, any integer), `gwp20`/`gwp50`/`gwp100`, `gtp20`/`gtp50`/`gtp100`, and `grid` (requires `--grid FILE`).

Blended objectives are written as a weighted sum:

```sh
opentop optimize EHAM EDDF -a A320 --phase all \
    --obj "0.3*fuel+0.7*grid" \
    --grid contrail.casadi
```

Common flags:

| flag | purpose |
|---|---|
| `-a`, `--aircraft` | aircraft type (required), e.g. `A320` |
| `--phase` | `all` (CompleteFlight, default), `cruise`, `climb`, `descent` |
| `--obj` | objective expression — single term or weighted sum |
| `--m0` | initial mass in kg or an MTOW fraction in `(0, 1]` (default `0.85`) |
| `--grid` | cost-grid file; `.casadi` cache preferred, `.parquet` accepted with a slow-path warning |
| `--max-iter` | IPOPT iteration cap (default `1500`) |
| `-o`, `--output` | write the trajectory DataFrame to a parquet file |
| `-v`, `--debug` | verbose IPOPT output |

### `opentop gengrid`

Build and cache a CasADi interpolant from a raw cost grid:

```sh
opentop gengrid --in raw_grid.parquet --out contrail.casadi \
    --bbox 35:57,-9:7 \
    --time 2022-02-20T10:00,2022-02-20T14:00 \
    --shape bspline
```

The resulting `.casadi` file loads in under a second (vs. minutes to rebuild a bspline from raw grid data), so keep it on disk and pass it to `opentop optimize --grid`.

Use `opentop --help`, `opentop optimize --help`, and `opentop gengrid --help` for the full option list.

### `opentop replay`

Replay a real flight by callsign: pulls trajectory from OpenSky, ERA5 meteo
from fastmeteo, and runs an optimization side-by-side with the actual flight.

```sh
# Install with the replay extras
pip install "opentop[replay]"

# Flag mode
opentop replay RYR880W --date 2023-01-05 --obj fuel -o ./results/

# Interactive wizard (no arguments)
opentop replay
```

Outputs under the output directory:
- `actual.parquet` — cleaned OpenSky trace
- `optimized.parquet` — optimized trajectory
- `trajectory.png` — actual-vs-optimized overlay

Requires OpenSky credentials at `~/.config/traffic/traffic.conf` (see
traffic's docs). ERA5 data is fetched via fastmeteo's ArcoEra5; the default
local store is `<system tmp>/opentop-era5` — first use will download several
GB on a large bbox.

## Accessing Solver Results

After calling `.trajectory()`, the optimizer exposes:

```python
optimizer.stats           # solver statistics dict ("success", "iter_count", ...)
optimizer.success         # True if the most recent solve succeeded
optimizer.objective_value # final objective value (float)
```

For full structured access including status, iteration count, fuel, and grid cost in one dataclass:

```python
result = optimizer.trajectory(objective="fuel", result_object=True)
# result.df, result.success, result.status, result.objective,
# result.iters, result.fuel, result.grid_cost, result.stats
```

## Benchmarks

Run benchmarks across versions to verify performance:

```sh
./benchmark.sh                 # Benchmark HEAD (local dev code)
./benchmark.sh v2.0.0          # Benchmark a specific PyPI release
./benchmark.sh v1.11.0 v2.0.0  # Benchmark multiple versions sequentially
```

Reports land in `tests/benchmarks/<version>.txt`.

## Migrating to 2.2

Version 2.2 makes several structural changes that may require small updates to existing code:

| v2.1 | v2.2 |
|---|---|
| `opentop.MultiPhase(...)` | `opentop.CompleteFlight(...)` — `MultiPhase` has been removed |
| `trajectory(objective="fuel", foo=bar)` silently tolerates unknown kwargs | Unknown kwargs raise `TypeError`; only documented names are accepted |
| `optimizer.solver.stats()` | `optimizer.stats` (dict) or `optimizer.success` (bool) |
| `opentop.vis.map(df, ...)` | `opentop.vis.plot_map(df, ...)` |
| — | New: `trajectory(..., result_object=True)` returns a `TrajectoryResult` dataclass |
| — | New: `opentop replay CALLSIGN` CLI — fetches real flight from OpenSky, runs comparison optimization (see `### opentop replay` above) |
| — | New: `opentop.vis.trajectory(df | [df1, df2])` accepts a list for overlay plotting |

Internally, `opentop/` has been split into focused modules: `_dynamics.py`, `_objectives.py`, `_trajectory.py`, `_options.py`, `_multi_start.py`. If you were importing internal helpers, they have moved — use the public API on `Base` / `Cruise` / `CompleteFlight` where possible.

`MultiPhase` was rarely used and its functionality is fully covered by `CompleteFlight`. If you had code using `MultiPhase`, replace it with:

```python
full = opentop.CompleteFlight("A320", "EHAM", "LGAV", m0=0.85).trajectory()
```

Type annotations are now enforced in CI via pyright (basic mode). Public API signatures are fully annotated; see `opentop/_options.py` for the `TrajectoryResult` dataclass.

## What's New in 2.0

Version 2.0 is a major refactor. Most user code keeps the same shape, but a few things have moved:

| v1.x | v2.0 |
|---|---|
| `pip install openap-top` | `pip install opentop` |
| `from openap import top` | `import opentop` |
| `top.Cruise(...)` | `opentop.Cruise(...)` |
| `optimizer.change_engine()` dropped | `opentop.Cruise(..., engine="CFM56-5B4")` |
| `optimizer.solution["f"]` | `optimizer.objective_value` |
| `optimizer.solver` was a `ca.nlpsol` callable | use `optimizer.stats` and `optimizer.success` |
| `setup(max_iteration=...)` | `setup(max_iter=...)` |
| — | new CLI: `opentop optimize ORIGIN DEST ...` and `opentop gengrid ...` |
| — | new `opentop.tools.cached_interpolant_from_dataframe()` for disk-cached bspline interpolants |

The NLP construction moved to CasADi's Opti stack, which removed ~400 lines of boilerplate and cleaned up several bugs. The module rename from `openap.top` to `opentop` eliminates the namespace-extension install mode that used to require `.pth` tricks.

See the [changelog](https://github.com/junzis/opentop/releases) for details.

## License

MIT License. See [LICENSE](LICENSE) for details.

### Optional neural fuel model

The Python optimizers `Cruise`, `CompleteFlight`, `Climb`, and `Descent` accept
`fuel_model=` alongside `performance_model=`. The performance model supplies
forces and default fuel flow; a fuel override replaces only fuel flow throughout
the optimization and reporting. Omitting it preserves the default behavior.

Install `opentop[neural]` and supply your Acropole model/parameter files:

```python
fuel = top.AcropoleFuelFlow(
    "A320",
    model_path="acropole_fuel_model.onnx",
    aircraft_params_path="aircraft_params.csv",
    activation_smoothing=0.01,
)
optimizer = top.CompleteFlight(
    "A320", "LFPG", "EDDM", performance_model="openap", fuel_model=fuel,
)
```

Neural optimization currently requires still air and uses zero speed-derivative
features. Smoothing changes the released network and requires validation.
`engine=` does not change Acropole's training parameters or an independently
constructed physics correction. See [the neural fuel notes](docs/design/neural_fuel.md)
for correction configuration and limitations.
