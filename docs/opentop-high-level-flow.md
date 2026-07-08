# OpenTOP High-Level Flow

This diagram summarizes the main execution paths in OpenTOP: the Python API,
the CLI, optional grid/wind/replay inputs, NLP construction, solving, and
trajectory reporting.

```mermaid
flowchart TD
    user[User code or CLI] --> entry{Entry point}

    entry -->|Python API| api[opentop.CompleteFlight / Cruise / Climb / Descent]
    entry -->|opentop optimize| cli_opt[CLI optimize command]
    entry -->|opentop replay| cli_replay[CLI replay command]
    entry -->|opentop gengrid| cli_grid[CLI gengrid command]

    cli_opt --> phase_select[Select phase class]
    cli_opt --> objective_parse[Parse objective expression]
    cli_opt --> grid_load[Load optional grid interpolant]
    phase_select --> api
    objective_parse --> objective_input[Objective spec or callable]
    grid_load --> optional_inputs

    cli_grid --> raw_grid[Raw parquet cost grid]
    raw_grid --> grid_filter[Optional bbox/time filtering and altitude padding]
    grid_filter --> build_grid[Build CasADi interpolant]
    build_grid --> cache_grid[Save .casadi cache]
    cache_grid --> grid_load

    cli_replay --> flight_fetch[Fetch/load actual flight]
    cli_replay --> meteo_fetch[Optional ERA5 meteo]
    flight_fetch --> replay_endpoints[Infer endpoints and aircraft]
    meteo_fetch --> wind_input[Wind DataFrame]
    meteo_fetch --> contrail_grid[Contrail cost grid]
    contrail_grid --> build_grid
    replay_endpoints --> api
    wind_input --> optional_inputs

    api --> base_init[Base.__init__]
    base_init --> airport_projection[Resolve airports and projection]
    base_init --> performance[Build OpenAP/BADA models]
    base_init --> setup[setup: nodes, collocation degree, IPOPT options]

    api --> trajectory[trajectory]
    optional_inputs[Optional inputs: wind, grid, initial guess, payload, max fuel] --> trajectory
    objective_input --> trajectory

    trajectory --> init_conditions[Phase init_conditions]
    init_conditions --> bounds[State/control bounds and guesses]

    bounds --> build_opti[Base._build_opti]
    build_opti --> init_model[init_model: states, controls, objective, dynamics]
    init_model --> dynamics[_dynamics.xdot]
    init_model --> objectives[_objectives resolver and objective functions]
    build_opti --> collocation[Legendre direct-collocation variables and equations]
    collocation --> shared_constraints[Shared state/control bounds and continuity]

    shared_constraints --> phase_constraints{Phase constraints}
    phase_constraints --> complete[CompleteFlight: climb/cruise/descent shaping, energy, smoothing, fuel]
    phase_constraints --> cruise[Cruise: cruise performance, optional fixed Mach/alt/track]
    phase_constraints --> climb[Climb: preliminary cruise context, energy, range, climb filtering]
    phase_constraints --> descent[Descent: preliminary cruise context, energy, descent filtering]

    complete --> solve[Base._solve]
    cruise --> solve
    climb --> solve
    descent --> solve

    solve --> ipopt[CasADi Opti + IPOPT]
    ipopt --> numeric_solution[Optimized X/U, final time, solver stats]
    numeric_solution --> dataframe[_trajectory.to_dataframe]
    dataframe --> traj_df[Trajectory DataFrame]
    numeric_solution --> result_obj[Optional TrajectoryResult]

    traj_df --> downstream{Downstream use}
    result_obj --> downstream
    downstream --> analysis[User analysis or saved parquet]
    downstream --> vis[opentop.vis trajectory plots]
    downstream --> multistart[_multi_start repeated solves and best candidate]
```

## Main Pipeline

The package API is exported from `opentop/__init__.py`. Users normally create
one of the phase optimizers and call `trajectory()`.

All phase classes inherit from `Base`. The phase method first calls
`init_conditions()` to set state and control bounds, endpoint constraints, and
initial guesses. It then calls `Base._build_opti()`, which creates the CasADi
Opti problem, symbolic state/control variables, free final time, objective
quadrature, collocation equations, and continuity constraints.

After the shared NLP scaffold is built, the phase class adds its own
constraints. `CompleteFlight` adds full-flight shaping and energy constraints,
`Cruise` adds cruise-specific and optional fixed-profile constraints, while
`Climb` and `Descent` can first run a preliminary cruise solve to obtain the
phase boundary altitude and Mach context.

Finally, `Base._solve()` configures IPOPT, solves the NLP, stores solver stats
and the physical objective value, and delegates numeric output conversion to
`_trajectory.to_dataframe()`.

## Data And Objective Inputs

Objectives can be built-in strings such as `fuel`, `time`, `ci:N`, climate
metrics, or `grid_cost`, or they can be user callables. `_objectives.py`
contains the pure objective implementations and registry. Grid-cost objectives
use CasADi interpolants from `tools.py`; bspline grid costs trigger exact
Hessian handling in the optimizer setup.

Wind enters through `Base.enable_wind()`, which fits a `tools.PolyWind` model.
That wind model is used inside `_dynamics.xdot()` so both numeric and symbolic
state derivatives include wind components.

## CLI And Replay

`opentop optimize` is a thin CLI wrapper around the same phase classes. It
parses weighted objective expressions, optionally loads a grid interpolant, runs
`trajectory()`, and prints solver and trajectory summaries.

`opentop gengrid` prepares raw parquet grids by slicing time/space, padding
altitudes, building a CasADi interpolant, and saving a reusable `.casadi` cache.

`opentop replay` fetches or loads an actual flight, optionally fetches ERA5
weather, enables wind, builds a contrail grid interpolant when needed, runs the
optimizer, and saves actual/optimized trajectories plus a plot.
