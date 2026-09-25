# Multi-start trajectory optimization

Current behavior reviewed on 2026-09-25. The wrapper runs sequential local solves
with different initial guesses; it does not make IPOPT a global optimizer.
Even fuel-optimal, still-air trajectory problems need not be convex.

## API

```python
import opentop as top

optimizer = top.Cruise("A320", "LFPG", "EDDM")
best, candidates = optimizer.multi_start_trajectory(
    objective="fuel", n_starts=5, seed=42,
    lateral_jitter_km=100, altitude_jitter_ft=3000, mach_jitter=0.04,
)
```

Defaults are five starts, 100 km lateral jitter, 3,000 ft altitude jitter, zero
Mach jitter, and no fixed random seed. Additional arguments are forwarded to
`trajectory()`. The return is always `(DataFrame, candidates)`, including when
`result_object=True` is supplied.

## Initial guesses

Start 0 uses the supplied `initial_guess`, or the optimizer's default guess.
Later starts perturb the supplied guess when present; otherwise they perturb
start 0's **returned trajectory**, not its original great-circle initialization.

Each later start draws a sinusoidal lateral offset, a constant altitude offset,
and optionally a constant additive Mach offset. The lateral perturbation
preserves horizontal endpoints. Mach offsets are uniform within
`[-mach_jitter, mach_jitter]`. Supplied guesses need a `mach` column when jitter
is enabled. The NLP clips Mach guesses to each node's bounds; jitter changes
initialization, not the constraints or the fuel model.

Mass, elapsed time, vertical-rate controls and heading controls are not randomly
perturbed. With Mach jitter disabled, the established lateral/altitude random
sequence is retained. Use identical settings and a fixed seed for repeatable
perturbations; solver outcomes can still depend on numerical environment.

Standalone `Climb` uses supplied Mach guesses, but still constructs its own
state guess; lateral/altitude perturbations do not initialize its states.
An exception raised by `trajectory()`
is not converted into a candidate by the wrapper and can stop the run.

## Ranking and reporting

Candidates are sorted by solver success first, then objective value;
start index breaks ties. Failed iterates can therefore be returned if no start
succeeds. Check `candidates[0]["success"]` before using the winner.

Each candidate contains `start_index`, `objective`, `fuel`, `grid_cost`,
`grid_cost_exact`, `success`, `return_status`, `iters`, `perturbation`,
`wall_time_s`, and `trajectory`. `perturbation` includes `mach_offset` only when
speed jitter is enabled. `grid_cost` sums the DataFrame column;
`grid_cost_exact` uses the recorded solver quadrature. These can differ.

After selection, `optimizer.objective_value` and `optimizer.grid_cost_value`
describe the winner. The optimizer's solver state and stats still describe the
most recent solve; use the winning candidate's diagnostics instead.

With a fixed seed and otherwise identical settings, adding starts retains the
previous candidates, subject to numerical reproducibility. Multi-start improves
coverage of initial guesses, but gives no certificate of global optimality.
