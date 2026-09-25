# Acropole neural fuel integration and experiment history

Current API contract reviewed 2026-09-25. See the [API guide](../api-guide.md).
The dated results below record earlier experiments, not the current paper
benchmark or a guarantee of convergence. Images under `examples/figures/` are
local generated artifacts; run the adjacent commands to reproduce them.

This branch adds an experimental `AcropoleFuelFlow` adapter and a `fuel_model`
argument to `Cruise`, `CompleteFlight`, `Climb`, and `Descent`. OpenAP (or the selected performance model) still supplies
drag, thrust, and aircraft operating constraints. Acropole supplies fuel flow
in the mass dynamics, fuel objective, numeric objective evaluation, emissions
fuel input, and trajectory `fuelflow` column. Integrated `fuel_cost` remains
derived from mass loss.

The original experiment was **still-air, constant-Mach, constant-altitude cruise**.
Altitude and Mach are chosen by the optimizer but held constant along each
trajectory. The route, initial mass, altitude bounds, mesh, and force models are
identical for OpenAP and Acropole. This makes zero speed derivatives consistent
with the optimized flight, and isolates fuel-model effects.

## Model import

Upstream sources:

- [Acropole](https://github.com/DGAC/Acropole), inspected at
  `37f8d182dc7dce69fb0274576fefa4139bc77411`.
- [Acropole preprocessing](https://github.com/DGAC/Acropole/blob/37f8d182dc7dce69fb0274576fefa4139bc77411/src/acropole/estimator.py).
- [CasADi 3.8 release notes](https://web.casadi.org/get/#notesTab-38).

Acropole already ships a dense ONNX network with four ReLU hidden activations
and a sigmoid output. The tested CasADi 3.8.1 Python wheel imports the affine
layers but rejects `Relu` and `Sigmoid`. The adapter rewrites these operators in
a temporary ONNX file as `Max(x, 0)` and `1 / (1 + Exp(-x))`, fixes the batch
dimension to one, and calls `GraphBuilder.create(..., {"symbolic": True})`.
By default, weights and activations remain unchanged; no retraining, smoothing,
finite-difference callback, or ONNX Runtime dependency is used in the optimizer.
An explicit `activation_smoothing=epsilon` replaces ReLU by
`(x + sqrt(x*x + epsilon*epsilon))/2` for initialization studies. This changes
predictions and is not enabled automatically. Epsilon is in hidden preactivation
units; its effect must be measured for the intended operating conditions. Floating-point evaluation
differs from the original float32 runtime, so numerical parity is checked.

Acropole weights and aircraft CSV are supplied by path, not bundled in opentop.
Their SHA-256 hashes are recorded by the experiment. Acropole's repository
declares AGPL-3.0; retain its license and provenance with those external assets.

## Feature contract and limitations

The 12 features, in order, are engine type, altitude derivative (ft/s), ground
and air speed derivatives (knots/s), wing area, maximum operating altitude and
speed, altitude (ft), groundspeed and TAS (knots), vertical rate (ft/min), and
mass normalized between Acropole's OEW and MTOW. Feature min/max normalization
and output scaling by takeoff fuel flow times engine count follow upstream.

- The adapter uses the supplied mass, not Acropole's unknown-mass sentinel.
- Altitude derivative is vertical rate / 60. Speed derivatives are zero,
  matching Acropole's no-timestamp defaults. In changing-speed/altitude flight
  this is an approximation; acceleration-aware optimization is future work.
- Still-air groundspeed is the horizontal projection of TAS, using opentop's
  flight-path angle convention. Wind is rejected rather than silently using
  TAS as wind-affected groundspeed.
- Temperature is represented only through Mach-to-TAS conversion; Acropole
  has no separate temperature input. Engine overrides affect the force model,
  not the aircraft-specific Acropole output scale.
- ReLU kinks remain. Automatic gradients and Hessians are valid within smooth
  regions; this does not make the network globally twice differentiable.
  Compare exact and limited-memory IPOPT Hessians experimentally.
- Normalization bounds are not evidence of training-data coverage. No training
  domain is inferred or enforced by the adapter. The example's 30,000--37,000 ft
  cruise envelope is an explicit experiment choice, not a validated domain.
  Research use needs validation on the aircraft, mass, speed, and altitude range.
- Python `Cruise`, `CompleteFlight`, `Climb`, and `Descent` accept `fuel_model`.
  Climb/descent pass the same override to their preliminary cruise optimizer.
  Fleet specs carry an already configured Cruise or Descent optimizer; the CLI
  does not expose the option.
  All phases retain the zero speed-derivative feature approximation.

## Run

The core adapter supports opentop's Python >=3.11 requirement. The validation
script additionally imports current Acropole source, which requires Python
>=3.12, plus its numpy/polars/ONNX Runtime dependencies.

```sh
git clone https://github.com/DGAC/Acropole.git /tmp/Acropole
git -C /tmp/Acropole checkout 37f8d182dc7dce69fb0274576fefa4139bc77411
uv sync --extra neural
uv run --extra neural --with onnxruntime --with polars \
  python examples/neural_fuel.py \
  --acropole-dir /tmp/Acropole --output-dir /tmp/neural-fuel-exact
```

Repeat with `--hessian limited-memory`, and vary `--nodes` for mesh sensitivity.
The script checks 1,000 deterministic samples against upstream's
`AircraftFuelEstimator`, finite-difference gradients and Hessians at a point
away from ReLU kinks, and then solves the two cruise problems. It writes
`report.json`, `openap.csv`, and `acropole.csv`, with versions, artifact hashes,
upstream commit, solver status, iterations, and elapsed build/solve time.
It exits unsuccessfully if validation or either solve fails.

Minimal integration:

```python
import opentop as top
from openap.aero import ft

fuel = top.AcropoleFuelFlow(
    "A320",
    model_path="/tmp/Acropole/src/acropole/models/acropole_fuel_model.onnx",
    aircraft_params_path="/tmp/Acropole/src/acropole/data/aircraft_params.csv",
)
optimizer = top.Cruise(
    "A320", "EHAM", "EDDF", fuel_model=fuel,
    h_min=30_000 * ft, h_max=37_000 * ft,
)
optimizer.fix_cruise_altitude()
optimizer.fix_mach_number()
result = optimizer.trajectory(objective="fuel", result_object=True)
assert result.success, result.status
```

## Conference-paper direction

The useful research question is whether a QAR-trained fuel model gives reliable
and useful trajectory decisions when differentiated inside an optimal-control
problem. Prediction accuracy alone does not answer that question.

1. Establish forward and derivative correctness, including boundary/kink cases.
2. Compare OpenAP, original Acropole/ReLU, and a separately identified smooth
   approximation or retrained smooth network. Report any approximation error.
3. Use identical constraints, route, mass, and environmental assumptions. Score
   each optimized trajectory under every model, with consistent mass replay,
   and evaluate against independent data where available.
4. Measure convergence/failure rates, time, mesh sensitivity, and multiple
   initial guesses across routes and aircraft/mass conditions. A single local
   solver result does not establish global optimality.
5. Inspect extrapolation and optimizer exploitation of model errors. Validate
   an operating envelope before extending to acceleration-aware climb/descent
   and wind-aware trajectories.

Fuel totals computed under different models are **not measured fuel savings**.
This first experiment demonstrates integration feasibility and supplies a
reproducible starting point, not a paper-ready performance claim.

## Initial observed results (2026-09-18)

The checked-in [smoke report](../../examples/data/neural_fuel_smoke.json) records
CasADi 3.8.1, OpenAP 2.6.0, the above Acropole revision, and the exact model and
CSV hashes. On 1,000 sampled conditions the maximum absolute difference from
upstream Acropole was **5.03e-6 kg/s**, maximum relative difference **6.96e-6**.
At the derivative-check point, maximum gradient/Hessian errors in scaled
coordinates were below **1.8e-11**. These are implementation checks, not
prediction-accuracy measurements against flight data.

The 12-interval, degree-3, EHAM--EDDF level-cruise experiment produced:

| Fuel model | Chosen altitude (ft) | Mach | Model-predicted fuel (kg) | Exact-Hessian iterations | Build + solve (s) |
|---|---:|---:|---:|---:|---:|
| OpenAP | 33,584 | 0.778730 | 1,211.18 | 19 | 2.58 |
| Acropole | 32,055 | 0.820000 | 1,076.62 | 23 | 3.04 |

Both returned `Solve_Succeeded`. Limited-memory Hessians converged to the same
reported altitude/Mach and fuel totals (18/27 iterations; 1.54/1.79 seconds).
Timings are single-run local observations and exclude neural import (~0.22 s),
not controlled performance benchmarks. Acropole selects the maximum allowed
Mach, making speed sensitivity and independent trajectory evaluation a useful
next investigation.

Validation performed: 29 targeted tests covering the new adapter, cruise,
performance setup, fuel accounting, and climate objectives; Ruff and Pyright.
The full solver-heavy suite was not run. Offline tests generate their own
small ONNX network; the real Acropole model is only used by the explicit
experiment script.

## BADA 4 comparison

To isolate the fuel model, use BADA 4 drag and thrust constraints in **both**
runs, with BADA 4 fuel flow in the baseline and Acropole fuel flow in the
comparison. The adapter matches Acropole's `A320` typecode to the BADA
`A320-214` variant; it still uses Acropole's aircraft-level parameter row.

```sh
uv run --extra neural --with onnxruntime --with polars \
  python examples/neural_fuel.py \
  --acropole-dir /tmp/Acropole --output-dir /tmp/neural-bada4 \
  --performance-model bada4 --bada-path /path/to/bada_4.2/tables
uv run python examples/plot_neural_fuel.py \
  --results-dir /tmp/neural-bada4 --output-dir examples/figures
```

The plot is a 150 dpi PNG. BADA model files remain external.
The [recorded run](../../examples/data/neural_fuel_bada4_smoke.json) used the
same route, mass, mesh, altitude bounds, and exact Hessians as the previous
experiment. Both solves returned `Solve_Succeeded`:

| Fuel model (BADA 4 forces in both) | Altitude (ft) | Mach | Duration (min) | Own-model fuel (kg) | Iterations |
|---|---:|---:|---:|---:|---:|
| BADA 4 | 33,198 | 0.736518 | 27.73 | 1,035.84 | 21 |
| Acropole | 32,012 | 0.787374 | 25.80 | 1,120.77 | 25 |

Acropole chooses a faster, lower cruise than BADA 4. Its choice also differs
from the earlier Acropole run with OpenAP forces (Mach 0.82), demonstrating
that the surrounding performance constraints matter. These are single local
solver results; the own-model totals cannot establish which trajectory burns
less fuel in reality. Cross-evaluation with mass replay, multi-start checks,
and independent flight data remain necessary for a research conclusion.

## High-Mach diagnosis

Reproduce the fixed-state sweeps, feasible-region plots, upstream parity checks,
and eight seeded solves with:

```sh
uv run --extra neural --with onnxruntime --with polars \
  python examples/debug_neural_fuel.py \
  --acropole-dir /tmp/Acropole --bada-path /path/to/bada_4.2/tables \
  --output-dir examples/figures/neural_debug --multistart
```

All plots are 150 dpi PNGs. The script uses the actual symbolic force and fuel
models used by the optimizer, and records raw sweeps and constraint slacks.

### What the speed sweep shows

At 66,300 kg and 32,000 ft, in steady level flight:

| Model | Fuel flow at M 0.78 (kg/s) | Fuel flow at M 0.82 (kg/s) | Fuel/distance at M 0.78 (kg/km) | Fuel/distance at M 0.82 (kg/km) |
|---|---:|---:|---:|---:|
| Acropole | 0.72786 | 0.72867 | 3.10498 | 2.95678 |
| OpenAP, wave drag on | 0.79278 | 0.84938 | 3.38192 | 3.44660 |
| OpenAP, wave drag off | 0.79017 | 0.82306 | 3.37078 | 3.33979 |
| BADA 4 | 0.70085 | 0.85542 | 2.98974 | 3.47112 |

Acropole's predicted fuel flow increases by only about 0.11% while speed rises
by about 5.13%. Its fuel per kilometre falls by about 4.77%, giving the optimizer
an incentive to keep increasing speed. OpenAP's wave-drag ablation confirms
that its explicit wave penalty reverses the fuel-per-distance trend near the
upper speed limit. BADA 4 has a much stronger high-speed rise in this slice.

This is evidence of a weak high-speed fuel penalty in Acropole's quasi-steady
response, consistent with the wave-drag hypothesis. It does not establish that
wave drag is absent from Acropole's training data or isolate wave drag as the
only cause. These diagnostics use the original Acropole model. Optional
physics corrections are described below; no retraining has been applied.

Local generated figure: `examples/figures/neural_debug/mach_diagnostic.png` (Mach diagnostic).

### Which constraints select the neural solutions?

Both neural solutions touch the extra lift-margin constraint
`0.90 * available_thrust >= lift_margin_drag`, where lift coefficient is
evaluated with the existing 0.8 margin. The ordinary drag/thrust constraint
remains slack:

| Force model | Neural solution | Minimum ordinary drag slack | Minimum lift-margin slack | Mach-limit slack |
|---|---|---:|---:|---:|
| OpenAP | M 0.8200, 32,055 ft | 6.95 kN | approximately 0 | approximately 0 |
| BADA 4 | M 0.7874, 32,012 ft | 10.00 kN | approximately 0 | 0.0326 |

Slacks were evaluated at every boundary and collocation point using unrounded
solver values. The BADA 4 result is therefore shaped by the **additional
lift-margin constraint**, not by ordinary drag reaching available thrust.
The fixed-mass feasible-region maps independently locate nearby solutions.
These margins and their physical justification deserve sensitivity analysis
before drawing conclusions for a paper.

Local generated figure: `examples/figures/neural_debug/feasible_landscape.png` (Feasible regions).

### Conversion and local-minimum checks

- Another 363 points at M 0.78--0.84 and 32,000/33,500/35,000 ft match the
  original Acropole runtime within 1e-6 kg/s absolute error. The high-speed
  response is present in the original network, not introduced by the importer.
- Four Mach/altitude seeds per force model all returned `Solve_Succeeded`.
  OpenAP-force runs agree within 0.00001 kg of fuel and BADA-force runs within
  0.000002 kg. This reduces concern about local trapping for this experiment,
  without proving a global optimum.

Local generated figure: `examples/figures/neural_debug/multistart.png` (Seed sensitivity).

### Reporting bug found and fixed

The numeric OpenAP fuel model used by trajectory reporting and numeric
objective evaluation previously defaulted to `wave_drag=False`, while the
symbolic optimization used `wave_drag=True`. At M 0.82, 32,000 ft, 66,300 kg,
this produced about 0.82306 kg/s in reporting versus 0.84938 kg/s in dynamics
(roughly 3.1% too low). The numeric factory now enables wave drag and numeric
emissions use that same selected fuel model. Backend evaluations agree within
1e-5 relative tolerance; regression tests cover three Mach values.

This fix affects reported instantaneous fuel flow and numeric cost/emissions
evaluation. It does not change optimization dynamics, the existing optimal
trajectories, or mass-derived integrated fuel. The earlier OpenAP comparison
PNG has been regenerated with corrected fuel flow. Acropole/BADA fuel totals
are unaffected.

Verification: 49 targeted tests passed across performance, trajectory costs,
climate objectives, cruise, neural fuel, and BADA integration. The full
solver-heavy suite was not run.

## Optional physics correction experiment

**Both corrections use OpenAP.** BADA 4 remains available as a comparison
fuel model and force-constraint model; it does not supply either correction.

`PhysicsCorrectedFuelFlow` wraps an existing `AcropoleFuelFlow` without modifying
its weights. Both methods are opt-in and use the same CasADi expression for mass
dynamics, fuel objectives, numeric trajectory fuel flow and emissions. Force
constraints still come from the optimizer's selected performance model. Inputs
retain the existing kg, knots, feet, feet/minute and kg/s conventions. Still-air
cruise remains the supported experimental use case.

### Speed-increment physics correction

Let `N` be neural fuel flow, `P` physical fuel flow, and `Mr` a user-supplied
reference Mach. Evaluate the reference states at the current mass, altitude,
vertical rate and temperature, with TAS (and Acropole groundspeed) recomputed
for `Mr`. Define:

```text
F_extension(M) = N(Mr) + P(M) - P(Mr)
t = clip((M - Mr) / transition_width, 0, 1)
w = 10*t^3 - 15*t^4 + 6*t^5
F_corrected(M) = (1-w)*N(M) + w*F_extension(M)
```

Below `Mr`, the original neural prediction is preserved. Above the transition,
the speed derivative comes from the physical model. The quintic weight has
zero first and second derivatives at both endpoints, avoiding additional
endpoint discontinuities where the component models are smooth. It does not
remove the original network's ReLU kinks or guarantee monotonicity within the
transition. The corrected total remains a fuel-flow prediction, not a drag
force or a penalty added only to the objective.

Example after loading `neural` from the external Acropole files:

```python
baseline = top.Cruise(
    "A320", "EHAM", "EDDF",
    performance_model="openap",
)
corrected = top.PhysicsCorrectedFuelFlow(
    neural, baseline.fuelflow,
    reference_mach=0.76, transition_width=0.02,
)
optimizer = top.Cruise(
    "A320", "EHAM", "EDDF",
    performance_model="openap",
    fuel_model=corrected,
)
```

The API requires an explicit reference Mach. **0.76 is an experimental
assumption, not an identified boundary of Acropole training coverage.** A
constant Mach boundary is only a first sensitivity experiment: a calibrated
boundary should account for the joint mass/altitude/speed distribution. The
OpenAP increment includes all modeled speed effects, not only wave drag.

### Wave-increment physics correction

```python
corrected = top.PhysicsCorrectedFuelFlow(
    neural, physical_with_wave_drag,
    method="wave_increment",
    wave_free_model=physical_without_wave_drag,
    strength=1.0,
)
```

This evaluates `N + strength * (P_wave - P_no_wave)` at identical conditions.
The experiment uses OpenAP with identical aircraft/engine settings and only
the wave-drag switch changed. The caller must supply compatible physical
models; arbitrary differences between models do not isolate wave drag.
There is no clipping of the increment, retraining, or inferred calibration.
This method can double-count wave effects already represented in the network.

### Reproduction and interpretation

```sh
uv run --with-editable '.[neural]' python examples/physics_corrected_fuel.py \
  --acropole-dir /path/to/Acropole \
  --bada-path /path/to/bada_4.2/tables \
  --output-dir examples/figures/physics_correction
```

The script compares OpenAP fuel, BADA 4 fuel, original Acropole, and the two
OpenAP-based corrections under each of OpenAP and BADA 4 force constraints.
Every main solve uses the same aircraft, route, mass, altitude bounds, discretization and fixed-altitude/fixed-Mach assumptions. It also varies
the transition start by +/-0.02 Mach and repeats the main speed correction from
a second initial guess. Reports include asset hashes, software versions, model
choices, transition settings, convergence and fuel-accounting checks. Figures
are PNG only at 150 dpi; raw sweeps and trajectories are also saved.

Neither agreement with a comparison model nor a lower cruise Mach establishes
improved real-world accuracy. Using OpenAP to construct a correction and then
comparing against OpenAP is a consistency experiment. BADA 4 provides an
additional model comparison, not observational ground truth. Independent
high-Mach data and validation of low-Mach accuracy are needed before claims of improved fuel
prediction or fuel savings.

### OpenAP-based correction results

Both corrections now use OpenAP, including the speed correction and its
transition sensitivity runs. BADA 4 supplies only comparison fuel predictions
and the force constraints in the second set of experiments.

| Fuel model | Mach with OpenAP forces | Mach with BADA 4 forces |
|---|---:|---:|
| OpenAP + wave drag | 0.778730 | 0.770373 |
| BADA 4 | 0.738491 | 0.736518 |
| Original Acropole | 0.820000 | 0.787374 |
| Acropole + OpenAP wave increment | 0.799654 | 0.783201 |
| Acropole + OpenAP speed correction | 0.769032 | 0.768053 |

The nominal speed correction transitions from M 0.76 to M 0.78. Its
sensitivity to the assumed transition start remains important:

| Force constraints | Start M 0.74 | Start M 0.76 | Start M 0.78 |
|---|---:|---:|---:|
| OpenAP | 0.771833 | 0.769032 | 0.786912 |
| BADA 4 | 0.766622 | 0.768053 | 0.783761 |

These settings are experimental assumptions, not calibrated training-coverage
boundaries. All 16 solves succeeded. The nominal speed correction was repeated
from M 0.81 / 31,000 ft instead of M 0.74 / 33,000 ft; the maximum fuel
difference across starts was 0.00000101 kg. This supports repeatability
for these starts without proving a global optimum or physical accuracy.

Local generated figure: `examples/figures/physics_correction/physics_correction_sweep.png` (Physics correction speed response).

Local generated figure: `examples/figures/physics_correction/physics_correction_optimization.png` (Physics correction optima).

Verification for this change: 34 neural-fuel and cruise tests passed, including
derivatives, transition continuity, temperature handling and fuel accounting.
Ruff and `git diff --check` pass. Each of the 16 experiment solves also checks
objective versus mass loss and reported fuel flow against the selected model.
The full solver-heavy suite was not rerun; this change affects the experiment
configuration, reporting and documentation rather than the correction API.

### Original and corrected optimized profiles

Plot all five main fuel-model solutions from the saved trajectories:

```sh
uv run python examples/plot_physics_corrected_profiles.py \
  --results-dir examples/figures/physics_correction
```

This creates two 150 dpi PNGs, one per force model, showing altitude, Mach,
fuel flow and cumulative mass-derived fuel burn against elapsed cruise time.
All curves within a figure share force constraints. Both figures use the same
axis scales. Altitude and Mach are flat because these experiments explicitly
hold them constant along the cruise; no climb or descent is included. Curve
endpoints show different flight durations, and fuel flow/burn is evaluated
with each solution's own fuel model. These are model predictions, not measured
fuel savings. The script verifies finite state/rate data, increasing times,
trajectory/report agreement and fuel accounting before rendering.

Local generated figure: `examples/figures/physics_correction/optimized_profiles_openap.png` (Profiles with OpenAP constraints).

Local generated figure: `examples/figures/physics_correction/optimized_profiles_bada4.png` (Profiles with BADA 4 constraints).

## Complete-flight paper experiment

`CompleteFlight` now accepts the same fuel override as `Cruise`. The physical
force model is retained. Zero speed-derivative network features remain an
approximation during climb and descent. Experimental full-flight scripts
initialize both controls and states from converged physical baselines.

A BADA3 adapter defect was identified in installed OpenAP 2.6.0: `nominal`
compares its kg/min nominal value to `idle()` in kg/s, then divides by 60.
The opentop BADA3 enroute wrapper enforces the idle lower bound in returned
kg/s units, for both symbolic and numeric paths. Symbolic smoothing uses the
width of 0.001 kg/s (at most 0.0005 kg/s smoothing increment at the crossing). Baselines computed before this correction
are diagnostic artifacts and must not be used as paper comparison results.

## Model configuration contract

`performance_model` chooses the baseline thrust, drag and default fuel models.
`fuel_model=None` retains that default fuel model; an explicit override replaces
fuel flow in every consumer, including mass dynamics, objectives, emissions and
reporting. Temperature is passed to the active fuel model regardless of which
baseline supplies the forces. Numeric fuel is resolved once per optimizer.

Aircraft limits and mass bounds still use OpenAP aircraft data. Acropole's CSV
parameters are its training normalization and output scale, not replacement
operating limits. `engine=` configures the baseline models where supported and
emissions; it does not recalibrate Acropole or reconfigure physical models
already supplied to a correction. BADA performance data determine their own
engine variant. Build a new optimizer/model for a different configuration.

Correction constructors reject mismatched aircraft and, for a wave-model pair,
mismatched engines when the models expose this metadata. Exposed wave-drag
settings must be enabled/disabled in the respective models. Custom models without
metadata are supported, but the caller must ensure they use the same aircraft,
engine, units, and settings other than wave drag. A correction can intentionally
use OpenAP while the optimizer uses BADA forces.
