# Continuous pulse step climb

The current isolated prototype uses one NLP call with optional nonnegative cruise
pulses. The user removed the 500-fpm cruise requirement and the second solve.
Public API integration has not been implemented.

Each pulse shares one amplitude across two consecutive control nodes, producing
linear ramps and a constant-rate hold. Amplitudes range from zero to 1500 fpm,
and interval durations are optimized. Pulse capacity is small, at most three,
and initialization bounds can reduce it. Inactive slots retain mesh and duration
restrictions. There is no amplitude gap or epsilon parameter, prior reference
NLP, DP, second solve, or outer NLP schedule enumeration.
[Current experiment](../../debug/step_climb_speed/one_pass_pulses/)

| One-pass A320 case | Steps | Positive amplitudes fpm | Whole wall time | Fuel kg |
| --- | ---: | --- | ---: | ---: |
| EHAM-LGAV, 70 t, Full40 | 2 | 449.447 / 471.160 | 120.487 s | 7649.766827 |
| EHAM-LGAV, 70 t, Cruise30 | 2 | 443.232 / 450.732 | 17.601 s | 7325.555374 |
| EHAM-EDDF, 60 t, Cruise20 | 0 | none | 9.205 s | 1094.348042 |

All three use one successful solve and pass revised profile and 81/161-point
sampled physical checks. Full40 adds 65 guard points per climb/descent interval
and 17 per cruise interval before the solve. Its 161-point minimum normalized
margin is +8.343753e-5. Each call retains 1500 iterations and 90 CPU seconds
maximum, with single BLAS/OpenMP threads and a 360-second external process cap.
Frozen sources, initialization/raw arrays, solver statistics, and acceptance
records accompany each run. [Full40 report](../../debug/step_climb_speed/one_pass_pulses/full40/report.json), [Cruise30 report](../../debug/step_climb_speed/one_pass_pulses/cruise30/report.json), [Short20 report](../../debug/step_climb_speed/one_pass_pulses/short20/report.json)

The current profile figure compares all three one-pass cases with the earlier
two-stage results. [Profile comparison](../../debug/step_climb_speed/one_pass_pulses/figures/one_pass_profiles.png)

Revised profile checks accept positive flat plateaus above the .01-fpm numerical
zero tolerance. They retain 30-second holds, 300-second level dwells, ramps at
most 300 seconds, monotonic ramps, rate bounds, and acceleration checks. The
former 500-fpm pulse minimum and derived 1083.333-ft minimum gain no longer apply.
The original initial-climb departure bound remains 500..2500 fpm. IPOPT tol=1e-6,
acceptable_tol=1e-4, and audit threshold -1e-7 are unchanged.
[Independent review](../../debug/step_climb_speed/two_stage_pulses/review.md)

The fuel-plus-regularization objective is unchanged for this comparison. Its
750-fpm scale and 100-second TV weight retain a soft preference for rate shape.
The duration penalty vanishes at zero and at rates >=750 fpm. Initial amplitudes
of 500 fpm, 100-second ramp guesses, and their 530-second capacity allocation
remain seed choices. Selected capacity restricts the basis, while solved pulse
rates are free within their nonnegative bounds. These retained choices are
explicit and have not been retuned for the relaxed requirement.

Full40 uses coherent generated position, altitude, mass, time, and collocation
initialization. Numerical endpoint roots choose time and heading guesses without
adding equality constraints. CasADi expand=True retains exact Hessians. Matched
primitive values, Jacobians, and weighted Hessians agree at three flight
conditions within rtol=1e-10 and atol=1e-9. This local evidence does not establish
complete-NLP identity or a controlled speedup.
[Derivative evidence](../../debug/step_climb_speed/two_stage_pulses/derivative_microbenchmark/results.json)

The accepted Full has a 219.721-second pre-TOC taper gaining 915.505 ft within
the historical .01..499.99-fpm comparison band, entirely above 30000 ft. This
band is descriptive only after removal of the rate requirement. Nominal TOC is
at 1771.231 seconds and 33056.051 ft. The taper is permitted by the phase
constraints and is not a demonstrated operational violation.
[Operational diagnostics](../../debug/step_climb_speed/two_stage_pulses/review_diagnostics/one_pass_full40_metrics.json)

Earlier expanded two-stage results remain historical evidence. Relaxed epsilon
.30 then 1e-8 selected two approximately 500-fpm steps for Cruise30 in 29.135
seconds, zero for Short20 in 17.338 seconds, and two for Full40 in 76.351 seconds.
Full40 used staged noncruise guards and approximately 2.59 GiB peak RSS. Earlier
unexpanded and limited-memory full attempts timed out, and initial library
position/time/control and mass guesses were inconsistent. One pass is not
universally faster than the earlier two-stage implementation.
[Two-stage experiment history](../../debug/step_climb_speed/two_stage_pulses/README.md)

Physical audits sample clean drag, lift margin, and energy. Finite samples do
not certify continuous feasibility or all state bounds and dynamics defects.
There is no audited no-step fuel comparison or global step-count proof. BADA4,
wind, and new-mesh sensitivity remain untested. Promotion still requires broader
numerical evidence and compatible public integration with existing checks.
Standalone climb/descent use under this method has not been demonstrated.
