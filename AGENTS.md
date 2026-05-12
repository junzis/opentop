# AGENTS.md

Guidance for coding agents working in this repository.

## Project Overview

`opentop` is a Python package supporting Python 3.11 and newer. Linting and type-checking target Python 3.13. It uses OpenAP aircraft performance models, CasADi Opti, and IPOPT to solve direct-collocation optimal-control problems for complete flights and individual climb, cruise, and descent phases.

The public package API is exported from `opentop/__init__.py`. Keep backwards compatibility in mind: users import `opentop` directly and call phase optimizer classes such as `CompleteFlight`, `Cruise`, `Climb`, and `Descent`.

## Repository Map

- `opentop/base.py`: shared optimizer setup, projection, OpenAP model wiring, collocation NLP construction, solve execution, objective helper methods, and result conversion.
- `opentop/full.py`: complete flight optimizer.
- `opentop/cruise.py`: cruise phase optimizer and optional fixed Mach/altitude/track constraints.
- `opentop/climb.py`: climb phase optimizer.
- `opentop/descent.py`: descent phase optimizer.
- `opentop/_dynamics.py`: pure dynamics and initial-guess helpers.
- `opentop/_objectives.py`: pure objective functions and objective resolver.
- `opentop/_trajectory.py`: numeric solver output to trajectory DataFrame conversion.
- `opentop/_multi_start.py`: multi-start wrapper and initial-guess perturbation logic.
- `opentop/_options.py`: structured result dataclasses.
- `opentop/tools.py`: wind-field preprocessing and CasADi grid interpolant utilities.
- `opentop/replay.py`: reusable replay/data-fetching logic.
- `opentop/vis.py`: Matplotlib/Cartopy visualization helpers.
- `opentop/cli/`: Click CLI commands for `optimize`, `gengrid`, and `replay`.
- `tests/`: pytest suite with both fast unit tests and solver-heavy integration tests.
- `docs/design/`: design notes for larger features.
- `examples/`: notebooks and quick examples.

## Development Commands

Use `uv` for local commands:

```sh
uv run pytest --collect-only -q
uv run pytest tests/test_cli.py -n auto -q
uv run pytest tests/test_cruise.py -n auto -q
uv run pytest tests/ -n auto -q
uv run ruff check .
uv run pyright
```

Prefer targeted tests while developing, then broaden verification according to the risk of the change. The suite is dominated by IPOPT solves; always include `-n auto` for `pytest tests/...` runs unless you are only collecting tests.

## Testing Guidance

- For parser, CLI helper, dataclass, replay normalization, or cache changes, run the directly related test file first.
- For objective, trajectory conversion, or fuel/mass accounting changes, run the targeted tests plus at least one phase optimizer test.
- For changes in `Base._build_opti`, `Base._solve`, phase constraints, or dynamics, expect solver-heavy tests to be relevant.
- `uv run pytest --collect-only -q` is cheap and useful for confirming test discovery.
- Run test files under `tests/` with xdist, for example `uv run pytest tests/test_cruise.py -n auto -q`.
- Avoid adding tests that require live network access. Existing replay tests use fixtures and mocks where possible.
- Grid-cost tests should reuse cached `.casadi` interpolants instead of rebuilding bsplines on every run. Prefer `opentop.tools.cached_interpolant_from_dataframe(...)` with a fixture path under `tests/fixtures/` when size permits.

## Coding Guidelines

- Preserve the public API unless the task explicitly asks for a breaking change.
- Keep `trajectory()` kwargs compatibility in mind. Several methods intentionally accept related kwargs for API symmetry across phases.
- Prefer the existing split between phase classes and pure helper modules. Put reusable math/objective/conversion logic in `_dynamics.py`, `_objectives.py`, `_trajectory.py`, or `_multi_start.py` rather than growing phase classes unnecessarily.
- Keep numerical changes small and justified. Bounds, smoothness constraints, scaling, and performance constraints can change solver feasibility.
- Do not casually change DataFrame column names or units. Public trajectory output uses columns such as `mass`, `ts`, `x`, `y`, `h`, `latitude`, `longitude`, `altitude`, `mach`, `tas`, `vertical_rate`, `heading`, `fuel_cost`, and `grid_cost`.
- Use structured APIs for data handling. Avoid ad hoc string parsing when pandas, pathlib, Click, or existing helpers already cover the case.
- Keep comments focused on non-obvious numerical or compatibility decisions.

## Solver And Numerics Notes

- The NLP is built with CasADi Opti and solved with IPOPT.
- States are `[xp, yp, h, mass, ts]`; controls are `[mach, vs, psi]`.
- Distances and positions use the optimizer projection in `Base.proj`.
- Fuel burn in trajectory output is derived from mass differences so `fuel_cost.sum()` matches `m0 - m_final` up to floating-point tolerance.
- Grid-cost objectives need careful handling of `interpolant`, `n_dim`, and `time_dependent`.
- Bspline grid-cost interpolants generally need exact Hessian behavior for stable IPOPT solves. Existing code marks grid objectives accordingly.
- `auto_rescale_objective` exists for very small objective magnitudes, especially climate/grid-style objectives.
- Manual NLP variable scaling was investigated and not adopted. IPOPT's default `gradient-based` NLP scaling is generally sufficient; manual scaling changed some grid-cost local optima and did not justify the added complexity.
- Wind support uses `tools.PolyWind`, which must work for both numeric and CasADi symbolic inputs.

## CLI And Replay Notes

- The CLI entry point is `opentop.cli:main`.
- `opentop optimize` supports built-in and weighted objective expressions through `opentop/cli/_helpers.py`.
- `opentop gengrid` builds cached `.casadi` interpolants from parquet grids.
- `opentop replay` has optional dependencies from the `replay` extra: `traffic`, `fastmeteo`, `scipy`, and `pandas<3`.
- Replay and visualization can run in headless environments. Preserve the Matplotlib backend handling in `opentop/cli/replay.py`.
- Do not make replay tests depend on OpenSky or ERA5 live services unless explicitly requested.
- When testing the current checkout as an installed CLI/package, prefer `uv run --with-editable . ...` over `uv run --with '.' ...`; the latter can reuse a stale uv cache. For example: `uv run --with-editable . opentop replay ...`.

## uv Environment Notes

- For comparing against released versions, use an isolated uv environment such as `uv run --no-project --with opentop==2.2.0 ...`. This is closer to how users consume the package and avoids local worktree or `sys.path` contamination.
- Reserve git worktrees for comparisons involving unreleased branches or local experiments.

## Style And Tooling

- Runtime support starts at Python 3.11; linting and type-checking target Python 3.13.
- Ruff is configured in `pyproject.toml`; keep imports sorted and lint-clean.
- Pyright runs in basic mode for `opentop`.
- CasADi, OpenAP, pyproj, and sklearn have incomplete typing in places. Existing `# type: ignore[...]` comments are often intentional; do not remove them unless verified.
- Prefer ASCII in source files unless an existing file already uses non-ASCII for a clear reason.

## Versioning And Releases

- Do not bump `pyproject.toml` version, create release tags, or push tags unless the user explicitly asks for a release action. Tag pushes trigger the PyPI publish workflow, and PyPI artifacts cannot be deleted or re-uploaded.
- If the user asks to bump the version but does not specify patch/minor/major, ask which level they want. Do not infer it from semver reasoning alone.
- Even during an explicit release task, stop before pushing a tag and confirm the tag name and target commit with the user.
- After a release tag is pushed, check the GitHub release and write concise release notes manually, for example with `gh release edit <tag> --notes ...`. Auto-generated release notes are usually empty because this repo often ships direct commits rather than merged PRs.

## Before Finishing A Change

Report what changed and which checks were run. If a check was skipped because it is slow, solver-heavy, or requires optional dependencies/network data, say so explicitly and name the targeted checks that did run.
