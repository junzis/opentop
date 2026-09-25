# OpenTOP documentation

Reviewed against the `feature/neural-fuel` working tree on 2026-09-25, including
its merged OpenTOP 2.7.0 changes and the pending neural-fuel API fixes. Features
on this branch are not necessarily available in the latest published package.

## Current API and architecture

- [API guide](api-guide.md): model selection, loading, cruise constraints, results,
  multi-start, route choice, and fleet scope.
- [Execution flow](opentop-high-level-flow.md): how the Python API and CLI reach
  the shared transcription, solver, and reporting code.
- [Functionality map](opentop-functionalities-map.html) and its
  [PNG export](opentop-functionalities-map.png).
- [Multi-start](design/multi_start_trajectory.md): initialization, speed jitter,
  ranking, and returned diagnostics.
- [Route choice](design/route_choice_optimization.md): route alternatives and
  continuous optimization, with selectable ranking criteria.
- [RAD integration](design/rad_route_optimization.md): supported network/rule
  inputs and remaining conformance limitations.

## Research notes and proposals

- [Neural fuel](design/neural_fuel.md): current integration contract plus dated
  experimental results. Historical results are not current benchmark claims.
  Figure files under `examples/figures/` are generated locally and Git-ignored.
- [Fleet scaling](design/scalable_fleet_optimization.md): deferred architecture
  proposal, not implemented large-fleet decomposition.

The screenshots in `_static/` are illustrative examples, not API references.
The repository [README](../README.md) contains installation and quick-start
instructions. No Sphinx or MkDocs build is configured in this repository;
Markdown guides and the standalone HTML map are the documentation sources.
