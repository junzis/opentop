"""Internal representation of one aircraft in a CasADi Opti problem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AircraftTranscription:
    """Symbolic variables and expressions for one discretized trajectory.

    The object deliberately owns the build-specific expressions that used to
    be available only through mutable ``Base`` attributes.  A single-aircraft
    solve and a shared multi-aircraft solve can therefore use the same
    transcription code.
    """

    optimizer: Any
    opti: Any
    X: list[Any]
    Xc: list[list[Any]]
    U: list[Any]
    ts_final: Any
    interval_dts: list[Any]
    objective_raw: Any
    objective_scaled: Any
    objective_scale: float
    objective_kwargs: dict[str, Any]
    projection_center: tuple[float, float] | None = None
