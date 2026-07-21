"""Normalized data models for RAD route selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from ..routing import RouteEdge as RouteEdge
from ..routing import RouteNode as RouteNode
from ..routing import RoutePath as RoutePath


class Conformance(str, Enum):
    """Outcome of parsing or evaluating a RAD construct."""

    VALID = "valid"
    INVALID = "invalid"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True, slots=True)
class Provenance:
    """Location of a source record in an input dataset."""

    path: Path
    line_number: int
    raw_line: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """A structured parser or conformance diagnostic."""

    status: Conformance
    message: str
    provenance: Provenance | None = None
    code: str = ""


@dataclass(frozen=True, slots=True)
class ParseResult:
    """Records and diagnostics returned by a RAD reader."""

    records: tuple[Any, ...]
    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def status(self) -> Conformance:
        statuses = {diagnostic.status for diagnostic in self.diagnostics}
        if Conformance.INVALID in statuses:
            return Conformance.INVALID
        if Conformance.INDETERMINATE in statuses:
            return Conformance.INDETERMINATE
        return Conformance.VALID


@dataclass(frozen=True, slots=True)
class NavPoint:
    """A navigation point from an NNPT file."""

    point_id: str
    point_type: str
    latitude: float
    longitude: float
    name: str | None
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class Airport:
    """An airport reference point from an ARP file."""

    icao: str
    latitude: float
    longitude: float
    fir_id: str | None
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class RoutePoint:
    """One ordered point in an RTS route definition."""

    route_id: str
    route_type: str
    valid_from: str
    valid_to: str
    point_id: str
    point_type: str
    sequence: int
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class AseSegment:
    """A losslessly parsed ASE segment with resolved endpoints.

    The meanings of ``raw_field_1`` through ``raw_field_3`` are deliberately
    not named more specifically until a versioned SAAM/Gasel schema is supplied.
    """

    source_id: str
    target_id: str
    source_latitude: float
    source_longitude: float
    target_latitude: float
    target_longitude: float
    raw_field_1: int
    raw_field_2: int
    raw_field_3: int
    raw_source_latitude_minutes: float
    raw_source_longitude_minutes: float
    raw_target_latitude_minutes: float
    raw_target_longitude_minutes: float
    endpoint_token: str
    layer: str
    provenance: Provenance

    @property
    def raw_fields(self) -> tuple[int, int, int]:
        return self.raw_field_1, self.raw_field_2, self.raw_field_3


class EdgeDirection(str, Enum):
    """Directions produced by a verified ASE metadata schema."""

    FORWARD = "forward"
    REVERSE = "reverse"
    BOTH = "both"


@dataclass(frozen=True, slots=True)
class AseSemantics:
    """Verified interpretation of ASE metadata."""

    direction: EdgeDirection
    usable: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class AseCodeSchema:
    """A named, versioned mapping from raw ASE codes to verified semantics."""

    name: str
    version: str
    codes: Mapping[tuple[int, int, int], AseSemantics]

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise ValueError("ASE schema name and version must be non-empty")
        object.__setattr__(self, "codes", MappingProxyType(dict(self.codes)))

    @property
    def schema_id(self) -> str:
        return f"{self.name}:{self.version}"

    def __call__(self, segment: AseSegment) -> AseSemantics | None:
        return self.codes.get(segment.raw_fields)


@dataclass(frozen=True, slots=True)
class FlightContext:
    """Static planning context used to construct a per-flight RAD graph."""

    departure: str
    arrival: str
    requested_flight_level: int | None = None
    departure_time: datetime | None = None
    callsign: str | None = None
    enabled_layers: frozenset[str] = frozenset({"base"})
