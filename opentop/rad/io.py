"""Readers for the route-network subset of SAAM/Gasel RAD data.

The NNPT, ARP, and RTS readers are Python adaptations informed by the private
``alazarovski/rddr`` R package, used with the repository owner's permission.
ASE parsing is native to OpenTOP because rddr does not currently cover it.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable, Mapping
from pathlib import Path
from typing import cast

from ._models import (
    Airport,
    AseSegment,
    Conformance,
    Diagnostic,
    NavPoint,
    ParseResult,
    Provenance,
    RoutePoint,
)
from ._text import read_legacy_lines


class RadParseError(ValueError):
    """Raised when strict RAD ingestion encounters an invalid record."""


def _finish(
    records: Iterable[object], diagnostics: list[Diagnostic], strict: bool
) -> ParseResult:
    result = ParseResult(tuple(records), tuple(diagnostics))
    if strict and result.status is Conformance.INVALID:
        first = next(
            diagnostic
            for diagnostic in result.diagnostics
            if diagnostic.status is Conformance.INVALID
        )
        location = ""
        if first.provenance is not None:
            location = f" ({first.provenance.path}:{first.provenance.line_number})"
        raise RadParseError(f"{first.message}{location}")
    return result


def _invalid(message: str, provenance: Provenance, *, code: str) -> Diagnostic:
    return Diagnostic(Conformance.INVALID, message, provenance, code)


def read_nnpt(path: str | Path, *, strict: bool = True) -> ParseResult:
    """Read navigation points from a semicolon-delimited NNPT file."""

    source = Path(path)
    lines = read_legacy_lines(source)
    if not lines:
        provenance = Provenance(source, 1, "")
        return _finish(
            (), [_invalid("NNPT file is empty", provenance, code="empty_file")], strict
        )

    diagnostics: list[Diagnostic] = []
    records: list[NavPoint] = []
    try:
        declared_count = int(lines[0].strip())
    except ValueError:
        declared_count = -1
        diagnostics.append(
            _invalid(
                "NNPT first line must contain a record count",
                Provenance(source, 1, lines[0]),
                code="invalid_count",
            )
        )

    seen: set[str] = set()
    for line_number, raw_line in enumerate(lines[1:], start=2):
        provenance = Provenance(source, line_number, raw_line)
        fields = raw_line.split(";")
        if len(fields) != 5:
            diagnostics.append(
                _invalid(
                    "NNPT record must contain five fields",
                    provenance,
                    code="field_count",
                )
            )
            continue
        point_id, point_type, lat_text, lon_text, name = (
            field.strip() for field in fields
        )
        try:
            latitude = float(lat_text)
            longitude = float(lon_text)
        except ValueError:
            diagnostics.append(
                _invalid(
                    "NNPT coordinates must be numeric",
                    provenance,
                    code="coordinate_type",
                )
            )
            continue
        if not (
            point_id
            and math.isfinite(latitude)
            and math.isfinite(longitude)
            and -90 <= latitude <= 90
            and -180 <= longitude <= 180
        ):
            diagnostics.append(
                _invalid(
                    "NNPT identifier or coordinates are invalid",
                    provenance,
                    code="invalid_point",
                )
            )
            continue
        if point_id in seen:
            diagnostics.append(
                _invalid(
                    f"duplicate NNPT identifier {point_id!r}",
                    provenance,
                    code="duplicate_point",
                )
            )
            continue
        seen.add(point_id)
        records.append(
            NavPoint(
                point_id,
                point_type,
                latitude,
                longitude,
                None if name == "_" else name,
                provenance,
            )
        )

    if declared_count >= 0 and declared_count != len(lines) - 1:
        diagnostics.append(
            _invalid(
                f"NNPT declares {declared_count} records but contains {len(lines) - 1}",
                Provenance(source, 1, lines[0]),
                code="count_mismatch",
            )
        )
    return _finish(records, diagnostics, strict)


def read_arp(path: str | Path, *, strict: bool = True) -> ParseResult:
    """Read airport reference points; source coordinates are arc-minutes."""

    source = Path(path)
    records: list[Airport] = []
    diagnostics: list[Diagnostic] = []
    for line_number, raw_line in enumerate(read_legacy_lines(source), start=1):
        if not raw_line.strip():
            continue
        provenance = Provenance(source, line_number, raw_line)
        fields = raw_line.split()
        if len(fields) not in (3, 4):
            diagnostics.append(
                _invalid(
                    "ARP record must contain three or four fields",
                    provenance,
                    code="field_count",
                )
            )
            continue
        try:
            latitude = float(fields[1]) / 60.0
            longitude = float(fields[2]) / 60.0
        except ValueError:
            diagnostics.append(
                _invalid(
                    "ARP coordinates must be numeric",
                    provenance,
                    code="coordinate_type",
                )
            )
            continue
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            diagnostics.append(
                _invalid(
                    "ARP coordinates are outside valid bounds",
                    provenance,
                    code="coordinate_bounds",
                )
            )
            continue
        records.append(
            Airport(
                fields[0],
                latitude,
                longitude,
                fields[3] if len(fields) == 4 else None,
                provenance,
            )
        )
    return _finish(records, diagnostics, strict)


def read_routes(path: str | Path, *, strict: bool = True) -> ParseResult:
    """Read ordered airway points from an RTS file."""

    source = Path(path)
    records: list[RoutePoint] = []
    diagnostics: list[Diagnostic] = []
    lines = read_legacy_lines(source)
    for line_number, raw_line in enumerate(lines[1:], start=2):
        if not raw_line.strip():
            continue
        provenance = Provenance(source, line_number, raw_line)
        fields = [field.strip() for field in raw_line.split(";")]
        if len(fields) < 8 or fields[0] != "L":
            diagnostics.append(
                _invalid(
                    "RTS route record must be an eight-field L record",
                    provenance,
                    code="invalid_route_record",
                )
            )
            continue
        try:
            sequence = int(fields[7])
        except ValueError:
            diagnostics.append(
                _invalid(
                    "RTS sequence must be an integer",
                    provenance,
                    code="sequence_type",
                )
            )
            continue
        records.append(
            RoutePoint(
                route_id=fields[1],
                route_type=fields[2],
                valid_from=fields[4],
                valid_to=fields[3],
                point_id=fields[5],
                point_type=fields[6],
                sequence=sequence,
                provenance=provenance,
            )
        )
    records.sort(key=lambda record: (record.route_id, record.sequence))
    return _finish(records, diagnostics, strict)


def navpoint_index(
    navpoints: Collection[NavPoint] | Mapping[str, NavPoint],
) -> Mapping[str, NavPoint]:
    """Return a point-ID index while rejecting duplicate identifiers."""

    if isinstance(navpoints, Mapping):
        return cast(Mapping[str, NavPoint], navpoints)
    index: dict[str, NavPoint] = {}
    for point in navpoints:
        if point.point_id in index:
            raise ValueError(f"duplicate navigation point {point.point_id!r}")
        index[point.point_id] = point
    return index


def _resolve_endpoint_token(
    token: str, points: Mapping[str, NavPoint]
) -> tuple[str, str] | None:
    candidates: list[tuple[str, str]] = []
    for position, character in enumerate(token):
        if character != "_":
            continue
        source_id = token[:position]
        target_id = token[position + 1 :]
        if source_id in points and target_id in points:
            candidates.append((source_id, target_id))
    if len(candidates) == 1:
        return candidates[0]
    return None


def read_ase(
    path: str | Path,
    navpoints: Collection[NavPoint] | Mapping[str, NavPoint],
    *,
    layer: str | None = None,
    coordinate_tolerance_degrees: float = 2e-5,
    strict: bool = True,
) -> ParseResult:
    """Read ASE segments without inventing meanings for undocumented fields.

    Endpoint coordinates in ASE are stored in arc-minutes. The endpoint token is
    resolved against NNPT identifiers, allowing identifiers that contain an
    underscore as long as the split remains unique.
    """

    if coordinate_tolerance_degrees < 0:
        raise ValueError("coordinate tolerance must be non-negative")
    source = Path(path)
    points = navpoint_index(navpoints)
    records: list[AseSegment] = []
    diagnostics: list[Diagnostic] = []
    segment_layer = layer or source.stem

    for line_number, raw_line in enumerate(read_legacy_lines(source), start=1):
        if not raw_line.strip():
            continue
        provenance = Provenance(source, line_number, raw_line)
        fields = raw_line.split()
        if len(fields) != 8:
            diagnostics.append(
                _invalid(
                    "ASE record must contain eight fields",
                    provenance,
                    code="field_count",
                )
            )
            continue
        try:
            raw_fields = tuple(int(value) for value in fields[:3])
            raw_coordinates = tuple(float(value) for value in fields[3:7])
        except ValueError:
            diagnostics.append(
                _invalid(
                    "ASE metadata and coordinates must be numeric",
                    provenance,
                    code="numeric_field",
                )
            )
            continue
        endpoint_ids = _resolve_endpoint_token(fields[7], points)
        if endpoint_ids is None:
            diagnostics.append(
                Diagnostic(
                    Conformance.INDETERMINATE,
                    f"ASE endpoint token {fields[7]!r} does not resolve uniquely",
                    provenance,
                    "endpoint_resolution",
                )
            )
            continue
        source_id, target_id = endpoint_ids
        source_point, target_point = points[source_id], points[target_id]
        coordinates = tuple(value / 60.0 for value in raw_coordinates)
        if not all(math.isfinite(value) for value in coordinates) or not (
            -90 <= coordinates[0] <= 90
            and -180 <= coordinates[1] <= 180
            and -90 <= coordinates[2] <= 90
            and -180 <= coordinates[3] <= 180
        ):
            diagnostics.append(
                _invalid(
                    "ASE coordinates are non-finite or outside valid bounds",
                    provenance,
                    code="coordinate_bounds",
                )
            )
            continue
        expected = (
            source_point.latitude,
            source_point.longitude,
            target_point.latitude,
            target_point.longitude,
        )
        if any(
            abs(actual - reference) > coordinate_tolerance_degrees
            for actual, reference in zip(coordinates, expected)
        ):
            diagnostics.append(
                Diagnostic(
                    Conformance.INDETERMINATE,
                    "ASE coordinates do not match the resolved NNPT endpoints",
                    provenance,
                    "coordinate_mismatch",
                )
            )
            continue
        records.append(
            AseSegment(
                source_id=source_id,
                target_id=target_id,
                source_latitude=source_point.latitude,
                source_longitude=source_point.longitude,
                target_latitude=target_point.latitude,
                target_longitude=target_point.longitude,
                raw_field_1=raw_fields[0],
                raw_field_2=raw_fields[1],
                raw_field_3=raw_fields[2],
                raw_source_latitude_minutes=raw_coordinates[0],
                raw_source_longitude_minutes=raw_coordinates[1],
                raw_target_latitude_minutes=raw_coordinates[2],
                raw_target_longitude_minutes=raw_coordinates[3],
                endpoint_token=fields[7],
                layer=segment_layer,
                provenance=provenance,
            )
        )
    return _finish(records, diagnostics, strict)
