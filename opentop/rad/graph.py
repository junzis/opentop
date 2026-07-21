"""Identity-preserving directed multigraphs for RAD routing."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from hashlib import sha256
from itertools import pairwise

from ..routing import DirectedMultiGraph, geodesic_distance_m
from ._models import (
    AseSegment,
    AseSemantics,
    EdgeDirection,
    FlightContext,
    NavPoint,
    RouteEdge,
    RouteNode,
    RoutePoint,
)

AseDecoder = Callable[[AseSegment], AseSemantics | None]


class UnsupportedAseCode(ValueError):
    """Raised when an ASE schema cannot interpret a segment in strict mode."""


def _stable_edge_id(prefix: str, source: str, target: str, discriminator: str) -> str:
    value = f"{prefix}\0{source}\0{target}\0{discriminator}".encode()
    return f"{prefix}:{sha256(value).hexdigest()[:20]}"


def graph_from_ase(
    segments: Iterable[AseSegment],
    navpoints: Iterable[NavPoint],
    *,
    decode: AseDecoder,
    strict: bool = True,
) -> DirectedMultiGraph:
    """Build a graph using an explicit, versioned ASE semantic decoder.

    A decoder must return ``None`` for unsupported metadata. Such records are
    omitted rather than being guessed into a production route graph.
    """

    graph = DirectedMultiGraph()
    point_index = {point.point_id: point for point in navpoints}
    for point in point_index.values():
        graph.add_node(
            RouteNode(
                point.point_id,
                point.latitude,
                point.longitude,
                point.point_type,
            )
        )

    for segment in segments:
        semantics = decode(segment)
        if semantics is None:
            if strict:
                raise UnsupportedAseCode(
                    f"ASE schema does not support fields {segment.raw_fields} "
                    f"at {segment.provenance.path}:{segment.provenance.line_number}"
                )
            continue
        if not semantics.usable:
            continue
        directions: tuple[tuple[str, str], ...]
        if semantics.direction is EdgeDirection.FORWARD:
            directions = ((segment.source_id, segment.target_id),)
        elif semantics.direction is EdgeDirection.REVERSE:
            directions = ((segment.target_id, segment.source_id),)
        else:
            directions = (
                (segment.source_id, segment.target_id),
                (segment.target_id, segment.source_id),
            )
        for direction_index, (source, target) in enumerate(directions):
            source_point, target_point = point_index[source], point_index[target]
            discriminator = (
                f"{segment.provenance.path}:{segment.provenance.line_number}:"
                f"{direction_index}"
            )
            graph.add_edge(
                RouteEdge(
                    edge_id=_stable_edge_id(
                        segment.layer, source, target, discriminator
                    ),
                    source=source,
                    target=target,
                    distance_m=geodesic_distance_m(
                        source_point.latitude,
                        source_point.longitude,
                        target_point.latitude,
                        target_point.longitude,
                    ),
                    layer=segment.layer,
                    metadata={
                        "ase_raw_fields": segment.raw_fields,
                        **dict(semantics.metadata),
                    },
                    provenance=segment.provenance,
                )
            )
    return graph


def graph_from_routes(
    routes: Iterable[RoutePoint], navpoints: Iterable[NavPoint]
) -> DirectedMultiGraph:
    """Build directed airway edges from consecutive points in each RTS route."""

    graph = DirectedMultiGraph()
    point_index = {point.point_id: point for point in navpoints}
    for point in point_index.values():
        graph.add_node(
            RouteNode(
                point.point_id,
                point.latitude,
                point.longitude,
                point.point_type,
            )
        )

    grouped: dict[str, list[RoutePoint]] = defaultdict(list)
    for route_point in routes:
        grouped[route_point.route_id].append(route_point)
    for route_id, route_points in grouped.items():
        ordered = sorted(route_points, key=lambda point: point.sequence)
        for index, (source_record, target_record) in enumerate(pairwise(ordered)):
            if source_record.point_id not in point_index:
                continue
            if target_record.point_id not in point_index:
                continue
            source = point_index[source_record.point_id]
            target = point_index[target_record.point_id]
            edge_id = _stable_edge_id(
                f"rts:{route_id}", source.point_id, target.point_id, str(index)
            )
            graph.add_edge(
                RouteEdge(
                    edge_id=edge_id,
                    source=source.point_id,
                    target=target.point_id,
                    distance_m=geodesic_distance_m(
                        source.latitude,
                        source.longitude,
                        target.latitude,
                        target.longitude,
                    ),
                    layer="base",
                    metadata={"route_id": route_id},
                    provenance=source_record.provenance,
                )
            )
    return graph


def edge_available(edge: RouteEdge, context: FlightContext) -> bool:
    """Evaluate the currently supported static edge predicates."""

    if edge.layer not in context.enabled_layers:
        return False
    level = context.requested_flight_level
    if level is not None:
        if edge.min_flight_level is not None and level < edge.min_flight_level:
            return False
        if edge.max_flight_level is not None and level > edge.max_flight_level:
            return False
    return True
