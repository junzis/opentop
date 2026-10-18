"""High-level RAD dataset and route-optimization API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType

from ..routes import RouteOption
from ._models import (
    Airport,
    AseCodeSchema,
    FlightContext,
    ParseResult,
    RoutePath,
)
from .altitude import city_pair_altitude_limit
from .graph import DirectedMultiGraph, graph_from_ase
from .integration import route_option
from .io import read_arp, read_ase, read_nnpt
from .planner import (
    RoutePlanner,
    RouteSelectionConfig,
    add_airport_connectors,
    airport_index,
    flight_graph,
)
from .rules import RadRule, read_flc2_rules
from .search import EdgeCost


@dataclass(frozen=True, slots=True)
class RadRouteSelection:
    """RAD-specific search output expressed as generic route options."""

    graph: DirectedMultiGraph
    candidates: tuple[RoutePath, ...]
    options: tuple[RouteOption, ...]


@dataclass(frozen=True, slots=True)
class RadDataset:
    """A route-ready RAD graph, airports, and source parse results."""

    graph: DirectedMultiGraph
    airports: Mapping[str, Airport]
    parse_results: Mapping[str, ParseResult] = field(default_factory=dict)
    flight_level_rules: tuple[RadRule, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "airports", MappingProxyType(dict(self.airports)))
        object.__setattr__(
            self,
            "parse_results",
            MappingProxyType(dict(self.parse_results)),
        )

    @classmethod
    def from_ase_files(
        cls,
        navpoints_path: str | Path,
        segments_path: str | Path,
        airports_path: str | Path,
        *,
        schema: AseCodeSchema,
        layer: str,
        strict: bool = True,
        flight_level_constraints_path: str | Path | None = None,
    ) -> RadDataset:
        """Read NNPT/ASE/ARP files and build a route-ready dataset."""

        navpoints = read_nnpt(navpoints_path, strict=strict)
        segments = read_ase(
            segments_path,
            navpoints.records,
            layer=layer,
            strict=strict,
        )
        airports = read_arp(airports_path, strict=strict)
        graph = graph_from_ase(
            segments.records,
            navpoints.records,
            decode=schema,
            strict=strict,
        )
        flight_levels = (
            read_flc2_rules(flight_level_constraints_path)
            if flight_level_constraints_path is not None
            else None
        )
        return cls(
            graph,
            airport_index(airports.records),
            {
                "navpoints": navpoints,
                "segments": segments,
                "airports": airports,
                **(
                    {"flight_levels": flight_levels}
                    if flight_levels is not None
                    else {}
                ),
            },
            () if flight_levels is None else flight_levels.records,
        )

    def select_routes(
        self,
        context: FlightContext,
        *,
        edge_cost: EdgeCost = lambda edge: edge.distance_m,
        config: RouteSelectionConfig = RouteSelectionConfig(),
        connector_count: int = 5,
        maximum_connector_distance_m: float = 150_000.0,
        airport_groups: Mapping[str, Sequence[str]] | None = None,
    ) -> RadRouteSelection:
        """Select RAD candidates and convert them to generic route options."""

        applicable = flight_graph(self.graph, context)
        connected, source, target = add_airport_connectors(
            applicable,
            self.airports[context.departure],
            self.airports[context.arrival],
            connector_count=connector_count,
            maximum_distance_m=maximum_connector_distance_m,
        )
        candidates = tuple(
            RoutePlanner(connected, edge_cost=edge_cost).candidates(
                source,
                target,
                config=config,
            )
        )
        options = tuple(
            route_option(connected, route, name=f"RAD route {index + 1}")
            for index, route in enumerate(candidates)
        )
        if self.flight_level_rules:
            constrained = []
            for option, route in zip(options, candidates):
                limit = city_pair_altitude_limit(
                    self.flight_level_rules,
                    context,
                    route_points=route.nodes,
                    airport_groups=airport_groups,
                )
                limit.require_resolved()
                constrained.append(
                    replace(
                        option,
                        maximum_altitude_m=limit.maximum_altitude_m,
                        metadata={
                            **option.metadata,
                            "altitude_rule_ids": tuple(
                                rule.rule_id for rule in limit.applied_rules
                            ),
                        },
                    )
                )
            options = tuple(constrained)
        return RadRouteSelection(connected, candidates, options)
