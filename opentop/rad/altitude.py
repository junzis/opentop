"""Resolve flight-wide RAD city-pair altitude ceilings from FLC2 actions.

Signed point/profile assignments have separate SAAM semantics and are not
interpreted as global ceilings. Their source records remain available.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from openap.aero import ft

from ._models import FlightContext
from .rules import RadRule, RuleContext, TruthValue


@dataclass(frozen=True, slots=True)
class CityPairAltitudeLimit:
    """The tightest applicable ceiling, with source rules and unknown rules."""

    maximum_flight_level: int | None
    applied_rules: tuple[RadRule, ...] = ()
    unresolved_rules: tuple[RadRule, ...] = ()

    @property
    def maximum_altitude_m(self) -> float | None:
        return (
            None
            if self.maximum_flight_level is None
            else self.maximum_flight_level * 100 * ft
        )

    def require_resolved(self) -> None:
        if self.unresolved_rules:
            identifiers = ", ".join(
                rule.rule_id or str(rule.provenance.line_number)
                for rule in self.unresolved_rules
            )
            raise ValueError(f"unresolved RAD city-pair altitude rules: {identifiers}")


def city_pair_altitude_limit(
    rules: Sequence[RadRule],
    flight: FlightContext,
    *,
    route_points: Sequence[str] = (),
    route_segments: Sequence[str] = (),
    airport_groups: Mapping[str, Sequence[str]] | None = None,
) -> CityPairAltitudeLimit:
    """Combine applicable unsigned flight-wide ceilings using their minimum.

    The FLC2 $3 == 0 guard denotes initialization of the flight-wide ceiling.
    It is evaluated once here, rather than against trajectory timestamps.
    Unknown rules remain unresolved when they could tighten the ceiling.
    A known stricter cap already satisfies any looser unknown ceiling.
    """
    values = dict(
        RuleContext.from_flight(
            flight, route_points=route_points, route_segments=route_segments
        ).values
    )
    values["$3"] = 0
    context = RuleContext(values, airport_groups or {})
    applied, unresolved = [], []
    for rule in rules:
        if rule.flight_level_scope != "flight":
            continue
        applicability = rule.applicability(context)
        if applicability is TruthValue.TRUE:
            if (
                rule.flight_level_constraint is None
                or rule.flight_level_constraint <= 0
            ):
                unresolved.append(rule)
            else:
                applied.append(rule)
        elif applicability is TruthValue.INDETERMINATE:
            unresolved.append(rule)
    ceiling = min(
        (
            rule.flight_level_constraint
            for rule in applied
            if rule.flight_level_constraint is not None
        ),
        default=None,
    )
    unresolved = [
        rule
        for rule in unresolved
        if ceiling is None
        or rule.flight_level_constraint is None
        or rule.flight_level_constraint < ceiling
    ]
    return CityPairAltitudeLimit(ceiling, tuple(applied), tuple(unresolved))
