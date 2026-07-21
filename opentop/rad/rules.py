"""Fail-closed parsing of the supported subset of RAD AWK rules."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType

from ._models import (
    Conformance,
    Diagnostic,
    FlightContext,
    ParseResult,
    Provenance,
)
from ._text import read_legacy_lines


class TruthValue(str, Enum):
    """Three-valued result used when a rule needs unavailable context."""

    TRUE = "true"
    FALSE = "false"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True, slots=True)
class RuleContext:
    """Values made available to supported AWK predicates."""

    values: Mapping[str, str | int | Sequence[str]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            MappingProxyType(
                {key.upper(): value for key, value in self.values.items()}
            ),
        )

    @classmethod
    def from_flight(
        cls,
        flight: FlightContext,
        *,
        route_points: Sequence[str] = (),
        route_segments: Sequence[str] = (),
    ) -> RuleContext:
        values: dict[str, str | int | Sequence[str]] = {
            "DEP": flight.departure,
            "ARR": flight.arrival,
            "CP": f"{flight.departure} {flight.arrival}",
            "POINT": route_points,
            "VIA": route_segments or route_points,
        }
        if flight.requested_flight_level is not None:
            values["RFL"] = flight.requested_flight_level
        if flight.callsign is not None:
            values["CALLSIGN"] = flight.callsign
        return cls(values)


class _Expression:
    def evaluate(self, context: RuleContext) -> TruthValue:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class _Constant(_Expression):
    value: TruthValue

    def evaluate(self, context: RuleContext) -> TruthValue:
        del context
        return self.value


@dataclass(frozen=True, slots=True)
class _Atom(_Expression):
    field_name: str
    operator: str
    value: str | int
    regex: re.Pattern[str] | None = field(default=None, repr=False)

    def evaluate(self, context: RuleContext) -> TruthValue:
        actual = context.values.get(self.field_name)
        if actual is None:
            return TruthValue.INDETERMINATE
        values: Sequence[str | int]
        if isinstance(actual, str | int):
            values = (actual,)
        else:
            values = actual
        if self.operator == "~":
            assert self.regex is not None
            joined = " ".join(str(item) for item in values)
            return (
                TruthValue.TRUE
                if self.regex.search(joined) is not None
                else TruthValue.FALSE
            )
        if self.operator == "==":
            return (
                TruthValue.TRUE
                if any(str(item) == str(self.value) for item in values)
                else TruthValue.FALSE
            )
        try:
            number = float(values[0])
            reference = float(self.value)
        except (TypeError, ValueError, IndexError):
            return TruthValue.INDETERMINATE
        comparisons = {
            ">": number > reference,
            ">=": number >= reference,
            "<": number < reference,
            "<=": number <= reference,
        }
        return TruthValue.TRUE if comparisons[self.operator] else TruthValue.FALSE


@dataclass(frozen=True, slots=True)
class _Not(_Expression):
    child: _Expression

    def evaluate(self, context: RuleContext) -> TruthValue:
        value = self.child.evaluate(context)
        if value is TruthValue.INDETERMINATE:
            return value
        return TruthValue.FALSE if value is TruthValue.TRUE else TruthValue.TRUE


@dataclass(frozen=True, slots=True)
class _Boolean(_Expression):
    operator: str
    left: _Expression
    right: _Expression

    def evaluate(self, context: RuleContext) -> TruthValue:
        left = self.left.evaluate(context)
        right = self.right.evaluate(context)
        if self.operator == "&&":
            if TruthValue.FALSE in (left, right):
                return TruthValue.FALSE
            if TruthValue.INDETERMINATE in (left, right):
                return TruthValue.INDETERMINATE
            return TruthValue.TRUE
        if TruthValue.TRUE in (left, right):
            return TruthValue.TRUE
        if TruthValue.INDETERMINATE in (left, right):
            return TruthValue.INDETERMINATE
        return TruthValue.FALSE


@dataclass(frozen=True, slots=True)
class RadPredicate:
    """A parsed supported AWK condition."""

    raw: str
    _expression: _Expression = field(repr=False)

    def evaluate(self, context: RuleContext) -> TruthValue:
        return self._expression.evaluate(context)


@dataclass(frozen=True, slots=True)
class RadRule:
    """One RAD routing action and its applicability predicate."""

    rule_id: str
    section: tuple[str, ...]
    airac: str | None
    description: str
    predicate: RadPredicate | None
    routing_constraint: str | None
    flight_level_constraint: int | None
    active: bool
    status: Conformance
    provenance: Provenance

    def applicability(self, context: RuleContext) -> TruthValue:
        if not self.active:
            return TruthValue.FALSE
        if self.status is not Conformance.VALID or self.predicate is None:
            return TruthValue.INDETERMINATE
        return self.predicate.evaluate(context)


class _ConditionParser:
    def __init__(self, source: str) -> None:
        self.source = source
        self.position = 0

    def parse(self) -> RadPredicate:
        expression = self._parse_or()
        self._skip_whitespace()
        if self.position != len(self.source):
            raise ValueError(
                f"unsupported condition near {self.source[self.position :]!r}"
            )
        return RadPredicate(self.source, expression)

    def _skip_whitespace(self) -> None:
        while self.position < len(self.source) and self.source[self.position].isspace():
            self.position += 1

    def _consume(self, value: str) -> bool:
        self._skip_whitespace()
        if self.source.startswith(value, self.position):
            self.position += len(value)
            return True
        return False

    def _parse_or(self) -> _Expression:
        expression = self._parse_and()
        while self._consume("||"):
            expression = _Boolean("||", expression, self._parse_and())
        return expression

    def _parse_and(self) -> _Expression:
        expression = self._parse_unary()
        while self._consume("&&"):
            expression = _Boolean("&&", expression, self._parse_unary())
        return expression

    def _parse_unary(self) -> _Expression:
        if self._consume("!"):
            return _Not(self._parse_unary())
        if self._consume("("):
            expression = self._parse_or()
            if not self._consume(")"):
                raise ValueError("unclosed predicate parenthesis")
            return expression
        return self._parse_atom()

    def _parse_atom(self) -> _Expression:
        self._skip_whitespace()
        field_match = re.match(
            r"(?:[A-Za-z_][A-Za-z0-9_]*|\$[0-9]+)",
            self.source[self.position :],
        )
        if field_match is None:
            raise ValueError("expected predicate field")
        field_name = field_match.group(0).upper()
        self.position += len(field_match.group(0))
        self._skip_whitespace()
        if self._consume("~"):
            self._skip_whitespace()
            if not self._consume("/"):
                raise ValueError("regex predicate must use /pattern/")
            pattern_chars: list[str] = []
            escaped = False
            while self.position < len(self.source):
                character = self.source[self.position]
                self.position += 1
                if character == "/" and not escaped:
                    break
                pattern_chars.append(character)
                escaped = character == "\\" and not escaped
                if character != "\\":
                    escaped = False
            else:
                raise ValueError("unclosed regex predicate")
            pattern = "".join(pattern_chars)
            try:
                compiled = re.compile(pattern)
            except re.error as error:
                raise ValueError(f"invalid predicate regex: {error}") from error
            return _Atom(field_name, "~", pattern, compiled)
        for operator in (">=", "<=", "==", ">", "<"):
            if not self._consume(operator):
                continue
            self._skip_whitespace()
            if operator == "==" and self._consume('"'):
                end = self.source.find('"', self.position)
                if end < 0:
                    raise ValueError("unclosed string predicate")
                value = self.source[self.position : end]
                self.position = end + 1
                return _Atom(field_name, operator, value)
            number = re.match(r"[-+]?\d+", self.source[self.position :])
            if number is None:
                raise ValueError("comparison predicate requires an integer")
            self.position += len(number.group(0))
            return _Atom(field_name, operator, int(number.group(0)))
        raise ValueError(f"unsupported operator for {field_name}")


def parse_predicate(source: str) -> RadPredicate:
    """Parse a supported boolean AWK condition or raise ``ValueError``."""

    return _ConditionParser(source.strip()).parse()


def read_awk_rules(path: str | Path) -> ParseResult:
    """Read routing rules while preserving unsupported rules as indeterminate."""

    source = Path(path)
    lines = read_legacy_lines(source)
    rules: list[RadRule] = []
    diagnostics: list[Diagnostic] = []
    section_by_depth: dict[int, str] = {}
    rule_id: str | None = None
    airac: str | None = None
    description = ""
    block_conditions: list[str] = []

    def add_rule(
        condition: str,
        action: str,
        *,
        active: bool,
        provenance: Provenance,
    ) -> None:
        conditions = (
            (*block_conditions, condition) if condition else tuple(block_conditions)
        )
        combined = " && ".join(f"({item})" for item in conditions)
        status = Conformance.VALID
        predicate: RadPredicate | None
        constraint_match = re.search(
            r'REGLE\s*=\s*REGLE\s*"([^"]+)"', action, re.IGNORECASE
        )
        routing_constraint = (
            constraint_match.group(1).strip() if constraint_match is not None else None
        )
        flight_level_match = re.search(r"FL_CONT\s*=\s*(\d+)", action, re.IGNORECASE)
        flight_level_constraint = (
            int(flight_level_match.group(1)) if flight_level_match is not None else None
        )
        try:
            predicate = (
                parse_predicate(combined)
                if combined
                else RadPredicate("true", _Constant(TruthValue.TRUE))
            )
        except ValueError as error:
            predicate = None
            status = Conformance.INDETERMINATE
            diagnostics.append(
                Diagnostic(
                    status,
                    f"unsupported RAD predicate: {error}",
                    provenance,
                    "unsupported_predicate",
                )
            )
        if routing_constraint is None and flight_level_constraint is None:
            status = Conformance.INDETERMINATE
            diagnostics.append(
                Diagnostic(
                    status,
                    "unsupported RAD action",
                    provenance,
                    "unsupported_action",
                )
            )
        rules.append(
            RadRule(
                rule_id=rule_id or "",
                section=tuple(
                    section_by_depth[depth] for depth in sorted(section_by_depth)
                ),
                airac=airac,
                description=description,
                predicate=predicate,
                routing_constraint=routing_constraint,
                flight_level_constraint=flight_level_constraint,
                active=active,
                status=status,
                provenance=provenance,
            )
        )

    section_pattern = re.compile(r"^#SECTION_(\d+)\s*-?\s*(.*)$")
    rule_pattern = re.compile(r"^#RAD\s+(\S+)\s*(.*)$")
    inline_pattern = re.compile(r"^if\s*\((.*)\)\s*\{\s*(.*?)\s*\}\s*$")
    block_pattern = re.compile(r"^if\s*\((.*)\)\s*\{\s*$")
    direct_pattern = re.compile(r"^if\s*\((.*)\)\s+([^{}]+)$")

    for line_number, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("#END"):
            break
        section_match = section_pattern.match(stripped)
        if section_match is not None:
            depth = int(section_match.group(1))
            section_by_depth = {
                key: value for key, value in section_by_depth.items() if key < depth
            }
            section_by_depth[depth] = section_match.group(2).strip()
            block_conditions.clear()
            continue
        rule_match = rule_pattern.match(stripped)
        if rule_match is not None:
            rule_id = rule_match.group(1)
            description = rule_match.group(2).strip()
            airac_matches = re.findall(r"\bAIRAC\s+(\d+)\b", description)
            airac = airac_matches[-1] if airac_matches else None
            block_conditions.clear()
            continue
        if rule_id is None:
            continue

        active = True
        code = stripped
        if code.startswith("#"):
            if not re.match(r"^#\s*if\s*\(", code):
                continue
            active = False
            code = re.sub(r"^#\s*", "", code)
        inline_match = inline_pattern.match(code)
        provenance = Provenance(source, line_number, raw_line)
        if inline_match is not None:
            add_rule(
                inline_match.group(1),
                inline_match.group(2),
                active=active,
                provenance=provenance,
            )
            continue
        direct_match = direct_pattern.match(code)
        if direct_match is not None:
            add_rule(
                direct_match.group(1),
                direct_match.group(2),
                active=active,
                provenance=provenance,
            )
            continue
        block_match = block_pattern.match(code)
        if block_match is not None:
            block_conditions.append(block_match.group(1))
            continue
        if code == "}":
            if block_conditions:
                block_conditions.pop()
            continue
        if "REGLE" in code:
            add_rule("", code, active=active, provenance=provenance)

    return ParseResult(tuple(rules), tuple(diagnostics))


def read_flc2_rules(path: str | Path) -> ParseResult:
    """Read FLC2 rules with the same structural and fail-closed parser."""

    return read_awk_rules(path)


def applicable_rules(
    rules: Sequence[RadRule], context: RuleContext
) -> tuple[tuple[RadRule, ...], tuple[RadRule, ...]]:
    """Return applicable and indeterminate active rules separately."""

    applicable: list[RadRule] = []
    indeterminate: list[RadRule] = []
    for rule in rules:
        result = rule.applicability(context)
        if result is TruthValue.TRUE:
            applicable.append(rule)
        elif result is TruthValue.INDETERMINATE:
            indeterminate.append(rule)
    return tuple(applicable), tuple(indeterminate)
