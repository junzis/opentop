"""Tests for fail-closed RAD AWK rule parsing."""

from pathlib import Path

from opentop import rad

FIXTURES = Path(__file__).parent / "fixtures" / "rad"


def test_supported_predicate_uses_three_valued_logic():
    predicate = rad.parse_predicate('DEP ~/^EH/ && (RFL >= 300 || ARR == "LIRF")')

    assert (
        predicate.evaluate(rad.RuleContext({"DEP": "EHAM", "RFL": 320}))
        is rad.TruthValue.TRUE
    )
    assert (
        predicate.evaluate(rad.RuleContext({"DEP": "EGLL", "RFL": 320}))
        is rad.TruthValue.FALSE
    )
    assert (
        predicate.evaluate(rad.RuleContext({"DEP": "EHAM"}))
        is rad.TruthValue.INDETERMINATE
    )


def test_awk_reader_retains_unsupported_rules_as_indeterminate():
    result = rad.read_awk_rules(FIXTURES / "rules.awk")

    assert len(result.records) == 4
    assert result.status is rad.Conformance.INDETERMINATE
    assert result.records[0].rule_id == "RULE1"
    assert result.records[0].routing_constraint == ">A_B >C"
    assert result.records[0].status is rad.Conformance.VALID
    assert result.records[1].active is False
    assert result.records[2].status is rad.Conformance.INDETERMINATE
    assert result.records[3].flight_level_constraint == 285
    assert result.records[3].status is rad.Conformance.VALID


def test_rule_applicability_separates_unknown_rules():
    rules = rad.read_awk_rules(FIXTURES / "rules.awk").records
    flight = rad.FlightContext("AAAA", "DDDD", 320)
    context = rad.RuleContext.from_flight(flight)

    applicable, indeterminate = rad.applicable_rules(rules, context)

    assert [rule.rule_id for rule in applicable] == ["RULE1"]
    assert [rule.rule_id for rule in indeterminate] == ["RULE3", "RULE4"]


def test_flc2_reader_accepts_runtime_guard_when_context_supplies_it():
    rules = rad.read_flc2_rules(FIXTURES / "rules.awk").records
    rule = rules[3]
    context = rad.RuleContext({"DEP": "AAAA", "ARR": "DDDD", "$3": 0})

    assert rule.applicability(context) is rad.TruthValue.TRUE
