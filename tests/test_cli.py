"""Objective expressions accepted by the CLI."""

import click
import pytest

from opentop.cli import parse_objective


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("fuel", [(1.0, "fuel", None)]),
        ("ci:30", [(1.0, "ci", "30")]),
        ("gwp100", [(1.0, "gwp100", None)]),
        ("0.3*fuel+0.7*grid", [(0.3, "fuel", None), (0.7, "grid", None)]),
        (
            "0.4*fuel+0.4*grid+0.2*time",
            [(0.4, "fuel", None), (0.4, "grid", None), (0.2, "time", None)],
        ),
        ("0.5*fuel+0.5*gwp100", [(0.5, "fuel", None), (0.5, "gwp100", None)]),
        (" 0.3 * fuel + 0.7 * grid ", [(0.3, "fuel", None), (0.7, "grid", None)]),
        ("0.5*fuel+grid", [(0.5, "fuel", None), (1.0, "grid", None)]),
    ],
)
def test_parse_objective(expression, expected):
    assert parse_objective(expression) == expected


@pytest.mark.parametrize(
    "expression,message",
    [
        ("bogus", "unknown objective term"),
        ("ci", "'ci' requires a parameter"),
        ("0.3*", "cannot parse objective term"),
        ("", None),
    ],
)
def test_invalid_objective(expression, message):
    with pytest.raises(click.UsageError, match=message):
        parse_objective(expression)
