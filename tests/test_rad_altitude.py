"""RAD city-pair ceilings reach the continuous trajectory NLP."""

from pathlib import Path
from types import SimpleNamespace

import casadi as ca
import pytest
from openap.aero import ft

import numpy as np
import opentop as top
from opentop import rad
from opentop._transcription import AircraftTranscription


def _rules(tmp_path, text):
    path = tmp_path / "rules.flc2"
    path.write_text(text)
    return rad.read_flc2_rules(path).records


def test_city_pair_tightest_cap_and_route_condition(tmp_path):
    rules = _rules(
        tmp_path,
        """
#SECTION_1 City Pair Level Capping (RAD ANNEX 2A)
#RAD CAP1 - Ceiling
if (DEP ~/EBBR/ && ARR ~/EDDK/ && $3==0) FL_CONT=175
#RAD CAP2 - Lower via cap
if (DEP ~/EBBR/ && ARR ~/EDDK/ && VIA ~/FIXA/) FL_CONT=165
#RAD OTHER - Other destination
if (ARR ~/EGLL/) FL_CONT=125
#RAD PROFILE - Signed point constraint is separate
if (ARR ~/EDDK/ && POINT == "FIXA") FL_CONT=-140
""",
    )
    limit = rad.city_pair_altitude_limit(
        rules, rad.FlightContext("EBBR", "EDDK"), route_points=("FIXA",)
    )
    limit.require_resolved()
    assert limit.maximum_flight_level == 165
    assert limit.maximum_altitude_m == pytest.approx(16500 * ft)
    assert [r.rule_id for r in limit.applied_rules] == ["CAP1", "CAP2"]
    assert rules[-1].flight_level_scope == "profile"
    assert rules[-1].flight_level_constraint == -140
    assert (
        rad.city_pair_altitude_limit(
            rules, rad.FlightContext("EHAM", "LIRF")
        ).maximum_altitude_m
        is None
    )


def test_groups_and_missing_predicates_are_unresolved(tmp_path):
    rules = _rules(
        tmp_path,
        """
#RAD GROUP - Cap
if (DEP ~/BRUSSELS_GROUP/ && ARR ~/PARIS_GROUP/ && $3==0) FL_CONT=195
""",
    )
    flight = rad.FlightContext("EBBR", "LFPG")
    with pytest.raises(ValueError, match="GROUP"):
        rad.city_pair_altitude_limit(rules, flight).require_resolved()
    groups = {"BRUSSELS_GROUP": ["EBBR"], "PARIS_GROUP": ["LFPG"]}
    limit = rad.city_pair_altitude_limit(rules, flight, airport_groups=groups)
    limit.require_resolved()
    assert limit.maximum_flight_level == 195
    assert not rad.city_pair_altitude_limit(
        rules, rad.FlightContext("EHAM", "LFPG"), airport_groups=groups
    ).applied_rules


def test_inactive_caps_and_missing_rfl(tmp_path):
    rules = _rules(
        tmp_path,
        """
#RAD INACTIVE - Cap
#if (DEP ~/EBBR/) FL_CONT=100
#RAD UNKNOWN - Needs RFL
if (DEP ~/EBBR/ && RFL > 200) FL_CONT=175
""",
    )
    with pytest.raises(ValueError, match="UNKNOWN"):
        rad.city_pair_altitude_limit(
            rules, rad.FlightContext("EBBR", "EDDK")
        ).require_resolved()
    limit = rad.city_pair_altitude_limit(rules, rad.FlightContext("EBBR", "EDDK", 250))
    assert limit.maximum_flight_level == 175


def test_dataset_loads_and_attaches_altitude_ceiling(tmp_path):
    fixtures = Path(__file__).parent / "fixtures/rad"
    path = tmp_path / "caps.flc2"
    path.write_text(
        "#RAD CAP - Cap\nif (DEP ~/AAAA/ && ARR ~/DDDD/ && $3==0) FL_CONT=125\n"
    )
    schema = rad.AseCodeSchema(
        "fixture",
        "1",
        {
            (0, 1, 10): rad.AseSemantics(rad.EdgeDirection.FORWARD),
            (0, 2, 20): rad.AseSemantics(rad.EdgeDirection.FORWARD),
        },
    )
    dataset = rad.RadDataset.from_ase_files(
        fixtures / "network.nnpt",
        fixtures / "segments.ase",
        fixtures / "airports.arp",
        schema=schema,
        layer="base",
        strict=False,
        flight_level_constraints_path=path,
    )
    selection = dataset.select_routes(
        rad.FlightContext("AAAA", "DDDD"), connector_count=1
    )
    assert selection.options
    for option in selection.options:
        assert option.maximum_altitude_m == pytest.approx(12500 * ft)
        assert option.metadata["altitude_rule_ids"] == ("CAP",)


def test_route_ceiling_cannot_be_relaxed_and_failed_validation_is_excluded():
    calls = []

    class Optimizer:
        def __init__(self):
            self.aircraft = {"cruise": {"height": 10000.0}}
            self.mass_init = 60000.0

        def setup(self, **kwargs):
            pass

        def trajectory(self, **kwargs):
            calls.append(kwargs)
            guess = kwargs["initial_guess"].copy()
            guess["altitude"] = 20000.0
            return SimpleNamespace(df=guess, success=True, fuel=10.0, status="ok")

    option = top.RouteOption(
        "cap", ((50.0, 2.0), (51.0, 4.0)), maximum_altitude_m=12500 * ft
    )
    result = top.optimize_routes(
        [option],
        Optimizer,
        config=top.RouteOptimizationConfig(trajectory_kwargs={"h_max": 30000 * ft}),
    )
    assert calls[0]["h_max"] == pytest.approx(12500 * ft)
    assert calls[0]["initial_guess"].altitude.max() == pytest.approx(12500)
    assert not result.successful


@pytest.mark.parametrize("phase", [top.CompleteFlight, top.Cruise])
@pytest.mark.parametrize("ceiling_ft", [4500, 12500])
def test_low_altitude_cap_converges_and_bounds_between_nodes(phase, ceiling_ft):
    opt = phase("A320", "EBBR", "EDDK", m0=60000)
    opt.setup(nodes=20, ipopt_kwargs={"max_cpu_time": 45})
    result = opt.trajectory(h_max=ceiling_ft * ft, result_object=True)
    assert result.success, result.status
    assert result.df.altitude.max() <= ceiling_ft + 0.1
    transcription = opt._last_transcription
    heights = [
        opt._last_solution.value(transcription.state_at(k, tau)[2])
        for k in range(opt.nodes)
        for tau in np.linspace(0, 1, 51)
    ]
    assert max(heights) <= ceiling_ft * ft + 0.1


def test_bernstein_cap_detects_hidden_peak():
    # A degree-3 polynomial can exceed the cap between its Legendre points.
    roots = tuple(ca.collocation_points(3, "legendre"))
    opti = ca.Opti()
    offset = opti.variable()

    def polynomial(tau):
        return 0.9 + 4 * tau * (0.5 - tau) * (1 - tau)

    nodes = [0.0, *roots]
    assert max(polynomial(tau) for tau in nodes) < 1.08
    assert polynomial(0.2) > 1.08
    X = [
        ca.vertcat(0, 0, polynomial(0) + offset),
        ca.vertcat(0, 0, polynomial(1) + offset),
    ]
    Xc = [[ca.vertcat(0, 0, polynomial(tau) + offset) for tau in roots]]
    transcription = AircraftTranscription(
        None, opti, X, Xc, [], None, [], None, None, 1.0, {}, roots
    )
    transcription.constrain_altitude_ceiling(1.08)
    # At zero offset, the Bernstein coefficients reveal the violation.
    constraints = np.asarray(opti.debug.value(opti.g, [offset == 0])).ravel()
    upper = np.asarray(opti.debug.value(opti.ubg)).ravel()
    assert (constraints > upper).any()


@pytest.mark.parametrize("assignment", ["195.5", "195-5", "LIMIT"])
def test_unsupported_altitude_action_is_not_silently_accepted(tmp_path, assignment):
    rules = _rules(
        tmp_path,
        f"#RAD BAD - Unsupported action\nif (DEP ~/EBBR/) FL_CONT={assignment}\n",
    )
    limit = rad.city_pair_altitude_limit(rules, rad.FlightContext("EBBR", "EDDK"))
    assert limit.maximum_flight_level is None
    with pytest.raises(ValueError, match="BAD"):
        limit.require_resolved()


def test_explicit_cruise_floor_conflicting_with_cap_is_rejected():
    opt = top.Cruise("A320", "EBBR", "EDDK", h_min=15000 * ft)
    with pytest.raises(ValueError, match="h_min"):
        opt.trajectory(h_max=12500 * ft)


@pytest.mark.parametrize("alias", ["EDGG_FIR", "GENEVA_AREA", "V_LFEECTA"])
def test_named_regional_sets_require_definitions(tmp_path, alias):
    rules = _rules(
        tmp_path, f"#RAD REGION - Cap\nif (DEP ~/{alias}/ && ARR ~/EDDK/) FL_CONT=175\n"
    )
    flight = rad.FlightContext("EBBR", "EDDK")
    with pytest.raises(ValueError, match="REGION"):
        rad.city_pair_altitude_limit(rules, flight).require_resolved()
    limit = rad.city_pair_altitude_limit(
        rules, flight, airport_groups={alias: ("EBBR",)}
    )
    limit.require_resolved()
    assert limit.maximum_flight_level == 175


@pytest.mark.parametrize(
    "unknown_cap, unresolved", [(195, False), (175, False), (125, True)]
)
def test_unknown_rule_only_blocks_when_it_can_tighten_the_cap(
    tmp_path, unknown_cap, unresolved
):
    rules = _rules(
        tmp_path,
        f"""
#RAD KNOWN - Explicit cap
if (DEP ~/EBBR/ && ARR ~/EDDK/) FL_CONT=175
#RAD UNKNOWN - Airport group
if (DEP ~/SOME_GROUP/ && ARR ~/EDDK/) FL_CONT={unknown_cap}
""",
    )
    limit = rad.city_pair_altitude_limit(rules, rad.FlightContext("EBBR", "EDDK"))
    assert bool(limit.unresolved_rules) is unresolved
    assert limit.maximum_flight_level == 175
    if not unresolved:
        limit.require_resolved()
