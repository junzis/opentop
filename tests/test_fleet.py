import openap
import pytest

import numpy as np
import opentop as top


def _small_cruise(origin="EHAM", destination="EDDF", *, nodes=8):
    optimizer = top.Cruise("A320", origin, destination, 0.85)
    optimizer.setup(nodes=nodes, max_iter=800)
    return optimizer


def test_flight_spec_validates_identity_and_weight():
    optimizer = _small_cruise()

    with pytest.raises(ValueError, match="non-empty"):
        top.FlightSpec("", optimizer)
    with pytest.raises(ValueError, match="weight"):
        top.FlightSpec("AC1", optimizer, weight=0)


def test_multi_aircraft_uses_default_separation_config():
    multi = top.MultiAircraft([top.FlightSpec("AC1", _small_cruise())])

    assert multi.separation == top.SeparationConfig()


def test_multi_aircraft_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="unique"):
        top.MultiAircraft(
            [
                top.FlightSpec("AC1", _small_cruise()),
                top.FlightSpec("AC1", _small_cruise()),
            ]
        )


def test_multi_aircraft_uses_limited_memory_hessian_by_default():
    multi = top.MultiAircraft([top.FlightSpec("AC1", _small_cruise())])

    assert multi._solver_options()["ipopt.hessian_approximation"] == "limited-memory"


def test_multi_aircraft_respects_explicit_exact_hessian():
    multi = top.MultiAircraft(
        [top.FlightSpec("AC1", _small_cruise())],
        ipopt_kwargs={"hessian_approximation": "exact"},
    )

    assert multi._solver_options()["ipopt.hessian_approximation"] == "exact"


def test_common_altitude_requires_fixed_cruise_altitude():
    with pytest.raises(ValueError, match="fix_cruise_altitude"):
        top.MultiAircraft(
            [top.FlightSpec("AC1", _small_cruise())],
            common_altitude=True,
        )


def test_crossing_aircraft_can_avoid_conflict_at_one_common_altitude():
    center_lat, center_lon = 51.0, 7.0
    half_route_m = 150_000.0
    flights = []
    for index, bearing in enumerate((0.0, 90.0)):
        origin = openap.aero.latlon(center_lat, center_lon, half_route_m, bearing)
        destination = openap.aero.latlon(
            center_lat, center_lon, half_route_m, bearing + 180.0
        )
        optimizer = _small_cruise(
            (float(origin[0]), float(origin[1])),
            (float(destination[0]), float(destination[1])),
        )
        optimizer.fix_cruise_altitude()
        flights.append(top.FlightSpec(f"AC{index + 1}", optimizer))

    result = top.MultiAircraft(
        flights, common_altitude=True, max_iter=2_000
    ).trajectory()

    assert result.success
    first = result.trajectories["AC1"]
    second = result.trajectories["AC2"]
    np.testing.assert_allclose(first.altitude, first.altitude.iloc[0], atol=1e-6)
    np.testing.assert_allclose(second.altitude, second.altitude.iloc[0], atol=1e-6)
    assert first.altitude.iloc[0] == pytest.approx(second.altitude.iloc[0], abs=1e-6)
    assert result.pair_reports[0].vertical_m == pytest.approx(0.0, abs=1e-6)


def test_single_aircraft_shared_opti_matches_direct_cruise():
    direct = _small_cruise()
    direct_df = direct.trajectory(objective="fuel")
    direct_objective = direct.objective_value

    child = _small_cruise()
    original_center = child._projection_center
    result = top.MultiAircraft(
        [top.FlightSpec("AC1", child, start_time=120.0)], max_iter=800
    ).trajectory()
    fleet_df = result.trajectories["AC1"]

    assert result.success
    assert result.separation_constraints == 0
    assert result.aircraft_objectives["AC1"] == pytest.approx(
        direct_objective, rel=2e-4
    )
    assert fleet_df.mass.iloc[-1] == pytest.approx(direct_df.mass.iloc[-1], rel=2e-4)
    np.testing.assert_allclose(fleet_df.absolute_ts, fleet_df.ts + 120.0)
    assert child._projection_center == original_center


def test_two_far_apart_aircraft_need_no_separation_constraints():
    flights = [
        top.FlightSpec("north", _small_cruise("EHAM", "EDDF")),
        top.FlightSpec("south", _small_cruise("LEMD", "LEBL")),
    ]

    result = top.MultiAircraft(flights, max_iter=800).trajectory()

    assert result.success
    assert result.separation_constraints == 0
    assert len(result.pair_reports) == 1
    assert result.pair_reports[0].margin > 0


def test_crossing_aircraft_satisfy_high_order_separation():
    center_lat, center_lon = 51.0, 7.0
    half_route_m = 150_000.0
    flights = []
    for index, bearing in enumerate((45.0, 135.0)):
        origin = openap.aero.latlon(
            center_lat, center_lon, half_route_m, (bearing + 180.0) % 360.0
        )
        destination = openap.aero.latlon(center_lat, center_lon, half_route_m, bearing)
        optimizer = _small_cruise(
            (float(origin[0]), float(origin[1])),
            (float(destination[0]), float(destination[1])),
            nodes=12,
        )
        flights.append(top.FlightSpec(f"AC{index}", optimizer))

    config = top.SeparationConfig(max_refinements=2)
    result = top.MultiAircraft(flights, separation=config, max_iter=1200).trajectory()

    assert result.solver_success
    assert result.separation_success
    assert result.success
    assert result.separation_constraints > 0
    assert result.pair_reports[0].minimum_metric >= (
        config.minimum_metric - config.verification_tolerance
    )
