from collections.abc import MutableMapping
from typing import Any, cast

import openap
import pytest

import numpy as np
import opentop as top
import pandas as pd


def _small_cruise(
    origin: str | tuple[float, float] = "EHAM",
    destination: str | tuple[float, float] = "EDDF",
    *,
    nodes: int = 8,
) -> top.Cruise:
    optimizer = top.Cruise("A320", origin, destination, 0.85)
    optimizer.setup(nodes=nodes, max_iter=800)
    return optimizer


def test_flight_spec_validates_identity_and_weight():
    optimizer = _small_cruise()

    with pytest.raises(ValueError, match="non-empty"):
        top.FlightSpec("", optimizer)
    with pytest.raises(ValueError, match="weight"):
        top.FlightSpec("AC1", optimizer, weight=0)


def test_flight_spec_keeps_joint_options_separate_and_immutable():
    spec = top.FlightSpec(
        "AC1",
        _small_cruise(),
        options={"route_margin_m": 20_000.0},
        joint_options={"inbound_route_side": "right"},
    )

    assert "inbound_route_side" not in spec.options
    assert spec.joint_options["inbound_route_side"] == "right"
    with pytest.raises(TypeError):
        cast(MutableMapping[str, Any], spec.joint_options)["inbound_route_side"] = (
            "left"
        )


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


def test_arrival_gap_requires_descent_optimizers():
    with pytest.raises(ValueError, match="Descent"):
        top.MultiAircraft(
            [top.FlightSpec("AC1", _small_cruise())],
            minimum_arrival_gap_s=120.0,
        )


def test_arrival_gap_must_be_positive():
    optimizer = top.Descent("A320", "EHAM", "EDDF", 0.85)
    with pytest.raises(ValueError, match="finite and positive"):
        top.MultiAircraft([top.FlightSpec("AC1", optimizer)], minimum_arrival_gap_s=0.0)


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


def test_two_descent_arrivals_are_sequenced_and_separated():
    runway = (52.360258, 4.711725)
    approach_waypoints: tuple[tuple[float, float], ...] = tuple(
        (float(point[0]), float(point[1]))
        for point in (
            openap.aero.latlon(*runway, distance_m, 0.0)
            for distance_m in (40_000.0, 20_000.0)
        )
    )
    flights = []
    for index, (distance_km, bearing, start_time) in enumerate(
        ((130.0, 340.0, 0.0), (125.0, 20.0, 30.0))
    ):
        origin = openap.aero.latlon(runway[0], runway[1], distance_km * 1000.0, bearing)
        optimizer = top.Descent(
            "A320",
            (float(origin[0]), float(origin[1])),
            runway,
            0.82,
        )
        optimizer.setup(nodes=10, max_iter=3_000)
        descent_entry = pd.DataFrame(
            {
                "mass": [optimizer.mass_init],
                "mach": [0.75],
                "h": [20_000.0 * openap.aero.ft],
            }
        )
        flights.append(
            top.FlightSpec(
                f"AC{index + 1}",
                optimizer,
                start_time=start_time,
                objective="ci:30",
                options={
                    "df_cruise": descent_entry,
                    "alt_start": 20_000.0,
                    "remove_cruise": False,
                    "route_margin_m": 50_000.0,
                    "max_duration_s": 2_400.0,
                    "waypoints": approach_waypoints,
                    "waypoint_node_indices": [6, 8],
                    "waypoint_tolerance_m": 500.0,
                    "route_heading_tolerance_deg": 15.0,
                    "variable_timestep": False,
                },
            )
        )

    gap_s = 180.0
    result = top.MultiAircraft(
        flights,
        minimum_arrival_gap_s=gap_s,
        max_iter=3_000,
        ipopt_kwargs={"hessian_approximation": "exact"},
    ).trajectory()

    assert result.success
    first, second = result.trajectories.values()
    assert first.ts.iloc[-1] <= 2_400.0 + 1e-3
    assert second.ts.iloc[-1] <= 2_400.0 + 1e-3
    assert second.absolute_ts.iloc[-1] - first.absolute_ts.iloc[-1] >= (gap_s - 1e-3)
    for trajectory in (first, second):
        for node_index, waypoint in zip((6, 8), approach_waypoints):
            assert (
                openap.aero.distance(
                    trajectory.latitude.iloc[node_index],
                    trajectory.longitude.iloc[node_index],
                    waypoint[0],
                    waypoint[1],
                )
                <= 500.0 + 1e-3
            )
    assert result.pair_reports[0].minimum_metric >= (
        result.pair_reports[0].required_metric
        - top.SeparationConfig().verification_tolerance
    )


def test_single_aircraft_shared_opti_matches_direct_cruise():
    direct = _small_cruise()
    direct_df = direct.trajectory(objective="fuel")
    assert isinstance(direct_df, pd.DataFrame)
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
