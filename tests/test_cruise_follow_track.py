"""Validation and numerical checks for prescribed cruise ground tracks."""

import casadi as ca
import openap.casadi as oc
import pytest

import numpy as np
import opentop as top
import pandas as pd
from opentop._options import TrajectoryResult

# ---- helpers ----------------------------------------------------------------


def _track_between(opt, n=15, bow_deg=0.0):
    """Build a (lat, lon) track from opt's origin to destination.

    Points are linearly interpolated in lat/lon so the endpoints match the
    boundary conditions exactly. A non-zero ``bow_deg`` adds a half-sine
    lateral offset in latitude (zero at both ends) to make the path a
    meaningful detour rather than the great-circle default.
    """
    frac = np.linspace(0.0, 1.0, n)
    lat = opt.lat1 + (opt.lat2 - opt.lat1) * frac
    lon = opt.lon1 + (opt.lon2 - opt.lon1) * frac
    lat = lat + bow_deg * np.sin(np.pi * frac)
    return lat, lon


# ---- fixtures ---------------------------------------------------------------


@pytest.fixture()
def opt(aircraft_type, short_flight):
    return top.Cruise(
        aircraft_type,
        short_flight["origin"],
        short_flight["destination"],
        short_flight["m0"],
    )


# ---- unit tests: track_ref construction -------------------------------------


class TestTrackRefConstruction:
    def test_track_ref_none_by_default(self, opt):
        assert opt.track_ref is None

    def test_accepts_python_lists(self, opt):
        """lat/lon are passed through np.asarray, so plain lists must work."""
        lat, lon = _track_between(opt)
        opt.follow_track(lat.tolist(), lon.tolist())
        assert opt.track_ref is not None

    def test_s_max_equals_total_arc_length(self, opt):
        lat, lon = _track_between(opt)
        opt.follow_track(lat, lon)
        _, _, s_max = opt.track_ref

        x, y = opt.proj(lon, lat)
        expected = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
        assert s_max == pytest.approx(expected, rel=1e-9)

    def test_accepts_non_uniform_spacing(self, opt):
        """The path is parametrized by cumulative arc length, not by index, so
        unevenly spaced points must build and still reproduce both endpoints."""
        frac = np.array([0.0, 0.02, 0.05, 0.1, 0.5, 0.85, 0.95, 1.0])
        lat = opt.lat1 + (opt.lat2 - opt.lat1) * frac
        lon = opt.lon1 + (opt.lon2 - opt.lon1) * frac

        opt.follow_track(lat, lon)
        x_ref, y_ref, s_max = opt.track_ref

        x0, y0 = opt.proj(opt.lon1, opt.lat1)
        xf, yf = opt.proj(opt.lon2, opt.lat2)
        assert float(x_ref(0.0)) == pytest.approx(x0, abs=1.0)
        assert float(y_ref(0.0)) == pytest.approx(y0, abs=1.0)
        assert float(x_ref(s_max)) == pytest.approx(xf, abs=1.0)
        assert float(y_ref(s_max)) == pytest.approx(yf, abs=1.0)

    def test_rejects_duplicate_consecutive_points(self, opt):
        """Reject duplicate points before passing the grid to CasADi."""
        lat, lon = _track_between(opt)
        lat = np.insert(lat, 1, lat[0])  # duplicate the first point
        lon = np.insert(lon, 1, lon[0])
        with pytest.raises(ValueError, match="duplicate consecutive"):
            opt.follow_track(lat, lon)

    def test_interpolants_reproduce_projected_endpoints(self, opt):
        lat, lon = _track_between(opt)
        opt.follow_track(lat, lon)
        x_ref, y_ref, s_max = opt.track_ref

        x0, y0 = opt.proj(opt.lon1, opt.lat1)
        xf, yf = opt.proj(opt.lon2, opt.lat2)

        # A bspline interpolant passes through the data at the knots (s=0, s=s_max).
        assert float(x_ref(0.0)) == pytest.approx(x0, abs=1.0)
        assert float(y_ref(0.0)) == pytest.approx(y0, abs=1.0)
        assert float(x_ref(s_max)) == pytest.approx(xf, abs=1.0)
        assert float(y_ref(s_max)) == pytest.approx(yf, abs=1.0)


# ---- unit tests: endpoint validation ----------------------------------------


class TestEndpointValidation:
    def test_rejects_start_far_from_origin(self, opt):
        lat, lon = _track_between(opt)
        lat[0] += 1.0  # ~111 km off — well over the 1 km tolerance
        with pytest.raises(ValueError, match="track endpoints do not match"):
            opt.follow_track(lat, lon)

    def test_rejects_end_far_from_destination(self, opt):
        lat, lon = _track_between(opt)
        lon[-1] += 1.0
        with pytest.raises(ValueError, match="track endpoints do not match"):
            opt.follow_track(lat, lon)

    def test_accepts_small_endpoint_offset(self, opt):
        """An offset under the 1 km tolerance must be accepted."""
        lat, lon = _track_between(opt)
        # ~50 m north of the origin — inside tolerance.
        lat[0] += 50.0 / oc.geo.distance(opt.lat1, opt.lon1, opt.lat1 + 1, opt.lon1)
        opt.follow_track(lat, lon)
        assert opt.track_ref is not None


# ---- integration test: constraint is actually enforced ----------------------


class TestFollowTrackTrajectory:
    def test_trajectory_follows_bowed_track(self, opt, monkeypatch):
        tolerance_m = 1000.0
        lat, lon = _track_between(opt, n=25, bow_deg=0.3)
        opt.follow_track(lat, lon)
        captured = []
        original = opt._add_formulation

        def capture(*args, **kwargs):
            transcription = original(*args, **kwargs)
            captured.append(transcription)
            return transcription

        monkeypatch.setattr(opt, "_add_formulation", capture)
        df = opt.trajectory(objective="fuel")
        assert opt.success, opt.stats["return_status"]
        assert isinstance(df, pd.DataFrame) and len(df) > 0
        transcription = captured[0]
        solution = opt._last_solution
        x_ref, y_ref, length = opt.track_ref
        # A dense reference grid bounds nearest-point measurement error to ~2 m.
        reference_s = np.linspace(0, length, int(length / 2) + 1)
        reference = np.column_stack(
            [
                np.asarray(x_ref(reference_s)).ravel(),
                np.asarray(y_ref(reference_s)).ravel(),
            ]
        )
        from scipy.spatial import KDTree

        tree = KDTree(reference)
        dense = []
        for k in range(opt.nodes):
            for tau in np.linspace(0, 1, 21):
                dense.append(solution.value(transcription.state_at(k, float(tau)))[:2])
        distances, _ = tree.query(np.asarray(dense))
        assert distances.max() <= tolerance_m + 2.0
        boundary_distances, _ = tree.query(df[["x", "y"]].to_numpy())
        assert boundary_distances.max() < 2.0
        rates = [
            abs(float(solution.value(opt._control_change_rate(transcription.U, k, 2))))
            for k in range(opt.nodes)
        ]
        assert max(rates) <= opt.MAX_TURN_RATE + 1e-7
        assert df.heading.max() - df.heading.min() > 2.0


def test_track_with_fixed_mach_and_grid_objective(opt):
    lat, lon = _track_between(opt, n=15, bow_deg=0.15)
    opt.follow_track(lat, lon)
    opt.mach_value = 0.76
    # A constant linear grid has a known integral and no expensive spline setup.
    grid = ca.interpolant(
        "track_test_grid",
        "linear",
        [[0.0, 15.0], [45.0, 60.0], [0.0, 15000.0]],
        [1.0] * 8,
    )
    result = opt.trajectory(
        objective="grid_cost", interpolant=grid, time_dependent=True, result_object=True
    )
    assert isinstance(result, TrajectoryResult)
    assert result.success, result.status
    np.testing.assert_allclose(result.df.mach, 0.76, atol=1e-6)
    assert result.grid_cost == pytest.approx(result.objective, rel=1e-6)
    assert result.grid_cost == pytest.approx(result.df.ts.iloc[-1], rel=1e-6)


@pytest.mark.parametrize(
    "lat, lon",
    [
        ([], []),
        ([1, 2, 3], [1, 2, 3]),
        ([[1, 2], [3, 4]], [[1, 2], [3, 4]]),
        ([1, 2, 3, 4], [1, 2, 3]),
        ([1, np.nan, 3, 4], [1, 2, 3, 4]),
        ([1, 2, 3, 4], [1, np.inf, 3, 4]),
        ([1, 91, 3, 4], [1, 2, 3, 4]),
        ([1, 2, 3, 4], [1, 181, 3, 4]),
    ],
)
def test_invalid_track_coordinates(opt, lat, lon):
    with pytest.raises(ValueError):
        opt.follow_track(lat, lon)
    assert opt.track_ref is None


@pytest.mark.parametrize("tolerance", [0, -1, np.nan, np.inf])
def test_invalid_tolerance(opt, tolerance):
    lat, lon = _track_between(opt)
    with pytest.raises(ValueError, match="tolerance_m"):
        opt.follow_track(lat, lon, tolerance_m=tolerance)


@pytest.mark.parametrize("track_first", [False, True])
def test_fixed_heading_and_track_are_mutually_exclusive(opt, track_first):
    lat, lon = _track_between(opt)
    if track_first:
        opt.follow_track(lat, lon)
        with pytest.raises(ValueError, match="cannot be combined"):
            opt.fix_track_angle()
    else:
        opt.fix_track_angle()
        with pytest.raises(ValueError, match="cannot be combined"):
            opt.follow_track(lat, lon)


def test_track_rebuilds_in_fleet_projection_and_restores_on_next_solve(opt):
    from opentop.fleet import _fleet_projection

    lat, lon = _track_between(opt, bow_deg=0.3)
    opt.follow_track(lat, lon)
    original_xy = opt.proj(lon, lat)
    flights = (top.FlightSpec("AC1", opt),)
    with _fleet_projection(flights, (48.0, 12.0)):
        opt._add_formulation(ca.Opti())
        expected_x, expected_y = opt.proj(opt.lon1, opt.lat1)
        x_ref, y_ref, _ = opt.track_ref
        assert float(x_ref(0)) == pytest.approx(expected_x, abs=1e-5)
        assert float(y_ref(0)) == pytest.approx(expected_y, abs=1e-5)
        assert abs(expected_x - original_xy[0][0]) > 1000
    opt.init_conditions()
    x_ref, y_ref, _ = opt.track_ref
    assert float(x_ref(0)) == pytest.approx(original_xy[0][0], abs=1e-5)
    assert float(y_ref(0)) == pytest.approx(original_xy[1][0], abs=1e-5)


def test_track_bounds_and_default_guess_include_large_detour(opt):
    lat, lon = _track_between(opt, bow_deg=2.0)
    opt.follow_track(lat, lon)
    opt.init_conditions()
    x_ref, y_ref, length = opt.track_ref
    samples = np.linspace(0, length, 100)
    x, y = np.asarray(x_ref(samples)), np.asarray(y_ref(samples))
    assert x.min() >= opt.x_lb[0] and x.max() <= opt.x_ub[0]
    assert y.min() >= opt.x_lb[1] and y.max() <= opt.x_ub[1]
    nodes = np.linspace(0, length, opt.nodes + 1)
    np.testing.assert_allclose(opt.x_guess[:, 0], np.asarray(x_ref(nodes)).ravel())
    np.testing.assert_allclose(opt.x_guess[:, 1], np.asarray(y_ref(nodes)).ravel())


def test_endpoint_snapping_does_not_mutate_input(opt):
    lat, lon = _track_between(opt)
    lat[0] += 0.001
    original_lat = lat.copy()
    opt.follow_track(lat, lon)
    np.testing.assert_array_equal(lat, original_lat)
    x_ref, y_ref, _ = opt.track_ref
    expected_x, expected_y = opt.proj(opt.lon1, opt.lat1)
    assert float(x_ref(0)) == pytest.approx(expected_x, abs=1e-5)
    assert float(y_ref(0)) == pytest.approx(expected_y, abs=1e-5)


@pytest.mark.parametrize("tolerance_m", [100.0, 1000.0])
def test_interior_constraints_reject_an_off_track_arc(tolerance_m):
    from types import SimpleNamespace

    from opentop._track import constrain_track

    problem = ca.Opti()
    amplitude = problem.variable()
    problem.set_initial(amplitude, 5000.0)
    # Endpoints lie on the straight reference, but the interior bows 5 km away.
    track = (
        ca.interpolant("straight_x", "linear", [[0.0, 10000.0]], [0.0, 10000.0]),
        ca.interpolant("straight_y", "linear", [[0.0, 10000.0]], [0.0, 0.0]),
        10000.0,
    )
    transcription = SimpleNamespace(
        opti=problem,
        X=[ca.vertcat(0, 0), ca.vertcat(10000, 0)],
        collocation_roots=(0.2, 0.5, 0.8),
        state_at=lambda k, tau: ca.vertcat(
            10000 * tau, 4 * tau * (1 - tau) * amplitude
        ),
    )
    constrain_track(transcription, track, tolerance_m=tolerance_m)
    constraint = np.asarray(problem.debug.value(problem.g, problem.initial()))
    upper = np.asarray(problem.debug.value(problem.ubg, problem.initial()))
    assert np.max(constraint - upper) > 1.0
    problem.set_initial(amplitude, tolerance_m / 2)
    constraint = np.asarray(problem.debug.value(problem.g, problem.initial()))
    lower = np.asarray(problem.debug.value(problem.lbg, problem.initial()))
    assert np.all(constraint <= upper + 1e-8)
    assert np.all(constraint >= lower - 1e-8)


def test_track_with_wind(opt):
    lat, lon = _track_between(opt, n=15, bow_deg=0.15)
    opt.follow_track(lat, lon)
    lon_g, lat_g, h_g, ts_g = np.meshgrid(
        [0, 7, 15], [48, 52, 55], [1000, 7000, 12000], [0, 10000, 20000], indexing="ij"
    )
    opt.enable_wind(
        pd.DataFrame(
            {
                "longitude": lon_g.ravel(),
                "latitude": lat_g.ravel(),
                "h": h_g.ravel(),
                "ts": ts_g.ravel(),
                "u": 10.0,
                "v": 0.0,
            }
        )
    )
    result = opt.trajectory(result_object=True)
    assert isinstance(result, TrajectoryResult)
    assert result.success, result.status


def test_follow_track_fleet_solve(opt):
    lat, lon = _track_between(opt, n=15, bow_deg=0.15)
    opt.follow_track(lat, lon)
    result = top.MultiAircraft(
        [top.FlightSpec("tracked", opt)], enforce_separation=False
    ).trajectory()
    assert result.success, result.stats
