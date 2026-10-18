"""Tests for route constraint thinning before problem construction."""

import pytest

import numpy as np
import opentop as top


def route():
    opt = top.CompleteFlight("A320", "EHAM", "LIRF", 0.85)
    opt.setup(nodes=20)
    opt._projection_center = ((opt.lat1 + opt.lat2) / 2, (opt.lon1 + opt.lon2) / 2)
    x, y = opt.proj(opt.lon1, opt.lat1)

    def point(distance):
        lon, lat = opt.proj(x + distance, y, inverse=True)
        return float(lat), float(lon)

    opt.lat2, opt.lon2 = point(100_000)
    return opt, [point(d) for d in (1_000, 25_000, 30_000, 50_000, 85_000, 95_000)]


def test_spacing_preserves_endpoints_and_explicit_assignments():
    opt, points = route()
    original = list(points)
    indices = [1, 3, 6, 9, 12, 15]
    filtered, mapped = opt._filter_waypoints(points, indices, 20_000)
    assert filtered == [points[1], points[3]]
    assert mapped == [3, 9]
    assert opt.retained_waypoint_positions == [1, 3]
    assert opt.skipped_waypoints == [points[i] for i in (0, 2, 4, 5)]
    assert points == original
    assert indices == [1, 3, 6, 9, 12, 15]
    assert opt.nodes == 20


def test_disabled_filter_preserves_route_and_clears_previous_skips():
    opt, points = route()
    opt._filter_waypoints(points, None, 20_000)
    filtered, indices = opt._filter_waypoints(points, None, 0)
    assert filtered == points
    assert indices is None
    assert opt.skipped_waypoints == []


@pytest.mark.parametrize("spacing", [-1, np.nan, np.inf])
def test_invalid_spacing_rejected(spacing):
    opt, points = route()
    with pytest.raises(ValueError, match="finite and nonnegative"):
        opt._filter_waypoints(points, None, spacing)


@pytest.mark.parametrize("indices", [[1], [1, 3, 6, 9, 12, 20], [1, 3, 6, 9, 12, 12]])
def test_invalid_assignments_not_hidden_by_filter(indices):
    opt, points = route()
    with pytest.raises(ValueError):
        opt._filter_waypoints(points, indices, 200_000)


def test_all_waypoints_removed_and_none_preserved():
    opt, points = route()
    assert opt._filter_waypoints(points, list(range(1, 7)), 200_000) == ([], [])
    assert opt._filter_waypoints(None, None, 20_000) == (None, None)
    assert opt.active_waypoints == []


@pytest.mark.parametrize(
    "optimizer", [top.CompleteFlight, top.Climb, top.Cruise, top.Descent]
)
def test_public_trajectory_filters_before_initialization(monkeypatch, optimizer):
    opt = optimizer("A320", "EHAM", "LIRF", 0.85)
    calls = []

    class Prepared(Exception):
        pass

    def prepare(waypoints, indices, spacing):
        calls.append((waypoints, indices, spacing))
        raise Prepared

    monkeypatch.setattr(opt, "_filter_waypoints", prepare)
    with pytest.raises(Prepared):
        opt.trajectory(waypoints=[(52.0, 5.0)], waypoint_min_spacing_m=20_000)
    assert calls == [([(52.0, 5.0)], None, 20_000)]


def test_point_at_spacing_threshold_is_retained():
    opt, points = route()
    origin = opt.proj(opt.lon1, opt.lat1)
    position = opt.proj(points[1][1], points[1][0])
    spacing = float(np.hypot(position[0] - origin[0], position[1] - origin[1]))
    assert opt._filter_waypoints([points[1]], None, spacing) == ([points[1]], None)
