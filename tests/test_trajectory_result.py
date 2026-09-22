"""Integration coverage for structured solver results."""

import math

import pytest

import opentop as top
import pandas as pd
from opentop._options import TrajectoryResult


def _fast_cruise():
    opt = top.Cruise("A320", (52.308, 4.764), (50.033, 8.570), m0=0.85)
    opt.setup(max_iter=500)
    return opt


def test_trajectory_returns_result_object_when_flag_set():
    opt = _fast_cruise()
    r = opt.trajectory(objective="fuel", result_object=True)
    assert isinstance(r, TrajectoryResult)
    assert isinstance(r.df, pd.DataFrame)
    assert len(r.df) > 0
    assert r.success is True
    assert math.isfinite(r.objective)
    assert r.iters > 0
    assert math.isfinite(r.fuel)
    assert r.fuel > 0
    # No interpolant was passed → grid_cost should be NaN.
    assert math.isnan(r.grid_cost)
    assert isinstance(r.stats, dict)
    assert "iter_count" in r.stats or "success" in r.stats


def test_stats_before_solve_raises_runtime_error():
    opt = top.Cruise("A320", (52.308, 4.764), (50.033, 8.570), m0=0.85)
    with pytest.raises(RuntimeError, match="call trajectory"):
        _ = opt.stats


def test_success_before_solve_raises_runtime_error():
    opt = top.Cruise("A320", (52.308, 4.764), (50.033, 8.570), m0=0.85)
    with pytest.raises(RuntimeError, match="call trajectory"):
        _ = opt.success
