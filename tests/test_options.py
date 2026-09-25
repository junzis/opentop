"""Unit tests for the _options dataclasses."""

import dataclasses
import math

import pytest

import pandas as pd
from opentop._options import TrajectoryResult, build_result


def test_trajectory_result_is_frozen():
    r = TrajectoryResult(
        df=pd.DataFrame(),
        success=True,
        status="ok",
        objective=0.0,
        iters=0,
        fuel=0.0,
        grid_cost=0.0,
        stats={},
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.success = False  # type: ignore[misc]


@pytest.mark.parametrize(
    "exact_grid_cost, expected", [(0.0, 0.0), (2.5, 2.5), (None, 9.0)]
)
def test_build_result_prefers_exact_grid_cost(exact_grid_cost, expected):
    df = pd.DataFrame({"mass": [100.0, 99.0], "grid_cost": [9.0, float("nan")]})
    result = build_result(df, {"success": True}, 3.0, exact_grid_cost)
    assert result.grid_cost == expected


def test_build_result_preserves_dataframe_and_solver_metadata():
    df = pd.DataFrame({"mass": [100.0, 97.0], "grid_cost": [float("nan")] * 2})
    stats = {"success": True, "return_status": "Solve_Succeeded", "iter_count": 7}
    result = build_result(df, stats, 12.5)
    assert result.df is df
    assert result.stats is stats
    assert (result.success, result.status, result.iters) == (True, "Solve_Succeeded", 7)
    assert result.objective == 12.5
    assert result.fuel == 3.0
    assert math.isnan(result.grid_cost)


def test_build_result_handles_rejected_solve():
    result = build_result(None, {}, float("nan"))
    assert result.df.empty
    assert not result.success
    assert result.status == ""
    assert result.iters == 0
    assert all(math.isnan(v) for v in (result.objective, result.fuel, result.grid_cost))
