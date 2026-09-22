"""Structured options for trajectory optimization and result packaging.

These dataclasses replace the **kwargs dict plumbing in Base.trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import casadi as ca

import pandas as pd


@dataclass(frozen=True, slots=True)
class SolveOptions:
    """Options consumed by Base._run / Base._build_opti / Base._solve."""

    max_iter: int = 1500
    max_fuel: Optional[float] = None
    auto_rescale_objective: bool = True
    return_failed: bool = False
    initial_guess: Optional[pd.DataFrame] = None
    alt_start: Optional[float] = None
    alt_stop: Optional[float] = None
    remove_cruise: bool = False
    exact_hessian: bool = False


@dataclass(frozen=True, slots=True)
class GridOptions:
    """Options for grid-cost objectives."""

    interpolant: Optional[ca.Function] = None
    n_dim: int = 3
    time_dependent: bool = False


@dataclass(frozen=True, slots=True)
class TrajectoryResult:
    """Rich result of a trajectory optimization.

    Returned when ``trajectory(result_object=True)``."""

    df: pd.DataFrame
    success: bool
    status: str
    objective: float
    iters: int
    fuel: float
    grid_cost: float  # NaN if no interpolant
    stats: dict[str, Any]


def build_result(
    df: pd.DataFrame | None,
    stats: dict,
    objective: float,
    exact_grid_cost: float | None = None,
) -> TrajectoryResult:
    """Package a DataFrame + solver stats + objective into a TrajectoryResult.

    Used by Base._make_result after a trajectory() call. ``df`` may be None
    (from a rejected solve); it is coerced to an empty DataFrame in the result.
    """
    result_df = df if df is not None else pd.DataFrame()
    has_df = len(result_df) > 0
    fuel = (
        float(result_df["mass"].iloc[0] - result_df["mass"].iloc[-1])
        if has_df and "mass" in result_df.columns
        else float("nan")
    )
    has_grid_cost = (
        has_df
        and "grid_cost" in result_df.columns
        and bool(result_df["grid_cost"].notna().any())
    )
    # Prefer the solver's own collocation quadrature when the solve recorded
    # one. Summing the per-node column is a left-endpoint rectangle rule and is
    # not the quantity IPOPT minimised.
    if exact_grid_cost is not None:
        grid_cost = float(exact_grid_cost)
    elif has_grid_cost:
        grid_cost = float(result_df["grid_cost"].sum(skipna=True))
    else:
        grid_cost = float("nan")
    return TrajectoryResult(
        df=result_df,
        success=bool(stats.get("success", False)),
        status=str(stats.get("return_status", "")),
        objective=objective,
        iters=int(stats.get("iter_count", 0)),
        fuel=fuel,
        grid_cost=grid_cost,
        stats=stats,
    )
