"""Integration: time-dependent (4D) grid cost on a short cruise.

Uses a cached bspline interpolant under tests/fixtures/ so this test re-runs
in ~1s plus solver time.
"""

from pathlib import Path

import pytest

import opentop as top
import opentop.tools as tools

FIXTURE = Path(__file__).parent / "fixtures" / "contrail_4d.casadi"


@pytest.fixture(scope="module")
def interp_4d():
    if not FIXTURE.exists():
        pytest.skip(
            f"{FIXTURE} not found; "
            "copy the real ERA5/contrail interpolant to "
            "tests/fixtures/contrail_4d.casadi"
        )
    return tools.load_interpolant(str(FIXTURE))


def test_cruise_with_4d_grid_cost_converges(interp_4d):
    opt = top.Cruise("A320", (52.362, 13.501), (40.472, -3.563), m0=0.85)
    opt.setup(max_iter=800)

    def blended(x, u, dt, **kwargs):
        grid = opt.obj_grid_cost(
            x,
            u,
            dt,
            interpolant=kwargs["interpolant"],
            time_dependent=True,
        )
        return grid + opt.obj_fuel(x, u, dt)

    df = opt.trajectory(
        objective=blended,
        interpolant=interp_4d,
        time_dependent=True,
    )
    assert df is not None, "trajectory returned None"
    assert opt.success, f"solver failed: {opt.stats}"
    assert "grid_cost" in df.columns, "grid_cost column missing from trajectory"  # type: ignore[union-attr]  # trajectory() without result_object always returns DataFrame


def test_reported_grid_cost_is_the_minimised_quadrature(interp_4d):
    """Reported grid cost equals the physical objective minimized by IPOPT."""
    opt = top.Cruise("A320", (52.362, 13.501), (40.472, -3.563), m0=0.85)
    opt.setup(max_iter=800)

    def pure_grid(x, u, dt, **kwargs):
        return opt.obj_grid_cost(
            x, u, dt, interpolant=kwargs["interpolant"], time_dependent=True
        )

    result = opt.trajectory(
        objective=pure_grid,
        interpolant=interp_4d,
        time_dependent=True,
        result_object=True,
    )
    assert result.success, f"solver failed: {result.status}"
    assert result.grid_cost == pytest.approx(result.objective, rel=1e-6)
    assert opt.grid_cost_value == pytest.approx(result.objective, rel=1e-6)
