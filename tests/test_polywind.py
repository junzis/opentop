"""Pin PolyWind numeric and symbolic output. Regression guard for the eval() rewrite."""

import casadi as ca
import pytest

import numpy as np
import opentop.tools as tools
import pandas as pd
import pyproj


def _fake_wind_df():
    """Synthetic wind field: u = 5 + 0.01*h, v = 2 - 0.001*ts."""
    rows = []
    for lon in np.linspace(0, 10, 4):
        for lat in np.linspace(45, 55, 4):
            for h in (1000, 5000, 10000):
                for ts in (0, 3600):
                    rows.append(
                        {
                            "longitude": lon,
                            "latitude": lat,
                            "h": h,
                            "ts": ts,
                            "u": 5 + 0.01 * h,
                            "v": 2 - 0.001 * ts,
                        }
                    )
    return pd.DataFrame(rows)


def _proj():
    return pyproj.Proj(proj="lcc", lat_1=46, lat_2=54, lat_0=50, lon_0=5)


@pytest.fixture(scope="module")
def wind():
    proj = _proj()
    return tools.PolyWind(_fake_wind_df(), proj, 46.0, 1.0, 54.0, 9.0), proj


def test_polywind_numeric_output_sane(wind):
    w, proj = wind
    x, y = proj(5.0, 50.0)
    # Ridge regularization means the fitted field approximates the input.
    assert 20 < float(w.calc_u(x, y, 5000, 1800)) < 90
    assert -10 < float(w.calc_v(x, y, 5000, 1800)) < 10


@pytest.mark.parametrize("component", ["calc_u", "calc_v"])
@pytest.mark.parametrize(
    "point",
    [
        (5.0, 50.0, 5000.0, 1800.0),
        (2.0, 47.0, 2000.0, 0.0),
        (8.0, 53.0, 9000.0, 3600.0),
    ],
)
@pytest.mark.parametrize("symbol_type", [ca.SX, ca.MX])
def test_polywind_numeric_matches_symbolic_evaluation(
    wind, component, point, symbol_type
):
    w, proj = wind
    lon, lat, h, ts = point
    x, y = proj(lon, lat)
    calc = getattr(w, component)
    symbols = [symbol_type.sym(name) for name in ("x", "y", "h", "ts")]
    f = ca.Function("wind", symbols, [calc(*symbols)])
    evaluated = f(x, y, h, ts)
    assert isinstance(evaluated, ca.DM)
    assert float(evaluated) == pytest.approx(float(calc(x, y, h, ts)), abs=1e-6)
