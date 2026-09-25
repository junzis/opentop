"""Tests for Cruise(mach_value=...) — pinning the cruise Mach to a given value."""

from typing import cast

import numpy as np
import opentop as top
import pandas as pd


def _cruise(aircraft_type, short_flight, **kwargs):
    return top.Cruise(
        aircraft_type,
        short_flight["origin"],
        short_flight["destination"],
        short_flight["m0"],
        **kwargs,
    )


def test_mach_value_none_by_default(aircraft_type, short_flight):
    assert _cruise(aircraft_type, short_flight).mach_value is None


def test_trajectory_holds_given_mach(aircraft_type, short_flight):
    opt = _cruise(aircraft_type, short_flight, mach_value=0.76)
    opt.fix_mach_number()
    df = opt.trajectory(objective="fuel")

    assert opt.success, opt.stats
    assert isinstance(df, pd.DataFrame)
    df = cast(pd.DataFrame, df)
    assert len(df) > 0
    np.testing.assert_allclose(df.mach.to_numpy(), 0.76, atol=1e-4)
