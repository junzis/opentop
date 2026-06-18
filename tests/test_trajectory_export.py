from openap.aero import fpm, ft

import numpy as np
import opentop as top


def test_to_trajectory_repeats_final_interval_control():
    opt = top.Cruise("A320", "EHAM", "EDDF", 0.85)
    opt.nodes = 2

    x_opt = np.array(
        [
            [0.0, 1_000.0, 2_000.0],
            [0.0, 0.0, 0.0],
            [30_000 * ft, 30_000 * ft, 30_000 * ft],
            [70_000.0, 69_900.0, 69_800.0],
            [0.0, 50.0, 100.0],
        ]
    )
    u_opt = np.array(
        [
            [0.5, 0.3],
            [0.0, -500 * fpm],
            [0.0, 0.0],
        ]
    )

    df = opt.to_trajectory(100.0, x_opt, u_opt)

    assert df.mach.iloc[-1] == 0.3
    assert df.vertical_rate.iloc[-1] == -500
