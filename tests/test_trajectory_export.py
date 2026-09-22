import numpy as np
import opentop as top


def test_to_trajectory_preserves_independent_terminal_control():
    opt = top.Cruise("A320", "EHAM", "EDDF", 0.85)
    opt.nodes = 2
    states = np.array(
        [[0, 1000, 2000], [0, 0, 0], [9000] * 3, [65000, 64900, 64800], [0, 50, 100]],
        dtype=float,
    )
    controls = np.array([[0.7, 0.72, 0.74], [1, 0.5, 0], [0, 0.1, 0.2]])
    df = opt.to_trajectory(100, states, controls)
    np.testing.assert_allclose(opt.U, controls)
    assert len(df) == 3
    assert df.mach.iloc[-1] == 0.74
    assert df.vertical_rate.iloc[-1] == 0
    assert df.fuel_cost.sum() == 200
