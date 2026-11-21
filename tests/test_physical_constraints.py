"""Units and native algebra of named physical constraint rows."""

import casadi as ca
import openap.casadi as oc
from openap.aero import ft, kts

import numpy as np
import opentop as top
from opentop._physical_constraints import energy_row, physical_rows


def test_named_rows_preserve_audit_and_native_algebra():
    optimizer = top.Cruise("A320", "EHAM", "LGAV", m0=70000)
    optimizer.setup(nodes=3)
    x, u = ca.MX.sym("x", 5), ca.MX.sym("u", 3)
    rows = physical_rows(optimizer, x, u)
    v = oc.aero.mach2tas(u[0], x[2], dT=optimizer.dT)
    thrust = optimizer._thrust_climb(v / kts, x[2] / ft)
    drag = optimizer.drag.clean(x[3], v / kts, x[2] / ft, dT=optimizer.dT)
    lift = optimizer._lift_margin_drag(x[3], v / kts, x[2] / ft)
    old_audit = ca.vertcat(
        0.95 * thrust - drag,
        0.9 * thrust - lift,
        thrust - drag - x[3] * oc.aero.g0 * u[1] / v,
    ) / ca.fmax(1, thrust)
    old_native = ca.vertcat(
        0.95 * thrust - drag,
        0.9 * thrust - lift,
        (thrust - drag) * v - x[3] * oc.aero.g0 * u[1],
    )
    f = ca.Function(
        "parity",
        [x, u],
        [
            ca.vertcat(*(r.audit_residual for r in rows)) - old_audit,
            ca.vertcat(*(r.lhs - r.rhs for r in rows)) - old_native,
        ],
    )
    for h in (1000, 8000, 11000):
        for rate in (-5, 0, 5):
            audit, native = f([0, 0, h, 65000, 0], [0.7, rate, 0])
            np.testing.assert_array_equal(audit, 0)
            np.testing.assert_array_equal(native, 0)
    assert [r.raw_unit for r in rows] == ["N", "N", "W"]


def test_energy_buffer_uses_watts_scale():
    r = energy_row(40000.0, 30000.0, 220.0, 65000.0, 2.0, buffer=1e-4)
    assert float(r.native_buffer_scale) == 40000 * 220
    assert r.buffer == 1e-4


def test_delayed_buffer_parameter_changes_bounds_without_new_rows():
    from opentop._physical_constraints import ConstraintRow

    opti = ca.Opti()
    x = opti.variable()
    b = opti.parameter()
    row = ConstraintRow("force", "N", "1", x, 20.0, 100.0, (x - 20) / 100, buffer=b)
    row.constrain(opti)
    opti.set_initial(x, 30.0)
    opti.set_value(b, 0.0)
    first = float(opti.debug.value(opti.g - opti.lbg, opti.initial()))
    opti.set_value(b, 1e-4)
    second = float(opti.debug.value(opti.g - opti.lbg, opti.initial()))
    assert opti.ng == 1 and opti.nx == 1 and opti.np == 1
    assert abs(abs(second - first) - 0.01) < 1e-12
