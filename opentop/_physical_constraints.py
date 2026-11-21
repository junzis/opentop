"""Named physical inequalities, retaining native solver units and conditioning."""

from dataclasses import dataclass
from typing import Any

import casadi as ca
import openap.casadi as oc
from openap.aero import ft, kts


@dataclass(frozen=True)
class ConstraintRow:
    name: str
    raw_unit: str
    audit_unit: str
    lhs: Any
    rhs: Any
    native_buffer_scale: Any
    audit_residual: Any
    audit_atol: float = 1e-7
    buffer: float = 0.0

    def constrain(self, opti):
        # Preserve the exact unbuffered native inequality and its units.
        if isinstance(self.buffer, (int, float)) and self.buffer == 0:
            opti.subject_to(self.lhs >= self.rhs)
        else:
            opti.subject_to(
                self.lhs >= self.rhs + self.buffer * self.native_buffer_scale
            )


def clean_rows(
    optimizer,
    mass,
    tas,
    altitude,
    thrust,
    *,
    drag_margin=0.95,
    lift_drag_margin=0.9,
    buffer=0.0,
):
    drag = optimizer.drag.clean(mass, tas, altitude, dT=optimizer.dT)
    lift_drag = optimizer._lift_margin_drag(mass, tas, altitude)
    scale = ca.fmax(1, thrust)
    rows = (
        ConstraintRow(
            "thrust95_drag",
            "N",
            "1",
            thrust * drag_margin,
            drag,
            scale,
            (thrust * drag_margin - drag) / scale,
            buffer=buffer,
        ),
        ConstraintRow(
            "thrust90_lift_drag",
            "N",
            "1",
            thrust * lift_drag_margin,
            lift_drag,
            scale,
            (thrust * lift_drag_margin - lift_drag) / scale,
            buffer=buffer,
        ),
    )
    return rows, drag


def energy_row(thrust, drag, speed, mass, vertical_speed, *, buffer=0.0):
    # Native W algebra is intentionally not replaced by the audit force form.
    lhs = (thrust - drag) * speed - mass * oc.aero.g0 * vertical_speed
    scale = ca.fmax(1, thrust)
    audit = (thrust - drag - mass * oc.aero.g0 * vertical_speed / speed) / scale
    return ConstraintRow(
        "energy", "W", "1", lhs, 0, scale * speed, audit, buffer=buffer
    )


def fuel_row(optimizer, derivative):
    flow = -derivative[3]
    minimum = optimizer.BADA4_MIN_FUELFLOW_KG_S
    return ConstraintRow(
        "fuel_domain", "kg/s", "kg/s", flow, minimum, 1, flow - minimum, buffer=0.0
    )


def physical_rows(
    optimizer, state, control, *, buffer=0.0, energy=True, thrust_kind="climb"
):
    speed = oc.aero.mach2tas(control[0], state[2], dT=optimizer.dT)
    tas, altitude = speed / kts, state[2] / ft
    thrust = (
        optimizer.thrust.cruise(tas, altitude, dT=optimizer.dT)
        if thrust_kind == "cruise"
        else optimizer._thrust_climb(tas, altitude)
    )
    clean, drag = clean_rows(optimizer, state[3], tas, altitude, thrust, buffer=buffer)
    rows = list(clean)
    if energy:
        rows.append(
            energy_row(thrust, drag, speed, state[3], control[1], buffer=buffer)
        )
    if optimizer.performance_model == "bada4":
        derivative, _ = optimizer.func_dynamics(state, control, 1.0)
        rows.append(fuel_row(optimizer, derivative))
    return rows
