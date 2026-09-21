"""Experimental Acropole fuel flow through CasADi's symbolic ONNX importer.

Model weights and aircraft parameters remain external, user-supplied artifacts.
The adapter follows DGAC/Acropole's 12-feature preprocessing contract.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import casadi as ca
from openap.aero import fpm, ft, kts
from openap.casadi import aero

import numpy as np


def _evaluate(function: Any, *values: Any) -> Any:
    """Keep scalar, array and symbolic evaluations on the same expression."""
    if any(isinstance(value, (ca.MX, ca.SX, ca.DM)) for value in values):
        return function(*values)
    arrays = np.broadcast_arrays(*[np.asarray(value, dtype=float) for value in values])
    if not arrays[0].size:
        return np.empty(arrays[0].shape)
    result = np.asarray(function(*[a.reshape(1, -1) for a in arrays]))
    result = result.reshape(arrays[0].shape)
    return float(result) if result.ndim == 0 else result


def _import_acropole(path: Path, activation_smoothing: float = 0.0) -> Any:
    if not hasattr(ca, "GraphBuilder"):
        raise ImportError("AcropoleFuelFlow requires CasADi >=3.8.1 (opentop[neural]).")
    try:
        import onnx
        from onnx import helper, numpy_helper
    except ImportError as exc:
        raise ImportError(
            "Install opentop[neural] to import Acropole ONNX models."
        ) from exc

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("Expected one Acropole input and one output.")
    shape = model.graph.input[0].type.tensor_type.shape.dim
    if len(shape) != 2 or shape[1].dim_value != 12:
        raise ValueError("Expected Acropole input shape [batch, 12].")
    # Fix batch size for a single collocation evaluation; CasADi handles mapping.
    shape[0].dim_value = 1
    output_shape = model.graph.output[0].type.tensor_type.shape.dim
    if len(output_shape) != 2 or output_shape[1].dim_value != 1:
        raise ValueError("Expected Acropole output shape [batch, 1].")
    output_shape[0].dim_value = 1

    # CasADi 3.8.1 lacks Relu/Sigmoid import. By default the rewrites preserve
    # the network, including ReLU kinks. Smoothing is an explicit approximation.
    names = {v.name for v in model.graph.initializer}
    names.update(s for node in model.graph.node for s in (*node.input, *node.output))

    def fresh(label: str) -> str:
        while label in names:
            label += "_"
        names.add(label)
        return label

    nodes = []
    for index, node in enumerate(model.graph.node):
        if node.op_type not in {"Relu", "Sigmoid"} or node.domain:
            nodes.append(node)
            continue
        constant = fresh(f"opentop_constant_{index}")
        value = 0.0 if node.op_type == "Relu" else 1.0
        model.graph.initializer.append(
            numpy_helper.from_array(np.array(value, dtype=np.float32), constant)
        )
        if node.op_type == "Relu":
            if activation_smoothing == 0:
                nodes.append(
                    helper.make_node("Max", [node.input[0], constant], node.output)
                )
            else:
                # Optional smooth ReLU: (x + sqrt(x*x + eps*eps))/2.
                # This changes predictions and is never enabled implicitly.
                eps2, half = fresh(f"eps2_{index}"), fresh(f"half_{index}")
                for name, number in ((eps2, activation_smoothing**2), (half, 0.5)):
                    model.graph.initializer.append(
                        numpy_helper.from_array(
                            np.array(number, dtype=np.float32), name
                        )
                    )
                square, total, root, positive = [
                    fresh(f"smooth_{part}_{index}")
                    for part in ("square", "total", "root", "positive")
                ]
                nodes.extend(
                    [
                        helper.make_node(
                            "Mul", [node.input[0], node.input[0]], [square]
                        ),
                        helper.make_node("Add", [square, eps2], [total]),
                        helper.make_node("Sqrt", [total], [root]),
                        helper.make_node("Add", [node.input[0], root], [positive]),
                        helper.make_node("Mul", [positive, half], node.output),
                    ]
                )
        else:
            neg, exp, den = [
                fresh(f"opentop_{s}_{index}") for s in ("neg", "exp", "den")
            ]
            nodes.extend(
                [
                    helper.make_node("Neg", [node.input[0]], [neg]),
                    helper.make_node("Exp", [neg], [exp]),
                    helper.make_node("Add", [constant, exp], [den]),
                    helper.make_node("Div", [constant, den], node.output),
                ]
            )
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    with TemporaryDirectory(prefix="opentop-onnx-") as directory:
        rewritten = Path(directory) / "acropole.onnx"
        onnx.save(model, str(rewritten))
        try:
            network = ca.GraphBuilder(str(rewritten)).create(
                "acropole_network", {"symbolic": True}
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "CasADi could not symbolically import the Acropole model; "
                "use an ONNX-enabled CasADi >=3.8.1 build and supported operators."
            ) from exc
    if network.size_in(0) != (1, 12) or network.numel_out(0) != 1:
        raise ValueError("Unexpected imported Acropole input/output dimensions.")
    return network


class AcropoleFuelFlow:
    """Acropole adapter for experimental, still-air trajectory optimization.

    Inputs to ``enroute`` match OpenAP: mass kg, TAS knots, altitude ft,
    vertical rate ft/min; output is total aircraft fuel flow kg/s.
    Speed derivatives are zero (Acropole's quasi-steady approximation).
    Groundspeed is horizontal TAS; wind integration is not supported yet.
    ReLU activations are retained by default, so derivatives are piecewise smooth.
    Positive ``activation_smoothing`` replaces each ReLU with
    ``(x + sqrt(x*x + epsilon*epsilon))/2``. This changes predictions and must
    be validated separately; epsilon is in hidden preactivation units.
    """

    def __init__(
        self,
        actype: str,
        *,
        model_path: str | Path,
        aircraft_params_path: str | Path,
        activation_smoothing: float = 0.0,
    ) -> None:
        if not np.isfinite(activation_smoothing) or activation_smoothing < 0:
            raise ValueError("activation_smoothing must be finite and nonnegative.")
        self.activation_smoothing = activation_smoothing
        self.actype = actype.upper()
        self.model_path = Path(model_path).resolve()
        self.aircraft_params_path = Path(aircraft_params_path).resolve()
        with self.aircraft_params_path.open(newline="") as stream:
            rows = [
                r for r in csv.DictReader(stream) if r["ACFT_ICAO_TYPE"] == self.actype
            ]
        if len(rows) != 1:
            raise ValueError(f"Expected one aircraft parameter row for {self.actype}.")
        fields = (
            "ENGINE_TYPE",
            "SURFACE",
            "MAX_OPE_ALTI",
            "MAX_OPE_SPEED",
            "OPE_EMPTY_WEIGHT",
            "MAX_TO_WEIGHT",
            "FUEL_FLOW_TO",
            "ENGINE_NUM",
        )
        self.params = {key: float(rows[0][key]) for key in fields}
        p = self.params
        if not all(np.isfinite(list(p.values()))):
            raise ValueError("Aircraft parameters must be finite.")
        if p["MAX_TO_WEIGHT"] <= p["OPE_EMPTY_WEIGHT"]:
            raise ValueError("MAX_TO_WEIGHT must exceed OPE_EMPTY_WEIGHT.")
        if p["FUEL_FLOW_TO"] <= 0 or p["ENGINE_NUM"] <= 0:
            raise ValueError("Fuel scale and engine count must be positive.")
        self.model_sha256 = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        self.params_sha256 = hashlib.sha256(
            self.aircraft_params_path.read_bytes()
        ).hexdigest()
        self.network = _import_acropole(self.model_path, activation_smoothing)
        mass, tas, alt, vs = [ca.MX.sym(name) for name in ("mass", "tas", "alt", "vs")]
        gs = tas * ca.cos(ca.atan2(vs * fpm, tas * kts))
        mass_norm = (mass - p["OPE_EMPTY_WEIGHT"]) / (
            p["MAX_TO_WEIGHT"] - p["OPE_EMPTY_WEIGHT"]
        )
        features = ca.vertcat(
            p["ENGINE_TYPE"],
            vs / 60,
            0,
            0,
            p["SURFACE"],
            p["MAX_OPE_ALTI"],
            p["MAX_OPE_SPEED"],
            alt,
            gs,
            tas,
            vs,
            mass_norm,
        )
        minima = ca.DM([0, -5000, -50, -50, 0, 0, 200, 0, 200, 200, -5000, 0])
        maxima = ca.DM([1, 5000, 50, 50, 600, 50000, 800, 50000, 800, 800, 5000, 1])
        normalized = (features - minima) / (maxima - minima)
        flow = self.network(normalized.T) * p["FUEL_FLOW_TO"] * p["ENGINE_NUM"]
        self.function = ca.Function(
            "acropole_fuel",
            [mass, tas, alt, vs],
            [flow],
            ["mass", "tas", "alt", "vs"],
            ["fuel_flow"],
        )

    def enroute(
        self, mass: Any, tas: Any, alt: Any, vs: Any = 0, *, dT: Any = 0
    ) -> Any:
        """Evaluate the same imported function for symbolic or numeric inputs.

        Temperature has no independent Acropole feature; opentop accounts for
        ``dT`` when converting Mach to TAS upstream. Numeric inputs broadcast.
        """
        return _evaluate(self.function, mass, tas, alt, vs)


class PhysicsCorrectedFuelFlow:
    """Optional, uncalibrated physics correction for Acropole cruise fuel.

    ``speed_increment`` retains Acropole below ``reference_mach`` and blends
    to N(M_ref) + P(M) - P(M_ref) over ``transition_width`` Mach. Mass,
    altitude, vertical rate and temperature are held fixed in the reference
    evaluations. The quintic blend has zero first/second derivatives at its
    endpoints; it does not remove ReLU kinks in the original network.

    ``wave_increment`` adds ``strength * (P_wave - P_no_wave)``. Supply the
    same physical aircraft/engine with only wave drag changed. This can
    double-count effects already learned by Acropole.

    Physical models must provide CasADi-compatible ``enroute`` in OpenAP
    units. No default transition Mach is inferred from training coverage.
    Neither correction is a validated fuel model or a force-model override.
    """

    def __init__(
        self,
        neural: AcropoleFuelFlow,
        physics_model: Any,
        *,
        method: Literal["speed_increment", "wave_increment"] = "speed_increment",
        reference_mach: float | None = None,
        transition_width: float = 0.02,
        wave_free_model: Any = None,
        strength: float = 1.0,
    ) -> None:
        if method not in {"speed_increment", "wave_increment"}:
            raise ValueError("Unknown physics correction method.")
        if not np.isfinite(strength) or strength < 0:
            raise ValueError("strength must be finite and nonnegative.")
        if method == "speed_increment":
            if (
                reference_mach is None
                or not np.isfinite(reference_mach)
                or reference_mach <= 0
                or not np.isfinite(transition_width)
                or transition_width <= 0
                or reference_mach + transition_width >= 1
            ):
                raise ValueError(
                    "Provide positive reference_mach and transition_width "
                    "with their sum below 1."
                )
            if wave_free_model is not None or strength != 1:
                raise ValueError(
                    "wave_free_model and strength apply to wave_increment."
                )
        elif wave_free_model is None or reference_mach is not None:
            raise ValueError(
                "wave_increment requires wave_free_model and no reference_mach."
            )

        self.actype = neural.actype
        self.neural = neural
        self.method = method
        self.reference_mach = reference_mach
        self.transition_width = transition_width
        self.strength = strength
        mass, tas, alt, vs, dT = [
            ca.MX.sym(name) for name in ("mass", "tas", "alt", "vs", "dT")
        ]
        original = neural.enroute(mass, tas, alt, vs, dT=dT)
        physical = physics_model.enroute(mass, tas, alt, vs, dT=dT)
        if method == "wave_increment":
            assert wave_free_model is not None
            no_wave = wave_free_model.enroute(mass, tas, alt, vs, dT=dT)
            corrected = original + strength * (physical - no_wave)
        else:
            mach = aero.tas2mach(tas * kts, alt * ft, dT=dT)
            reference_tas = aero.mach2tas(reference_mach, alt * ft, dT=dT) / kts
            reference_neural = neural.enroute(mass, reference_tas, alt, vs, dT=dT)
            reference_physical = physics_model.enroute(
                mass, reference_tas, alt, vs, dT=dT
            )
            t = ca.fmin(1, ca.fmax(0, (mach - reference_mach) / transition_width))
            weight = t**3 * (10 - 15 * t + 6 * t**2)
            corrected = original + weight * (
                reference_neural + physical - reference_physical - original
            )
        self.function = ca.Function(
            "physics_corrected_fuel",
            [mass, tas, alt, vs, dT],
            [corrected],
            ["mass", "tas", "alt", "vs", "dT"],
            ["fuel_flow"],
        )

    def enroute(
        self, mass: Any, tas: Any, alt: Any, vs: Any = 0, *, dT: Any = 0
    ) -> Any:
        """Total aircraft kg/s, from the same graph in dynamics and reporting."""
        return _evaluate(self.function, mass, tas, alt, vs, dT)
