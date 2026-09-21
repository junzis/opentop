from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import openap
import openap.casadi as oc
from openap.backends import CasadiBackend, NumpyBackend

PerformanceModelName = Literal["openap", "bada3", "bada4"]


@dataclass(frozen=True)
class PerformanceModels:
    name: PerformanceModelName
    aircraft: dict[str, Any]
    engtype: str
    engine: dict[str, Any] | None
    thrust: Any
    drag: Any
    fuelflow: Any
    emission: Any


def normalize_model_name(name: str) -> PerformanceModelName:
    key = name.lower()
    if key not in {"openap", "bada3", "bada4"}:
        raise ValueError(
            "performance_model must be one of 'openap', 'bada3', or 'bada4'"
        )
    return key  # type: ignore[return-value]


def _native_actype(actype: str) -> str:
    return actype.split("-", 1)[0]


class _Bada3EnrouteFuelFlow:
    """Enforce BADA3's idle floor in the returned kg/s units.

    OpenAP 2.6.0's nominal method compares an idle value in kg/s with a
    nominal value in kg/min before dividing by 60. Apply the floor after
    conversion so descent cannot exploit the resulting low fuel flow.
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)

    def enroute(
        self,
        mass: Any,
        tas: Any,
        alt: Any,
        vs: Any = 0,
        acc: Any = None,
        dT: Any = 0,
    ) -> Any:
        nominal = self.model.enroute(mass, tas, alt, vs, acc=acc, dT=dT)
        idle = self.model.idle(mass, tas, alt, vs)
        if self.model.smooth:
            # Keep the idle transition smooth in kg/s for the full-flight NLP.
            # The maximum smoothing increment at equality is 0.0005 kg/s.
            return self.model.backend.smooth_max(nominal, idle, softness=1e-3)
        return self.model.backend.maximum(nominal, idle)


def build_performance_models(
    actype: str,
    *,
    engine: str | None,
    use_synonym: bool,
    performance_model: str = "openap",
    bada_path: str | None = None,
    symbolic: bool = True,
) -> PerformanceModels:
    name = normalize_model_name(performance_model)
    backend = CasadiBackend() if symbolic else NumpyBackend()

    if name == "openap":
        aircraft = oc.prop.aircraft(actype, use_synonym=use_synonym)
        engtype = engine or aircraft["engine"]["default"]
        engine_data = oc.prop.engine(engtype)
        force_engine = engine is not None
        # OpenAP emits this warning once for Drag and again for FuelFlow on
        # every optimizer construction. Wave drag remains intentionally
        # enabled; only its repetitive experimental-status warning is hidden.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Warning: Wave drag is experimental\.",
                category=UserWarning,
                module=r"openap\.drag",
            )
            drag = oc.Drag(actype, wave_drag=True, use_synonym=use_synonym)
            fuelflow = oc.FuelFlow(
                actype,
                engtype,
                wave_drag=True,
                use_synonym=use_synonym,
                force_engine=force_engine,
            )
        return PerformanceModels(
            name=name,
            aircraft=aircraft,
            engtype=engtype,
            engine=engine_data,
            thrust=oc.Thrust(
                actype,
                engtype,
                use_synonym=use_synonym,
                force_engine=force_engine,
            ),
            drag=drag,
            fuelflow=fuelflow,
            emission=oc.Emission(actype, engtype, use_synonym=use_synonym),
        )

    if bada_path is None:
        raise ValueError("bada_path is required when performance_model is BADA")

    native_actype = _native_actype(actype)
    aircraft = oc.prop.aircraft(native_actype, use_synonym=use_synonym)
    engtype = engine or aircraft["engine"]["default"]
    emission = oc.Emission(native_actype, engtype, use_synonym=use_synonym)

    if name == "bada3":
        from openap.addon import bada3

        return PerformanceModels(
            name=name,
            aircraft=aircraft,
            engtype=engtype,
            engine=None,
            thrust=bada3.Thrust(
                actype,
                bada_path=bada_path,
                backend=backend,
                smooth=symbolic,
            ),
            drag=bada3.Drag(
                actype,
                bada_path=bada_path,
                backend=backend,
                smooth=symbolic,
            ),
            fuelflow=_Bada3EnrouteFuelFlow(
                bada3.FuelFlow(
                    actype,
                    bada_path=bada_path,
                    backend=backend,
                    smooth=symbolic,
                )
            ),
            emission=emission,
        )

    from openap.addon import bada4

    return PerformanceModels(
        name=name,
        aircraft=aircraft,
        engtype=engtype,
        engine=None,
        thrust=bada4.Thrust(
            actype,
            bada_path=bada_path,
            backend=backend,
            smooth=symbolic,
        ),
        drag=bada4.Drag(
            actype,
            bada_path=bada_path,
            backend=backend,
            smooth=symbolic,
        ),
        fuelflow=bada4.FuelFlow(
            actype,
            bada_path=bada_path,
            backend=backend,
            smooth=symbolic,
        ),
        emission=emission,
    )


@lru_cache(maxsize=None)
def build_numeric_fuelflow(
    actype: str,
    *,
    engtype: str,
    use_synonym: bool,
    performance_model: str,
    bada_path: str | None,
) -> Any:
    name = normalize_model_name(performance_model)
    if name == "openap":
        # Numeric reporting/objectives must use the same wave-drag setting
        # as build_performance_models, especially near the cruise Mach limit.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Warning: Wave drag is experimental\.",
                category=UserWarning,
                module=r"openap\.drag",
            )
            return openap.FuelFlow(
                actype,
                engtype,
                use_synonym=use_synonym,
                force_engine=True,
                wave_drag=True,
            )
    if bada_path is None:
        raise ValueError("bada_path is required when performance_model is BADA")
    if name == "bada3":
        from openap.addon import bada3

        return _Bada3EnrouteFuelFlow(bada3.FuelFlow(actype, bada_path=bada_path))

    from openap.addon import bada4

    return bada4.FuelFlow(actype, bada_path=bada_path)
