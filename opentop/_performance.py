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
            fuelflow=bada3.FuelFlow(
                actype,
                bada_path=bada_path,
                backend=backend,
                smooth=symbolic,
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
        return openap.FuelFlow(
            actype,
            engtype,
            use_synonym=use_synonym,
            force_engine=True,
        )
    if bada_path is None:
        raise ValueError("bada_path is required when performance_model is BADA")
    if name == "bada3":
        from openap.addon import bada3

        return bada3.FuelFlow(actype, bada_path=bada_path)

    from openap.addon import bada4

    return bada4.FuelFlow(actype, bada_path=bada_path)
