from importlib.metadata import version

from . import rad, tools, vis
from .base import Base
from .climb import Climb
from .cruise import Cruise
from .descent import Descent
from .fleet import (
    FlightSpec,
    MultiAircraft,
    MultiAircraftResult,
    PairSeparationReport,
    SeparationConfig,
)
from .full import CompleteFlight
from .routes import (
    OptimizedRouteOption,
    RouteChoiceResult,
    RouteNetwork,
    RouteNetworkSelection,
    RouteOptimizationConfig,
    RouteOption,
    optimize_routes,
)
from .routing import RouteSelectionConfig

__all__ = [
    "Base",
    "Climb",
    "CompleteFlight",
    "Cruise",
    "Descent",
    "FlightSpec",
    "MultiAircraft",
    "MultiAircraftResult",
    "OptimizedRouteOption",
    "PairSeparationReport",
    "RouteChoiceResult",
    "RouteNetwork",
    "RouteNetworkSelection",
    "RouteOptimizationConfig",
    "RouteOption",
    "RouteSelectionConfig",
    "SeparationConfig",
    "optimize_routes",
    "rad",
    "tools",
    "vis",
]

__version__ = version("opentop")
