from importlib.metadata import version

from . import tools, vis
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

__all__ = [
    "Base",
    "Climb",
    "CompleteFlight",
    "Cruise",
    "Descent",
    "FlightSpec",
    "MultiAircraft",
    "MultiAircraftResult",
    "PairSeparationReport",
    "SeparationConfig",
    "tools",
    "vis",
]

__version__ = version("opentop")
