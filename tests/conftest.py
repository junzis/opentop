"""Shared fixtures for opentop tests."""

import importlib
import importlib.util
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest  # noqa: E402  (sys.path must be set before importing project packages)


@pytest.fixture(scope="session")
def aircraft_type():
    return "A320"


@pytest.fixture(scope="session")
def short_flight():
    return {"origin": "EHAM", "destination": "EDDF", "m0": 0.85}


@pytest.fixture(scope="session")
def medium_flight():
    return {"origin": "EHAM", "destination": "LGAV", "m0": 0.85}


@pytest.fixture(scope="module")
def traffic_data():
    # Missing optional packages may skip; broken installed packages must fail.
    if importlib.util.find_spec("traffic") is None:
        pytest.skip("traffic is not installed")
    return importlib.import_module("traffic.data")
