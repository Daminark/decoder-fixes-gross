"""Shared pytest configuration: the --runslow option and the Hypothesis CI profile."""

import os

import pytest
from hypothesis import settings

# Starting points; tune with `pytest --durations=10`.
settings.register_profile("dev", max_examples=20, deadline=None)
settings.register_profile("ci", max_examples=200, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow test; use --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
