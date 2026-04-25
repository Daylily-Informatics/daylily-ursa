"""Pytest configuration and shared fixtures."""

from importlib import metadata as importlib_metadata
import os

import pytest

# Set WHITELIST_DOMAINS to the default base allowlist for tests.
# This must be set before importing any daylib_ursa modules
os.environ.setdefault(
    "WHITELIST_DOMAINS",
    "lsmc.com,lsmc.bio,lsmc.life,daylilyinformatics.com",
)


@pytest.fixture(autouse=True)
def _installed_dayec_211_for_unit_tests(monkeypatch):
    """Model the declared day-ec dependency in tests that construct the app."""

    from daylib_ursa.ephemeral_cluster import runner

    real_version = importlib_metadata.version

    def fake_version(distribution_name: str) -> str:
        if distribution_name == runner.DAYLILY_EC_DISTRIBUTION:
            return runner.REQUIRED_DAYLILY_EC_VERSION
        return real_version(distribution_name)

    monkeypatch.setattr(runner.importlib_metadata, "version", fake_version)
