"""Daylily Ursa API package exports."""

from importlib.metadata import PackageNotFoundError, version as package_version
from typing import Any

from daylib_ursa.analysis_store import AnalysisState, AnalysisStore, ReviewState
from daylib_ursa.bloom_resolver_client import BloomResolverClient
from daylib_ursa.integrations.dewey_client import DeweyClient

try:
    __version__ = package_version("daylily-ursa")
except PackageNotFoundError:
    __version__ = "0.0.0"


def create_app(*args: Any, **kwargs: Any):
    from daylib_ursa.workset_api import create_app as _create_app

    return _create_app(*args, **kwargs)


__all__ = [
    "__version__",
    "AnalysisStore",
    "AnalysisState",
    "ReviewState",
    "BloomResolverClient",
    "DeweyClient",
    "create_app",
]
