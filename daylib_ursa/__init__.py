"""Daylily Ursa API package exports."""

from importlib.metadata import PackageNotFoundError, version as package_version

from daylib_ursa.analysis_store import AnalysisState, AnalysisStore, ReviewState
from daylib_ursa.bloom_resolver_client import BloomResolverClient
from daylib_ursa.dewey_client import DeweyClient
from daylib_ursa.workset_api import create_app

try:
    __version__ = package_version("daylily-ursa")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "AnalysisStore",
    "AnalysisState",
    "ReviewState",
    "BloomResolverClient",
    "DeweyClient",
    "create_app",
]
