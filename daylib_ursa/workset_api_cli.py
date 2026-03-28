"""Console-script entrypoint for the Ursa beta analysis API."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional, Sequence

import uvicorn

from daylib_ursa.analysis_store import AnalysisStore
from daylib_ursa.atlas_result_client import AtlasResultClient
from daylib_ursa.bloom_resolver_client import BloomResolverClient
from daylib_ursa.config import get_settings
from daylib_ursa.dewey_client import DeweyClient
from daylib_ursa.workset_api import create_app

LOGGER = logging.getLogger("daylily.analysis_api.cli")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch Daylily Ursa beta analysis API",
    )
    parser.add_argument("--region", default="us-west-2", help="AWS region")
    parser.add_argument("--profile", help="AWS profile name")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8914, help="Port to bind to")
    parser.add_argument(
        "--bootstrap-tapdb",
        action="store_true",
        default=True,
        help="Bootstrap TapDB templates if needed (default: True)",
    )
    parser.add_argument(
        "--no-bootstrap-tapdb",
        action="store_false",
        dest="bootstrap_tapdb",
        help="Don't bootstrap TapDB templates automatically",
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--ssl-certfile", default=None, help="Path to TLS certificate file (PEM)")
    parser.add_argument("--ssl-keyfile", default=None, help="Path to TLS private key file (PEM)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    settings = get_settings()
    bloom_token = str(settings.bloom_api_token or "").strip()
    if not bloom_token:
        raise ValueError("BLOOM_API_TOKEN is required for authenticated Ursa->Bloom integration")
    atlas_api_key = str(settings.atlas_internal_api_key or "").strip()
    if not atlas_api_key:
        raise ValueError(
            "ATLAS_INTERNAL_API_KEY is required for authenticated Ursa->Atlas integration"
        )

    LOGGER.info("Initializing Ursa beta analysis store")
    store = AnalysisStore()
    if args.bootstrap_tapdb:
        LOGGER.info("Bootstrapping TapDB templates if needed")
        store.bootstrap()

    bloom_client = BloomResolverClient(
        base_url=settings.bloom_base_url,
        token=bloom_token,
        verify_ssl=settings.bloom_verify_ssl,
    )
    atlas_client = AtlasResultClient(
        base_url=settings.atlas_base_url,
        api_key=atlas_api_key,
        verify_ssl=settings.atlas_verify_ssl,
    )
    dewey_client = None
    if bool(getattr(settings, "dewey_enabled", False)):
        dewey_client = DeweyClient(
            base_url=str(getattr(settings, "dewey_base_url", "")),
            token=str(getattr(settings, "dewey_api_token", "") or "").strip(),
            verify_ssl=bool(getattr(settings, "dewey_verify_ssl", True)),
            timeout_seconds=float(getattr(settings, "dewey_timeout_seconds", 10.0)),
        )
    app = create_app(
        store,
        bloom_client=bloom_client,
        atlas_client=atlas_client,
        dewey_client=dewey_client,
        settings=settings,
    )

    LOGGER.info("Starting Ursa beta analysis API on %s:%d", args.host, args.port)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        ssl_certfile=args.ssl_certfile,
        ssl_keyfile=args.ssl_keyfile,
        log_level="debug" if args.verbose else "info",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
