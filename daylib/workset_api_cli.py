"""Console-script entrypoint for the Ursa beta analysis API."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional, Sequence

import uvicorn

from daylib.analysis_store import AnalysisStore
from daylib.atlas_result_client import AtlasResultClient
from daylib.bloom_resolver_client import BloomResolverClient
from daylib.config import get_settings
from daylib.workset_api import create_app

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

    LOGGER.info("Initializing Ursa beta analysis store")
    store = AnalysisStore()
    if args.bootstrap_tapdb:
        LOGGER.info("Bootstrapping TapDB templates if needed")
        store.bootstrap()

    bloom_client = BloomResolverClient(
        base_url=settings.bloom_base_url,
        token=settings.bloom_api_token,
        verify_ssl=settings.bloom_verify_ssl,
    )
    atlas_client = (
        AtlasResultClient(
            base_url=settings.atlas_base_url,
            api_key=settings.atlas_internal_api_key,
            verify_ssl=settings.atlas_verify_ssl,
        )
        if settings.atlas_internal_api_key
        else None
    )
    app = create_app(
        store,
        bloom_client=bloom_client,
        atlas_client=atlas_client,
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
