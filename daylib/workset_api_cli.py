"""Console-script entrypoint for the Daylily workset API.

This module exists so `pyproject.toml` can expose a stable, importable
`daylily-workset-api` console script.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional, Sequence

import uvicorn

from daylib.workset_api import create_app
from daylib.workset_scheduler import WorksetScheduler
from daylib.workset_state_db import WorksetStateDB

LOGGER = logging.getLogger("daylily.workset_api.cli")


def _str_to_bool(val: str) -> bool:
    """Convert string to boolean, handling common truthy values."""

    return val.lower() in ("true", "1", "yes", "on")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Launch Daylily workset monitoring web API",
    )
    parser.add_argument(
        "--region",
        default="us-west-2",
        help="AWS region",
    )
    parser.add_argument(
        "--profile",
        help="AWS profile name",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8914,
        help="Port to bind to",
    )
    parser.add_argument(
        "--enable-scheduler",
        action="store_true",
        help="Enable scheduler endpoints",
    )
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
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--ssl-certfile",
        default=None,
        help="Path to TLS certificate file (PEM)",
    )
    parser.add_argument(
        "--ssl-keyfile",
        default=None,
        help="Path to TLS private key file (PEM)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


def configure_logging(verbose: bool) -> None:
    """Configure logging."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main entry point."""

    from daylib.config import get_settings

    args = parse_args(argv)
    configure_logging(args.verbose)
    settings = get_settings()

    LOGGER.info("Initializing TapDB workset state store (namespace via TAPDB_* env vars)")
    state_db = WorksetStateDB()

    if args.bootstrap_tapdb:
        LOGGER.info("Bootstrapping TapDB templates if needed...")
        state_db.bootstrap()

    scheduler = None
    if args.enable_scheduler:
        LOGGER.info("Enabling scheduler")
        scheduler = WorksetScheduler(state_db)

    # Integration layer is created per-workset using bucket from cluster tags
    # (aws-parallelcluster-monitor-bucket). No global integration needed at startup.
    integration = None

    # Initialize CustomerManager for user registration and management
    customer_manager = None
    try:
        from daylib.workset_customer import CustomerManager

        customer_manager = CustomerManager(
            region=args.region,
            profile=args.profile,
        )
        try:
            customer_manager.bootstrap()
            LOGGER.info("CustomerManager TapDB templates bootstrapped")
        except Exception as exc:
            LOGGER.warning("Could not bootstrap customer templates: %s", exc)
            LOGGER.info("CustomerManager initialized (templates may require manual bootstrap)")
    except Exception as exc:
        LOGGER.warning("Failed to initialize CustomerManager: %s", exc)
        import traceback

        LOGGER.debug("CustomerManager init traceback: %s", traceback.format_exc())

    # Check if authentication is enabled via environment variable
    enable_auth = _str_to_bool(os.getenv("DAYLILY_ENABLE_AUTH", "false"))
    cognito_auth = None

    if enable_auth:
        try:
            from daylily_cognito.auth import CognitoAuth
        except ImportError:
            LOGGER.error(
                "Authentication enabled but python-jose not installed. "
                "Install with: pip install 'python-jose[cryptography]'"
            )
            return 1

        user_pool_id = settings.cognito_user_pool_id
        app_client_id = settings.cognito_app_client_id
        app_client_secret = settings.cognito_app_client_secret
        cognito_region = settings.cognito_region or args.region

        if not user_pool_id or not app_client_id:
            LOGGER.error(
                "Authentication enabled but Cognito not configured. "
                "Set cognito_user_pool_id and cognito_app_client_id in "
                "~/.config/ursa/ursa-config.yaml"
            )
            return 1

        LOGGER.info(
            "Initializing Cognito authentication (pool: %s, client: %s)",
            user_pool_id,
            app_client_id,
        )
        cognito_auth = CognitoAuth(
            region=cognito_region,
            user_pool_id=user_pool_id,
            app_client_id=app_client_id,
            app_client_secret=app_client_secret,
            profile=args.profile,
            settings=settings,
        )
        LOGGER.info("Cognito authentication initialized successfully")
    else:
        LOGGER.info("Authentication disabled (set DAYLILY_ENABLE_AUTH=true to enable)")

    file_registry = None
    try:
        from daylib.file_registry import FileRegistry

        try:
            file_registry = FileRegistry()
            if args.bootstrap_tapdb:
                file_registry.bootstrap()
                LOGGER.info("FileRegistry TapDB templates bootstrapped")
        except Exception as exc:
            LOGGER.warning("Could not bootstrap file registry templates: %s", exc)
    except Exception as exc:
        LOGGER.warning("Failed to initialize FileRegistry: %s", exc)
        import traceback

        LOGGER.debug("FileRegistry init traceback: %s", traceback.format_exc())

    LOGGER.info(
        "Creating FastAPI app with customer_manager=%s, file_registry=%s",
        "CONFIGURED" if customer_manager else "NONE",
        "CONFIGURED" if file_registry else "NONE",
    )
    app = create_app(
        state_db,
        scheduler,
        integration=integration,
        cognito_auth=cognito_auth,
        customer_manager=customer_manager,
        enable_auth=enable_auth,
        file_registry=file_registry,
        settings=settings,
    )

    auth_status = "ENABLED" if enable_auth else "DISABLED"
    LOGGER.info(
        "Starting API server on %s:%d (auth: %s)",
        args.host,
        args.port,
        auth_status,
    )
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


if __name__ == "__main__":
    sys.exit(main())
