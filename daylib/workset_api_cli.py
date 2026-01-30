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
        "--table-name",
        default="daylily-worksets",
        help="DynamoDB table name for workset state",
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
        default=8001,
        help="Port to bind to",
    )
    parser.add_argument(
        "--enable-scheduler",
        action="store_true",
        help="Enable scheduler endpoints",
    )
    parser.add_argument(
        "--create-table",
        action="store_true",
        default=True,
        help="Create DynamoDB tables if they don't exist (default: True)",
    )
    parser.add_argument(
        "--no-create-table",
        action="store_false",
        dest="create_table",
        help="Don't create DynamoDB tables automatically",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
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

    from daylib.ursa_config import get_ursa_config

    args = parse_args(argv)
    configure_logging(args.verbose)

    # Load UrsaConfig for centralized settings
    ursa_config = get_ursa_config()

    # Determine DynamoDB region: CLI arg > UrsaConfig > default
    # If --region was explicitly passed (not the default), use it.
    # Otherwise, prefer UrsaConfig's dynamo_db_region.
    if args.region != "us-west-2":
        dynamo_region = args.region
        LOGGER.info("Using DynamoDB region from CLI: %s", dynamo_region)
    else:
        dynamo_region = ursa_config.get_effective_dynamo_db_region()
        source = ursa_config.get_value_source("dynamo_db_region")
        LOGGER.info(
            "Using DynamoDB region from %s: %s",
            source if source != "not set" else "default",
            dynamo_region,
        )

    LOGGER.info(
        "Initializing workset state database: %s (region: %s)",
        args.table_name,
        dynamo_region,
    )
    state_db = WorksetStateDB(
        table_name=args.table_name,
        region=dynamo_region,
        profile=args.profile,
    )

    if args.create_table:
        LOGGER.info("Creating DynamoDB table if needed...")
        state_db.create_table_if_not_exists()

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
            customer_manager.create_customer_table_if_not_exists()
            LOGGER.info(
                "CustomerManager initialized (table: %s)",
                customer_manager.customer_table_name,
            )
        except Exception as table_err:
            LOGGER.warning("Could not verify/create customer table: %s", table_err)
            LOGGER.info(
                "CustomerManager initialized (table: %s, may need manual creation)",
                customer_manager.customer_table_name,
            )
    except Exception as exc:
        LOGGER.warning("Failed to initialize CustomerManager: %s", exc)
        import traceback

        LOGGER.debug("CustomerManager init traceback: %s", traceback.format_exc())

    # Check if authentication is enabled via environment variable
    enable_auth = _str_to_bool(os.getenv("DAYLILY_ENABLE_AUTH", "false"))
    cognito_auth = None

    if enable_auth:
        try:
            from daylib.workset_auth import CognitoAuth
        except ImportError:
            LOGGER.error(
                "Authentication enabled but python-jose not installed. "
                "Install with: pip install 'python-jose[cryptography]'"
            )
            return 1

        # Env vars take precedence, then ursa config.
        user_pool_id = os.getenv("COGNITO_USER_POOL_ID") or ursa_config.cognito_user_pool_id
        app_client_id = (
            os.getenv("COGNITO_CLIENT_ID")
            or os.getenv("COGNITO_APP_CLIENT_ID")
            or ursa_config.cognito_app_client_id
        )
        app_client_secret = (
            os.getenv("COGNITO_APP_CLIENT_SECRET")
            or ursa_config.cognito_app_client_secret
        )
        cognito_region = (
            os.getenv("COGNITO_REGION")
            or ursa_config.cognito_region
            or args.region
        )

        if not user_pool_id or not app_client_id:
            LOGGER.error(
                "Authentication enabled but Cognito not configured. "
                "Set COGNITO_USER_POOL_ID and COGNITO_APP_CLIENT_ID in environment "
                "or ~/.ursa/ursa-config.yaml"
            )
            return 1

        source_pool = ursa_config.get_value_source("cognito_user_pool_id")
        source_client = ursa_config.get_value_source("cognito_app_client_id")
        LOGGER.info(
            "Initializing Cognito authentication (pool: %s [%s], client: [%s])",
            user_pool_id,
            source_pool,
            source_client,
        )

        from daylib.config import get_settings

        settings = get_settings()
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

        file_registry = FileRegistry(
            region=args.region,
            profile=args.profile,
        )
        try:
            file_registry.create_tables_if_not_exist()
            LOGGER.info(
                "FileRegistry initialized (tables: %s, %s, %s)",
                file_registry.files_table_name,
                file_registry.filesets_table_name,
                file_registry.file_workset_usage_table_name,
            )
        except Exception as table_err:
            LOGGER.warning("Could not verify/create file registry tables: %s", table_err)
    except Exception as exc:
        LOGGER.warning("Failed to initialize FileRegistry: %s", exc)
        import traceback

        LOGGER.debug("FileRegistry init traceback: %s", traceback.format_exc())

    # Initialize BiospecimenRegistry for biospecimen management
    try:
        from daylib.biospecimen import BiospecimenRegistry

        biospecimen_registry = BiospecimenRegistry(
            region=args.region,
            profile=args.profile,
        )
        try:
            biospecimen_registry.create_tables_if_not_exist()
            LOGGER.info(
                "BiospecimenRegistry initialized (tables: %s, %s, %s, %s)",
                biospecimen_registry.subjects_table_name,
                biospecimen_registry.biospecimens_table_name,
                biospecimen_registry.biosamples_table_name,
                biospecimen_registry.libraries_table_name,
            )
        except Exception as table_err:
            LOGGER.warning("Could not verify/create biospecimen tables: %s", table_err)
    except Exception as exc:
        LOGGER.warning("Failed to initialize BiospecimenRegistry: %s", exc)
        import traceback

        LOGGER.debug("BiospecimenRegistry init traceback: %s", traceback.format_exc())

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
        log_level="debug" if args.verbose else "info",
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
