"""FastAPI web interface for workset monitoring and management.

Thin app factory that assembles routers and middleware.
Route handlers live in daylib/routes/*.py modules.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml  # type: ignore[import-untyped]

from fastapi import APIRouter, Body, Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from daylib.config import Settings, get_settings
from daylib.workset_state_db import WorksetStateDB
from daylib.workset_scheduler import WorksetScheduler

# Import shared models and utilities used in remaining inline endpoints
from daylib.routes.dependencies import (
    WorksetValidationResponse,
    WorkYamlGenerateRequest,
    calculate_cost_with_efficiency as _calculate_cost_with_efficiency,
    convert_customer_for_template as _convert_customer_for_template,
)

# Optional imports for authentication
try:
    from daylily_cognito.auth import CognitoAuth
    from daylily_cognito.fastapi import create_auth_dependency
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False
    CognitoAuth = None  # type: ignore[misc, assignment]
    create_auth_dependency = None  # type: ignore[assignment]

from daylib.workset_customer import CustomerManager, CustomerConfig
from daylib.workset_validation import WorksetValidator

# Optional integration layer import
try:
    from daylib.workset_integration import WorksetIntegration
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False
    WorksetIntegration = None  # type: ignore[misc, assignment]

# File management imports
try:
    from daylib.file_api import create_file_api_router
    from daylib.file_registry import FileRegistry, BucketFileDiscovery
    from daylib.s3_bucket_validator import S3BucketValidator, LinkedBucketManager
    FILE_MANAGEMENT_AVAILABLE = True
except ImportError:
    FILE_MANAGEMENT_AVAILABLE = False
    create_file_api_router = None  # type: ignore[assignment]
    FileRegistry = None  # type: ignore[misc, assignment]
    BucketFileDiscovery = None  # type: ignore[misc, assignment]
    S3BucketValidator = None  # type: ignore[misc, assignment]
    LinkedBucketManager = None  # type: ignore[misc, assignment]

# Biospecimen layer imports
try:
    from daylib.biospecimen import BiospecimenRegistry
    from daylib.biospecimen_api import create_biospecimen_router
    BIOSPECIMEN_AVAILABLE = True
except ImportError:
    BIOSPECIMEN_AVAILABLE = False
    BiospecimenRegistry = None  # type: ignore[misc, assignment]
    create_biospecimen_router = None  # type: ignore[assignment]

# Manifest storage imports
try:
    from daylib.manifest_registry import ManifestRegistry
    MANIFEST_STORAGE_AVAILABLE = True
except ImportError:
    MANIFEST_STORAGE_AVAILABLE = False
    ManifestRegistry = None  # type: ignore[misc, assignment]

LOGGER = logging.getLogger("daylily.workset_api")


def create_app(
    state_db: WorksetStateDB,
    scheduler: Optional[WorksetScheduler] = None,
    cognito_auth: Optional[CognitoAuth] = None,
    customer_manager: Optional[CustomerManager] = None,
    validator: Optional[WorksetValidator] = None,
    integration: Optional["WorksetIntegration"] = None,
    file_registry: Optional["FileRegistry"] = None,
    manifest_registry: Optional["ManifestRegistry"] = None,
    enable_auth: bool = False,
    settings: Optional[Settings] = None,
) -> FastAPI:
    """Create FastAPI application.

    Args:
        state_db: Workset state database
        scheduler: Optional workset scheduler
        cognito_auth: Optional Cognito authentication
        customer_manager: Optional customer manager
        validator: Optional workset validator
        integration: Optional integration layer for S3 sync
        file_registry: Optional file registry for file management
        enable_auth: Enable authentication (requires cognito_auth)
        settings: Optional Settings instance (uses get_settings() if not provided)

    Returns:
        FastAPI application instance
    """
    # Load settings from environment if not provided
    if settings is None:
        settings = get_settings()

    # Validate demo mode is not enabled in production
    settings.validate_demo_mode()
    if settings.demo_mode:
        LOGGER.warning(
            "DEMO MODE ENABLED - First-customer fallback is active. "
            "This should NEVER be used in production!"
        )

    # Configure logging based on settings
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Set specific loggers to DEBUG for file management (in development)
    if settings.is_development:
        logging.getLogger("daylily.file_api").setLevel(logging.DEBUG)
        logging.getLogger("daylily.file_registry").setLevel(logging.DEBUG)
        logging.getLogger("daylily.s3_bucket_validator").setLevel(logging.DEBUG)
        logging.getLogger("daylily.manifest_registry").setLevel(logging.DEBUG)

    LOGGER.info("Creating Daylily application (env=%s, log_level=%s)", settings.daylily_env, settings.log_level)

    # AWS configuration from settings
    region = settings.get_effective_region()
    profile = settings.aws_profile

    # Initialize manifest registry (optional but enabled by default when available)
    if MANIFEST_STORAGE_AVAILABLE and manifest_registry is None and ManifestRegistry is not None:
        try:
            manifest_registry = ManifestRegistry()
            manifest_registry.bootstrap()
            LOGGER.info("Manifest storage enabled (TapDB templates bootstrapped)")
        except Exception as e:
            LOGGER.warning("Failed to initialize ManifestRegistry: %s", str(e))
            manifest_registry = None

    # Initialize LinkedBucketManager early so portal routes can use it
    linked_bucket_manager = None
    # BucketFileDiscovery is optional; keep a stable binding for portal routes
    bucket_file_discovery = None
    if FILE_MANAGEMENT_AVAILABLE and LinkedBucketManager is not None:
        try:
            linked_bucket_manager = LinkedBucketManager(
                region=region,
                profile=profile,
            )
            linked_bucket_manager.bootstrap()
            LOGGER.info("LinkedBucketManager initialized for portal and file API")
        except Exception as e:
            LOGGER.warning("Failed to create LinkedBucketManager: %s", str(e))

    app = FastAPI(
        title="Daylily Workset Monitor API",
        description="REST API for workset monitoring and management with multi-tenant support",
        version="2.0.0",
    )

    # Enable CORS with settings-based configuration
    try:
        cors_origins = settings.get_cors_origins()
    except ValueError as e:
        LOGGER.error("CORS configuration error: %s", e)
        raise
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add session middleware for portal authentication
    if settings.is_production and settings.session_secret_key == "daylily-dev-secret-change-in-production":
        LOGGER.warning("Using default session secret in production - this is insecure!")
    app.add_middleware(SessionMiddleware, secret_key=settings.session_secret_key)

    # ========== Global Exception Handlers ==========
    # Import custom exceptions
    from daylib.exceptions import DaylilyException

    @app.exception_handler(DaylilyException)
    async def daylily_exception_handler(request: Request, exc: DaylilyException):
        """Handle all DaylilyException subclasses with consistent JSON response."""
        request_id = getattr(request.state, "request_id", None) or str(uuid.uuid4())[:8]
        LOGGER.error(
            "DaylilyException: code=%s, message=%s, request_id=%s, path=%s",
            exc.code, exc.message, request_id, request.url.path
        )
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(request_id=request_id),
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions with consistent JSON response."""
        # Don't intercept HTTPException - let FastAPI handle those
        if isinstance(exc, HTTPException):
            raise exc

        request_id = getattr(request.state, "request_id", None) or str(uuid.uuid4())[:8]
        LOGGER.exception(
            "Unhandled exception: %s, request_id=%s, path=%s",
            str(exc), request_id, request.url.path
        )
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=500,
            content={
                "error": "An internal error occurred",
                "code": "INTERNAL_ERROR",
                "details": {"exception_type": type(exc).__name__} if settings.is_development else {},
                "request_id": request_id,
            },
        )

    # Add request ID middleware
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        """Add request ID to all requests for tracing."""
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # Setup authentication dependency if enabled
    if enable_auth:
        if not AUTH_AVAILABLE:
            LOGGER.error(
                "Authentication requested but python-jose not installed. "
                "Install with: pip install 'python-jose[cryptography]'"
            )
            raise ImportError(
                "Authentication requires python-jose. "
                "Install with: pip install 'python-jose[cryptography]' "
                "or set enable_auth=False"
            )
        if not cognito_auth:
            raise ValueError("enable_auth=True requires cognito_auth parameter")
        # Create JWT auth as optional - we'll also accept session auth
        jwt_auth_dependency = create_auth_dependency(cognito_auth, optional=True)
        LOGGER.info("Authentication enabled - API endpoints will accept session or JWT auth")
    else:
        jwt_auth_dependency = None
        LOGGER.info("Authentication disabled - API endpoints will not require authentication")

    # Create a combined auth dependency that accepts either:
    # 1. Session-based auth from portal (user_email in session)
    # 2. JWT token auth from API calls (Authorization header)
    # 3. API key auth via X-API-Key header (resolved via CustomerManager)
    def get_current_user(request: Request) -> Optional[Dict]:
        """Combined auth dependency for portal session and API JWT auth."""
        # First try session-based auth (portal)
        if hasattr(request, "session"):
            user_email = request.session.get("user_email")
            if user_email:
                customer_id = request.session.get("customer_id")
                is_admin = bool(request.session.get("is_admin", False))
                return {
                    "email": user_email,
                    "customer_id": customer_id,
                    "is_admin": is_admin,
                    "auth_type": "session",
                    "authenticated": True,
                }

        # Then try JWT auth if available (check Authorization header)
        if jwt_auth_dependency and cognito_auth:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                try:
                    from fastapi.security import HTTPAuthorizationCredentials
                    token = auth_header[7:]  # Remove "Bearer " prefix
                    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
                    user: Optional[Dict[Any, Any]] = cognito_auth.get_current_user(credentials)
                    if user:
                        customer_id = user.get("custom:customer_id") or user.get("customer_id")
                        if customer_id is not None and "customer_id" not in user:
                            user["customer_id"] = str(customer_id)

                        if customer_manager and customer_id:
                            try:
                                customer_config = customer_manager.get_customer_config(str(customer_id))
                                if customer_config:
                                    user["is_admin"] = bool(customer_config.is_admin)
                            except Exception as e:  # pragma: no cover - defensive logging
                                LOGGER.debug("Failed to attach is_admin for JWT user: %s", str(e))

                        user["auth_type"] = "jwt"
                        return user
                except HTTPException:
                    # JWT auth failed, but that's ok - we already checked session
                    pass
                except Exception as e:
                    LOGGER.debug("JWT auth check failed: %s", str(e))

        # Finally, try API key auth via X-API-Key header if customer_manager is available
        if customer_manager:
            api_key = request.headers.get("X-API-Key") or request.headers.get("x-api-key")
            if api_key:
                try:
                    customer = customer_manager.get_customer_by_api_key(api_key)
                except Exception as e:  # pragma: no cover - defensive logging
                    LOGGER.warning("API key lookup failed: %s", str(e))
                    customer = None

                if customer:
                    return {
                        "email": customer.email,
                        "customer_id": customer.customer_id,
                        "is_admin": bool(getattr(customer, "is_admin", False)),
                        "auth_type": "api_key",
                        "authenticated": True,
                    }

        # No valid authentication found
        if enable_auth:
            # For API endpoints that require auth, raise an error
            # But only for non-portal routes (portal uses session redirects)
            if not request.url.path.startswith("/portal"):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required - provide session cookie or Bearer token",
                    headers={"WWW-Authenticate": "Bearer"},
                )

        return None

    LOGGER.info("Combined session/JWT authentication configured")

    def get_customer_for_session(request: Request) -> Tuple[Optional[Dict[Any, Any]], Optional[CustomerConfig]]:
        """Get the customer for the currently logged-in user.

        Looks up the customer by the user's email from the session.
        Returns (customer, customer_config) tuple or (None, None) if not found.
        """
        if not customer_manager:
            return None, None

        user_email = None
        if hasattr(request, "session"):
            user_email = request.session.get("user_email")

        if not user_email:
            return None, None

        customer_config = customer_manager.get_customer_by_email(user_email)
        if customer_config:
            return _convert_customer_for_template(customer_config), customer_config

        return None, None

    def get_customer_with_demo_fallback(
        request: Request,
        require_auth: bool = True,
    ) -> Optional[Dict[Any, Any]]:
        """Get customer from session with optional demo mode fallback.

        Args:
            request: The HTTP request
            require_auth: If True, raise HTTPException when no customer found (unless demo mode)

        Returns:
            Customer dict or None

        Raises:
            HTTPException(401): If require_auth=True and no customer found (and demo mode disabled)
        """
        # Try session first
        customer, _config = get_customer_for_session(request)
        if customer:
            return customer

        # Demo mode fallback - only if explicitly enabled
        if settings.demo_mode and customer_manager:
            customers = customer_manager.list_customers()
            if customers:
                LOGGER.debug("Demo mode: using first customer as fallback")
                result: Dict[Any, Any] = _convert_customer_for_template(customers[0])
                return result

        # No customer found
        if require_auth:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required. Please log in.",
            )

        return None

    def _health_payload(*, ready: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "status": "healthy",
            "service": "daylily-workset-monitor",
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        if ready:
            payload["ready"] = True
        return payload

    @app.get("/", tags=["health"], operation_id="api_health_root")
    async def root(request: Request):
        """Root health check endpoint."""
        accept = request.headers.get("accept", "")
        if "text/html" in accept:
            return RedirectResponse(url="/portal/login" if enable_auth else "/portal", status_code=302)
        return _health_payload()

    @app.get("/healthz", tags=["health"], operation_id="api_health_liveness")
    async def healthz() -> Dict[str, Any]:
        """Explicit liveness endpoint for orchestration probes."""
        return _health_payload()

    @app.get("/readyz", tags=["health"], operation_id="api_health_readiness")
    async def readyz() -> Dict[str, Any]:
        """Explicit readiness endpoint for orchestration probes."""
        return _health_payload(ready=True)
    
    # ========== Wire up extracted routers ==========

    # Workset CRUD + queue/scheduler/next (routes/worksets.py)
    from daylib.routes.worksets import create_worksets_router
    worksets_router = create_worksets_router(state_db=state_db, scheduler=scheduler)
    app.include_router(worksets_router)
    LOGGER.info("Workset routes registered via create_worksets_router")

    # Monitoring/admin routes (routes/monitoring.py)
    from daylib.routes.monitoring import create_monitoring_router, MonitoringDependencies
    mon_deps = MonitoringDependencies(state_db=state_db, settings=settings)
    mon_router = create_monitoring_router(mon_deps)
    app.include_router(mon_router)
    LOGGER.info("Monitoring routes registered via create_monitoring_router")

    # S3 utility routes (routes/s3.py)
    from daylib.routes.s3 import create_s3_router, S3Dependencies
    s3_deps = S3Dependencies(settings=settings, get_current_user=get_current_user)
    s3_router = create_s3_router(s3_deps)
    app.include_router(s3_router)
    LOGGER.info("S3 routes registered via create_s3_router")

    # Cluster management routes (routes/clusters.py)
    from daylib.routes.clusters import create_clusters_router, ClusterDependencies
    cluster_deps = ClusterDependencies(settings=settings, get_current_user=get_current_user)
    cluster_router = create_clusters_router(cluster_deps)
    app.include_router(cluster_router)
    LOGGER.info("Cluster routes registered via create_clusters_router")

    # Customer routes (routes/customers.py) - only if customer_manager available
    if customer_manager:
        from daylib.routes.customers import create_customers_router, CustomerDependencies
        cust_deps = CustomerDependencies(
            customer_manager=customer_manager,
            get_current_user=get_current_user,
        )
        cust_router = create_customers_router(cust_deps)
        app.include_router(cust_router)
        LOGGER.info("Customer routes registered via create_customers_router")

        # File management routes (routes/files.py)
        from daylib.routes.files import create_files_router, FileDependencies
        file_deps = FileDependencies(customer_manager=customer_manager)
        files_router = create_files_router(file_deps)
        app.include_router(files_router)
        LOGGER.info("File routes registered via create_files_router")

        # Manifest routes (routes/manifests.py) - registry may be None (routes return 503)
        from daylib.routes.manifests import create_manifests_router, ManifestDependencies
        manifest_deps = ManifestDependencies(
            customer_manager=customer_manager,
            manifest_registry=manifest_registry,
        )
        manifest_router = create_manifests_router(manifest_deps)
        app.include_router(manifest_router)
        LOGGER.info("Manifest routes registered via create_manifests_router")

    # ========== Cost Estimation Endpoint ==========

    @app.post("/api/estimate-cost", tags=["utilities"])
    async def estimate_workset_cost(
        pipeline_type: str = Body(..., embed=True),
        reference_genome: str = Body("GRCh38", embed=True),
        sample_count: int = Body(1, embed=True),
        estimated_coverage: float = Body(30.0, embed=True),
        priority: str = Body("normal", embed=True),
        data_size_gb: float = Body(0.0, embed=True),
    ):
        """Estimate cost for a workset based on parameters."""
        base_vcpu_hours_per_sample = {
            "germline": 4.0, "somatic": 8.0, "rnaseq": 2.0, "wgs": 12.0, "wes": 3.0,
        }
        base_hours = base_vcpu_hours_per_sample.get(pipeline_type, 4.0)
        coverage_factor = estimated_coverage / 30.0
        vcpu_hours = base_hours * sample_count * coverage_factor
        avg_vcpus = 16
        duration_hours = vcpu_hours / avg_vcpus
        cost_per_vcpu_hour = {"urgent": 0.08, "high": 0.08, "normal": 0.03, "low": 0.015}
        base_cost = cost_per_vcpu_hour.get(priority, 0.03)
        compute_cost = vcpu_hours * base_cost
        if data_size_gb <= 0:
            data_size_gb = sample_count * 50.0
        storage_cost = data_size_gb * 0.023 / 4
        fsx_cost = data_size_gb * 0.14 / 4
        transfer_cost = data_size_gb * 0.10 * 0.09
        efficiency_multiplier = _calculate_cost_with_efficiency(data_size_gb)
        adjusted_storage_cost = storage_cost * efficiency_multiplier if efficiency_multiplier > 0 else storage_cost
        total_cost = compute_cost + adjusted_storage_cost + fsx_cost + transfer_cost
        priority_multiplier = {"urgent": 2.0, "high": 1.5, "normal": 1.0, "low": 0.6}
        multiplier = priority_multiplier.get(priority, 1.0)

        return {
            "estimated_cost_usd": round(total_cost * multiplier, 2),
            "compute_cost_usd": round(compute_cost * multiplier, 2),
            "storage_cost_usd": round(adjusted_storage_cost + fsx_cost, 2),
            "transfer_cost_usd": round(transfer_cost, 2),
            "vcpu_hours": round(vcpu_hours, 1),
            "estimated_duration_hours": round(duration_hours, 1),
            "estimated_duration_minutes": int(duration_hours * 60),
            "data_size_gb": round(data_size_gb, 1),
            "efficiency_multiplier": round(efficiency_multiplier, 2),
            "pipeline_type": pipeline_type,
            "sample_count": sample_count,
            "priority": priority,
            "cost_breakdown": {
                "compute": f"${compute_cost * multiplier:.2f}",
                "storage": f"${adjusted_storage_cost:.2f}",
                "fsx": f"${fsx_cost:.2f}",
                "transfer": f"${transfer_cost:.2f}",
            },
            "notes": [
                "Costs are estimates based on typical workloads",
                f"Priority '{priority}' applies {multiplier}x multiplier",
                f"Storage efficiency multiplier: {efficiency_multiplier:.2f}x",
                "Actual costs depend on spot market and data complexity",
            ],
        }

    # ========== Workset Validation Endpoints ==========

    if validator:
        @app.post("/worksets/validate", response_model=WorksetValidationResponse, tags=["validation"])
        async def validate_workset(
            bucket: str = Query(..., description="S3 bucket name"),
            prefix: str = Query(..., description="S3 prefix"),
            dry_run: bool = Query(False, description="Dry-run mode"),
            current_user: Optional[Dict] = Depends(get_current_user),
        ):
            """Validate a workset configuration."""
            result = validator.validate_workset(bucket, prefix, dry_run)
            return WorksetValidationResponse(
                is_valid=result.is_valid,
                errors=result.errors,
                warnings=result.warnings,
                estimated_cost_usd=result.estimated_cost_usd,
                estimated_duration_minutes=result.estimated_duration_minutes,
                estimated_vcpu_hours=result.estimated_vcpu_hours,
                estimated_storage_gb=result.estimated_storage_gb,
            )

    # ========== YAML Generator Endpoint ==========

    @app.post("/worksets/generate-yaml", tags=["utilities"])
    async def generate_work_yaml(
        request: WorkYamlGenerateRequest,
        current_user: Optional[Dict] = Depends(get_current_user),
    ):
        """Generate daylily_work.yaml from form data."""
        work_config = {
            "samples": request.samples,
            "reference_genome": request.reference_genome,
            "pipeline": request.pipeline,
            "priority": request.priority,
            "max_retries": request.max_retries,
            "estimated_coverage": request.estimated_coverage,
        }
        yaml_content = yaml.dump(work_config, default_flow_style=False, sort_keys=False)
        return {"yaml_content": yaml_content, "config": work_config}

    # ========== Customer Portal Routes ==========

    # Setup templates directory
    templates_dir = Path(__file__).parent.parent / "templates"
    static_dir = Path(__file__).parent.parent / "static"
    cw_router: Optional[APIRouter] = None
    dash_router: Optional[APIRouter] = None
    billing_router: Optional[APIRouter] = None

    if templates_dir.exists():
        templates = Jinja2Templates(directory=str(templates_dir))

        # Mount static files
        if static_dir.exists():
            app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        # Initialize biospecimen registry for portal (optional)
        biospecimen_registry_for_portal = None
        if BIOSPECIMEN_AVAILABLE:
            try:
                biospecimen_registry_for_portal = BiospecimenRegistry()
                biospecimen_registry_for_portal.bootstrap()
                LOGGER.info("Biospecimen registry initialized for portal")
            except Exception as e:
                LOGGER.warning("Failed to initialize biospecimen registry for portal: %s", str(e))

        # Portal routes are now in daylib/routes/portal.py
        from daylib.routes.portal import create_portal_router, PortalDependencies
        portal_deps = PortalDependencies(
            state_db=state_db,
            templates=templates,
            settings=settings,
            enable_auth=enable_auth,
            cognito_auth=cognito_auth,
            customer_manager=customer_manager,
            file_registry=file_registry,
            linked_bucket_manager=linked_bucket_manager,
            biospecimen_registry=biospecimen_registry_for_portal,
            manifest_registry=manifest_registry,
        )
        portal_router = create_portal_router(portal_deps)
        app.include_router(portal_router)
        LOGGER.info("Portal routes registered via create_portal_router")

        if customer_manager is not None:
            # Customer workset routes (extracted from this file)
            from daylib.routes.customer_worksets import (
                create_customer_worksets_router,
                CustomerWorksetDependencies,
            )
            cw_deps = CustomerWorksetDependencies(
                state_db=state_db,
                settings=settings,
                customer_manager=customer_manager,
                integration=integration,
                manifest_registry=manifest_registry,
                get_current_user=get_current_user,
            )
            cw_router = create_customer_worksets_router(cw_deps)
            app.include_router(cw_router)
            LOGGER.info("Customer workset routes registered via create_customer_worksets_router")

            # Dashboard routes (extracted from this file)
            from daylib.routes.dashboard import (
                create_dashboard_router,
                DashboardDependencies,
            )
            dash_deps = DashboardDependencies(
                state_db=state_db,
                settings=settings,
                customer_manager=customer_manager,
            )
            dash_router = create_dashboard_router(dash_deps)
            app.include_router(dash_router)
            LOGGER.info("Dashboard routes registered via create_dashboard_router")

            # Billing routes (extracted from this file)
            from daylib.routes.billing import (
                create_billing_router,
                BillingDependencies,
            )
            billing_deps = BillingDependencies(
                state_db=state_db,
                settings=settings,
                customer_manager=customer_manager,
            )
            billing_router = create_billing_router(billing_deps)
            app.include_router(billing_router)
            LOGGER.info("Billing routes registered via create_billing_router")

    # ========== File Management API Integration ==========

    if file_registry and FILE_MANAGEMENT_AVAILABLE:
        LOGGER.info("Integrating file management API endpoints")
        try:
            # Pass auth dependency - use combined session/JWT auth
            auth_dep = get_current_user

            # Create S3 bucket validator for validation endpoints
            s3_bucket_validator = None
            bucket_file_discovery = None

            if S3BucketValidator is not None:
                try:
                    s3_bucket_validator = S3BucketValidator(region=region, profile=profile)
                    LOGGER.info("S3BucketValidator initialized for file API")
                except Exception as e:
                    LOGGER.warning("Failed to create LinkedBucketManager: %s", str(e))

            if BucketFileDiscovery is not None and s3_bucket_validator:
                try:
                    bucket_file_discovery = BucketFileDiscovery(
                        region=region,
                        profile=profile,
                    )
                    LOGGER.info("BucketFileDiscovery initialized for file API")
                except Exception as e:
                    LOGGER.warning("Failed to create BucketFileDiscovery: %s", str(e))

            file_router = create_file_api_router(
                file_registry,
                auth_dependency=auth_dep,
                s3_bucket_validator=s3_bucket_validator,
                linked_bucket_manager=linked_bucket_manager,
                bucket_file_discovery=bucket_file_discovery,
            )
            app.include_router(file_router)
            auth_status = "with combined session/JWT authentication"
            LOGGER.info(f"File management API endpoints registered at /api/files/* ({auth_status})")
        except Exception as e:
            LOGGER.error("Failed to integrate file management API: %s", str(e))
            LOGGER.warning("File management endpoints will not be available")
    elif file_registry and not FILE_MANAGEMENT_AVAILABLE:
        LOGGER.warning(
            "FileRegistry provided but file management modules not available. "
            "File management endpoints will not be registered."
        )
    else:
        LOGGER.info("File management not configured - file API endpoints not registered")

    # ========== Biospecimen API Integration ==========

    if BIOSPECIMEN_AVAILABLE:
        LOGGER.info("Integrating biospecimen API endpoints")
        try:
            biospecimen_registry = BiospecimenRegistry()
            biospecimen_registry.bootstrap()

            def get_customer_id_from_request(request: Request) -> str:
                """Get customer ID from request session or raise 401.

                Resolution order:
                1. Session customer_id (portal login)
                2. API key customer lookup (if implemented)

                Raises:
                    HTTPException(401): If customer cannot be resolved
                """
                # Try session first (portal login)
                session = getattr(request, 'session', None)
                if session:
                    customer_id = session.get('customer_id')
                    if customer_id:
                        return str(customer_id)

                # No valid authentication found
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required. Please log in to access biospecimen data.",
                )

            biospecimen_router = create_biospecimen_router(
                registry=biospecimen_registry,
                get_customer_id=get_customer_id_from_request,
            )
            app.include_router(biospecimen_router)
            LOGGER.info("Biospecimen API endpoints registered at /api/biospecimen/*")
        except Exception as e:
            LOGGER.error("Failed to integrate biospecimen API: %s", str(e))
            LOGGER.warning("Biospecimen endpoints will not be available")
    else:
        LOGGER.info("Biospecimen module not available - biospecimen API endpoints not registered")

    # ========== API Versioning (v1) ==========
    # Create a versioned router that mirrors all API routes under /v1/
    # Original routes remain for backward compatibility.
    v1_router = APIRouter(prefix="/v1", tags=["v1"])

    # Always-available API routers
    v1_router.include_router(worksets_router)
    v1_router.include_router(mon_router)
    v1_router.include_router(s3_router)
    v1_router.include_router(cluster_router)

    # Customer-dependent API routers
    if customer_manager:
        v1_router.include_router(cust_router)
        v1_router.include_router(files_router)
        v1_router.include_router(manifest_router)

    # Template-dependent API routers (customer worksets, dashboard, billing)
    if cw_router is not None and dash_router is not None and billing_router is not None:
        v1_router.include_router(cw_router)
        v1_router.include_router(dash_router)
        v1_router.include_router(billing_router)

    # File management API router (conditional)
    if file_registry and FILE_MANAGEMENT_AVAILABLE:
        try:
            v1_router.include_router(file_router)
        except NameError:
            pass  # file_router was not created due to earlier error

    # Biospecimen API router (conditional)
    if BIOSPECIMEN_AVAILABLE:
        try:
            v1_router.include_router(biospecimen_router)
        except NameError:
            pass  # biospecimen_router was not created due to earlier error

    app.include_router(v1_router)
    LOGGER.info("API v1 routes registered under /v1/ prefix")

    # Store settings in app state for access in route handlers
    app.state.settings = settings

    return app
