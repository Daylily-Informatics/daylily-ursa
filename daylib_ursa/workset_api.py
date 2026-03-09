"""FastAPI application for Ursa beta analysis flows."""

from __future__ import annotations

import hmac
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlencode

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field

from daylib_ursa.analysis_store import (
    AnalysisArtifact,
    AnalysisRecord,
    AnalysisState,
    AnalysisStore,
    ReviewState,
)
from daylib_ursa.atlas_result_client import (
    AtlasResultArtifact,
    AtlasResultClient,
    AtlasResultClientError,
)
from daylib_ursa.bloom_resolver_client import BloomResolverClient, BloomResolverError
from daylib_ursa.config import Settings, get_settings
from daylib_ursa.portal_auth import (
    PORTAL_SESSION_COOKIE_NAME,
    PORTAL_SESSION_MAX_AGE_SECONDS,
    decode_id_token_claims,
    decode_portal_session,
    derive_identity,
    encode_portal_session,
    exchange_code_for_tokens,
    fetch_userinfo,
)
from daylib_ursa.portal_onboarding import OnboardingError, ensure_customer_onboarding
from daylib_ursa.portal import mount_portal
from daylib_ursa.tapdb_mount import mount_tapdb_admin
from daylib_ursa.user_preferences import get_display_timezone_for_email

LOGGER = logging.getLogger("daylily.analysis_api")
_REPO_ROOT = Path(__file__).resolve().parent.parent
_TEMPLATES = Jinja2Templates(directory=str(_REPO_ROOT / "templates"))


class AnalysisIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_euid: str
    flowcell_id: str
    lane: str
    library_barcode: str
    analysis_type: str = "beta-default"
    artifact_bucket: str
    input_files: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnalysisArtifactRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_type: str
    storage_uri: str
    filename: str
    mime_type: str | None = None
    checksum_sha256: str | None = None
    size_bytes: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnalysisStatusRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: AnalysisState
    result_status: str | None = None
    result_payload: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None


class AnalysisReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_state: ReviewState
    reviewer: str | None = None
    notes: str | None = None


class AnalysisReturnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result_payload: dict[str, Any] = Field(default_factory=dict)
    result_status: str = "COMPLETED"


class AnalysisArtifactResponse(BaseModel):
    artifact_euid: str
    artifact_type: str
    storage_uri: str
    filename: str
    mime_type: str | None
    checksum_sha256: str | None
    size_bytes: int | None
    created_at: str
    metadata: dict[str, Any]


class AnalysisResponse(BaseModel):
    analysis_euid: str
    run_euid: str
    flowcell_id: str
    lane: str
    library_barcode: str
    sequenced_library_assignment_euid: str
    atlas_tenant_id: str
    atlas_trf_euid: str
    atlas_test_euid: str
    atlas_test_process_item_euid: str
    analysis_type: str
    state: str
    review_state: str
    result_status: str
    run_folder: str
    artifact_bucket: str
    result_payload: dict[str, Any]
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
    atlas_return: dict[str, Any]
    artifacts: list[AnalysisArtifactResponse]


def _artifact_response(artifact: AnalysisArtifact) -> AnalysisArtifactResponse:
    return AnalysisArtifactResponse(
        artifact_euid=artifact.artifact_euid,
        artifact_type=artifact.artifact_type,
        storage_uri=artifact.storage_uri,
        filename=artifact.filename,
        mime_type=artifact.mime_type,
        checksum_sha256=artifact.checksum_sha256,
        size_bytes=artifact.size_bytes,
        created_at=artifact.created_at,
        metadata=artifact.metadata,
    )


def _analysis_response(record: AnalysisRecord) -> AnalysisResponse:
    return AnalysisResponse(
        analysis_euid=record.analysis_euid,
        run_euid=record.run_euid,
        flowcell_id=record.flowcell_id,
        lane=record.lane,
        library_barcode=record.library_barcode,
        sequenced_library_assignment_euid=record.sequenced_library_assignment_euid,
        atlas_tenant_id=record.atlas_tenant_id,
        atlas_trf_euid=record.atlas_trf_euid,
        atlas_test_euid=record.atlas_test_euid,
        atlas_test_process_item_euid=record.atlas_test_process_item_euid,
        analysis_type=record.analysis_type,
        state=record.state,
        review_state=record.review_state,
        result_status=record.result_status,
        run_folder=record.run_folder,
        artifact_bucket=record.artifact_bucket,
        result_payload=record.result_payload,
        metadata=record.metadata,
        created_at=record.created_at,
        updated_at=record.updated_at,
        atlas_return=record.atlas_return,
        artifacts=[_artifact_response(artifact) for artifact in record.artifacts],
    )


def _build_hosted_ui_url(
    *,
    domain: str,
    endpoint: str,
    client_id: str,
    redirect_uri: str,
) -> str:
    normalized_domain = domain.strip()
    if not normalized_domain.startswith(("http://", "https://")):
        normalized_domain = f"https://{normalized_domain}"
    query = urlencode(
        {
            "client_id": client_id,
            "response_type": "code",
            "scope": "openid email profile",
            "redirect_uri": redirect_uri,
        }
    )
    return f"{normalized_domain.rstrip('/')}/{endpoint}?{query}"


def create_app(
    store: AnalysisStore,
    *,
    bloom_client: BloomResolverClient,
    atlas_client: AtlasResultClient | None = None,
    settings: Settings | None = None,
    require_api_key: bool | None = None,
) -> FastAPI:
    if settings is None:
        settings = get_settings()

    if require_api_key is None:
        require_api_key = True

    app = FastAPI(
        title="Daylily Ursa Beta Analysis API",
        description="Run-linked analysis execution, review, artifacts, and Atlas result return",
        version="3.0.0",
    )
    app.state.store = store
    app.state.bloom_client = bloom_client
    app.state.atlas_client = atlas_client
    app.state.require_api_key = require_api_key
    app.state.api_key = settings.ursa_internal_api_key

    if not any(getattr(route, "path", None) == "/static" for route in app.routes):
        app.mount("/static", StaticFiles(directory=str(_REPO_ROOT / "static")), name="static")

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        cors_origins = settings.get_cors_origins()
    except ValueError as exc:
        LOGGER.error("CORS configuration error: %s", exc)
        raise
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.on_event("startup")
    async def start_background_services() -> None:
        if settings.ursa_cost_monitor_enabled:
            app.state.pricing_monitor.start()

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        if isinstance(exc, HTTPException):
            raise exc
        LOGGER.exception("Unhandled exception on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": "An internal error occurred",
                "request_id": getattr(request.state, "request_id", ""),
            },
        )

    def require_write_api_key(
        x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    ) -> str:
        if not app.state.require_api_key:
            return ""
        expected = str(app.state.api_key or "")
        provided = str(x_api_key or "")
        if not provided or not hmac.compare_digest(provided, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
            )
        return provided

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/portal/login", status_code=status.HTTP_307_TEMPORARY_REDIRECT)

    @app.get("/portal/login", response_class=HTMLResponse, include_in_schema=False)
    async def portal_login(request: Request, error: str | None = None):
        session_cookie = request.cookies.get(PORTAL_SESSION_COOKIE_NAME)
        if session_cookie:
            session_identity = decode_portal_session(settings.session_secret_key, session_cookie) or {}
            if session_identity.get("logged_in"):
                return RedirectResponse(url="/portal", status_code=status.HTTP_307_TEMPORARY_REDIRECT)

        hosted_ui_enabled = bool(
            settings.enable_auth and settings.cognito_domain and settings.cognito_app_client_id
        )
        sso_login_url = ""
        sso_signup_url = ""
        reset_login_url = "/portal/login"
        if hosted_ui_enabled:
            callback_uri = f"{request.url.scheme}://{request.url.netloc}/auth/callback"
            sso_login_url = _build_hosted_ui_url(
                domain=str(settings.cognito_domain),
                endpoint="login",
                client_id=str(settings.cognito_app_client_id),
                redirect_uri=callback_uri,
            )
            sso_signup_url = _build_hosted_ui_url(
                domain=str(settings.cognito_domain),
                endpoint="signup",
                client_id=str(settings.cognito_app_client_id),
                redirect_uri=callback_uri,
            )
            reset_login_url = sso_login_url
        return _TEMPLATES.TemplateResponse(
            "auth/login.html",
            {
                "request": request,
                "auth_enabled": settings.enable_auth,
                "hosted_ui_enabled": hosted_ui_enabled,
                "sso_login_url": sso_login_url,
                "sso_signup_url": sso_signup_url,
                "reset_login_url": reset_login_url,
                "error": error,
                "cache_bust": "1",
            },
        )

    @app.get("/auth/callback", include_in_schema=False)
    async def auth_callback(
        request: Request,
        code: str | None = None,
        error: str | None = None,
        error_description: str | None = None,
    ) -> RedirectResponse:
        if error:
            LOGGER.warning("Cognito callback error: %s (%s)", error, error_description or "")
            return RedirectResponse(url="/portal/login?error=login_failed", status_code=307)
        if not code:
            LOGGER.warning("Cognito callback missing authorization code")
            return RedirectResponse(url="/portal/login?error=missing_code", status_code=307)
        if not settings.cognito_domain or not settings.cognito_app_client_id:
            LOGGER.error("Cognito callback received without Cognito domain/client configuration")
            return RedirectResponse(url="/portal/login?error=cognito_unconfigured", status_code=307)

        redirect_uri = f"{request.url.scheme}://{request.url.netloc}/auth/callback"
        try:
            token_payload = await exchange_code_for_tokens(
                domain=str(settings.cognito_domain),
                code=code,
                client_id=str(settings.cognito_app_client_id),
                client_secret=settings.cognito_app_client_secret,
                redirect_uri=redirect_uri,
            )
        except Exception:
            LOGGER.exception("Failed to exchange Cognito authorization code")
            return RedirectResponse(url="/portal/login?error=token_exchange_failed", status_code=307)

        id_claims = decode_id_token_claims(str(token_payload.get("id_token") or ""))
        userinfo: dict[str, Any] = {}
        try:
            userinfo = await fetch_userinfo(
                domain=str(settings.cognito_domain),
                access_token=str(token_payload.get("access_token") or ""),
            )
        except Exception:
            LOGGER.exception("Failed to fetch Cognito userinfo; using ID token claims only")

        identity = derive_identity(
            claims=id_claims,
            userinfo=userinfo,
            default_customer_id=settings.ursa_portal_default_customer_id,
        )
        try:
            identity = ensure_customer_onboarding(identity=identity, settings=settings)
        except OnboardingError:
            LOGGER.exception("Failed to complete customer onboarding during auth callback")
            return RedirectResponse(url="/portal/login?error=onboarding_failed", status_code=307)
        identity["display_timezone"] = get_display_timezone_for_email(identity.get("user_email"))

        response = RedirectResponse(url="/portal", status_code=307)
        response.set_cookie(
            key=PORTAL_SESSION_COOKIE_NAME,
            value=encode_portal_session(settings.session_secret_key, identity),
            httponly=True,
            secure=request.url.scheme == "https",
            samesite="lax",
            max_age=PORTAL_SESSION_MAX_AGE_SECONDS,
            path="/",
        )
        return response

    @app.get("/healthz", tags=["health"])
    async def healthz() -> dict[str, str]:
        return {
            "status": "healthy",
            "service": "daylily-ursa-beta-analysis",
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

    @app.post(
        "/api/analyses/ingest",
        response_model=AnalysisResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def ingest_analysis(
        request: AnalysisIngestRequest,
        _api_key: str = Depends(require_write_api_key),
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ) -> AnalysisResponse:
        if not str(idempotency_key or "").strip():
            raise HTTPException(status_code=400, detail="Idempotency-Key header is required")
        try:
            resolution = app.state.bloom_client.resolve_run_assignment(
                request.run_euid,
                request.flowcell_id,
                request.lane,
                request.library_barcode,
            )
        except BloomResolverError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        record = app.state.store.ingest_analysis(
            resolution=resolution,
            analysis_type=request.analysis_type,
            artifact_bucket=request.artifact_bucket,
            idempotency_key=str(idempotency_key),
            input_files=request.input_files,
            metadata=request.metadata,
        )
        return _analysis_response(record)

    @app.get("/api/analyses/{analysis_euid}", response_model=AnalysisResponse)
    async def get_analysis(analysis_euid: str) -> AnalysisResponse:
        record = app.state.store.get_analysis(analysis_euid)
        if record is None:
            raise HTTPException(status_code=404, detail="Analysis not found")
        return _analysis_response(record)

    @app.post("/api/analyses/{analysis_euid}/status", response_model=AnalysisResponse)
    async def update_analysis_status(
        analysis_euid: str,
        request: AnalysisStatusRequest,
        _api_key: str = Depends(require_write_api_key),
    ) -> AnalysisResponse:
        try:
            record = app.state.store.update_analysis_state(
                analysis_euid,
                state=request.state,
                result_status=request.result_status,
                result_payload=request.result_payload,
                metadata=request.metadata,
                reason=request.reason,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _analysis_response(record)

    @app.post(
        "/api/analyses/{analysis_euid}/artifacts",
        response_model=AnalysisArtifactResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def add_analysis_artifact(
        analysis_euid: str,
        request: AnalysisArtifactRequest,
        _api_key: str = Depends(require_write_api_key),
    ) -> AnalysisArtifactResponse:
        try:
            artifact = app.state.store.add_artifact(
                analysis_euid,
                artifact_type=request.artifact_type,
                storage_uri=request.storage_uri,
                filename=request.filename,
                mime_type=request.mime_type,
                checksum_sha256=request.checksum_sha256,
                size_bytes=request.size_bytes,
                metadata=request.metadata,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _artifact_response(artifact)

    @app.post("/api/analyses/{analysis_euid}/review", response_model=AnalysisResponse)
    async def review_analysis(
        analysis_euid: str,
        request: AnalysisReviewRequest,
        _api_key: str = Depends(require_write_api_key),
    ) -> AnalysisResponse:
        try:
            record = app.state.store.set_review_state(
                analysis_euid,
                review_state=request.review_state,
                reviewer=request.reviewer,
                notes=request.notes,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _analysis_response(record)

    @app.post("/api/analyses/{analysis_euid}/return", response_model=AnalysisResponse)
    async def return_analysis_result(
        analysis_euid: str,
        request: AnalysisReturnRequest,
        _api_key: str = Depends(require_write_api_key),
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ) -> AnalysisResponse:
        if not str(idempotency_key or "").strip():
            raise HTTPException(status_code=400, detail="Idempotency-Key header is required")
        record = app.state.store.get_analysis(analysis_euid)
        if record is None:
            raise HTTPException(status_code=404, detail="Analysis not found")
        if app.state.atlas_client is None:
            raise HTTPException(status_code=503, detail="Atlas result return client is not configured")
        if record.review_state != ReviewState.APPROVED.value:
            raise HTTPException(
                status_code=409,
                detail="Analysis cannot be returned before manual approval",
            )
        try:
            atlas_response = app.state.atlas_client.return_analysis_result(
                atlas_tenant_id=record.atlas_tenant_id,
                atlas_trf_euid=record.atlas_trf_euid,
                atlas_test_euid=record.atlas_test_euid,
                atlas_test_process_item_euid=record.atlas_test_process_item_euid,
                analysis_euid=record.analysis_euid,
                run_euid=record.run_euid,
                sequenced_library_assignment_euid=record.sequenced_library_assignment_euid,
                flowcell_id=record.flowcell_id,
                lane=record.lane,
                library_barcode=record.library_barcode,
                analysis_type=record.analysis_type,
                result_status=request.result_status,
                review_state=record.review_state,
                result_payload=request.result_payload,
                artifacts=[
                    AtlasResultArtifact(
                        artifact_type=artifact.artifact_type,
                        storage_uri=artifact.storage_uri,
                        filename=artifact.filename,
                        mime_type=artifact.mime_type,
                        checksum_sha256=artifact.checksum_sha256,
                        size_bytes=artifact.size_bytes,
                        metadata=artifact.metadata,
                    )
                    for artifact in record.artifacts
                ],
                idempotency_key=str(idempotency_key),
            )
        except AtlasResultClientError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        updated = app.state.store.mark_returned(
            analysis_euid,
            atlas_return={
                **atlas_response,
                "result_status": request.result_status,
            },
            idempotency_key=str(idempotency_key),
        )
        return _analysis_response(updated)

    mount_portal(app, settings)
    mount_tapdb_admin(app, settings)

    return app
