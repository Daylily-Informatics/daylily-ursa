"""FastAPI application for Ursa beta analysis flows."""

from __future__ import annotations

import hmac
import logging
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from daylib.analysis_store import (
    AnalysisArtifact,
    AnalysisRecord,
    AnalysisState,
    AnalysisStore,
    ReviewState,
)
from daylib.atlas_result_client import (
    AtlasResultArtifact,
    AtlasResultClient,
    AtlasResultClientError,
)
from daylib.bloom_resolver_client import BloomResolverClient, BloomResolverError
from daylib.config import Settings, get_settings

LOGGER = logging.getLogger("daylily.analysis_api")


class AnalysisIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_euid: str
    index_string: str
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
    index_string: str
    atlas_tenant_id: str
    atlas_order_euid: str
    atlas_test_order_euid: str
    source_euid: str
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
        index_string=record.index_string,
        atlas_tenant_id=record.atlas_tenant_id,
        atlas_order_euid=record.atlas_order_euid,
        atlas_test_order_euid=record.atlas_test_order_euid,
        source_euid=record.source_euid,
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

    @app.get("/", tags=["health"])
    async def root() -> dict[str, str]:
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
            resolution = app.state.bloom_client.resolve_run_index(
                request.run_euid,
                request.index_string,
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
        try:
            atlas_response = app.state.atlas_client.return_analysis_result(
                atlas_tenant_id=record.atlas_tenant_id,
                atlas_order_euid=record.atlas_order_euid,
                atlas_test_order_euid=record.atlas_test_order_euid,
                analysis_euid=record.analysis_euid,
                run_euid=record.run_euid,
                index_string=record.index_string,
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

    return app
