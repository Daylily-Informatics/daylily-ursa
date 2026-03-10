"""FastAPI application for Ursa analysis flows."""

from __future__ import annotations

import hmac
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

from botocore.exceptions import ClientError
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

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
from daylib_ursa.dewey_client import DeweyClient, DeweyClientError
from daylib_ursa.domain_access import (
    build_allowed_origin_regex,
    build_trusted_hosts,
    is_allowed_origin,
)
from daylib_ursa.s3_utils import RegionAwareS3Client
from daylib_ursa.tapdb_mount import mount_tapdb_admin

LOGGER = logging.getLogger("daylily.analysis_api")
_REPO_ROOT = Path(__file__).resolve().parent.parent


class AnalysisInputReferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference_type: Literal["s3_uri", "artifact_euid", "artifact_set_euid"]
    value: str

    @model_validator(mode="after")
    def validate_value(self) -> "AnalysisInputReferenceRequest":
        if not str(self.value or "").strip():
            raise ValueError("value is required")
        return self


class AnalysisIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_euid: str
    flowcell_id: str
    lane: str
    library_barcode: str
    analysis_type: str = "beta-default"
    input_references: list[AnalysisInputReferenceRequest] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_input_references(self) -> "AnalysisIngestRequest":
        if not self.input_references:
            raise ValueError("input_references is required")
        return self


class AnalysisArtifactRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_type: str | None = None
    artifact_euid: str | None = None
    storage_uri: str | None = None
    filename: str | None = None
    mime_type: str | None = None
    checksum_sha256: str | None = None
    size_bytes: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_reference_fields(self) -> "AnalysisArtifactRequest":
        has_artifact_ref = bool(str(self.artifact_euid or "").strip())
        has_storage_uri = bool(str(self.storage_uri or "").strip())
        if has_artifact_ref == has_storage_uri:
            raise ValueError("Exactly one of artifact_euid or storage_uri is required")
        return self


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
    atlas_test_fulfillment_item_euid: str
    analysis_type: str
    state: str
    review_state: str
    result_status: str
    run_folder: str
    internal_bucket: str
    input_references: list[dict[str, Any]]
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
        atlas_test_fulfillment_item_euid=record.atlas_test_fulfillment_item_euid,
        analysis_type=record.analysis_type,
        state=record.state,
        review_state=record.review_state,
        result_status=record.result_status,
        run_folder=record.run_folder,
        internal_bucket=record.internal_bucket,
        input_references=record.input_references,
        result_payload=record.result_payload,
        metadata=record.metadata,
        created_at=record.created_at,
        updated_at=record.updated_at,
        atlas_return=record.atlas_return,
        artifacts=[_artifact_response(artifact) for artifact in record.artifacts],
    )


def _parse_s3_object_uri(value: str) -> tuple[str, str]:
    parsed = urlparse(str(value or "").strip())
    bucket = str(parsed.netloc or "").strip()
    key = str(parsed.path or "").strip().lstrip("/")
    if parsed.scheme != "s3" or not bucket or not key:
        raise ValueError("Expected s3://<bucket>/<key> object URI")
    return bucket, key


def _ensure_s3_fetchable(s3_client: RegionAwareS3Client, storage_uri: str) -> None:
    bucket, key = _parse_s3_object_uri(storage_uri)
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code") or "")
        if code in {"404", "NoSuchKey"}:
            raise ValueError(f"Input object not found: {storage_uri}") from exc
        if code in {"403", "AccessDenied"}:
            raise ValueError(f"Input object is not fetchable: {storage_uri}") from exc
        raise ValueError(f"Input object validation failed: {storage_uri}") from exc


def create_app(
    store: AnalysisStore,
    *,
    bloom_client: BloomResolverClient,
    atlas_client: AtlasResultClient | None = None,
    dewey_client: DeweyClient | None = None,
    settings: Settings | None = None,
    require_api_key: bool | None = None,
) -> FastAPI:
    if settings is None:
        settings = get_settings()

    if require_api_key is False:
        raise ValueError("Ursa write API key enforcement cannot be disabled")

    internal_bucket = str(getattr(settings, "ursa_internal_output_bucket", "") or "").strip()
    if not internal_bucket:
        raise ValueError("ursa_internal_output_bucket is required")
    allow_local_domain_access = not settings.is_production

    app = FastAPI(
        title="Daylily Ursa Analysis API",
        description="Analysis execution, review, artifact registration, and Atlas result return",
        version="4.0.0",
    )
    app.state.store = store
    app.state.bloom_client = bloom_client
    app.state.atlas_client = atlas_client
    app.state.dewey_client = dewey_client
    app.state.s3_client = RegionAwareS3Client(
        default_region=settings.get_effective_region(),
        profile=settings.aws_profile,
    )
    app.state.internal_bucket = internal_bucket
    app.state.require_api_key = True
    app.state.api_key = settings.ursa_internal_api_key

    if not any(getattr(route, "path", None) == "/static" for route in app.routes):
        app.mount("/static", StaticFiles(directory=str(_REPO_ROOT / "static")), name="static")

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=build_trusted_hosts(allow_local=allow_local_domain_access),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[],
        allow_origin_regex=build_allowed_origin_regex(
            allow_local=allow_local_domain_access
        ),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def enforce_origin_allowlist(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and not is_allowed_origin(origin, allow_local=allow_local_domain_access):
            return PlainTextResponse("Origin not allowed", status_code=403)
        return await call_next(request)

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

    @app.get("/healthz", tags=["health"])
    async def healthz() -> dict[str, str]:
        return {
            "status": "healthy",
            "service": "daylily-ursa-analysis",
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

        resolved_references: list[dict[str, Any]] = []
        for ref in request.input_references:
            raw_value = str(ref.value or "").strip()
            if ref.reference_type == "s3_uri":
                try:
                    _ensure_s3_fetchable(app.state.s3_client, raw_value)
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                resolved_references.append(
                    {
                        "reference_type": "s3_uri",
                        "value": raw_value,
                        "storage_uri": raw_value,
                    }
                )
                continue

            if app.state.dewey_client is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Dewey integration is required for artifact_euid and artifact_set_euid"
                    ),
                )

            if ref.reference_type == "artifact_euid":
                try:
                    resolved = app.state.dewey_client.resolve_artifact(raw_value)
                except DeweyClientError as exc:
                    raise HTTPException(status_code=502, detail=str(exc)) from exc
                resolved_references.append(
                    {
                        "reference_type": "artifact_euid",
                        "value": raw_value,
                        "artifact_euid": str(resolved.get("artifact_euid") or raw_value),
                        "artifact_type": str(resolved.get("artifact_type") or ""),
                        "storage_uri": str(resolved.get("storage_uri") or ""),
                        "metadata": dict(resolved.get("metadata") or {}),
                    }
                )
                continue

            try:
                resolved_set = app.state.dewey_client.resolve_artifact_set(raw_value)
            except DeweyClientError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc

            members = resolved_set.get("members")
            member_payload = [
                {
                    "artifact_euid": str(member.get("artifact_euid") or ""),
                    "artifact_type": str(member.get("artifact_type") or ""),
                    "storage_uri": str(member.get("storage_uri") or ""),
                    "metadata": dict(member.get("metadata") or {}),
                }
                for member in (members if isinstance(members, list) else [])
                if isinstance(member, dict)
            ]
            resolved_references.append(
                {
                    "reference_type": "artifact_set_euid",
                    "value": raw_value,
                    "artifact_set_euid": str(resolved_set.get("artifact_set_euid") or raw_value),
                    "artifact_euids": [str(item.get("artifact_euid") or "") for item in member_payload],
                    "members": member_payload,
                }
            )

        record = app.state.store.ingest_analysis(
            resolution=resolution,
            analysis_type=request.analysis_type,
            internal_bucket=app.state.internal_bucket,
            idempotency_key=str(idempotency_key),
            input_references=resolved_references,
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
        if app.state.dewey_client is None:
            raise HTTPException(
                status_code=400,
                detail="Dewey integration configuration is required for artifact registration",
            )

        artifact_type = str(request.artifact_type or "").strip()
        storage_uri = ""
        filename = str(request.filename or "").strip()
        resolved_metadata: dict[str, Any] = {}

        source_artifact_euid = str(request.artifact_euid or "").strip()
        if source_artifact_euid:
            try:
                resolved = app.state.dewey_client.resolve_artifact(source_artifact_euid)
            except DeweyClientError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc

            artifact_type = artifact_type or str(resolved.get("artifact_type") or "").strip()
            storage_uri = str(resolved.get("storage_uri") or "").strip()
            filename = filename or str(resolved.get("filename") or "").strip()
            if not filename:
                filename = Path(storage_uri).name or f"{source_artifact_euid}.bin"

            resolved_metadata = dict(resolved.get("metadata") or {})
            resolved_metadata["dewey_artifact_euid"] = source_artifact_euid
            resolved_metadata["dewey_resolved"] = True
        else:
            storage_uri = str(request.storage_uri or "").strip()
            if not artifact_type:
                raise HTTPException(status_code=400, detail="artifact_type is required for storage_uri")
            try:
                bucket, key = _parse_s3_object_uri(storage_uri)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if bucket != app.state.internal_bucket:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "storage_uri must be in the configured Ursa internal output bucket "
                        f"({app.state.internal_bucket})"
                    ),
                )
            try:
                _ensure_s3_fetchable(app.state.s3_client, storage_uri)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if not filename:
                filename = Path(key).name or "artifact.bin"

            registration_metadata = {
                "producer_system": "ursa",
                "producer_object_euid": analysis_euid,
                **dict(request.metadata or {}),
            }
            try:
                registered_euid = app.state.dewey_client.register_artifact(
                    artifact_type=artifact_type,
                    storage_uri=storage_uri,
                    metadata=registration_metadata,
                    idempotency_key=f"{analysis_euid}:{storage_uri}",
                )
            except DeweyClientError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            resolved_metadata = {
                **registration_metadata,
                "dewey_artifact_euid": registered_euid,
                "dewey_registered": True,
            }

        if not artifact_type:
            raise HTTPException(status_code=400, detail="artifact_type is required")
        if not storage_uri:
            raise HTTPException(status_code=502, detail="resolved artifact storage_uri is empty")

        artifact_metadata = dict(resolved_metadata)
        artifact_metadata.update(dict(request.metadata or {}))

        try:
            artifact = app.state.store.add_artifact(
                analysis_euid,
                artifact_type=artifact_type,
                storage_uri=storage_uri,
                filename=filename,
                mime_type=request.mime_type,
                checksum_sha256=request.checksum_sha256,
                size_bytes=request.size_bytes,
                metadata=artifact_metadata,
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
            atlas_artifacts: list[AtlasResultArtifact] = []
            missing_dewey_refs: list[str] = []
            for artifact in record.artifacts:
                dewey_artifact_euid = str(artifact.metadata.get("dewey_artifact_euid") or "").strip()
                if not dewey_artifact_euid:
                    missing_dewey_refs.append(artifact.artifact_euid)
                    continue
                atlas_artifacts.append(
                    AtlasResultArtifact(
                        artifact_euid=dewey_artifact_euid,
                        artifact_type=artifact.artifact_type,
                        storage_uri=artifact.storage_uri,
                        filename=artifact.filename,
                        mime_type=artifact.mime_type,
                        checksum_sha256=artifact.checksum_sha256,
                        size_bytes=artifact.size_bytes,
                        metadata=artifact.metadata,
                    )
                )
            if missing_dewey_refs:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "All analysis artifacts must be Dewey-registered before Atlas return. "
                        f"Missing dewey_artifact_euid for: {', '.join(missing_dewey_refs)}"
                    ),
                )

            atlas_response = app.state.atlas_client.return_analysis_result(
                atlas_tenant_id=record.atlas_tenant_id,
                atlas_trf_euid=record.atlas_trf_euid,
                atlas_test_euid=record.atlas_test_euid,
                atlas_test_fulfillment_item_euid=record.atlas_test_fulfillment_item_euid,
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
                artifacts=atlas_artifacts,
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

    mount_tapdb_admin(app, settings)

    return app
