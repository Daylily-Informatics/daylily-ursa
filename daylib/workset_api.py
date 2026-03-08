"""FastAPI application for Ursa beta analysis flows."""

from __future__ import annotations

import hmac
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import boto3
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
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
from daylib.cluster_service import ClusterInfo, get_cluster_service
from daylib.config import Settings, get_settings
from daylib.ephemeral_cluster import list_cluster_create_jobs, start_create_job, tail_job_log
from daylib.pricing_monitor import PricingMonitor
from daylib.pricing_state import PricingState

LOGGER = logging.getLogger("daylily.analysis_api")


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


class ClusterCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region_az: str
    cluster_name: str
    ssh_key_name: str
    s3_bucket_name: str
    config_path: str | None = None
    pass_on_warn: bool = False
    debug: bool = False
    contact_email: str | None = None


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


def _pricing_db_path() -> Path:
    return Path.home() / ".ursa" / "pricing.sqlite3"


def _build_boto_session(settings: Settings, region: str | None = None) -> Any:
    kwargs: dict[str, Any] = {}
    if settings.aws_profile:
        kwargs["profile_name"] = settings.aws_profile
    if region:
        kwargs["region_name"] = region
    return boto3.Session(**kwargs)


def _load_create_options(settings: Settings, region: str) -> dict[str, list[str]]:
    session = _build_boto_session(settings, region=region)
    ec2 = session.client("ec2", region_name=region)
    s3 = session.client("s3")

    keypairs = sorted(
        str(item.get("KeyName") or "").strip()
        for item in ec2.describe_key_pairs().get("KeyPairs", [])
        if str(item.get("KeyName") or "").strip()
    )
    buckets = sorted(
        str(item.get("Name") or "").strip()
        for item in s3.list_buckets().get("Buckets", [])
        if str(item.get("Name") or "").strip()
    )
    return {"keypairs": keypairs, "buckets": buckets}


def _cluster_payload(cluster: ClusterInfo) -> dict[str, Any]:
    return cluster.to_dict(include_sensitive=True)


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
    app.state.pricing_store = PricingState(_pricing_db_path())
    app.state.pricing_monitor = PricingMonitor(settings=settings, store=app.state.pricing_store)

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

    def require_admin_api_key(
        x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    ) -> str:
        return require_write_api_key(x_api_key)

    @app.get("/", tags=["health"])
    async def root() -> dict[str, str]:
        return {
            "status": "healthy",
            "service": "daylily-ursa-beta-analysis",
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

    @app.get("/api/clusters")
    async def list_clusters(
        refresh: bool = False,
        _api_key: str = Depends(require_admin_api_key),
    ) -> dict[str, Any]:
        cluster_service = get_cluster_service(
            regions=settings.get_allowed_regions() or settings.get_cost_monitor_regions(),
            aws_profile=settings.aws_profile,
        )
        clusters = cluster_service.get_all_clusters(force_refresh=refresh)
        return {"clusters": [_cluster_payload(cluster) for cluster in clusters]}

    @app.delete("/api/clusters/{cluster_name}")
    async def delete_cluster(
        cluster_name: str,
        region: str,
        _api_key: str = Depends(require_admin_api_key),
    ) -> dict[str, Any]:
        cluster_service = get_cluster_service(
            regions=settings.get_allowed_regions() or settings.get_cost_monitor_regions(),
            aws_profile=settings.aws_profile,
        )
        result = cluster_service.delete_cluster(cluster_name, region)
        return {"success": True, "result": result}

    @app.get("/api/clusters/create/options")
    async def create_cluster_options(
        region: str,
        _api_key: str = Depends(require_admin_api_key),
    ) -> dict[str, Any]:
        try:
            return _load_create_options(settings, region)
        except Exception as exc:
            LOGGER.exception("Failed to load cluster create options")
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/api/clusters/create")
    async def create_cluster(
        request: ClusterCreateRequest,
        _api_key: str = Depends(require_admin_api_key),
    ) -> dict[str, Any]:
        job = start_create_job(
            region_az=request.region_az,
            cluster_name=request.cluster_name,
            ssh_key_name=request.ssh_key_name,
            s3_bucket_name=request.s3_bucket_name,
            aws_profile=settings.aws_profile,
            contact_email=request.contact_email,
            config_path_override=request.config_path,
            pass_on_warn=request.pass_on_warn,
            debug=request.debug,
        )
        return {
            "job_id": job.job_id,
            "cluster_name": job.cluster_name,
            "region_az": job.region_az,
            "status": job.status,
        }

    @app.get("/api/clusters/create/jobs")
    async def cluster_create_jobs(
        limit: int = 20,
        _api_key: str = Depends(require_admin_api_key),
    ) -> dict[str, Any]:
        return {"jobs": list_cluster_create_jobs(limit=limit)}

    @app.get("/api/clusters/create/jobs/{job_id}/logs")
    async def cluster_create_logs(
        job_id: str,
        lines: int = 200,
        _api_key: str = Depends(require_admin_api_key),
    ) -> dict[str, Any]:
        return {"job_id": job_id, "log": tail_job_log(job_id, lines=lines)}

    @app.get("/api/pricing-snapshots")
    async def pricing_snapshots(
        region: str | None = Query(default=None),
        partitions: str | None = Query(default=None),
        from_ts: str | None = Query(default=None, alias="from"),
        to_ts: str | None = Query(default=None, alias="to"),
        _api_key: str = Depends(require_admin_api_key),
    ) -> dict[str, Any]:
        requested_partitions = [part.strip() for part in partitions.split(",")] if partitions else None
        return app.state.pricing_monitor.get_snapshot_payload(
            region=region,
            partitions=requested_partitions,
            from_ts=from_ts,
            to_ts=to_ts,
        )

    @app.post("/api/pricing-snapshots/run")
    async def run_pricing_snapshot(
        _api_key: str = Depends(require_admin_api_key),
    ) -> dict[str, Any]:
        return app.state.pricing_monitor.queue_capture(trigger="manual", requested_by="admin")

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

    return app
