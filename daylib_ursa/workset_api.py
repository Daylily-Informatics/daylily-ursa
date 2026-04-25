"""FastAPI application for Ursa's versioned backend APIs."""

from __future__ import annotations

import gzip
import hashlib
import hmac
import io
import logging
import secrets
import tarfile
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, Sequence
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from daylily_auth_cognito import configure_session_middleware

from daylib_ursa import __version__
from daylib_ursa.analysis_commands import (
    analysis_command_payload,
    command_catalog_payload,
    preview_analysis_command,
)
from daylib_ursa.analysis_jobs import AnalysisJobManager
from daylib_ursa.analysis_store import (
    AnalysisArtifact,
    AnalysisRecord,
    AnalysisState,
    AnalysisStore,
    ReviewState,
)
from daylib_ursa.anomalies import open_anomaly_repository
from daylib_ursa.atlas_result_client import (
    AtlasResultArtifact,
    AtlasResultClient,
    AtlasResultClientError,
)
from daylib_ursa.auth import (
    AtlasUserDirectoryEntry,
    AuthError,
    build_web_session_config,
    CognitoAuthProvider,
    CognitoUserDirectoryService,
    CurrentUser,
    RequireAdmin,
    RequireAuth,
    RequireObservability,
    UserTokenRecord,
    UserTokenService,
    UserTokenUsageRecord,
)
from daylib_ursa.bloom_resolver_client import BloomResolverClient, BloomResolverError
from daylib_ursa.cluster_jobs import ClusterJobManager, region_from_region_az
from daylib_ursa.cluster_service import ClusterService
from daylib_ursa.config import Settings, get_settings
from daylib_ursa.dewey_client import DeweyClient, DeweyClientError
from daylib_ursa.domain_access import (
    build_allowed_origin_regex,
    build_trusted_hosts,
    is_allowed_origin,
)
from daylib_ursa.ephemeral_cluster.runner import (
    REQUIRED_DAYLILY_EC_VERSION,
    require_daylily_ec_version,
)
from daylib_ursa.gui_app import mount_gui
from daylib_ursa.observability import (
    UrsaObservabilityStore,
    build_api_health_payload,
    build_auth_health_payload,
    build_db_health_payload,
    build_endpoint_health_payload,
    build_health_payload,
    build_healthz_payload,
    build_my_health_payload,
    build_obs_services_payload,
    build_readyz_payload,
    install_sqlalchemy_observability,
)
from daylib_ursa.pricing_state import pricing_quantile
from daylib_ursa.resource_store import (
    ClientRegistrationRecord,
    AnalysisJobEventRecord,
    AnalysisJobRecord,
    ClusterJobEventRecord,
    ClusterJobRecord,
    DeweyImportRecord,
    LinkedBucketRecord,
    ManifestRecord,
    ResourceStore,
    StagingJobEventRecord,
    StagingJobRecord,
    WorksetRecord,
)
from daylib_ursa.s3_utils import RegionAwareS3Client, normalize_bucket_name
from daylib_ursa.analysis_samples_manifest import build_analysis_samples_manifest
from daylib_ursa.staging_jobs import StagingJobManager
from daylib_ursa.tapdb_mount import mount_tapdb_admin
from daylib_ursa.ursa_config import get_ursa_config, parse_regions_csv, update_config_regions

LOGGER = logging.getLogger("daylily.ursa.api")


class AnalysisInputReferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference_type: Literal["artifact_euid", "artifact_set_euid"]
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
    workset_euid: str | None = None
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
        if not has_artifact_ref or has_storage_uri:
            raise ValueError(
                "artifact_euid is required; import raw objects through /api/v1/artifacts/import"
            )
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


class ManifestCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workset_euid: str
    name: str
    artifact_set_euid: str | None = None
    artifact_euids: list[str] = Field(default_factory=list)
    input_references: list["ManifestInputReferenceRequest"] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_manifest_inputs(self) -> "ManifestCreateRequest":
        if self.input_references:
            return self
        editor_rows = dict(self.metadata or {}).get("editor_analysis_inputs")
        if isinstance(editor_rows, list) and editor_rows:
            return self
        raise ValueError("input_references or metadata.editor_analysis_inputs is required")
        return self


class ManifestInputReferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference_type: Literal["artifact_euid", "artifact_set_euid", "s3_uri"]
    value: str

    @model_validator(mode="after")
    def validate_value(self) -> "ManifestInputReferenceRequest":
        if not str(self.value or "").strip():
            raise ValueError("value is required")
        return self


class WorksetCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    artifact_set_euids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_type: str
    storage_uri: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactResolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_euid: str | None = None
    artifact_set_euid: str | None = None

    @model_validator(mode="after")
    def validate_choice(self) -> "ArtifactResolveRequest":
        has_artifact = bool(str(self.artifact_euid or "").strip())
        has_set = bool(str(self.artifact_set_euid or "").strip())
        if has_artifact == has_set:
            raise ValueError("Exactly one of artifact_euid or artifact_set_euid is required")
        return self


class LinkedBucketCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bucket_name: str
    display_name: str | None = None
    bucket_type: str = "secondary"
    description: str | None = None
    prefix_restriction: str | None = None
    read_only: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class LinkedBucketDeleteResponse(BaseModel):
    bucket_id: str
    state: str


class LinkedBucketUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str | None = None
    bucket_type: str | None = None
    description: str | None = None
    prefix_restriction: str | None = None
    read_only: bool | None = None
    metadata: dict[str, Any] | None = None


class LinkedBucketValidationResponse(BaseModel):
    bucket_name: str
    region: str | None
    is_validated: bool
    can_read: bool
    can_write: bool
    can_list: bool
    remediation_steps: list[str]


class AdminS3BucketItemResponse(BaseModel):
    bucket_name: str
    created_at: str | None = None


class AdminS3BucketListResponse(BaseModel):
    profile: str
    buckets: list[AdminS3BucketItemResponse] = Field(default_factory=list)


class BucketFolderCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    folder_name: str


ManifestCreateRequest.model_rebuild()


class UserTokenCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token_name: str
    scope: str = "internal_rw"
    expires_in_days: int = 30
    note: str | None = None


class AdminUserTokenCreateRequest(UserTokenCreateRequest):
    owner_user_id: str
    client_registration_euid: str | None = None


class TokenRevokeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    note: str | None = None


class ClientRegistrationCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_name: str
    owner_user_id: str
    scopes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClusterCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_name: str
    region: str | None = None
    region_az: str
    ssh_key_name: str
    s3_bucket_name: str
    owner_user_id: str | None = None
    contact_email: str | None = None
    pass_on_warn: bool = False
    debug: bool = False


class ClusterCreateOptionsResponse(BaseModel):
    keypairs: list[str] = Field(default_factory=list)
    buckets: list[str] = Field(default_factory=list)
    availability_zones: list[str] = Field(default_factory=list)


class ClusterScanRegionsUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    regions_csv: str

    @model_validator(mode="after")
    def validate_regions_csv(self) -> "ClusterScanRegionsUpdateRequest":
        normalized_regions = parse_regions_csv(self.regions_csv)
        self.regions_csv = ",".join(normalized_regions)
        return self


class ClusterScanRegionsResponse(BaseModel):
    regions: list[str] = Field(default_factory=list)
    regions_csv: str
    config_path: str


class ClusterPartitionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region: str
    region_az: str

    @model_validator(mode="after")
    def validate_region_inputs(self) -> "ClusterPartitionRequest":
        region = str(self.region or "").strip()
        region_az = str(self.region_az or "").strip()
        if not region or not region_az:
            raise ValueError("region and region_az are required")
        if region_az == region or not region_az.startswith(f"{region}"):
            raise ValueError("region_az must identify an availability zone within the region")
        self.region = region
        self.region_az = region_az
        return self


class ClusterPartitionPricingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region: str

    @model_validator(mode="after")
    def validate_region(self) -> "ClusterPartitionPricingRequest":
        region = str(self.region or "").strip()
        if not region:
            raise ValueError("region is required")
        self.region = region
        return self


class ClusterPartitionPricingPointResponse(BaseModel):
    instance_type: str
    hourly_spot_price: float


class ClusterPartitionVerificationItemResponse(BaseModel):
    partition: str
    expected_instance_types: list[str] = Field(default_factory=list)
    spot_available_instance_types: list[str] = Field(default_factory=list)
    missing_instance_types: list[str] = Field(default_factory=list)
    status: Literal["PASS", "WARN", "FAIL"]
    summary: str


class ClusterPartitionVerificationResponse(BaseModel):
    region: str
    region_az: str
    captured_at: str
    cluster_config_path: str
    has_failures: bool
    partitions: list[ClusterPartitionVerificationItemResponse] = Field(default_factory=list)


class ClusterPartitionPricingItemResponse(BaseModel):
    availability_zone: str
    count: int
    min: float | None = None
    q1: float | None = None
    median: float | None = None
    mean: float | None = None
    q3: float | None = None
    max: float | None = None
    points: list[ClusterPartitionPricingPointResponse] = Field(default_factory=list)


class ClusterPartitionPricingPartitionResponse(BaseModel):
    partition: str
    availability_zones: list[ClusterPartitionPricingItemResponse] = Field(default_factory=list)


class ClusterPartitionPricingResponse(BaseModel):
    region: str
    availability_zones: list[str] = Field(default_factory=list)
    captured_at: str
    cluster_config_path: str
    partitions: list[ClusterPartitionPricingPartitionResponse] = Field(default_factory=list)


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
    workset_euid: str | None = None
    run_euid: str
    flowcell_id: str
    lane: str
    library_barcode: str
    sequenced_library_assignment_euid: str
    tenant_id: uuid.UUID
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


class ManifestResponse(BaseModel):
    manifest_euid: str
    name: str
    workset_euid: str
    tenant_id: uuid.UUID
    owner_user_id: str
    artifact_set_euid: str | None
    artifact_euids: list[str]
    input_references: list[dict[str, Any]]
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
    state: str


class WorksetResponse(BaseModel):
    workset_euid: str
    name: str
    tenant_id: uuid.UUID
    owner_user_id: str
    state: str
    artifact_set_euids: list[str]
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
    manifests: list[ManifestResponse]
    analysis_euids: list[str]


class AnalysisCommandPreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    optional_features: list[str] = Field(default_factory=list)
    profile: str | None = None
    region: str | None = None
    cluster_name: str | None = None
    stage_dir: str | None = None
    session_name: str | None = None
    project: str | None = None
    dry_run: bool = False


class AnalysisCommandPreviewResponse(BaseModel):
    valid: bool
    command: dict[str, Any]
    argv: list[str] = Field(default_factory=list)
    shell_preview: str = ""


class AnalysisJobCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_name: str | None = None
    workset_euid: str
    manifest_euid: str
    cluster_name: str
    region: str
    reference_bucket: str | None = None
    analysis_command_id: str
    optional_features: list[str] = Field(default_factory=list)
    session_name: str | None = None
    project: str | None = None
    aws_profile: str | None = None
    dry_run: bool = False
    stage_target: str | None = None
    staging_job_euid: str | None = None

    @model_validator(mode="after")
    def validate_required_fields(self) -> "AnalysisJobCreateRequest":
        for field_name in (
            "workset_euid",
            "manifest_euid",
            "cluster_name",
            "region",
            "analysis_command_id",
        ):
            if not str(getattr(self, field_name) or "").strip():
                raise ValueError(f"{field_name} is required")
        if (
            not str(self.staging_job_euid or "").strip()
            and not str(self.reference_bucket or "").strip()
        ):
            raise ValueError("reference_bucket is required when staging_job_euid is omitted")
        return self


class AnalysisJobLaunchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnalysisJobEventResponse(BaseModel):
    event_euid: str
    job_euid: str
    event_type: str
    status: str
    summary: str
    details: dict[str, Any]
    created_by: str | None
    created_at: str


class AnalysisJobResponse(BaseModel):
    job_euid: str
    job_name: str
    workset_euid: str
    manifest_euid: str
    cluster_name: str
    region: str
    tenant_id: uuid.UUID
    owner_user_id: str
    state: str
    created_at: str
    updated_at: str
    started_at: str | None
    completed_at: str | None
    return_code: int | None
    error: str | None
    output_summary: str | None
    request: dict[str, Any]
    launch: dict[str, Any]
    events: list[AnalysisJobEventResponse]


class StagingJobCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_name: str | None = None
    workset_euid: str
    manifest_euid: str
    cluster_name: str
    region: str
    reference_bucket: str
    stage_target: str | None = None
    aws_profile: str | None = None
    debug: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_required_fields(self) -> "StagingJobCreateRequest":
        for field_name in (
            "workset_euid",
            "manifest_euid",
            "cluster_name",
            "region",
            "reference_bucket",
        ):
            if not str(getattr(self, field_name) or "").strip():
                raise ValueError(f"{field_name} is required")
        return self


class StagingJobRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StagingJobEventResponse(BaseModel):
    event_euid: str
    job_euid: str
    event_type: str
    status: str
    summary: str
    details: dict[str, Any]
    created_by: str | None
    created_at: str


class StagingJobResponse(BaseModel):
    job_euid: str
    job_name: str
    workset_euid: str
    manifest_euid: str
    cluster_name: str
    region: str
    tenant_id: uuid.UUID
    owner_user_id: str
    state: str
    created_at: str
    updated_at: str
    started_at: str | None
    completed_at: str | None
    return_code: int | None
    error: str | None
    output_summary: str | None
    request: dict[str, Any]
    stage: dict[str, Any]
    events: list[StagingJobEventResponse]


class ArtifactImportResponse(BaseModel):
    import_euid: str
    artifact_euid: str
    artifact_type: str
    storage_uri: str
    actor_user_id: str
    created_at: str
    metadata: dict[str, Any]


class LinkedBucketResponse(BaseModel):
    bucket_id: str
    bucket_name: str
    tenant_id: uuid.UUID
    owner_user_id: str
    display_name: str | None
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
    state: str
    bucket_type: str
    description: str | None = None
    prefix_restriction: str | None = None
    read_only: bool = False
    region: str | None = None
    is_validated: bool = False
    can_read: bool = False
    can_write: bool = False
    can_list: bool = False
    remediation_steps: list[str] = Field(default_factory=list)


class UserTokenResponse(BaseModel):
    token_euid: str
    owner_user_id: str
    token_name: str
    token_prefix: str
    scope: str
    status: str
    expires_at: str
    created_at: str
    updated_at: str
    created_by: str | None
    last_used_at: str | None
    revoked_at: str | None
    note: str | None
    client_registration_euid: str | None
    plaintext_token: str | None = None


class TokenUsageResponse(BaseModel):
    usage_euid: str
    token_euid: str
    actor_user_id: str
    endpoint: str
    http_method: str
    response_status: int
    ip_address: str | None
    user_agent: str | None
    request_metadata: dict[str, Any]
    created_at: str


class ClientRegistrationResponse(BaseModel):
    client_registration_euid: str
    client_name: str
    owner_user_id: str
    sponsor_user_id: str
    scopes: list[str]
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
    state: str


class MeResponse(BaseModel):
    user_id: str
    tenant_id: uuid.UUID
    roles: list[str]
    email: str | None
    display_name: str | None
    organization: str | None
    site: str | None
    auth_source: str
    token_euid: str | None
    token_scope: str | None
    client_registration_euid: str | None


class AtlasUserDirectoryResponse(BaseModel):
    user_id: str
    tenant_id: uuid.UUID
    organization_id: str
    organization_name: str | None
    site_id: str | None
    site_name: str | None
    roles: list[str]
    email: str | None
    display_name: str | None
    is_active: bool


class ClusterJobEventResponse(BaseModel):
    event_euid: str
    job_euid: str
    event_type: str
    status: str
    summary: str
    details: dict[str, Any]
    created_by: str | None
    created_at: str


class ClusterJobResponse(BaseModel):
    job_euid: str
    job_name: str
    cluster_name: str
    region: str
    region_az: str
    tenant_id: uuid.UUID
    owner_user_id: str
    sponsor_user_id: str
    state: str
    created_at: str
    updated_at: str
    started_at: str | None
    completed_at: str | None
    return_code: int | None
    error: str | None
    output_summary: str | None
    request: dict[str, Any]
    cluster: dict[str, Any]
    events: list[ClusterJobEventResponse]


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
        workset_euid=record.workset_euid,
        run_euid=record.run_euid,
        flowcell_id=record.flowcell_id,
        lane=record.lane,
        library_barcode=record.library_barcode,
        sequenced_library_assignment_euid=record.sequenced_library_assignment_euid,
        tenant_id=record.tenant_id,
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


def _manifest_response(record: ManifestRecord) -> ManifestResponse:
    return ManifestResponse(**record.__dict__)


def _workset_response(record: WorksetRecord) -> WorksetResponse:
    return WorksetResponse(
        workset_euid=record.workset_euid,
        name=record.name,
        tenant_id=record.tenant_id,
        owner_user_id=record.owner_user_id,
        state=record.state,
        artifact_set_euids=record.artifact_set_euids,
        metadata=record.metadata,
        created_at=record.created_at,
        updated_at=record.updated_at,
        manifests=[_manifest_response(item) for item in record.manifests],
        analysis_euids=record.analysis_euids,
    )


def _canonicalize_workset_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(metadata or {})
    raw_analysis_command = payload.get("analysis_command")
    command_id = str(payload.get("analysis_command_id") or "").strip()
    optional_features = list(payload.get("optional_features") or [])
    if isinstance(raw_analysis_command, dict):
        command_id = command_id or str(raw_analysis_command.get("command_id") or "").strip()
        optional_features = list(raw_analysis_command.get("optional_features") or optional_features)
    if not command_id:
        return payload

    try:
        command = analysis_command_payload(
            command_id,
            optional_features=[str(item) for item in optional_features],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    command_id = str(command.get("command_id") or command_id)
    payload["analysis_command"] = {
        "command_id": command_id,
        "repository": str(command.get("repository") or ""),
        "command_catalog_version": 1,
        "optional_features": [str(item) for item in optional_features],
        "profile": command,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    payload["analysis_command_id"] = command_id
    payload["pipeline_type"] = str(command.get("display_name") or command_id)
    payload["reference_genome"] = str(command.get("genome") or "")
    payload["analysis_repository"] = str(command.get("repository") or "")
    if not payload.get("sample_count"):
        payload["sample_count"] = int(command.get("sample_count") or 0)
    return payload


def _token_response(
    record: UserTokenRecord, *, plaintext_token: str | None = None
) -> UserTokenResponse:
    return UserTokenResponse(
        token_euid=record.token_euid,
        owner_user_id=record.owner_user_id,
        token_name=record.token_name,
        token_prefix=record.token_prefix,
        scope=record.scope,
        status=record.status,
        expires_at=record.expires_at,
        created_at=record.created_at,
        updated_at=record.updated_at,
        created_by=record.created_by,
        last_used_at=record.last_used_at,
        revoked_at=record.revoked_at,
        note=record.note,
        client_registration_euid=record.client_registration_euid,
        plaintext_token=plaintext_token,
    )


def _token_usage_response(record: UserTokenUsageRecord) -> TokenUsageResponse:
    return TokenUsageResponse(**record.__dict__)


def _client_registration_response(record: ClientRegistrationRecord) -> ClientRegistrationResponse:
    return ClientRegistrationResponse(**record.__dict__)


def _dewey_import_response(record: DeweyImportRecord) -> ArtifactImportResponse:
    return ArtifactImportResponse(**record.__dict__)


def _linked_bucket_response(record: LinkedBucketRecord) -> LinkedBucketResponse:
    return LinkedBucketResponse(**record.__dict__)


def _me_response(actor: CurrentUser) -> MeResponse:
    return MeResponse(
        user_id=actor.user_id,
        tenant_id=actor.tenant_id,
        roles=list(actor.roles),
        email=actor.email,
        display_name=actor.display_name,
        organization=actor.organization,
        site=actor.site,
        auth_source=actor.auth_source,
        token_euid=actor.token_euid,
        token_scope=actor.token_scope,
        client_registration_euid=actor.client_registration_euid,
    )


def _atlas_user_directory_response(entry: AtlasUserDirectoryEntry) -> AtlasUserDirectoryResponse:
    return AtlasUserDirectoryResponse(
        user_id=entry.user_id,
        tenant_id=entry.tenant_id,
        organization_id=entry.organization_id,
        organization_name=entry.organization_name,
        site_id=entry.site_id,
        site_name=entry.site_name,
        roles=list(entry.roles),
        email=entry.email,
        display_name=entry.display_name,
        is_active=entry.is_active,
    )


def _cluster_job_event_response(record: ClusterJobEventRecord) -> ClusterJobEventResponse:
    return ClusterJobEventResponse(**record.__dict__)


def _cluster_job_response(record: ClusterJobRecord) -> ClusterJobResponse:
    return ClusterJobResponse(
        job_euid=record.job_euid,
        job_name=record.job_name,
        cluster_name=record.cluster_name,
        region=record.region,
        region_az=record.region_az,
        tenant_id=record.tenant_id,
        owner_user_id=record.owner_user_id,
        sponsor_user_id=record.sponsor_user_id,
        state=record.state,
        created_at=record.created_at,
        updated_at=record.updated_at,
        started_at=record.started_at,
        completed_at=record.completed_at,
        return_code=record.return_code,
        error=record.error,
        output_summary=record.output_summary,
        request=record.request,
        cluster=record.cluster,
        events=[_cluster_job_event_response(item) for item in record.events],
    )


def _analysis_job_event_response(record: AnalysisJobEventRecord) -> AnalysisJobEventResponse:
    return AnalysisJobEventResponse(**record.__dict__)


def _analysis_job_response(record: AnalysisJobRecord) -> AnalysisJobResponse:
    return AnalysisJobResponse(
        job_euid=record.job_euid,
        job_name=record.job_name,
        workset_euid=record.workset_euid,
        manifest_euid=record.manifest_euid,
        cluster_name=record.cluster_name,
        region=record.region,
        tenant_id=record.tenant_id,
        owner_user_id=record.owner_user_id,
        state=record.state,
        created_at=record.created_at,
        updated_at=record.updated_at,
        started_at=record.started_at,
        completed_at=record.completed_at,
        return_code=record.return_code,
        error=record.error,
        output_summary=record.output_summary,
        request=record.request,
        launch=record.launch,
        events=[_analysis_job_event_response(item) for item in record.events],
    )


def _staging_job_event_response(record: StagingJobEventRecord) -> StagingJobEventResponse:
    return StagingJobEventResponse(**record.__dict__)


def _staging_job_response(record: StagingJobRecord) -> StagingJobResponse:
    return StagingJobResponse(
        job_euid=record.job_euid,
        job_name=record.job_name,
        workset_euid=record.workset_euid,
        manifest_euid=record.manifest_euid,
        cluster_name=record.cluster_name,
        region=record.region,
        tenant_id=record.tenant_id,
        owner_user_id=record.owner_user_id,
        state=record.state,
        created_at=record.created_at,
        updated_at=record.updated_at,
        started_at=record.started_at,
        completed_at=record.completed_at,
        return_code=record.return_code,
        error=record.error,
        output_summary=record.output_summary,
        request=record.request,
        stage=record.stage,
        events=[_staging_job_event_response(item) for item in record.events],
    )


def _parse_s3_object_uri(value: str) -> tuple[str, str]:
    parsed = urlparse(str(value or "").strip())
    bucket = str(parsed.netloc or "").strip()
    key = str(parsed.path or "").strip().lstrip("/")
    if parsed.scheme != "s3" or not bucket or not key:
        raise ValueError("Expected s3://<bucket>/<key> object URI")
    return bucket, key


def _guess_artifact_type(storage_uri: str) -> str:
    lower = str(storage_uri or "").lower()
    suffix_map = (
        (".fastq.gz", "fastq"),
        (".fq.gz", "fastq"),
        (".fastq", "fastq"),
        (".fq", "fastq"),
        (".bam", "bam"),
        (".cram", "cram"),
        (".vcf.gz", "vcf"),
        (".vcf", "vcf"),
        (".g.vcf.gz", "vcf"),
        (".gvcf.gz", "vcf"),
    )
    for suffix, artifact_type in suffix_map:
        if lower.endswith(suffix):
            return artifact_type
    return "file"


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


def _normalize_bucket_name(value: str) -> str:
    bucket_name = str(normalize_bucket_name(value) or "").strip()
    if not bucket_name:
        raise ValueError("bucket_name is required")
    return bucket_name


def _normalize_prefix(value: str | None) -> str | None:
    prefix = str(value or "").strip().lstrip("/")
    if not prefix:
        return None
    return prefix.rstrip("/") + "/"


def _object_within_prefix(*, key: str, prefix_restriction: str | None) -> bool:
    normalized_key = str(key or "").lstrip("/")
    prefix = _normalize_prefix(prefix_restriction)
    if prefix is None:
        return bool(normalized_key)
    return normalized_key.startswith(prefix)


def _preview_s3_object(
    s3_client: RegionAwareS3Client,
    *,
    bucket_name: str,
    key: str,
    lines: int = 20,
) -> dict[str, Any]:
    head = s3_client.head_object(Bucket=bucket_name, Key=key)
    file_size = int(head.get("ContentLength") or 0)
    content_type = str(head.get("ContentType") or "application/octet-stream")
    file_lower = key.lower()
    is_gzip = file_lower.endswith(".gz") or file_lower.endswith(".gzip")
    is_tar_gz = file_lower.endswith(".tar.gz") or file_lower.endswith(".tgz")
    is_zip = file_lower.endswith(".zip")

    text_extensions = {
        ".txt",
        ".log",
        ".csv",
        ".tsv",
        ".json",
        ".xml",
        ".html",
        ".htm",
        ".yaml",
        ".yml",
        ".md",
        ".rst",
        ".py",
        ".js",
        ".ts",
        ".sh",
        ".bash",
        ".r",
        ".pl",
        ".rb",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".fastq",
        ".fq",
        ".fasta",
        ".fa",
        ".sam",
        ".vcf",
        ".bed",
        ".gff",
        ".gtf",
    }

    base_name = key
    if is_gzip and not is_tar_gz:
        base_name = key[:-3] if file_lower.endswith(".gz") else key[:-5]
    ext = "." + base_name.split(".")[-1] if "." in base_name else ""
    is_text = ext.lower() in text_extensions or content_type.startswith("text/")
    max_download = 10 * 1024 * 1024
    if file_size > max_download:
        response = s3_client.get_object(
            Bucket=bucket_name, Key=key, Range=f"bytes=0-{max_download}"
        )
    else:
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
    body = response["Body"].read()
    preview_lines: list[str] = []
    file_type = "text"

    if is_tar_gz:
        file_type = "tar.gz"
        try:
            with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as archive:
                members = archive.getnames()
                preview_lines.append(f"=== Archive contents ({len(members)} files) ===")
                preview_lines.extend(members[:20])
                if len(members) > 20:
                    preview_lines.append(f"... and {len(members) - 20} more files")
        except Exception as exc:  # pragma: no cover - defensive parsing branch
            preview_lines = [f"Error reading tar.gz: {exc}"]
    elif is_gzip:
        file_type = "gzip"
        try:
            decompressed = gzip.decompress(body)
            text = decompressed.decode("utf-8", errors="replace")
            preview_lines = text.splitlines()[:lines]
        except Exception as exc:  # pragma: no cover - defensive parsing branch
            preview_lines = [f"Error decompressing: {exc}"]
    elif is_zip:
        file_type = "zip"
        try:
            with zipfile.ZipFile(io.BytesIO(body)) as archive:
                names = archive.namelist()
                preview_lines.append(f"=== Archive contents ({len(names)} files) ===")
                preview_lines.extend(names[:20])
                if len(names) > 20:
                    preview_lines.append(f"... and {len(names) - 20} more files")
        except Exception as exc:  # pragma: no cover - defensive parsing branch
            preview_lines = [f"Error reading zip: {exc}"]
    elif is_text or file_size < 1024 * 1024:
        try:
            text = body.decode("utf-8", errors="replace")
            preview_lines = text.splitlines()[:lines]
        except Exception:
            file_type = "binary"
            preview_lines = ["[Binary file - preview not available]"]
    else:
        file_type = "binary"
        preview_lines = ["[Binary file - preview not available]"]

    return {
        "filename": key.split("/")[-1],
        "file_type": file_type,
        "size": file_size,
        "lines": preview_lines,
        "total_lines": len(preview_lines),
        "truncated": len(preview_lines) >= lines,
    }


def _validate_bucket_access(
    s3_client: RegionAwareS3Client,
    *,
    bucket_name: str,
    prefix_restriction: str | None,
    read_only: bool,
) -> LinkedBucketValidationResponse:
    normalized_bucket = _normalize_bucket_name(bucket_name)
    normalized_prefix = _normalize_prefix(prefix_restriction)
    region: str | None = None
    can_read = False
    can_write = False
    can_list = False
    remediation_steps: list[str] = []

    try:
        location = s3_client.get_bucket_location(Bucket=normalized_bucket)
        region = str(location.get("LocationConstraint") or "us-east-1")
    except ClientError:
        remediation_steps.append(
            "Grant s3:GetBucketLocation on the bucket so Ursa can determine the region."
        )

    try:
        s3_client.list_objects_v2(
            Bucket=normalized_bucket,
            Prefix=normalized_prefix or "",
            Delimiter="/",
            MaxKeys=1,
        )
        can_list = True
        can_read = True
    except ClientError:
        remediation_steps.append(
            "Grant s3:ListBucket on the bucket and ensure the prefix restriction is correct."
        )

    if read_only:
        can_write = False
    else:
        validation_key = f"{normalized_prefix or ''}.ursa-validation-{secrets.token_hex(6)}"
        try:
            s3_client.put_object(
                Bucket=normalized_bucket,
                Key=validation_key,
                Body=b"ursa bucket validation",
                ContentType="text/plain",
            )
            can_write = True
            s3_client.delete_object(Bucket=normalized_bucket, Key=validation_key)
        except ClientError:
            remediation_steps.append(
                "Grant s3:PutObject and s3:DeleteObject on the bucket prefix for write-enabled buckets."
            )

    is_validated = can_list and can_read and (read_only or can_write)
    if not remediation_steps and is_validated:
        remediation_steps.append("Bucket access validated successfully.")
    return LinkedBucketValidationResponse(
        bucket_name=normalized_bucket,
        region=region,
        is_validated=is_validated,
        can_read=can_read,
        can_write=can_write,
        can_list=can_list,
        remediation_steps=remediation_steps,
    )


def _detect_file_format(filename: str) -> str | None:
    lower = str(filename or "").lower()
    format_map = (
        (".fastq.gz", "fastq"),
        (".fq.gz", "fastq"),
        (".fastq", "fastq"),
        (".fq", "fastq"),
        (".bam", "bam"),
        (".cram", "cram"),
        (".vcf.gz", "vcf"),
        (".vcf", "vcf"),
        (".tsv", "tsv"),
        (".csv", "csv"),
        (".txt", "txt"),
    )
    for suffix, label in format_map:
        if lower.endswith(suffix):
            return label
    return None


def _format_file_size(size_bytes: int | None) -> str:
    size = int(size_bytes or 0)
    if size < 1024:
        return f"{size} B"
    units = ["KB", "MB", "GB", "TB"]
    scaled = float(size)
    for unit in units:
        scaled /= 1024.0
        if scaled < 1024.0 or unit == units[-1]:
            precision = 0 if scaled >= 100 else 1
            return f"{scaled:.{precision}f} {unit}"
    return f"{size} B"


def _normalize_euid_list(values: list[str] | None, *, label: str) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in list(values or []):
        value = str(raw or "").strip()
        if not value:
            raise HTTPException(status_code=400, detail=f"{label} entries must not be empty")
        if value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def _load_daylily_ec_pricing_helpers() -> tuple[Any, Any, Any]:
    require_daylily_ec_version()
    try:
        from daylily_ec.aws.pricing_snapshots import (
            collect_pricing_snapshot,
            load_partition_instance_types,
            resolve_cluster_config_path,
        )
    except Exception as exc:  # pragma: no cover - exercised via integration
        raise RuntimeError(
            f"Failed to import daylily-ec {REQUIRED_DAYLILY_EC_VERSION} pricing helpers: {exc}"
        ) from exc
    return collect_pricing_snapshot, load_partition_instance_types, resolve_cluster_config_path


def resolve_daylily_cluster_config_path(settings: Settings) -> Path:
    _, _, resolve_cluster_config_path = _load_daylily_ec_pricing_helpers()
    configured = str(settings.ursa_cost_monitor_config_path or "").strip() or None
    return Path(resolve_cluster_config_path(configured))


def _load_cluster_partition_names(cluster_config_path: Path) -> list[str]:
    import yaml

    payload = yaml.safe_load(cluster_config_path.read_text(encoding="utf-8")) or {}
    queues = payload.get("Scheduling", {}).get("SlurmQueues", [])
    partitions: list[str] = []
    seen: set[str] = set()
    for queue in list(queues or []):
        if not isinstance(queue, dict):
            continue
        partition = str(queue.get("Name") or queue.get("QueueName") or "").strip()
        if not partition or partition in seen:
            continue
        seen.add(partition)
        partitions.append(partition)
    if not partitions:
        raise RuntimeError(f"No Slurm queues found in cluster config: {cluster_config_path}")
    return partitions


def load_daylily_partition_instance_types(settings: Settings) -> tuple[Path, dict[str, list[str]]]:
    _, load_partition_instance_types, _ = _load_daylily_ec_pricing_helpers()
    cluster_config_path = resolve_daylily_cluster_config_path(settings)
    partitions = _load_cluster_partition_names(cluster_config_path)
    partition_map = load_partition_instance_types(
        cluster_config_path=str(cluster_config_path),
        partitions=partitions,
    )
    return cluster_config_path, {
        partition: sorted(
            {
                str(instance_type).strip()
                for instance_type in list(partition_map.get(partition) or [])
                if str(instance_type).strip()
            }
        )
        for partition in partitions
    }


def collect_daylily_cluster_pricing_snapshot(
    settings: Settings,
    *,
    region: str,
    partitions: Sequence[str],
) -> dict[str, Any]:
    collect_pricing_snapshot, _, _ = _load_daylily_ec_pricing_helpers()
    cluster_config_path = resolve_daylily_cluster_config_path(settings)
    snapshot = collect_pricing_snapshot(
        regions=[region],
        partitions=list(partitions),
        cluster_config_path=str(cluster_config_path),
        profile=str(settings.aws_profile or "").strip() or None,
    )
    if isinstance(snapshot, dict):
        return snapshot
    if hasattr(snapshot, "to_dict"):
        return snapshot.to_dict()
    raise RuntimeError("Unsupported daylily-ec pricing snapshot payload")


def resolve_cluster_partition_selection(
    *,
    region: str | None,
    region_az: str,
) -> ClusterPartitionRequest:
    normalized_region_az = str(region_az or "").strip()
    normalized_region = str(region or "").strip() or region_from_region_az(normalized_region_az)
    return ClusterPartitionRequest(region=normalized_region, region_az=normalized_region_az)


def _partition_points_for_region_az(
    snapshot: dict[str, Any],
    *,
    region: str,
    region_az: str,
) -> dict[str, list[dict[str, Any]]]:
    points_by_partition: dict[str, list[dict[str, Any]]] = {}
    for raw_point in list(snapshot.get("points") or []):
        if str(raw_point.get("region") or "").strip() != region:
            continue
        if str(raw_point.get("availability_zone") or "").strip() != region_az:
            continue
        partition = str(raw_point.get("partition") or "").strip()
        instance_type = str(raw_point.get("instance_type") or "").strip()
        if not partition or not instance_type:
            continue
        try:
            hourly_spot_price = float(raw_point.get("hourly_spot_price"))
        except (TypeError, ValueError):
            continue
        points_by_partition.setdefault(partition, []).append(
            {
                "instance_type": instance_type,
                "hourly_spot_price": hourly_spot_price,
            }
        )
    for partition_points in points_by_partition.values():
        partition_points.sort(key=lambda item: str(item["instance_type"]))
    return points_by_partition


def _partition_points_for_region(
    snapshot: dict[str, Any],
    *,
    region: str,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], list[str]]:
    points_by_partition: dict[str, dict[str, list[dict[str, Any]]]] = {}
    availability_zones: set[str] = set()
    for raw_point in list(snapshot.get("points") or []):
        if str(raw_point.get("region") or "").strip() != region:
            continue
        availability_zone = str(raw_point.get("availability_zone") or "").strip()
        partition = str(raw_point.get("partition") or "").strip()
        instance_type = str(raw_point.get("instance_type") or "").strip()
        if not availability_zone or not partition or not instance_type:
            continue
        try:
            hourly_spot_price = float(raw_point.get("hourly_spot_price"))
        except (TypeError, ValueError):
            continue
        availability_zones.add(availability_zone)
        points_by_partition.setdefault(partition, {}).setdefault(availability_zone, []).append(
            {
                "instance_type": instance_type,
                "hourly_spot_price": hourly_spot_price,
            }
        )
    for partition_points in points_by_partition.values():
        for availability_zone_points in partition_points.values():
            availability_zone_points.sort(key=lambda item: str(item["instance_type"]))
    return points_by_partition, sorted(availability_zones)


def _pricing_stats(hourly_prices: list[float]) -> dict[str, float | int | None]:
    count = len(hourly_prices)
    return {
        "count": count,
        "min": round(min(hourly_prices), 8) if hourly_prices else None,
        "q1": round(pricing_quantile(hourly_prices, 0.25), 8) if hourly_prices else None,
        "median": round(pricing_quantile(hourly_prices, 0.5), 8) if hourly_prices else None,
        "mean": round(sum(hourly_prices) / count, 8) if hourly_prices else None,
        "q3": round(pricing_quantile(hourly_prices, 0.75), 8) if hourly_prices else None,
        "max": round(max(hourly_prices), 8) if hourly_prices else None,
    }


def build_cluster_partition_verification(
    *,
    region: str,
    region_az: str,
    cluster_config_path: Path,
    partition_instances: dict[str, list[str]],
    snapshot: dict[str, Any],
) -> ClusterPartitionVerificationResponse:
    points_by_partition = _partition_points_for_region_az(
        snapshot, region=region, region_az=region_az
    )
    items: list[ClusterPartitionVerificationItemResponse] = []
    has_failures = False

    for partition, expected_instance_types in partition_instances.items():
        available_instance_types = sorted(
            {
                str(point["instance_type"])
                for point in list(points_by_partition.get(partition) or [])
                if str(point.get("instance_type") or "").strip()
            }
        )
        missing_instance_types = [
            instance_type
            for instance_type in expected_instance_types
            if instance_type not in available_instance_types
        ]
        if expected_instance_types and not missing_instance_types:
            status_value: Literal["PASS", "WARN", "FAIL"] = "PASS"
            summary = (
                f"All {len(expected_instance_types)} configured instance types have current Spot "
                f"price data in {region_az}."
            )
        elif available_instance_types:
            status_value = "WARN"
            summary = (
                f"Spot price data is available for {len(available_instance_types)} of "
                f"{len(expected_instance_types)} configured instance types in {region_az}."
            )
        else:
            status_value = "FAIL"
            summary = f"No configured instance types for {partition} have current Spot price data in {region_az}."
            has_failures = True
        items.append(
            ClusterPartitionVerificationItemResponse(
                partition=partition,
                expected_instance_types=list(expected_instance_types),
                spot_available_instance_types=available_instance_types,
                missing_instance_types=missing_instance_types,
                status=status_value,
                summary=summary,
            )
        )

    return ClusterPartitionVerificationResponse(
        region=region,
        region_az=region_az,
        captured_at=str(snapshot.get("captured_at") or ""),
        cluster_config_path=str(cluster_config_path),
        has_failures=has_failures,
        partitions=items,
    )


def run_cluster_partition_verification(
    settings: Settings,
    *,
    region: str,
    region_az: str,
) -> ClusterPartitionVerificationResponse:
    cluster_config_path, partition_instances = load_daylily_partition_instance_types(settings)
    snapshot = collect_daylily_cluster_pricing_snapshot(
        settings,
        region=region,
        partitions=list(partition_instances.keys()),
    )
    return build_cluster_partition_verification(
        region=region,
        region_az=region_az,
        cluster_config_path=cluster_config_path,
        partition_instances=partition_instances,
        snapshot=snapshot,
    )


def build_cluster_partition_pricing(
    *,
    region: str,
    cluster_config_path: Path,
    partition_instances: dict[str, list[str]],
    snapshot: dict[str, Any],
) -> ClusterPartitionPricingResponse:
    points_by_partition, availability_zones = _partition_points_for_region(snapshot, region=region)
    partitions: list[ClusterPartitionPricingPartitionResponse] = []

    for partition in partition_instances:
        pricing_items: list[ClusterPartitionPricingItemResponse] = []
        for availability_zone in availability_zones:
            raw_points = [
                ClusterPartitionPricingPointResponse(
                    instance_type=str(point["instance_type"]),
                    hourly_spot_price=round(float(point["hourly_spot_price"]), 8),
                )
                for point in list(
                    points_by_partition.get(partition, {}).get(availability_zone) or []
                )
            ]
            hourly_prices = sorted(point.hourly_spot_price for point in raw_points)
            pricing_items.append(
                ClusterPartitionPricingItemResponse(
                    availability_zone=availability_zone,
                    points=raw_points,
                    **_pricing_stats(hourly_prices),
                )
            )
        partitions.append(
            ClusterPartitionPricingPartitionResponse(
                partition=partition,
                availability_zones=pricing_items,
            )
        )

    return ClusterPartitionPricingResponse(
        region=region,
        availability_zones=availability_zones,
        captured_at=str(snapshot.get("captured_at") or ""),
        cluster_config_path=str(cluster_config_path),
        partitions=partitions,
    )


def create_app(
    store: AnalysisStore,
    *,
    bloom_client: BloomResolverClient,
    atlas_client: AtlasResultClient | None = None,
    dewey_client: DeweyClient | None = None,
    resource_store: ResourceStore | None = None,
    token_service: UserTokenService | None = None,
    auth_provider: CognitoAuthProvider | None = None,
    user_directory: CognitoUserDirectoryService | None = None,
    cluster_service: ClusterService | None = None,
    analysis_job_manager: AnalysisJobManager | None = None,
    staging_job_manager: StagingJobManager | None = None,
    settings: Settings | None = None,
    require_api_key: bool | None = None,
    s3_client: Any | None = None,
) -> FastAPI:
    if settings is None:
        settings = get_settings()

    if require_api_key is False:
        raise ValueError("Ursa write API key enforcement cannot be disabled")

    internal_bucket = str(getattr(settings, "ursa_internal_output_bucket", "") or "").strip()
    if not internal_bucket:
        raise ValueError("ursa_internal_output_bucket is required")

    app = FastAPI(
        title="Daylily Ursa Backend API",
        description="Versioned backend APIs for analyses, worksets, manifests, tokens, and admin surfaces",
        version=__version__,
    )
    app.state.store = store
    app.state.bloom_client = bloom_client
    app.state.atlas_client = atlas_client
    app.state.dewey_client = dewey_client
    app.state.settings = settings
    app.state.ursa_config = get_ursa_config()
    app.state.s3_client = s3_client or RegionAwareS3Client(
        default_region=settings.get_effective_region(),
        profile=settings.aws_profile,
    )
    app.state.internal_bucket = internal_bucket
    app.state.require_api_key = True
    app.state.api_key = settings.ursa_internal_api_key
    app.state.observability = UrsaObservabilityStore(
        settings=settings,
        app_version=__version__,
    )

    if auth_provider is None:
        auth_provider = CognitoAuthProvider(
            user_pool_id=str(getattr(settings, "cognito_user_pool_id", "") or "").strip(),
            app_client_id=str(getattr(settings, "cognito_app_client_id", "") or "").strip(),
            region=str(
                getattr(settings, "cognito_region", "") or settings.get_effective_region()
            ).strip(),
        )
    app.state.auth_provider = auth_provider

    if (
        user_directory is None
        and str(getattr(settings, "cognito_user_pool_id", "") or "").strip()
        and str(getattr(settings, "cognito_region", "") or "").strip()
    ):
        user_directory = CognitoUserDirectoryService(
            user_pool_id=str(settings.cognito_user_pool_id or "").strip(),
            region=str(settings.cognito_region or "").strip(),
            profile=settings.aws_profile,
        )
    app.state.user_directory = user_directory

    if resource_store is None and hasattr(store, "backend"):
        resource_store = ResourceStore(backend=store.backend)
    app.state.resource_store = resource_store

    if cluster_service is None:
        cluster_service = ClusterService(
            regions=settings.get_allowed_regions(),
            aws_profile=settings.aws_profile,
        )
    app.state.cluster_service = cluster_service

    if token_service is None and resource_store is not None and hasattr(resource_store, "backend"):
        token_service = UserTokenService(
            backend=resource_store.backend,
            user_lookup=user_directory.get_user if user_directory is not None else None,
        )
    app.state.token_service = token_service
    app.state.cluster_job_manager = (
        ClusterJobManager(
            resource_store=resource_store,
            cluster_service=cluster_service,
            workspace_root=Path.cwd(),
        )
        if resource_store is not None
        else None
    )
    app.state.analysis_job_manager = analysis_job_manager or (
        AnalysisJobManager(
            resource_store=resource_store,
            client=cluster_service.client,
            workspace_root=Path.cwd(),
        )
        if resource_store is not None and hasattr(cluster_service, "client")
        else None
    )
    app.state.staging_job_manager = staging_job_manager or (
        StagingJobManager(
            resource_store=resource_store,
            client=cluster_service.client,
            workspace_root=Path.cwd(),
        )
        if resource_store is not None and hasattr(cluster_service, "client")
        else None
    )
    app.state.observability_cleanup = []

    def _anomaly_repository():
        resource_store = getattr(app.state, "resource_store", None)
        token_service = getattr(app.state, "token_service", None)
        backend = getattr(resource_store, "backend", None) or getattr(
            token_service, "backend", None
        )
        if resource_store is None or backend is None:
            raise HTTPException(status_code=503, detail="Anomaly repository is not configured")
        return open_anomaly_repository(
            resource_store=resource_store,
            settings=settings,
            backend=backend,
        )

    def _extract_sqlalchemy_engine(candidate: Any) -> Any | None:
        backend = getattr(candidate, "backend", None)
        if backend is None:
            return None
        for engine_candidate in (
            getattr(backend, "engine", None),
            getattr(getattr(backend, "bundle", None), "connection", None),
            getattr(getattr(backend, "_conn", None), "engine", None),
        ):
            engine = (
                getattr(engine_candidate, "engine", None) if engine_candidate is not None else None
            )
            if engine is not None:
                return engine
            if engine_candidate is not None and hasattr(engine_candidate, "connect"):
                return engine_candidate
        return None

    def _install_observability_hooks() -> None:
        seen_engines: set[int] = set()
        cleanup_callbacks: list[Any] = []
        for candidate in (app.state.store, app.state.resource_store, app.state.token_service):
            engine = _extract_sqlalchemy_engine(candidate)
            if engine is None:
                continue
            engine_id = id(engine)
            if engine_id in seen_engines:
                continue
            seen_engines.add(engine_id)
            cleanup_callbacks.append(
                install_sqlalchemy_observability(app.state.observability, engine)
            )
        app.state.observability_cleanup = cleanup_callbacks

    def _correlation_id(source: str) -> str:
        return hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]  # nosec B324 — non-security correlation ID

    def _route_template_for_request(request: Request) -> str:
        route = request.scope.get("route")
        template = getattr(route, "path", None)
        if template:
            return str(template)
        return "/__unmatched__"

    def _database_probe() -> dict[str, Any]:
        observed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        engine = _extract_sqlalchemy_engine(app.state.store) or _extract_sqlalchemy_engine(
            app.state.resource_store
        )
        if engine is None:
            payload = {
                "status": "unknown",
                "latency_ms": None,
                "detail": "sqlalchemy_engine_unavailable",
                "observed_at": observed_at,
            }
            app.state.observability.record_db_probe(
                status="unknown",
                latency_ms=0.0,
                detail="sqlalchemy_engine_unavailable",
            )
            return payload

        started_at = time.monotonic()
        try:
            with engine.connect() as connection:
                connection.exec_driver_sql("SELECT 1")
            latency_ms = (time.monotonic() - started_at) * 1000
            payload = {
                "status": "ok",
                "latency_ms": round(latency_ms, 3),
                "detail": "select_1_ok",
                "observed_at": observed_at,
            }
            app.state.observability.record_db_probe(
                status="ok",
                latency_ms=latency_ms,
                detail="select_1_ok",
            )
            return payload
        except Exception as exc:
            latency_ms = (time.monotonic() - started_at) * 1000
            detail = f"select_1_failed:{type(exc).__name__}"
            payload = {
                "status": "error",
                "latency_ms": round(latency_ms, 3),
                "detail": detail,
                "observed_at": observed_at,
            }
            app.state.observability.record_db_probe(
                status="error",
                latency_ms=latency_ms,
                detail=detail,
            )
            return payload

    _install_observability_hooks()

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    allow_local_domain_access = not settings.is_production
    app.state.server_instance_id = secrets.token_urlsafe(16)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=build_trusted_hosts(allow_local=allow_local_domain_access),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[],
        allow_origin_regex=build_allowed_origin_regex(allow_local=allow_local_domain_access),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    configure_session_middleware(
        app,
        build_web_session_config(settings, app.state.server_instance_id),
    )

    @app.middleware("http")
    async def enforce_origin_allowlist(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and not is_allowed_origin(origin, allow_local=allow_local_domain_access):
            return PlainTextResponse("Origin not allowed", status_code=403)
        return await call_next(request)

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or secrets.token_hex(4)
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.middleware("http")
    async def record_observability(request: Request, call_next):
        request_id = str(getattr(request.state, "request_id", "") or secrets.token_hex(4))
        correlation_source = (
            request.headers.get("X-Correlation-ID")
            or request.headers.get("X-Request-ID")
            or request_id
        )
        request.state.correlation_id = _correlation_id(str(correlation_source))
        started_at = time.monotonic()
        response = await call_next(request)
        route_template = _route_template_for_request(request)
        app.state.observability.record_http_request(
            method=request.method,
            route_template=route_template,
            status_code=response.status_code,
            duration_ms=(time.monotonic() - started_at) * 1000,
        )
        response.headers["X-Correlation-ID"] = request.state.correlation_id
        return response

    @app.middleware("http")
    async def log_ursa_token_usage(request: Request, call_next):
        response = await call_next(request)
        usage = getattr(request.state, "user_token_usage", None)
        service: UserTokenService | None = getattr(app.state, "token_service", None)
        if usage and service is not None:
            try:
                service.log_usage(
                    token_euid=str(usage.get("token_euid") or ""),
                    actor_user_id=str(usage.get("actor_user_id") or ""),
                    endpoint=request.url.path,
                    http_method=request.method,
                    response_status=response.status_code,
                    ip_address=request.client.host if request.client else None,
                    user_agent=request.headers.get("user-agent"),
                    request_metadata={"request_id": getattr(request.state, "request_id", "")},
                )
            except Exception:
                LOGGER.exception("Failed to log Ursa token usage for %s", request.url.path)
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
        expected = str(app.state.api_key or "")
        provided = str(x_api_key or "")
        if not provided or not hmac.compare_digest(provided, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
            )
        return provided

    def require_resource_store() -> ResourceStore:
        resource_backend = getattr(app.state, "resource_store", None)
        if resource_backend is None:
            raise HTTPException(status_code=503, detail="Resource store is not configured")
        return resource_backend

    def require_token_service() -> UserTokenService:
        service = getattr(app.state, "token_service", None)
        if service is None:
            raise HTTPException(status_code=503, detail="User token service is not configured")
        return service

    def require_user_directory() -> CognitoUserDirectoryService:
        directory = getattr(app.state, "user_directory", None)
        if directory is None:
            raise HTTPException(status_code=503, detail="User directory is not configured")
        return directory

    def require_cluster_service() -> ClusterService:
        service = getattr(app.state, "cluster_service", None)
        if service is None:
            raise HTTPException(status_code=503, detail="Cluster service is not configured")
        return service

    def require_cluster_job_manager() -> ClusterJobManager:
        manager = getattr(app.state, "cluster_job_manager", None)
        if manager is None:
            raise HTTPException(status_code=503, detail="Cluster job manager is not configured")
        return manager

    def require_analysis_job_manager() -> AnalysisJobManager:
        manager = getattr(app.state, "analysis_job_manager", None)
        if manager is None:
            raise HTTPException(status_code=503, detail="Analysis job manager is not configured")
        return manager

    def require_staging_job_manager() -> StagingJobManager:
        manager = getattr(app.state, "staging_job_manager", None)
        if manager is None:
            raise HTTPException(status_code=503, detail="Staging job manager is not configured")
        return manager

    def require_dewey_client() -> DeweyClient:
        client = getattr(app.state, "dewey_client", None)
        if client is None:
            raise HTTPException(status_code=503, detail="Dewey client is not configured")
        return client

    def record_observed_dependency(service_id: str) -> None:
        try:
            app.state.observability.record_observed_dependency(service_id)
        except Exception:
            return

    def resolve_dewey_artifact_euid(artifact_euid: str) -> str:
        dewey_client = require_dewey_client()
        try:
            resolved = dewey_client.resolve_artifact(artifact_euid)
        except DeweyClientError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        record_observed_dependency("dewey")
        canonical = str(resolved.get("artifact_euid") or "").strip()
        if not canonical:
            raise HTTPException(
                status_code=502, detail="Dewey resolve response missing artifact_euid"
            )
        return canonical

    def resolve_dewey_artifact_set_payload(artifact_set_euid: str) -> dict[str, Any]:
        dewey_client = require_dewey_client()
        try:
            resolved = dewey_client.resolve_artifact_set(artifact_set_euid)
        except DeweyClientError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        record_observed_dependency("dewey")
        canonical = str(resolved.get("artifact_set_euid") or "").strip()
        if not canonical:
            raise HTTPException(
                status_code=502,
                detail="Dewey artifact-set resolve response missing artifact_set_euid",
            )
        return resolved

    def validate_workset_artifact_sets(artifact_set_euids: list[str]) -> list[str]:
        normalized = _normalize_euid_list(artifact_set_euids, label="artifact_set_euids")
        canonical: list[str] = []
        for artifact_set_euid in normalized:
            resolved = resolve_dewey_artifact_set_payload(artifact_set_euid)
            canonical.append(str(resolved.get("artifact_set_euid") or artifact_set_euid).strip())
        return canonical

    def validate_manifest_artifact_references(
        artifact_set_euid: str,
        artifact_euids: list[str],
    ) -> tuple[str, list[str], list[dict[str, Any]]]:
        normalized_set_euid = str(artifact_set_euid or "").strip()
        if not normalized_set_euid:
            raise HTTPException(status_code=400, detail="artifact_set_euid is required")
        resolved_set = resolve_dewey_artifact_set_payload(normalized_set_euid)
        canonical_set_euid = str(
            resolved_set.get("artifact_set_euid") or normalized_set_euid
        ).strip()
        allowed_member_euids = {
            str(member.get("artifact_euid") or "").strip()
            for member in list(resolved_set.get("members") or [])
            if isinstance(member, dict) and str(member.get("artifact_euid") or "").strip()
        }

        canonical_artifact_euids: list[str] = []
        for artifact_euid in _normalize_euid_list(artifact_euids, label="artifact_euids"):
            canonical_artifact_euid = resolve_dewey_artifact_euid(artifact_euid)
            if allowed_member_euids and canonical_artifact_euid not in allowed_member_euids:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Artifact {canonical_artifact_euid} is not a member of artifact set "
                        f"{canonical_set_euid}"
                    ),
                )
            canonical_artifact_euids.append(canonical_artifact_euid)
        input_references: list[dict[str, Any]] = [
            {
                "reference_type": "artifact_set_euid",
                "value": normalized_set_euid,
                "artifact_set_euid": canonical_set_euid,
                "artifact_euids": sorted(allowed_member_euids),
            }
        ]
        input_references.extend(
            {
                "reference_type": "artifact_euid",
                "value": artifact_euid,
                "artifact_euid": artifact_euid,
            }
            for artifact_euid in canonical_artifact_euids
        )
        return canonical_set_euid, canonical_artifact_euids, input_references

    def resolve_manifest_input_references(
        *,
        actor: CurrentUser,
        resources: ResourceStore,
        request: ManifestCreateRequest,
    ) -> tuple[str | None, list[str], list[dict[str, Any]], dict[str, Any]]:
        if not request.input_references:
            metadata = dict(request.metadata or {})
            editor_rows = metadata.get("editor_analysis_inputs")
            if (
                isinstance(editor_rows, list)
                and editor_rows
                and not str(request.artifact_set_euid or "").strip()
            ):
                return (None, [], [], metadata)
            canonical_set_euid, canonical_artifact_euids, input_references = (
                validate_manifest_artifact_references(
                    str(request.artifact_set_euid or ""),
                    request.artifact_euids,
                )
            )
            return (
                canonical_set_euid,
                canonical_artifact_euids,
                input_references,
                dict(request.metadata or {}),
            )

        if app.state.dewey_client is None:
            raise HTTPException(status_code=503, detail="Dewey client is not configured")

        canonical_artifact_euids: list[str] = []
        dedupe: set[str] = set()
        input_references: list[dict[str, Any]] = []
        canonical_sets: list[str] = []

        for ref in request.input_references:
            raw_value = str(ref.value or "").strip()
            if ref.reference_type == "artifact_euid":
                artifact_euid = resolve_dewey_artifact_euid(raw_value)
                if artifact_euid not in dedupe:
                    dedupe.add(artifact_euid)
                    canonical_artifact_euids.append(artifact_euid)
                input_references.append(
                    {
                        "reference_type": "artifact_euid",
                        "value": raw_value,
                        "artifact_euid": artifact_euid,
                    }
                )
                continue

            if ref.reference_type == "artifact_set_euid":
                resolved_set = resolve_dewey_artifact_set_payload(raw_value)
                canonical_set = str(resolved_set.get("artifact_set_euid") or raw_value).strip()
                member_euids = [
                    str(member.get("artifact_euid") or "").strip()
                    for member in list(resolved_set.get("members") or [])
                    if isinstance(member, dict) and str(member.get("artifact_euid") or "").strip()
                ]
                canonical_sets.append(canonical_set)
                for artifact_euid in member_euids:
                    if artifact_euid in dedupe:
                        continue
                    dedupe.add(artifact_euid)
                    canonical_artifact_euids.append(artifact_euid)
                input_references.append(
                    {
                        "reference_type": "artifact_set_euid",
                        "value": raw_value,
                        "artifact_set_euid": canonical_set,
                        "artifact_euids": member_euids,
                    }
                )
                continue

            try:
                _ensure_s3_fetchable(app.state.s3_client, raw_value)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            artifact_type = _guess_artifact_type(raw_value)
            try:
                artifact_euid = app.state.dewey_client.register_artifact(
                    artifact_type=artifact_type,
                    storage_uri=raw_value,
                    metadata={
                        "producer_system": "ursa-manifest",
                        "actor_user_id": actor.user_id,
                        "tenant_id": str(actor.tenant_id),
                        "workset_euid": request.workset_euid,
                    },
                    idempotency_key=f"manifest:{actor.user_id}:{raw_value}",
                )
            except DeweyClientError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            record_observed_dependency("dewey")
            resources.record_dewey_import(
                artifact_euid=artifact_euid,
                artifact_type=artifact_type,
                storage_uri=raw_value,
                actor_user_id=actor.user_id,
                metadata={"source": "manifest_input_references"},
            )
            if artifact_euid not in dedupe:
                dedupe.add(artifact_euid)
                canonical_artifact_euids.append(artifact_euid)
            input_references.append(
                {
                    "reference_type": "s3_uri",
                    "value": raw_value,
                    "artifact_type": artifact_type,
                    "artifact_euid": artifact_euid,
                }
            )

        canonical_set_euid: str | None = None
        if canonical_sets and len(canonical_sets) == 1 and len(request.input_references) == 1:
            canonical_set_euid = canonical_sets[0]

        manifest_metadata = dict(request.metadata or {})
        manifest_metadata["input_references"] = [
            {"reference_type": ref.reference_type, "value": str(ref.value or "").strip()}
            for ref in request.input_references
        ]
        return canonical_set_euid, canonical_artifact_euids, input_references, manifest_metadata

    def require_linked_bucket_record(
        *,
        bucket_id: str,
        actor: CurrentUser,
        resources: ResourceStore,
    ) -> LinkedBucketRecord:
        record = resources.get_linked_bucket(bucket_id)
        if record is None or record.state == "DELETED":
            raise HTTPException(status_code=404, detail="Bucket not found")
        if not actor.is_admin and record.tenant_id != actor.tenant_id:
            raise HTTPException(status_code=403, detail="Bucket is outside the caller tenant")
        return record

    def list_bucket_items(
        *,
        bucket: LinkedBucketRecord,
        prefix: str = "",
        max_keys: int = 500,
    ) -> dict[str, Any]:
        normalized_prefix = str(prefix or "").lstrip("/")
        restricted_prefix = _normalize_prefix(bucket.prefix_restriction)
        if (
            restricted_prefix
            and normalized_prefix
            and not normalized_prefix.startswith(restricted_prefix)
        ):
            raise HTTPException(
                status_code=403, detail="Prefix is outside the linked bucket restriction"
            )
        effective_prefix = normalized_prefix or restricted_prefix or ""
        try:
            response = app.state.s3_client.list_objects_v2(
                Bucket=bucket.bucket_name,
                Prefix=effective_prefix,
                Delimiter="/",
                MaxKeys=max_keys,
            )
        except ClientError as exc:
            raise HTTPException(status_code=502, detail=f"Failed to browse bucket: {exc}") from exc

        items: list[dict[str, Any]] = []
        for common_prefix in response.get("CommonPrefixes", []):
            folder_path = str(common_prefix.get("Prefix") or "")
            folder_name = folder_path.rstrip("/").split("/")[-1]
            items.append(
                {
                    "name": folder_name,
                    "is_folder": True,
                    "key": folder_path,
                    "size_bytes": None,
                    "size_human": "--",
                    "last_modified": None,
                    "file_format": None,
                }
            )
        for obj in response.get("Contents", []):
            key = str(obj.get("Key") or "")
            if not key or key == effective_prefix or key.endswith("/") and key == effective_prefix:
                continue
            name = key.split("/")[-1]
            if not name:
                continue
            size_bytes = int(obj.get("Size") or 0)
            last_modified = obj.get("LastModified")
            items.append(
                {
                    "name": name,
                    "is_folder": False,
                    "key": key,
                    "size_bytes": size_bytes,
                    "size_human": _format_file_size(size_bytes),
                    "last_modified": last_modified.isoformat()
                    if hasattr(last_modified, "isoformat")
                    else None,
                    "file_format": _detect_file_format(name),
                }
            )
        breadcrumbs = [{"name": "/", "prefix": restricted_prefix or ""}]
        if effective_prefix:
            root_prefix = restricted_prefix or ""
            suffix = (
                effective_prefix[len(root_prefix) :]
                if root_prefix and effective_prefix.startswith(root_prefix)
                else effective_prefix
            )
            current_parts = [part for part in suffix.rstrip("/").split("/") if part]
            running_prefix = root_prefix
            for part in current_parts:
                running_prefix = f"{running_prefix}{part}/"
                breadcrumbs.append({"name": part, "prefix": running_prefix})
        if not effective_prefix:
            parent_prefix = None
        else:
            trimmed = effective_prefix.rstrip("/")
            parent_parts = trimmed.split("/")[:-1]
            parent_prefix = (
                "/".join(parent_parts) + "/" if parent_parts else (restricted_prefix or "")
            )
            if (
                restricted_prefix
                and parent_prefix
                and not parent_prefix.startswith(restricted_prefix)
            ):
                parent_prefix = restricted_prefix
        return {
            "bucket": _linked_bucket_response(bucket),
            "prefix": effective_prefix,
            "parent_prefix": parent_prefix,
            "breadcrumbs": breadcrumbs,
            "items": items,
        }

    def list_profile_s3_buckets(region: str | None = None) -> dict[str, Any]:
        settings = app.state.settings
        profile = str(getattr(settings, "aws_profile", "") or "").strip()
        session_kwargs: dict[str, Any] = {}
        if profile:
            session_kwargs["profile_name"] = profile
        target_region = str(region or settings.get_effective_region() or "").strip() or "us-west-2"
        session = boto3.Session(**session_kwargs)
        s3 = session.client("s3", region_name=target_region)
        response = s3.list_buckets()
        buckets = sorted(
            (
                {
                    "bucket_name": name,
                    "created_at": (
                        created.isoformat()
                        if isinstance(created, datetime)
                        else (str(created) if created else None)
                    ),
                }
                for item in list(response.get("Buckets") or [])
                if (name := str(item.get("Name") or "").strip())
                for created in [item.get("CreationDate")]
            ),
            key=lambda item: item["bucket_name"],
        )
        return {"profile": profile or "default", "buckets": buckets}

    def load_cluster_create_options(region: str) -> ClusterCreateOptionsResponse:
        session_kwargs: dict[str, Any] = {"region_name": region}
        profile = str(app.state.settings.aws_profile or "").strip()
        if profile:
            session_kwargs["profile_name"] = profile
        keypairs: list[str] = []
        buckets: list[str] = []
        availability_zones: list[str] = []
        try:
            session = boto3.Session(**session_kwargs)
            ec2 = session.client("ec2")
            response = ec2.describe_key_pairs()
            keypairs = sorted(
                str(item.get("KeyName") or "").strip()
                for item in list(response.get("KeyPairs") or [])
                if str(item.get("KeyName") or "").strip()
            )
            zone_response = ec2.describe_availability_zones(
                Filters=[{"Name": "state", "Values": ["available"]}]
            )
            availability_zones = sorted(
                str(item.get("ZoneName") or "").strip()
                for item in list(zone_response.get("AvailabilityZones") or [])
                if str(item.get("ZoneName") or "").strip()
            )
        except Exception:
            LOGGER.exception("Failed to load EC2 create options for %s", region)
        try:
            buckets = [
                item["bucket_name"] for item in list_profile_s3_buckets(region=region)["buckets"]
            ]
        except Exception:
            LOGGER.exception("Failed to list S3 buckets for cluster create options")
        return ClusterCreateOptionsResponse(
            keypairs=keypairs,
            buckets=buckets,
            availability_zones=availability_zones,
        )

    def update_cluster_scan_regions(regions_csv: str) -> ClusterScanRegionsResponse:
        normalized_regions = parse_regions_csv(regions_csv)
        current_config = getattr(app.state, "ursa_config", None)
        config_path = getattr(current_config, "config_path", None)
        if not isinstance(config_path, Path):
            raise RuntimeError("Ursa config path is not available for cluster region updates")

        refreshed_config = update_config_regions(
            regions=normalized_regions,
            config_path=config_path,
        )
        app.state.ursa_config = refreshed_config
        app.state.settings.ursa_allowed_regions = ",".join(refreshed_config.get_allowed_regions())

        aws_profile = (
            str(app.state.settings.aws_profile or refreshed_config.aws_profile or "").strip()
            or None
        )
        cluster_service = ClusterService(
            regions=refreshed_config.get_allowed_regions(),
            aws_profile=aws_profile,
        )
        app.state.cluster_service = cluster_service

        cluster_job_manager = getattr(app.state, "cluster_job_manager", None)
        if cluster_job_manager is not None:
            cluster_job_manager.cluster_service = cluster_service
        analysis_job_manager = getattr(app.state, "analysis_job_manager", None)
        if analysis_job_manager is not None and hasattr(cluster_service, "client"):
            analysis_job_manager.client = cluster_service.client

        return ClusterScanRegionsResponse(
            regions=refreshed_config.get_allowed_regions(),
            regions_csv=",".join(refreshed_config.get_allowed_regions()),
            config_path=str(refreshed_config.config_path or config_path),
        )

    @app.get("/healthz", tags=["health"])
    async def healthz(request: Request) -> dict[str, Any]:
        return build_healthz_payload(
            request,
            settings=settings,
            app_version=__version__,
            started_at=app.state.observability.started_at,
        )

    @app.get("/readyz", tags=["health"])
    async def readyz(request: Request) -> JSONResponse:
        database = _database_probe()
        ready = str(database.get("status") or "") == "ok"
        return JSONResponse(
            status_code=200 if ready else 503,
            content=build_readyz_payload(
                request,
                settings=settings,
                app_version=__version__,
                started_at=app.state.observability.started_at,
                database_check=database,
                ready=ready,
            ),
        )

    @app.get("/health", tags=["observability"])
    async def health(actor: RequireObservability, request: Request) -> dict[str, Any]:
        _ = actor
        snapshot = app.state.observability.health_snapshot()
        observed_at = snapshot.get("checks", {}).get("database", {}).get(
            "observed_at"
        ) or snapshot.get("checks", {}).get("auth", {}).get("observed_at")
        projection = app.state.observability.projection(observed_at=observed_at)
        return build_health_payload(
            request,
            settings=settings,
            app_version=__version__,
            projection=projection,
            health_snapshot=snapshot,
        )

    @app.get("/obs_services", tags=["observability"])
    async def obs_services(actor: RequireObservability, request: Request) -> dict[str, Any]:
        _ = actor
        projection, snapshot = app.state.observability.obs_services_snapshot()
        return build_obs_services_payload(
            request,
            settings=settings,
            app_version=__version__,
            projection=projection,
            snapshot=snapshot,
        )

    @app.get("/api_health", tags=["observability"])
    async def api_health(actor: RequireObservability, request: Request) -> dict[str, Any]:
        _ = actor
        projection, families = app.state.observability.api_health()
        return build_api_health_payload(
            request,
            settings=settings,
            app_version=__version__,
            projection=projection,
            families=families,
        )

    @app.get("/endpoint_health", tags=["observability"])
    async def endpoint_health(
        actor: RequireObservability,
        request: Request,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=200),
    ) -> dict[str, Any]:
        _ = actor
        projection, page = app.state.observability.endpoint_health(offset=offset, limit=limit)
        return build_endpoint_health_payload(
            request,
            settings=settings,
            app_version=__version__,
            projection=projection,
            total=int(page["total"]),
            offset=int(page["offset"]),
            limit=int(page["limit"]),
            items=list(page["items"]),
        )

    @app.get("/db_health", tags=["observability"])
    async def db_health(actor: RequireObservability, request: Request) -> dict[str, Any]:
        _ = actor
        _database_probe()
        projection, rollup = app.state.observability.db_health()
        if str(rollup.get("status") or "") == "error":
            latest_probe = dict(rollup.get("latest") or {})
            _anomaly_repository().record_db_probe_failure(
                detail=str(latest_probe.get("detail") or "database probe failed"),
                latency_ms=float(latest_probe.get("latency_ms") or 0.0),
            )
        return build_db_health_payload(
            request,
            settings=settings,
            app_version=__version__,
            projection=projection,
            db_health=rollup,
        )

    @app.get("/api/anomalies", tags=["anomalies"])
    async def list_anomalies(actor: RequireObservability) -> dict[str, Any]:
        _ = actor
        items = [item.__dict__ for item in _anomaly_repository().list()]
        observed_at = str(items[0].get("last_seen_at") or "") if items else ""
        projection = app.state.observability.projection(observed_at=observed_at or None)
        return {
            "service": "ursa",
            "contract_version": "v3",
            "observed_at": projection.observed_at,
            "projection": projection.model_dump(),
            "count": len(items),
            "items": items,
        }

    @app.get("/api/anomalies/{anomaly_id}", tags=["anomalies"])
    async def get_anomaly(anomaly_id: str, actor: RequireObservability) -> dict[str, Any]:
        _ = actor
        item = _anomaly_repository().get(anomaly_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Anomaly not found")
        projection = app.state.observability.projection(observed_at=item.last_seen_at)
        return {
            "service": "ursa",
            "contract_version": "v3",
            "observed_at": projection.observed_at,
            "projection": projection.model_dump(),
            "item": item.__dict__,
        }

    @app.get("/my_health", tags=["observability"])
    async def my_health(actor: RequireAuth, request: Request) -> dict[str, Any]:
        if actor.auth_source == "service_token":
            raise HTTPException(status_code=401, detail="Service tokens cannot access /my_health")
        return build_my_health_payload(
            request,
            settings=settings,
            app_version=__version__,
            user=actor,
        )

    @app.get("/auth_health", tags=["observability"])
    async def auth_health(actor: RequireObservability, request: Request) -> dict[str, Any]:
        _ = actor
        projection, rollup = app.state.observability.auth_health()
        return build_auth_health_payload(
            request,
            settings=settings,
            app_version=__version__,
            projection=projection,
            auth_rollup=rollup,
        )

    @app.get("/api/v1/me", response_model=MeResponse)
    async def get_me(actor: RequireAuth) -> MeResponse:
        return _me_response(actor)

    @app.get("/api/v1/analyses", response_model=list[AnalysisResponse])
    async def list_analyses(
        actor: RequireAuth,
        workset_euid: str | None = Query(default=None),
    ) -> list[AnalysisResponse]:
        records = app.state.store.list_analyses(
            tenant_id=None if actor.is_admin else actor.tenant_id,
            workset_euid=workset_euid,
        )
        return [_analysis_response(record) for record in records]

    @app.post(
        "/api/v1/analyses/ingest",
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
        record_observed_dependency("bloom")

        resolved_references: list[dict[str, Any]] = []
        if app.state.dewey_client is None:
            raise HTTPException(
                status_code=503,
                detail="Dewey integration is required for analysis ingest",
            )

        for ref in request.input_references:
            raw_value = str(ref.value or "").strip()
            if ref.reference_type == "artifact_euid":
                try:
                    resolved = app.state.dewey_client.resolve_artifact(raw_value)
                except DeweyClientError as exc:
                    raise HTTPException(status_code=502, detail=str(exc)) from exc
                record_observed_dependency("dewey")
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
            record_observed_dependency("dewey")

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
                    "artifact_euids": [
                        str(item.get("artifact_euid") or "") for item in member_payload
                    ],
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
        if request.workset_euid:
            resources = require_resource_store()
            try:
                resources.link_analysis(
                    workset_euid=str(request.workset_euid),
                    analysis_euid=record.analysis_euid,
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            record = app.state.store.get_analysis(record.analysis_euid) or record
        return _analysis_response(record)

    @app.get("/api/v1/analyses/{analysis_euid}", response_model=AnalysisResponse)
    async def get_analysis(
        analysis_euid: str,
        actor: RequireAuth,
    ) -> AnalysisResponse:
        record = app.state.store.get_analysis(analysis_euid)
        if record is None:
            raise HTTPException(status_code=404, detail="Analysis not found")
        if not actor.is_admin and record.tenant_id != actor.tenant_id:
            raise HTTPException(status_code=403, detail="Analysis is outside the caller tenant")
        return _analysis_response(record)

    @app.post("/api/v1/analyses/{analysis_euid}/status", response_model=AnalysisResponse)
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
        "/api/v1/analyses/{analysis_euid}/artifacts",
        response_model=AnalysisArtifactResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def add_analysis_artifact(
        analysis_euid: str,
        request: AnalysisArtifactRequest,
        _api_key: str = Depends(require_write_api_key),
    ) -> AnalysisArtifactResponse:
        dewey_client = require_dewey_client()
        filename = str(request.filename or "").strip()
        resolved_metadata: dict[str, Any] = {}

        source_artifact_euid = str(request.artifact_euid or "").strip()
        try:
            resolved = dewey_client.resolve_artifact(source_artifact_euid)
        except DeweyClientError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        record_observed_dependency("dewey")

        artifact_type = str(resolved.get("artifact_type") or "").strip()
        storage_uri = str(resolved.get("storage_uri") or "").strip()
        filename = filename or str(resolved.get("filename") or "").strip()
        if not filename:
            filename = Path(storage_uri).name or f"{source_artifact_euid}.bin"
        registered_euid = str(resolved.get("artifact_euid") or source_artifact_euid)
        resolved_metadata = {
            **dict(resolved.get("metadata") or {}),
            "dewey_artifact_euid": registered_euid,
            "dewey_resolved": True,
        }

        if not artifact_type or not storage_uri or not registered_euid:
            raise HTTPException(status_code=502, detail="Dewey artifact resolution failed")

        try:
            artifact = app.state.store.add_artifact(
                analysis_euid,
                artifact_type=artifact_type,
                storage_uri=storage_uri,
                filename=filename,
                mime_type=request.mime_type,
                checksum_sha256=request.checksum_sha256,
                size_bytes=request.size_bytes,
                metadata={**resolved_metadata, **dict(request.metadata or {})},
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _artifact_response(artifact)

    @app.post("/api/v1/analyses/{analysis_euid}/review", response_model=AnalysisResponse)
    async def review_analysis(
        analysis_euid: str,
        request: AnalysisReviewRequest,
        actor: RequireAuth,
    ) -> AnalysisResponse:
        existing = app.state.store.get_analysis(analysis_euid)
        if existing is None:
            raise HTTPException(status_code=404, detail="Analysis not found")
        if not actor.is_admin and existing.tenant_id != actor.tenant_id:
            raise HTTPException(status_code=403, detail="Analysis is outside the caller tenant")
        try:
            record = app.state.store.set_review_state(
                analysis_euid,
                review_state=request.review_state,
                reviewer=request.reviewer or actor.user_id,
                notes=request.notes,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _analysis_response(record)

    @app.post("/api/v1/analyses/{analysis_euid}/return", response_model=AnalysisResponse)
    async def return_analysis_result(
        analysis_euid: str,
        payload: AnalysisReturnRequest,
        request: Request,
        actor: RequireAuth,
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ) -> AnalysisResponse:
        if not str(idempotency_key or "").strip():
            raise HTTPException(status_code=400, detail="Idempotency-Key header is required")
        record = app.state.store.get_analysis(analysis_euid)
        if record is None:
            raise HTTPException(status_code=404, detail="Analysis not found")
        if not actor.is_admin and record.tenant_id != actor.tenant_id:
            raise HTTPException(status_code=403, detail="Analysis is outside the caller tenant")
        if app.state.atlas_client is None:
            raise HTTPException(
                status_code=503, detail="Atlas result return client is not configured"
            )
        if record.review_state != ReviewState.APPROVED.value:
            raise HTTPException(
                status_code=409,
                detail="Analysis cannot be returned before manual approval",
            )
        try:
            atlas_artifacts: list[AtlasResultArtifact] = []
            missing_dewey_refs: list[str] = []
            for artifact in record.artifacts:
                dewey_artifact_euid = str(
                    artifact.metadata.get("dewey_artifact_euid") or ""
                ).strip()
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
                atlas_tenant_id=str(record.tenant_id),
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
                result_status=payload.result_status,
                review_state=record.review_state,
                result_payload=payload.result_payload,
                artifacts=atlas_artifacts,
                idempotency_key=str(idempotency_key),
                request_id=str(getattr(request.state, "request_id", "") or ""),
            )
            record_observed_dependency("atlas")
        except AtlasResultClientError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        updated = app.state.store.mark_returned(
            analysis_euid,
            atlas_return={
                **atlas_response,
                "result_status": payload.result_status,
                "returned_by_user_id": actor.user_id,
            },
            idempotency_key=str(idempotency_key),
        )
        return _analysis_response(updated)

    @app.get("/api/v1/worksets", response_model=list[WorksetResponse])
    async def list_worksets(
        actor: RequireAuth,
        resources: ResourceStore = Depends(require_resource_store),
    ) -> list[WorksetResponse]:
        items = resources.list_worksets(
            tenant_id=actor.tenant_id,
        )
        return [_workset_response(item) for item in items]

    @app.get("/api/v1/analysis-commands")
    async def list_analysis_commands(actor: RequireAuth) -> dict[str, Any]:
        _ = actor
        try:
            return command_catalog_payload()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/v1/analysis-commands/{command_id}")
    async def get_analysis_command(command_id: str, actor: RequireAuth) -> dict[str, Any]:
        _ = actor
        try:
            return analysis_command_payload(command_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post(
        "/api/v1/analysis-commands/{command_id}/preview",
        response_model=AnalysisCommandPreviewResponse,
    )
    async def preview_analysis_launch_command(
        command_id: str,
        request: AnalysisCommandPreviewRequest,
        actor: RequireAuth,
    ) -> AnalysisCommandPreviewResponse:
        _ = actor
        try:
            return AnalysisCommandPreviewResponse.model_validate(
                preview_analysis_command(
                    command_id,
                    optional_features=request.optional_features,
                    profile=request.profile,
                    region=request.region,
                    cluster_name=request.cluster_name,
                    stage_dir=request.stage_dir,
                    session_name=request.session_name,
                    project=request.project,
                    dry_run=request.dry_run,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post(
        "/api/v1/worksets", response_model=WorksetResponse, status_code=status.HTTP_201_CREATED
    )
    async def create_workset(
        request: WorksetCreateRequest,
        actor: RequireAuth,
        resources: ResourceStore = Depends(require_resource_store),
    ) -> WorksetResponse:
        artifact_set_euids = validate_workset_artifact_sets(request.artifact_set_euids)
        metadata = _canonicalize_workset_metadata(request.metadata)
        record = resources.create_workset(
            name=request.name,
            tenant_id=actor.tenant_id,
            owner_user_id=actor.user_id,
            artifact_set_euids=artifact_set_euids,
            metadata=metadata,
        )
        return _workset_response(record)

    @app.get("/api/v1/worksets/{workset_euid}", response_model=WorksetResponse)
    async def get_workset(
        workset_euid: str,
        actor: RequireAuth,
        resources: ResourceStore = Depends(require_resource_store),
    ) -> WorksetResponse:
        record = resources.get_workset(workset_euid)
        if record is None:
            raise HTTPException(status_code=404, detail="Workset not found")
        if not actor.is_admin and record.tenant_id != actor.tenant_id:
            raise HTTPException(status_code=403, detail="Workset is outside the caller tenant")
        return _workset_response(record)

    @app.get("/api/v1/manifests", response_model=list[ManifestResponse])
    async def list_manifests(
        actor: RequireAuth,
        resources: ResourceStore = Depends(require_resource_store),
    ) -> list[ManifestResponse]:
        records = resources.list_manifests(tenant_id=actor.tenant_id)
        return [_manifest_response(item) for item in records]

    @app.post(
        "/api/v1/manifests", response_model=ManifestResponse, status_code=status.HTTP_201_CREATED
    )
    async def create_manifest(
        request: ManifestCreateRequest,
        actor: RequireAuth,
        resources: ResourceStore = Depends(require_resource_store),
    ) -> ManifestResponse:
        workset = resources.get_workset(request.workset_euid)
        if workset is None:
            raise HTTPException(status_code=404, detail="Workset not found")
        if not actor.is_admin and workset.tenant_id != actor.tenant_id:
            raise HTTPException(status_code=403, detail="Workset is outside the caller tenant")
        artifact_set_euid, artifact_euids, input_references, metadata = (
            resolve_manifest_input_references(
                actor=actor,
                resources=resources,
                request=request,
            )
        )
        metadata = dict(metadata)
        if "editor_manifest_tsv" in metadata:
            raise HTTPException(
                status_code=400,
                detail="editor_manifest_tsv is not supported; provide editor_analysis_inputs",
            )
        if "stable_manifest" in metadata:
            raise HTTPException(
                status_code=400,
                detail="metadata.stable_manifest is not supported; Ursa writes metadata.analysis_samples_manifest",
            )
        if "analysis_samples_manifest" in metadata:
            raise HTTPException(
                status_code=400,
                detail="metadata.analysis_samples_manifest is generated by Ursa",
            )
        try:
            analysis_samples_manifest = build_analysis_samples_manifest(
                metadata=metadata,
                input_references=input_references,
                artifact_euids=artifact_euids,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        metadata["analysis_samples_manifest"] = analysis_samples_manifest.metadata()
        record = resources.create_manifest(
            workset_euid=request.workset_euid,
            name=request.name,
            artifact_set_euid=artifact_set_euid,
            artifact_euids=artifact_euids,
            input_references=input_references,
            metadata=metadata,
        )
        return _manifest_response(record)

    @app.get("/api/v1/manifests/{manifest_euid}", response_model=ManifestResponse)
    async def get_manifest(
        manifest_euid: str,
        actor: RequireAuth,
        resources: ResourceStore = Depends(require_resource_store),
    ) -> ManifestResponse:
        record = resources.get_manifest(manifest_euid)
        if record is None:
            raise HTTPException(status_code=404, detail="Manifest not found")
        if not actor.is_admin and record.tenant_id != actor.tenant_id:
            raise HTTPException(status_code=403, detail="Manifest is outside the caller tenant")
        return _manifest_response(record)

    @app.get("/api/v1/manifests/{manifest_euid}/download")
    async def download_manifest(
        manifest_euid: str,
        actor: RequireAuth,
        resources: ResourceStore = Depends(require_resource_store),
    ) -> PlainTextResponse:
        record = resources.get_manifest(manifest_euid)
        if record is None:
            raise HTTPException(status_code=404, detail="Manifest not found")
        if not actor.is_admin and record.tenant_id != actor.tenant_id:
            raise HTTPException(status_code=403, detail="Manifest is outside the caller tenant")
        metadata = dict(record.metadata or {})
        analysis_samples_manifest = dict(metadata.get("analysis_samples_manifest") or {})
        tsv_content = str(analysis_samples_manifest.get("content") or "")
        if not tsv_content:
            raise HTTPException(
                status_code=409,
                detail="This manifest does not have analysis_samples_manifest.content",
            )
        filename = str(analysis_samples_manifest.get("filename") or "analysis_samples.tsv").replace(
            "/", "-"
        )
        return PlainTextResponse(
            tsv_content,
            media_type="text/tab-separated-values",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    def require_staging_job_access(
        *,
        job_euid: str,
        actor: CurrentUser,
        resources: ResourceStore,
    ) -> StagingJobRecord:
        record = resources.get_staging_job(job_euid)
        if record is None:
            raise HTTPException(status_code=404, detail="Staging job not found")
        if record.tenant_id != actor.tenant_id:
            raise HTTPException(status_code=403, detail="Staging job is outside the caller tenant")
        return record

    @app.get("/api/v1/staging-jobs", response_model=list[StagingJobResponse])
    async def list_staging_jobs(
        actor: RequireAuth,
        resources: ResourceStore = Depends(require_resource_store),
    ) -> list[StagingJobResponse]:
        records = resources.list_staging_jobs(tenant_id=actor.tenant_id)
        return [_staging_job_response(item) for item in records]

    @app.post(
        "/api/v1/staging-jobs",
        response_model=StagingJobResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_staging_job(
        request: StagingJobCreateRequest,
        actor: RequireAuth,
        resources: ResourceStore = Depends(require_resource_store),
    ) -> StagingJobResponse:
        workset = resources.get_workset(request.workset_euid)
        if workset is None:
            raise HTTPException(status_code=404, detail="Workset not found")
        if workset.tenant_id != actor.tenant_id:
            raise HTTPException(status_code=403, detail="Workset is outside the caller tenant")
        manifest = resources.get_manifest(request.manifest_euid)
        if manifest is None:
            raise HTTPException(status_code=404, detail="Manifest not found")
        if manifest.tenant_id != actor.tenant_id:
            raise HTTPException(status_code=403, detail="Manifest is outside the caller tenant")
        if manifest.workset_euid != workset.workset_euid:
            raise HTTPException(status_code=400, detail="Manifest does not belong to workset")
        request_payload = {
            "reference_bucket": request.reference_bucket,
            "stage_target": request.stage_target or "/data/staged_sample_data",
            "aws_profile": request.aws_profile,
            "debug": bool(request.debug),
            "metadata": dict(request.metadata or {}),
        }
        job_name = str(request.job_name or "").strip() or f"{workset.name}:staging"
        try:
            record = resources.create_staging_job(
                job_name=job_name,
                workset_euid=workset.workset_euid,
                manifest_euid=manifest.manifest_euid,
                cluster_name=request.cluster_name,
                region=request.region,
                tenant_id=actor.tenant_id,
                owner_user_id=actor.user_id,
                request=request_payload,
            )
            resources.add_staging_job_event(
                job_euid=record.job_euid,
                event_type="defined",
                status="DEFINED",
                summary="Defined staging job",
                details={"manifest_euid": manifest.manifest_euid},
                created_by=actor.user_id,
            )
            record = resources.get_staging_job(record.job_euid) or record
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _staging_job_response(record)

    @app.get("/api/v1/staging-jobs/{job_euid}", response_model=StagingJobResponse)
    async def get_staging_job(
        job_euid: str,
        actor: RequireAuth,
        resources: ResourceStore = Depends(require_resource_store),
    ) -> StagingJobResponse:
        return _staging_job_response(
            require_staging_job_access(job_euid=job_euid, actor=actor, resources=resources)
        )

    @app.post(
        "/api/v1/staging-jobs/{job_euid}/run",
        response_model=StagingJobResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def run_staging_job(
        job_euid: str,
        request: StagingJobRunRequest,
        actor: RequireAuth,
        resources: ResourceStore = Depends(require_resource_store),
        manager: StagingJobManager = Depends(require_staging_job_manager),
    ) -> StagingJobResponse:
        _ = request
        require_staging_job_access(job_euid=job_euid, actor=actor, resources=resources)
        try:
            record = manager.run_job(job_euid, actor_user_id=actor.user_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return _staging_job_response(record)

    @app.get("/api/v1/staging-jobs/{job_euid}/logs")
    async def get_staging_job_logs(
        job_euid: str,
        actor: RequireAuth,
        lines: int = Query(default=200, ge=1, le=5000),
        resources: ResourceStore = Depends(require_resource_store),
        manager: StagingJobManager = Depends(require_staging_job_manager),
    ) -> dict[str, Any]:
        require_staging_job_access(job_euid=job_euid, actor=actor, resources=resources)
        try:
            return manager.logs(job_euid, lines=lines)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    def require_analysis_job_access(
        *,
        job_euid: str,
        actor: CurrentUser,
        resources: ResourceStore,
    ) -> AnalysisJobRecord:
        record = resources.get_analysis_job(job_euid)
        if record is None:
            raise HTTPException(status_code=404, detail="Analysis job not found")
        if not actor.is_admin and record.tenant_id != actor.tenant_id:
            raise HTTPException(status_code=403, detail="Analysis job is outside the caller tenant")
        return record

    @app.get("/api/v1/analysis-jobs", response_model=list[AnalysisJobResponse])
    async def list_analysis_jobs(
        actor: RequireAuth,
        resources: ResourceStore = Depends(require_resource_store),
    ) -> list[AnalysisJobResponse]:
        records = resources.list_analysis_jobs(
            tenant_id=None if actor.is_admin else actor.tenant_id
        )
        return [_analysis_job_response(item) for item in records]

    @app.post(
        "/api/v1/analysis-jobs",
        response_model=AnalysisJobResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_analysis_job(
        request: AnalysisJobCreateRequest,
        actor: RequireAuth,
        resources: ResourceStore = Depends(require_resource_store),
    ) -> AnalysisJobResponse:
        workset = resources.get_workset(request.workset_euid)
        if workset is None:
            raise HTTPException(status_code=404, detail="Workset not found")
        if not actor.is_admin and workset.tenant_id != actor.tenant_id:
            raise HTTPException(status_code=403, detail="Workset is outside the caller tenant")
        manifest = resources.get_manifest(request.manifest_euid)
        if manifest is None:
            raise HTTPException(status_code=404, detail="Manifest not found")
        if manifest.workset_euid != workset.workset_euid:
            raise HTTPException(status_code=400, detail="Manifest does not belong to workset")
        if not actor.is_admin and manifest.tenant_id != actor.tenant_id:
            raise HTTPException(status_code=403, detail="Manifest is outside the caller tenant")
        staging_job: StagingJobRecord | None = None
        staging_job_euid = str(request.staging_job_euid or "").strip()
        if staging_job_euid:
            staging_job = resources.get_staging_job(staging_job_euid)
            if staging_job is None:
                raise HTTPException(status_code=404, detail="Staging job not found")
            if staging_job.tenant_id != workset.tenant_id:
                raise HTTPException(
                    status_code=400,
                    detail="Staging job tenant does not match analysis job tenant",
                )
            if staging_job.workset_euid != workset.workset_euid:
                raise HTTPException(
                    status_code=400,
                    detail="Staging job does not belong to workset",
                )
            if staging_job.manifest_euid != manifest.manifest_euid:
                raise HTTPException(
                    status_code=400,
                    detail="Staging job does not belong to manifest",
                )
            if staging_job.state != "COMPLETED":
                raise HTTPException(status_code=409, detail="Staging job is not completed")
            if not str((staging_job.stage or {}).get("stage_dir") or "").strip():
                raise HTTPException(status_code=409, detail="Staging job has no stage_dir")
        try:
            command = analysis_command_payload(
                request.analysis_command_id,
                optional_features=request.optional_features,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        request_payload = {
            "analysis_command_id": request.analysis_command_id,
            "optional_features": list(request.optional_features),
            "reference_bucket": request.reference_bucket,
            "session_name": request.session_name,
            "project": request.project,
            "aws_profile": request.aws_profile,
            "dry_run": bool(request.dry_run),
            "stage_target": request.stage_target
            or (staging_job.request.get("stage_target") if staging_job else None)
            or "/data/staged_sample_data",
            "staging_job_euid": staging_job_euid or None,
            "command": command,
        }
        job_name = str(request.job_name or "").strip() or (
            f"{workset.name}:{command.get('command_id')}"
        )
        try:
            record = resources.create_analysis_job(
                job_name=job_name,
                workset_euid=workset.workset_euid,
                manifest_euid=manifest.manifest_euid,
                cluster_name=request.cluster_name,
                region=request.region,
                tenant_id=workset.tenant_id,
                owner_user_id=actor.user_id,
                request=request_payload,
            )
            resources.add_analysis_job_event(
                job_euid=record.job_euid,
                event_type="defined",
                status="DEFINED",
                summary=f"Defined analysis job for {command.get('command_id')}",
                details={"manifest_euid": manifest.manifest_euid},
                created_by=actor.user_id,
            )
            record = resources.get_analysis_job(record.job_euid) or record
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _analysis_job_response(record)

    @app.get("/api/v1/analysis-jobs/{job_euid}", response_model=AnalysisJobResponse)
    async def get_analysis_job(
        job_euid: str,
        actor: RequireAuth,
        resources: ResourceStore = Depends(require_resource_store),
    ) -> AnalysisJobResponse:
        return _analysis_job_response(
            require_analysis_job_access(job_euid=job_euid, actor=actor, resources=resources)
        )

    @app.post(
        "/api/v1/analysis-jobs/{job_euid}/launch",
        response_model=AnalysisJobResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def launch_analysis_job(
        job_euid: str,
        request: AnalysisJobLaunchRequest,
        actor: RequireAuth,
        resources: ResourceStore = Depends(require_resource_store),
        manager: AnalysisJobManager = Depends(require_analysis_job_manager),
    ) -> AnalysisJobResponse:
        _ = request
        require_analysis_job_access(job_euid=job_euid, actor=actor, resources=resources)
        try:
            record = manager.launch_job(job_euid, actor_user_id=actor.user_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _analysis_job_response(record)

    @app.post("/api/v1/analysis-jobs/{job_euid}/refresh", response_model=AnalysisJobResponse)
    async def refresh_analysis_job(
        job_euid: str,
        actor: RequireAuth,
        resources: ResourceStore = Depends(require_resource_store),
        manager: AnalysisJobManager = Depends(require_analysis_job_manager),
    ) -> AnalysisJobResponse:
        require_analysis_job_access(job_euid=job_euid, actor=actor, resources=resources)
        try:
            return _analysis_job_response(
                manager.refresh_job(job_euid, actor_user_id=actor.user_id)
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/v1/analysis-jobs/{job_euid}/logs")
    async def get_analysis_job_logs(
        job_euid: str,
        actor: RequireAuth,
        lines: int = Query(default=200, ge=1, le=5000),
        resources: ResourceStore = Depends(require_resource_store),
        manager: AnalysisJobManager = Depends(require_analysis_job_manager),
    ) -> dict[str, Any]:
        require_analysis_job_access(job_euid=job_euid, actor=actor, resources=resources)
        try:
            return manager.logs(job_euid, lines=lines)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post(
        "/api/v1/artifacts/import",
        response_model=ArtifactImportResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def import_artifact_to_dewey(
        request: ArtifactImportRequest,
        actor: RequireAuth,
        resources: ResourceStore = Depends(require_resource_store),
    ) -> ArtifactImportResponse:
        if app.state.dewey_client is None:
            raise HTTPException(status_code=503, detail="Dewey client is not configured")
        try:
            _ensure_s3_fetchable(app.state.s3_client, request.storage_uri)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            artifact_euid = app.state.dewey_client.register_artifact(
                artifact_type=request.artifact_type,
                storage_uri=request.storage_uri,
                metadata={
                    **dict(request.metadata or {}),
                    "producer_system": "ursa",
                    "actor_user_id": actor.user_id,
                    "tenant_id": str(actor.tenant_id),
                },
                idempotency_key=f"{actor.user_id}:{request.storage_uri}",
            )
        except DeweyClientError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        record_observed_dependency("dewey")
        record = resources.record_dewey_import(
            artifact_euid=artifact_euid,
            artifact_type=request.artifact_type,
            storage_uri=request.storage_uri,
            actor_user_id=actor.user_id,
            metadata=request.metadata,
        )
        return _dewey_import_response(record)

    @app.post("/api/v1/artifacts/resolve")
    async def resolve_artifact(
        request: ArtifactResolveRequest,
        actor: RequireAuth,
    ) -> dict[str, Any]:
        _ = actor
        if app.state.dewey_client is None:
            raise HTTPException(status_code=503, detail="Dewey client is not configured")
        try:
            if request.artifact_euid:
                resolved = app.state.dewey_client.resolve_artifact(request.artifact_euid)
                record_observed_dependency("dewey")
                return resolved
            resolved = app.state.dewey_client.resolve_artifact_set(str(request.artifact_set_euid))
            record_observed_dependency("dewey")
            return resolved
        except DeweyClientError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.get("/api/v1/buckets", response_model=list[LinkedBucketResponse])
    async def list_linked_buckets(
        actor: RequireAuth,
        resources: ResourceStore = Depends(require_resource_store),
    ) -> list[LinkedBucketResponse]:
        records = resources.list_linked_buckets(tenant_id=actor.tenant_id)
        return [_linked_bucket_response(item) for item in records]

    @app.post("/api/v1/buckets/validate", response_model=LinkedBucketValidationResponse)
    async def validate_linked_bucket(
        request: LinkedBucketCreateRequest,
        actor: RequireAuth,
    ) -> LinkedBucketValidationResponse:
        _ = actor
        try:
            return _validate_bucket_access(
                app.state.s3_client,
                bucket_name=request.bucket_name,
                prefix_restriction=request.prefix_restriction,
                read_only=bool(request.read_only),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post(
        "/api/v1/buckets", response_model=LinkedBucketResponse, status_code=status.HTTP_201_CREATED
    )
    async def create_linked_bucket(
        request: LinkedBucketCreateRequest,
        actor: RequireAuth,
        resources: ResourceStore = Depends(require_resource_store),
    ) -> LinkedBucketResponse:
        try:
            validation = _validate_bucket_access(
                app.state.s3_client,
                bucket_name=request.bucket_name,
                prefix_restriction=request.prefix_restriction,
                read_only=bool(request.read_only),
            )
            record = resources.create_linked_bucket(
                bucket_name=validation.bucket_name,
                tenant_id=actor.tenant_id,
                owner_user_id=actor.user_id,
                display_name=request.display_name,
                bucket_type=request.bucket_type,
                description=request.description,
                prefix_restriction=request.prefix_restriction,
                read_only=bool(request.read_only),
                region=validation.region,
                is_validated=validation.is_validated,
                can_read=validation.can_read,
                can_write=validation.can_write,
                can_list=validation.can_list,
                remediation_steps=validation.remediation_steps,
                metadata=request.metadata,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _linked_bucket_response(record)

    @app.get("/api/v1/buckets/{bucket_id}", response_model=LinkedBucketResponse)
    async def get_linked_bucket(
        bucket_id: str,
        actor: RequireAuth,
        resources: ResourceStore = Depends(require_resource_store),
    ) -> LinkedBucketResponse:
        record = require_linked_bucket_record(bucket_id=bucket_id, actor=actor, resources=resources)
        return _linked_bucket_response(record)

    @app.patch("/api/v1/buckets/{bucket_id}", response_model=LinkedBucketResponse)
    async def update_linked_bucket(
        bucket_id: str,
        request: LinkedBucketUpdateRequest,
        actor: RequireAuth,
        resources: ResourceStore = Depends(require_resource_store),
    ) -> LinkedBucketResponse:
        existing = require_linked_bucket_record(
            bucket_id=bucket_id, actor=actor, resources=resources
        )
        validation = _validate_bucket_access(
            app.state.s3_client,
            bucket_name=existing.bucket_name,
            prefix_restriction=request.prefix_restriction
            if request.prefix_restriction is not None
            else existing.prefix_restriction,
            read_only=bool(existing.read_only if request.read_only is None else request.read_only),
        )
        updated = resources.update_linked_bucket(
            bucket_id=bucket_id,
            display_name=request.display_name,
            bucket_type=request.bucket_type,
            description=request.description,
            prefix_restriction=request.prefix_restriction,
            read_only=request.read_only,
            region=validation.region,
            is_validated=validation.is_validated,
            can_read=validation.can_read,
            can_write=validation.can_write,
            can_list=validation.can_list,
            remediation_steps=validation.remediation_steps,
            metadata=request.metadata,
        )
        if updated is None:
            raise HTTPException(status_code=404, detail="Bucket not found")
        return _linked_bucket_response(updated)

    @app.post("/api/v1/buckets/{bucket_id}/revalidate", response_model=LinkedBucketResponse)
    async def revalidate_linked_bucket(
        bucket_id: str,
        actor: RequireAuth,
        resources: ResourceStore = Depends(require_resource_store),
    ) -> LinkedBucketResponse:
        existing = require_linked_bucket_record(
            bucket_id=bucket_id, actor=actor, resources=resources
        )
        validation = _validate_bucket_access(
            app.state.s3_client,
            bucket_name=existing.bucket_name,
            prefix_restriction=existing.prefix_restriction,
            read_only=existing.read_only,
        )
        updated = resources.update_linked_bucket(
            bucket_id=bucket_id,
            region=validation.region,
            is_validated=validation.is_validated,
            can_read=validation.can_read,
            can_write=validation.can_write,
            can_list=validation.can_list,
            remediation_steps=validation.remediation_steps,
        )
        if updated is None:
            raise HTTPException(status_code=404, detail="Bucket not found")
        return _linked_bucket_response(updated)

    @app.delete("/api/v1/buckets/{bucket_id}", response_model=LinkedBucketDeleteResponse)
    async def delete_linked_bucket(
        bucket_id: str,
        actor: RequireAuth,
        resources: ResourceStore = Depends(require_resource_store),
    ) -> LinkedBucketDeleteResponse:
        existing = {
            item.bucket_id: item
            for item in resources.list_linked_buckets(tenant_id=actor.tenant_id)
        }.get(bucket_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="Bucket not found")
        deleted = resources.delete_linked_bucket(bucket_id=bucket_id)
        if deleted is None:
            raise HTTPException(status_code=404, detail="Bucket not found")
        return LinkedBucketDeleteResponse(bucket_id=deleted.bucket_id, state=deleted.state)

    @app.get("/api/v1/admin/s3-buckets", response_model=AdminS3BucketListResponse)
    async def admin_list_s3_buckets(actor: RequireAdmin) -> AdminS3BucketListResponse:
        _ = actor
        try:
            payload = list_profile_s3_buckets()
        except Exception as exc:
            LOGGER.exception("Failed to list S3 buckets for admin bucket management")
            raise HTTPException(
                status_code=502, detail=f"Failed to list S3 buckets: {exc}"
            ) from exc
        return AdminS3BucketListResponse(**payload)

    @app.get("/api/v1/buckets/{bucket_id}/objects")
    async def list_bucket_objects(
        bucket_id: str,
        actor: RequireAuth,
        prefix: str = Query(default=""),
        max_keys: int = Query(default=500, ge=1, le=1000),
        resources: ResourceStore = Depends(require_resource_store),
    ) -> dict[str, Any]:
        record = require_linked_bucket_record(bucket_id=bucket_id, actor=actor, resources=resources)
        return list_bucket_items(bucket=record, prefix=prefix, max_keys=max_keys)

    @app.post("/api/v1/buckets/{bucket_id}/folders")
    async def create_bucket_folder(
        bucket_id: str,
        request: BucketFolderCreateRequest,
        actor: RequireAuth,
        prefix: str = Query(default=""),
        resources: ResourceStore = Depends(require_resource_store),
    ) -> dict[str, Any]:
        record = require_linked_bucket_record(bucket_id=bucket_id, actor=actor, resources=resources)
        if record.read_only or not record.can_write:
            raise HTTPException(status_code=400, detail="Bucket is read-only")
        folder_name = str(request.folder_name or "").strip().strip("/")
        if not folder_name:
            raise HTTPException(status_code=400, detail="folder_name is required")
        current_prefix = str(prefix or "").lstrip("/")
        if current_prefix and not _object_within_prefix(
            key=current_prefix,
            prefix_restriction=record.prefix_restriction,
        ):
            raise HTTPException(
                status_code=403, detail="Prefix is outside the linked bucket restriction"
            )
        folder_key = f"{current_prefix}{folder_name}/"
        if not _object_within_prefix(key=folder_key, prefix_restriction=record.prefix_restriction):
            raise HTTPException(
                status_code=403, detail="Folder is outside the linked bucket restriction"
            )
        try:
            app.state.s3_client.put_object(Bucket=record.bucket_name, Key=folder_key, Body=b"")
            app.state.s3_client.put_object(
                Bucket=record.bucket_name,
                Key=f"{folder_key}.hold",
                Body=b"",
            )
        except ClientError as exc:
            raise HTTPException(status_code=502, detail=f"Failed to create folder: {exc}") from exc
        return {"success": True, "folder": folder_key}

    @app.post("/api/v1/buckets/{bucket_id}/upload")
    async def upload_bucket_file(
        bucket_id: str,
        actor: RequireAuth,
        file: UploadFile = File(...),
        prefix: str = Form(""),
        resources: ResourceStore = Depends(require_resource_store),
    ) -> dict[str, Any]:
        record = require_linked_bucket_record(bucket_id=bucket_id, actor=actor, resources=resources)
        if record.read_only or not record.can_write:
            raise HTTPException(status_code=400, detail="Bucket is read-only")
        filename = str(file.filename or "").strip()
        if not filename:
            raise HTTPException(status_code=400, detail="Uploaded file must have a filename")
        current_prefix = str(prefix or "").lstrip("/")
        key = f"{current_prefix}{filename}"
        if not _object_within_prefix(key=key, prefix_restriction=record.prefix_restriction):
            raise HTTPException(
                status_code=403, detail="File is outside the linked bucket restriction"
            )
        try:
            extra_args = {"ContentType": file.content_type or "application/octet-stream"}
            app.state.s3_client.upload_fileobj(
                file.file, Bucket=record.bucket_name, Key=key, ExtraArgs=extra_args
            )
        except ClientError as exc:
            raise HTTPException(status_code=502, detail=f"Failed to upload file: {exc}") from exc
        return {"success": True, "key": key, "bucket": record.bucket_name}

    @app.get("/api/v1/buckets/{bucket_id}/objects/download-url")
    async def get_bucket_object_download_url(
        bucket_id: str,
        actor: RequireAuth,
        key: str = Query(...),
        resources: ResourceStore = Depends(require_resource_store),
    ) -> dict[str, str]:
        record = require_linked_bucket_record(bucket_id=bucket_id, actor=actor, resources=resources)
        normalized_key = str(key or "").lstrip("/")
        if not _object_within_prefix(
            key=normalized_key, prefix_restriction=record.prefix_restriction
        ):
            raise HTTPException(
                status_code=403, detail="Object is outside the linked bucket restriction"
            )
        try:
            url = app.state.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": record.bucket_name, "Key": normalized_key},
                ExpiresIn=3600,
            )
        except ClientError as exc:
            raise HTTPException(
                status_code=502, detail=f"Failed to generate download URL: {exc}"
            ) from exc
        return {"url": url}

    @app.get("/api/v1/buckets/{bucket_id}/objects/preview")
    async def preview_bucket_object(
        bucket_id: str,
        actor: RequireAuth,
        key: str = Query(...),
        lines: int = Query(default=20, ge=1, le=200),
        resources: ResourceStore = Depends(require_resource_store),
    ) -> dict[str, Any]:
        record = require_linked_bucket_record(bucket_id=bucket_id, actor=actor, resources=resources)
        normalized_key = str(key or "").lstrip("/")
        if not _object_within_prefix(
            key=normalized_key, prefix_restriction=record.prefix_restriction
        ):
            raise HTTPException(
                status_code=403, detail="Object is outside the linked bucket restriction"
            )
        try:
            return _preview_s3_object(
                app.state.s3_client,
                bucket_name=record.bucket_name,
                key=normalized_key,
                lines=lines,
            )
        except ClientError as exc:
            raise HTTPException(status_code=502, detail=f"Failed to preview object: {exc}") from exc

    @app.delete("/api/v1/buckets/{bucket_id}/objects")
    async def delete_bucket_object(
        bucket_id: str,
        actor: RequireAuth,
        key: str = Query(...),
        resources: ResourceStore = Depends(require_resource_store),
    ) -> dict[str, Any]:
        record = require_linked_bucket_record(bucket_id=bucket_id, actor=actor, resources=resources)
        if record.read_only or not record.can_write:
            raise HTTPException(status_code=400, detail="Bucket is read-only")
        normalized_key = str(key or "").lstrip("/")
        if not _object_within_prefix(
            key=normalized_key, prefix_restriction=record.prefix_restriction
        ):
            raise HTTPException(
                status_code=403, detail="Object is outside the linked bucket restriction"
            )
        try:
            app.state.s3_client.delete_object(Bucket=record.bucket_name, Key=normalized_key)
        except ClientError as exc:
            raise HTTPException(status_code=502, detail=f"Failed to delete object: {exc}") from exc
        return {"success": True, "deleted": normalized_key}

    def resolve_cluster_region(
        cluster_name: str,
        *,
        region: str | None,
        service: ClusterService,
    ) -> str:
        explicit_region = str(region or "").strip()
        if explicit_region:
            return explicit_region
        cached_region = service.get_region_for_cluster(cluster_name)
        if cached_region:
            return cached_region
        cluster = service.get_cluster_by_name(cluster_name, force_refresh=True)
        if cluster is not None:
            return cluster.region
        raise HTTPException(status_code=404, detail=f"Cluster not found: {cluster_name}")

    @app.get("/api/v1/clusters/jobs", response_model=list[ClusterJobResponse])
    async def list_cluster_jobs(
        actor: RequireAdmin,
        resources: ResourceStore = Depends(require_resource_store),
    ) -> list[ClusterJobResponse]:
        records = resources.list_cluster_jobs(tenant_id=None if actor.is_admin else actor.tenant_id)
        return [_cluster_job_response(item) for item in records]

    @app.get("/api/v1/clusters/jobs/{job_euid}", response_model=ClusterJobResponse)
    async def get_cluster_job(
        job_euid: str,
        actor: RequireAdmin,
        resources: ResourceStore = Depends(require_resource_store),
    ) -> ClusterJobResponse:
        record = resources.get_cluster_job(job_euid)
        if record is None:
            raise HTTPException(status_code=404, detail="Cluster job not found")
        if not actor.is_admin and record.tenant_id != actor.tenant_id:
            raise HTTPException(status_code=403, detail="Cluster job is outside the caller tenant")
        return _cluster_job_response(record)

    @app.get("/api/v1/clusters/create-options", response_model=ClusterCreateOptionsResponse)
    async def get_cluster_create_options(
        actor: RequireAdmin,
        region: str = Query(...),
    ) -> ClusterCreateOptionsResponse:
        _ = actor
        normalized_region = str(region or "").strip()
        if not normalized_region:
            raise HTTPException(status_code=400, detail="region is required")
        return load_cluster_create_options(normalized_region)

    @app.post(
        "/api/v1/clusters/scan-regions",
        response_model=ClusterScanRegionsResponse,
    )
    async def set_cluster_scan_regions(
        request: ClusterScanRegionsUpdateRequest,
        actor: RequireAdmin,
    ) -> ClusterScanRegionsResponse:
        _ = actor
        try:
            return update_cluster_scan_regions(request.regions_csv)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("Failed to update cluster scan regions")
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post(
        "/api/v1/clusters/verify-partitions",
        response_model=ClusterPartitionVerificationResponse,
    )
    async def verify_cluster_partitions(
        request: ClusterPartitionRequest,
        actor: RequireAdmin,
    ) -> ClusterPartitionVerificationResponse:
        _ = actor
        try:
            selection = resolve_cluster_partition_selection(
                region=request.region,
                region_az=request.region_az,
            )
            return run_cluster_partition_verification(
                app.state.settings,
                region=selection.region,
                region_az=selection.region_az,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception(
                "Failed to verify partition instances for %s in %s",
                request.region,
                request.region_az,
            )
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post(
        "/api/v1/clusters/partition-pricing",
        response_model=ClusterPartitionPricingResponse,
    )
    async def cluster_partition_pricing(
        request: ClusterPartitionPricingRequest,
        actor: RequireAdmin,
    ) -> ClusterPartitionPricingResponse:
        _ = actor
        try:
            cluster_config_path, partition_instances = load_daylily_partition_instance_types(
                app.state.settings
            )
            snapshot = collect_daylily_cluster_pricing_snapshot(
                app.state.settings,
                region=request.region,
                partitions=list(partition_instances.keys()),
            )
            return build_cluster_partition_pricing(
                region=request.region,
                cluster_config_path=cluster_config_path,
                partition_instances=partition_instances,
                snapshot=snapshot,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception(
                "Failed to calculate partition pricing for %s",
                request.region,
            )
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/v1/clusters")
    async def list_clusters(
        actor: RequireAdmin,
        refresh: bool = Query(default=False),
        fetch_ssh_status: bool = Query(default=False),
        service: ClusterService = Depends(require_cluster_service),
    ) -> dict[str, list[dict[str, Any]]]:
        _ = actor
        items = service.get_all_clusters_with_status(
            force_refresh=refresh,
            fetch_ssh_status=fetch_ssh_status,
        )
        return {"items": [item.to_dict(include_sensitive=fetch_ssh_status) for item in items]}

    @app.post(
        "/api/v1/clusters",
        response_model=ClusterJobResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def create_cluster(
        request: ClusterCreateRequest,
        actor: RequireAdmin,
        manager: ClusterJobManager = Depends(require_cluster_job_manager),
    ) -> ClusterJobResponse:
        owner_user_id = str(request.owner_user_id or actor.user_id).strip()
        if not owner_user_id:
            raise HTTPException(status_code=400, detail="owner_user_id is required")
        cluster_name = str(request.cluster_name or "").strip()
        region_az = str(request.region_az or "").strip()
        ssh_key_name = str(request.ssh_key_name or "").strip()
        s3_bucket_name = str(request.s3_bucket_name or "").strip()
        if not cluster_name or not region_az or not ssh_key_name or not s3_bucket_name:
            raise HTTPException(
                status_code=400,
                detail="cluster_name, region_az, ssh_key_name, and s3_bucket_name are required",
            )
        try:
            selection = resolve_cluster_partition_selection(
                region=request.region,
                region_az=region_az,
            )
            verification = run_cluster_partition_verification(
                app.state.settings,
                region=selection.region,
                region_az=selection.region_az,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception(
                "Failed pre-create partition verification for %s in %s",
                request.region or region_from_region_az(region_az),
                region_az,
            )
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if verification.has_failures:
            failing_partitions = (
                ", ".join(
                    item.partition for item in verification.partitions if item.status == "FAIL"
                )
                or "unknown partitions"
            )
            raise HTTPException(
                status_code=400,
                detail=(
                    "Create blocked because partition verification found no current Spot "
                    f"availability for: {failing_partitions}."
                ),
            )
        record = manager.start_create_job(
            cluster_name=cluster_name,
            region_az=selection.region_az,
            ssh_key_name=ssh_key_name,
            s3_bucket_name=s3_bucket_name,
            tenant_id=actor.tenant_id,
            owner_user_id=owner_user_id,
            sponsor_user_id=actor.user_id,
            aws_profile=app.state.settings.aws_profile,
            contact_email=str(request.contact_email or "").strip() or actor.email,
            pass_on_warn=bool(request.pass_on_warn),
            debug=bool(request.debug),
        )
        return _cluster_job_response(record)

    @app.get("/api/v1/clusters/{cluster_name}")
    async def get_cluster(
        cluster_name: str,
        actor: RequireAdmin,
        region: str | None = Query(default=None),
        refresh: bool = Query(default=False),
        fetch_ssh_status: bool = Query(default=False),
        service: ClusterService = Depends(require_cluster_service),
    ) -> dict[str, Any]:
        _ = actor
        resolved_region = resolve_cluster_region(cluster_name, region=region, service=service)
        cluster = service.describe_cluster(cluster_name, resolved_region)
        if fetch_ssh_status:
            cluster = service.fetch_headnode_status(cluster)
        payload = cluster.to_dict(include_sensitive=fetch_ssh_status)
        if refresh:
            service.clear_cache()
        if cluster.error_message and payload.get("cluster_status") == "UNKNOWN":
            raise HTTPException(status_code=404, detail=cluster.error_message)
        return payload

    @app.post("/api/v1/clusters/{cluster_name}/delete-plan")
    async def create_cluster_delete_plan(
        cluster_name: str,
        actor: RequireAdmin,
        region: str | None = Query(default=None),
        service: ClusterService = Depends(require_cluster_service),
    ) -> dict[str, Any]:
        _ = actor
        resolved_region = resolve_cluster_region(cluster_name, region=region, service=service)
        try:
            return service.create_delete_plan(cluster_name, resolved_region)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.delete("/api/v1/clusters/{cluster_name}")
    async def delete_cluster(
        cluster_name: str,
        actor: RequireAdmin,
        region: str | None = Query(default=None),
        confirmation_token: str = Query(...),
        confirm_cluster_name: str = Query(...),
        service: ClusterService = Depends(require_cluster_service),
    ) -> dict[str, Any]:
        _ = actor
        resolved_region = resolve_cluster_region(cluster_name, region=region, service=service)
        try:
            result = service.delete_cluster(
                cluster_name,
                resolved_region,
                confirmation_token=confirmation_token,
                confirm_cluster_name=confirm_cluster_name,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "cluster_name": cluster_name,
            "region": resolved_region,
            "result": result,
        }

    @app.get("/api/v1/user-tokens", response_model=list[UserTokenResponse])
    async def list_user_tokens(
        actor: RequireAuth,
        service: UserTokenService = Depends(require_token_service),
    ) -> list[UserTokenResponse]:
        return [_token_response(item) for item in service.list_tokens(actor=actor)]

    @app.post(
        "/api/v1/user-tokens", response_model=UserTokenResponse, status_code=status.HTTP_201_CREATED
    )
    async def create_user_token(
        request: UserTokenCreateRequest,
        actor: RequireAuth,
        service: UserTokenService = Depends(require_token_service),
    ) -> UserTokenResponse:
        record, plaintext = service.create_token(
            actor=actor,
            owner_user_id=actor.user_id,
            token_name=request.token_name,
            scope=request.scope,
            expires_in_days=request.expires_in_days,
            note=request.note,
        )
        return _token_response(record, plaintext_token=plaintext)

    @app.post("/api/v1/user-tokens/{token_euid}/revoke", response_model=UserTokenResponse)
    async def revoke_user_token(
        token_euid: str,
        request: TokenRevokeRequest,
        actor: RequireAuth,
        service: UserTokenService = Depends(require_token_service),
    ) -> UserTokenResponse:
        try:
            record = service.revoke_token(actor=actor, token_euid=token_euid, note=request.note)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except AuthError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return _token_response(record)

    @app.get("/api/v1/user-tokens/{token_euid}/usage", response_model=list[TokenUsageResponse])
    async def list_user_token_usage(
        token_euid: str,
        actor: RequireAuth,
        service: UserTokenService = Depends(require_token_service),
    ) -> list[TokenUsageResponse]:
        try:
            records = service.list_usage(actor=actor, token_euid=token_euid)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except AuthError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return [_token_usage_response(item) for item in records]

    @app.get("/api/v1/admin/user-tokens", response_model=list[UserTokenResponse])
    async def admin_list_user_tokens(
        actor: RequireAdmin,
        owner_user_id: str = Query(default="*"),
        service: UserTokenService = Depends(require_token_service),
    ) -> list[UserTokenResponse]:
        return [
            _token_response(item)
            for item in service.list_tokens(actor=actor, owner_user_id=owner_user_id)
        ]

    @app.post(
        "/api/v1/admin/user-tokens",
        response_model=UserTokenResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def admin_create_user_token(
        request: AdminUserTokenCreateRequest,
        actor: RequireAdmin,
        service: UserTokenService = Depends(require_token_service),
    ) -> UserTokenResponse:
        record, plaintext = service.create_token(
            actor=actor,
            owner_user_id=request.owner_user_id,
            token_name=request.token_name,
            scope=request.scope,
            expires_in_days=request.expires_in_days,
            note=request.note,
            client_registration_euid=request.client_registration_euid,
        )
        return _token_response(record, plaintext_token=plaintext)

    @app.post("/api/v1/admin/user-tokens/{token_euid}/revoke", response_model=UserTokenResponse)
    async def admin_revoke_user_token(
        token_euid: str,
        request: TokenRevokeRequest,
        actor: RequireAdmin,
        service: UserTokenService = Depends(require_token_service),
    ) -> UserTokenResponse:
        try:
            record = service.revoke_token(actor=actor, token_euid=token_euid, note=request.note)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _token_response(record)

    @app.get("/api/v1/admin/users", response_model=list[AtlasUserDirectoryResponse])
    async def admin_list_atlas_users(
        actor: RequireAdmin,
        tenant_id: uuid.UUID | None = Query(default=None),
        search: str | None = Query(default=None),
        active_only: bool = Query(default=True),
        limit: int = Query(default=50, ge=1, le=200),
        skip: int = Query(default=0, ge=0),
        directory: CognitoUserDirectoryService = Depends(require_user_directory),
    ) -> list[AtlasUserDirectoryResponse]:
        _ = actor
        try:
            results = directory.list_users(
                tenant_id=tenant_id,
                search=search,
                active_only=active_only,
                limit=limit,
                skip=skip,
            )
        except AuthError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return [_atlas_user_directory_response(item) for item in results]

    @app.get("/api/v1/admin/client-registrations", response_model=list[ClientRegistrationResponse])
    async def admin_list_client_registrations(
        actor: RequireAdmin,
        owner_user_id: str | None = Query(default=None),
        resources: ResourceStore = Depends(require_resource_store),
    ) -> list[ClientRegistrationResponse]:
        _ = actor
        records = resources.list_client_registrations(owner_user_id=owner_user_id)
        return [_client_registration_response(item) for item in records]

    @app.get(
        "/api/v1/admin/client-registrations/{client_registration_euid}",
        response_model=ClientRegistrationResponse,
    )
    async def admin_get_client_registration(
        client_registration_euid: str,
        actor: RequireAdmin,
        resources: ResourceStore = Depends(require_resource_store),
    ) -> ClientRegistrationResponse:
        _ = actor
        record = resources.get_client_registration(client_registration_euid)
        if record is None:
            raise HTTPException(status_code=404, detail="Client registration not found")
        return _client_registration_response(record)

    @app.post(
        "/api/v1/admin/client-registrations",
        response_model=ClientRegistrationResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def admin_create_client_registration(
        request: ClientRegistrationCreateRequest,
        actor: RequireAdmin,
        resources: ResourceStore = Depends(require_resource_store),
    ) -> ClientRegistrationResponse:
        record = resources.create_client_registration(
            client_name=request.client_name,
            owner_user_id=request.owner_user_id,
            sponsor_user_id=actor.user_id,
            scopes=request.scopes,
            metadata=request.metadata,
        )
        return _client_registration_response(record)

    @app.get(
        "/api/v1/admin/client-registrations/{client_registration_euid}/tokens",
        response_model=list[UserTokenResponse],
    )
    async def admin_list_client_registration_tokens(
        client_registration_euid: str,
        actor: RequireAdmin,
        resources: ResourceStore = Depends(require_resource_store),
        service: UserTokenService = Depends(require_token_service),
    ) -> list[UserTokenResponse]:
        _ = actor
        registration = resources.get_client_registration(client_registration_euid)
        if registration is None:
            raise HTTPException(status_code=404, detail="Client registration not found")
        tokens = [
            item
            for item in service.list_tokens(actor=actor, owner_user_id="*")
            if item.client_registration_euid == client_registration_euid
        ]
        return [_token_response(item) for item in tokens]

    @app.post(
        "/api/v1/admin/client-registrations/{client_registration_euid}/tokens",
        response_model=UserTokenResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def admin_create_client_registration_token(
        client_registration_euid: str,
        request: UserTokenCreateRequest,
        actor: RequireAdmin,
        resources: ResourceStore = Depends(require_resource_store),
        service: UserTokenService = Depends(require_token_service),
    ) -> UserTokenResponse:
        registration = resources.get_client_registration(client_registration_euid)
        if registration is None:
            raise HTTPException(status_code=404, detail="Client registration not found")
        requested_scope = str(request.scope or "internal_ro").strip().lower()
        if registration.scopes and requested_scope not in {
            str(item).strip().lower() for item in registration.scopes
        }:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Scope {requested_scope!r} is not allowed for client registration "
                    f"{client_registration_euid}"
                ),
            )
        record, plaintext = service.create_token(
            actor=actor,
            owner_user_id=registration.owner_user_id,
            token_name=request.token_name,
            scope=requested_scope,
            expires_in_days=request.expires_in_days,
            note=request.note,
            client_registration_euid=client_registration_euid,
        )
        return _token_response(record, plaintext_token=plaintext)

    mount_gui(app)
    mount_tapdb_admin(app, settings)
    return app
