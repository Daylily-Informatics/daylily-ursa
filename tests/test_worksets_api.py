from __future__ import annotations

import io
from types import SimpleNamespace
import uuid
from unittest.mock import patch

from fastapi.testclient import TestClient

from daylib_ursa.auth import CurrentUser, Role
from daylib_ursa.analysis_samples_manifest import ANALYSIS_SAMPLES_SOURCE_COLUMNS
from daylib_ursa.config import Settings
from daylib_ursa.integrations.dewey_client import DeweyClientError
from daylib_ursa.file_metadata import ANALYSIS_SAMPLES_COLUMNS, DEFAULT_STAGE_TARGET
from daylib_ursa.resource_store import (
    AnalysisJobEventRecord,
    AnalysisJobRecord,
    LinkedBucketRecord,
    ManifestEditorOptionRecord,
    ManifestRecord,
    StagingJobEventRecord,
    StagingJobRecord,
    WorksetRecord,
)
from daylib_ursa.workset_api import create_app

TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
TENANT_TWO_ID = uuid.UUID("00000000-0000-0000-0000-000000000002")
USER_ID = "00000000-0000-0000-0000-000000000101"


class DummyAuthProvider:
    def resolve_access_token(self, access_token: str) -> CurrentUser:
        if access_token == "tenant-two-token":
            return CurrentUser(
                sub="00000000-0000-0000-0000-000000000202",
                email="tenant-two@example.test",
                name="Tenant Two",
                tenant_id=TENANT_TWO_ID,
                roles=[Role.ADMIN.value],
                auth_source="cognito",
            )
        assert access_token == "atlas-token"
        return CurrentUser(
            sub=USER_ID,
            email="user@example.test",
            name="User One",
            tenant_id=TENANT_ID,
            roles=[Role.ADMIN.value],
            auth_source="cognito",
        )


class MemoryResourceStore:
    def __init__(self) -> None:
        self.worksets: dict[str, WorksetRecord] = {}
        self.manifests: dict[str, ManifestRecord] = {}
        self.manifest_options: dict[str, ManifestEditorOptionRecord] = {}
        self.buckets: dict[str, LinkedBucketRecord] = {}
        self.analysis_jobs: dict[str, AnalysisJobRecord] = {}
        self.staging_jobs: dict[str, StagingJobRecord] = {}
        self._workset_seq = 0
        self._manifest_seq = 0
        self._manifest_option_seq = 0
        self._bucket_seq = 0
        self._analysis_job_seq = 0
        self._analysis_event_seq = 0
        self._staging_job_seq = 0
        self._staging_event_seq = 0

    def list_worksets(self, *, tenant_id: uuid.UUID, limit: int = 100):
        _ = limit
        return [item for item in self.worksets.values() if item.tenant_id == tenant_id]

    def create_workset(
        self, *, name: str, tenant_id: uuid.UUID, owner_user_id: str, artifact_set_euids, metadata
    ):
        self._workset_seq += 1
        record = WorksetRecord(
            workset_euid=f"WS-{self._workset_seq}",
            name=name,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            state="ACTIVE",
            artifact_set_euids=list(artifact_set_euids or []),
            metadata=dict(metadata or {}),
            created_at="2026-03-25T00:00:00Z",
            updated_at="2026-03-25T00:00:00Z",
            manifests=[],
            analysis_euids=[],
        )
        self.worksets[record.workset_euid] = record
        return record

    def get_workset(self, workset_euid: str):
        return self.worksets.get(workset_euid)

    def list_manifests(self, *, tenant_id: uuid.UUID, limit: int = 200):
        _ = limit
        return [item for item in self.manifests.values() if item.tenant_id == tenant_id]

    def create_manifest(
        self,
        *,
        workset_euid: str,
        name: str,
        artifact_set_euid: str | None,
        artifact_euids,
        input_references,
        metadata,
    ):
        workset = self.worksets[workset_euid]
        self._manifest_seq += 1
        manifest = ManifestRecord(
            manifest_euid=f"MF-{self._manifest_seq}",
            name=name,
            workset_euid=workset_euid,
            tenant_id=workset.tenant_id,
            owner_user_id=workset.owner_user_id,
            artifact_set_euid=artifact_set_euid,
            artifact_euids=list(artifact_euids or []),
            input_references=list(input_references or []),
            metadata=dict(metadata or {}),
            created_at="2026-03-25T00:10:00Z",
            updated_at="2026-03-25T00:10:00Z",
            state="ACTIVE",
        )
        self.manifests[manifest.manifest_euid] = manifest
        updated = WorksetRecord(
            workset_euid=workset.workset_euid,
            name=workset.name,
            tenant_id=workset.tenant_id,
            owner_user_id=workset.owner_user_id,
            state=workset.state,
            artifact_set_euids=workset.artifact_set_euids,
            metadata=workset.metadata,
            created_at=workset.created_at,
            updated_at="2026-03-25T00:10:00Z",
            manifests=[*workset.manifests, manifest],
            analysis_euids=workset.analysis_euids,
        )
        self.worksets[workset_euid] = updated
        return manifest

    def get_manifest(self, manifest_euid: str):
        return self.manifests.get(manifest_euid)

    def list_manifest_editor_options(
        self,
        *,
        tenant_id: uuid.UUID,
        option_type: str | None = None,
        limit: int = 1000,
    ):
        _ = limit
        records = [
            record
            for record in self.manifest_options.values()
            if record.tenant_id == tenant_id and record.state == "ACTIVE"
        ]
        if option_type is not None:
            records = [record for record in records if record.option_type == option_type]
        return records

    def upsert_manifest_editor_option(
        self,
        *,
        tenant_id: uuid.UUID,
        option_type: str,
        value: str,
        actor_user_id: str,
    ):
        cleaned = " ".join(str(value or "").strip().split())
        normalized = cleaned.casefold()
        option_key = f"{tenant_id}:{option_type}:{normalized}"
        existing = self.manifest_options.get(option_key)
        if existing is not None:
            updated = ManifestEditorOptionRecord(
                option_euid=existing.option_euid,
                tenant_id=tenant_id,
                option_type=option_type,
                value=existing.value,
                normalized_value=existing.normalized_value,
                created_by=existing.created_by,
                created_at=existing.created_at,
                updated_at="2026-03-25T00:12:00Z",
                state="ACTIVE",
            )
            self.manifest_options[option_key] = updated
            return updated
        self._manifest_option_seq += 1
        record = ManifestEditorOptionRecord(
            option_euid=f"MO-{self._manifest_option_seq}",
            tenant_id=tenant_id,
            option_type=option_type,
            value=cleaned,
            normalized_value=normalized,
            created_by=actor_user_id,
            created_at="2026-03-25T00:11:00Z",
            updated_at="2026-03-25T00:11:00Z",
            state="ACTIVE",
        )
        self.manifest_options[option_key] = record
        return record

    def create_analysis_job(
        self,
        *,
        job_name: str,
        workset_euid: str,
        manifest_euid: str,
        cluster_name: str,
        region: str,
        tenant_id: uuid.UUID,
        owner_user_id: str,
        request,
    ):
        self._analysis_job_seq += 1
        record = AnalysisJobRecord(
            job_euid=f"AJ-{self._analysis_job_seq}",
            job_name=job_name,
            workset_euid=workset_euid,
            manifest_euid=manifest_euid,
            cluster_name=cluster_name,
            region=region,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            state="DEFINED",
            created_at="2026-03-25T00:40:00Z",
            updated_at="2026-03-25T00:40:00Z",
            started_at=None,
            completed_at=None,
            return_code=None,
            error=None,
            output_summary=None,
            request=dict(request or {}),
            launch={},
            events=[],
        )
        self.analysis_jobs[record.job_euid] = record
        return record

    def add_analysis_job_event(
        self,
        *,
        job_euid: str,
        event_type: str,
        status: str,
        summary: str,
        details=None,
        created_by=None,
    ):
        self._analysis_event_seq += 1
        event = AnalysisJobEventRecord(
            event_euid=f"AJE-{self._analysis_event_seq}",
            job_euid=job_euid,
            event_type=event_type,
            status=status,
            summary=summary,
            details=dict(details or {}),
            created_by=created_by,
            created_at="2026-03-25T00:40:01Z",
        )
        record = self.analysis_jobs[job_euid]
        updated = AnalysisJobRecord(
            **{
                **record.__dict__,
                "events": [event, *record.events],
                "updated_at": event.created_at,
            }
        )
        self.analysis_jobs[job_euid] = updated
        return event

    def update_analysis_job_status(self, *, job_euid: str, state: str, created_by: str, **updates):
        _ = created_by
        record = self.analysis_jobs[job_euid]
        updated = AnalysisJobRecord(
            **{
                **record.__dict__,
                "state": state,
                "updated_at": "2026-03-25T00:41:00Z",
                "started_at": updates.get("started_at", record.started_at),
                "completed_at": updates.get("completed_at", record.completed_at),
                "return_code": updates.get("return_code", record.return_code),
                "error": updates.get("error", record.error),
                "output_summary": updates.get("output_summary", record.output_summary),
                "launch": updates.get("launch", record.launch),
            }
        )
        self.analysis_jobs[job_euid] = updated
        return updated

    def get_analysis_job(self, job_euid: str):
        return self.analysis_jobs.get(job_euid)

    def list_analysis_jobs(self, *, tenant_id: uuid.UUID | None = None, limit: int = 200):
        _ = limit
        return [
            item
            for item in self.analysis_jobs.values()
            if tenant_id is None or item.tenant_id == tenant_id
        ]

    def create_staging_job(
        self,
        *,
        job_name: str,
        workset_euid: str,
        manifest_euid: str,
        cluster_name: str,
        region: str,
        tenant_id: uuid.UUID,
        owner_user_id: str,
        request,
    ):
        workset = self.worksets[workset_euid]
        manifest = self.manifests[manifest_euid]
        if workset.tenant_id != tenant_id or manifest.tenant_id != tenant_id:
            raise ValueError("staging job tenant mismatch")
        if manifest.workset_euid != workset_euid:
            raise ValueError("manifest does not belong to workset")
        self._staging_job_seq += 1
        record = StagingJobRecord(
            job_euid=f"SJ-{self._staging_job_seq}",
            job_name=job_name,
            workset_euid=workset_euid,
            manifest_euid=manifest_euid,
            cluster_name=cluster_name,
            region=region,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            state="DEFINED",
            created_at="2026-03-25T00:35:00Z",
            updated_at="2026-03-25T00:35:00Z",
            started_at=None,
            completed_at=None,
            return_code=None,
            error=None,
            output_summary=None,
            request=dict(request or {}),
            stage={},
            events=[],
        )
        self.staging_jobs[record.job_euid] = record
        return record

    def add_staging_job_event(
        self,
        *,
        job_euid: str,
        event_type: str,
        status: str,
        summary: str,
        details=None,
        created_by=None,
    ):
        self._staging_event_seq += 1
        event = StagingJobEventRecord(
            event_euid=f"SJE-{self._staging_event_seq}",
            job_euid=job_euid,
            event_type=event_type,
            status=status,
            summary=summary,
            details=dict(details or {}),
            created_by=created_by,
            created_at="2026-03-25T00:35:01Z",
        )
        record = self.staging_jobs[job_euid]
        updated = StagingJobRecord(
            **{
                **record.__dict__,
                "events": [event, *record.events],
                "updated_at": event.created_at,
            }
        )
        self.staging_jobs[job_euid] = updated
        return event

    def update_staging_job_status(self, *, job_euid: str, state: str, created_by: str, **updates):
        _ = created_by
        record = self.staging_jobs[job_euid]
        updated = StagingJobRecord(
            **{
                **record.__dict__,
                "state": state,
                "updated_at": "2026-03-25T00:36:00Z",
                "started_at": updates.get("started_at", record.started_at),
                "completed_at": updates.get("completed_at", record.completed_at),
                "return_code": updates.get("return_code", record.return_code),
                "error": updates.get("error", record.error),
                "output_summary": updates.get("output_summary", record.output_summary),
                "stage": updates.get("stage", record.stage),
            }
        )
        self.staging_jobs[job_euid] = updated
        return updated

    def get_staging_job(self, job_euid: str):
        return self.staging_jobs.get(job_euid)

    def list_staging_jobs(self, *, tenant_id: uuid.UUID, limit: int = 200):
        _ = limit
        return [item for item in self.staging_jobs.values() if item.tenant_id == tenant_id]

    def list_linked_buckets(self, *, tenant_id: uuid.UUID, limit: int = 200):
        _ = limit
        return [
            item
            for item in self.buckets.values()
            if item.tenant_id == tenant_id and item.state != "DELETED"
        ]

    def create_linked_bucket(
        self,
        *,
        bucket_name: str,
        tenant_id: uuid.UUID,
        owner_user_id: str,
        display_name=None,
        bucket_type="secondary",
        description=None,
        prefix_restriction=None,
        read_only=False,
        region=None,
        is_validated=False,
        can_read=False,
        can_write=False,
        can_list=False,
        remediation_steps=None,
        metadata=None,
    ):
        for item in self.buckets.values():
            if (
                item.tenant_id == tenant_id
                and item.bucket_name == bucket_name
                and item.state != "DELETED"
            ):
                raise ValueError(f"Bucket already linked: {bucket_name}")
        self._bucket_seq += 1
        record = LinkedBucketRecord(
            bucket_id=f"BK-{self._bucket_seq}",
            bucket_name=bucket_name,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            display_name=display_name,
            metadata=dict(metadata or {}),
            created_at="2026-03-25T00:20:00Z",
            updated_at="2026-03-25T00:20:00Z",
            state="ACTIVE",
            bucket_type=bucket_type,
            description=description,
            prefix_restriction=prefix_restriction,
            read_only=read_only,
            region=region,
            is_validated=is_validated,
            can_read=can_read,
            can_write=can_write,
            can_list=can_list,
            remediation_steps=list(remediation_steps or []),
        )
        self.buckets[record.bucket_id] = record
        return record

    def get_linked_bucket(self, bucket_id: str):
        return self.buckets.get(bucket_id)

    def update_linked_bucket(self, *, bucket_id: str, **updates):
        record = self.buckets.get(bucket_id)
        if record is None:
            return None
        updated = LinkedBucketRecord(
            bucket_id=record.bucket_id,
            bucket_name=record.bucket_name,
            tenant_id=record.tenant_id,
            owner_user_id=record.owner_user_id,
            display_name=updates.get("display_name", record.display_name),
            metadata=record.metadata
            if updates.get("metadata") is None
            else updates.get("metadata"),
            created_at=record.created_at,
            updated_at="2026-03-25T00:20:30Z",
            state=record.state,
            bucket_type=record.bucket_type
            if updates.get("bucket_type") is None
            else updates.get("bucket_type"),
            description=updates.get("description", record.description),
            prefix_restriction=updates.get("prefix_restriction", record.prefix_restriction),
            read_only=record.read_only
            if updates.get("read_only") is None
            else updates.get("read_only"),
            region=updates.get("region", record.region),
            is_validated=record.is_validated
            if updates.get("is_validated") is None
            else updates.get("is_validated"),
            can_read=record.can_read
            if updates.get("can_read") is None
            else updates.get("can_read"),
            can_write=record.can_write
            if updates.get("can_write") is None
            else updates.get("can_write"),
            can_list=record.can_list
            if updates.get("can_list") is None
            else updates.get("can_list"),
            remediation_steps=updates.get("remediation_steps", record.remediation_steps),
        )
        self.buckets[bucket_id] = updated
        return updated

    def delete_linked_bucket(self, *, bucket_id: str):
        record = self.buckets.get(bucket_id)
        if record is None:
            return None
        deleted = LinkedBucketRecord(
            bucket_id=record.bucket_id,
            bucket_name=record.bucket_name,
            tenant_id=record.tenant_id,
            owner_user_id=record.owner_user_id,
            display_name=record.display_name,
            metadata=record.metadata,
            created_at=record.created_at,
            updated_at="2026-03-25T00:21:00Z",
            state="DELETED",
            bucket_type=record.bucket_type,
            description=record.description,
            prefix_restriction=record.prefix_restriction,
            read_only=record.read_only,
            region=record.region,
            is_validated=record.is_validated,
            can_read=record.can_read,
            can_write=record.can_write,
            can_list=record.can_list,
            remediation_steps=record.remediation_steps,
        )
        self.buckets[bucket_id] = deleted
        return deleted

    def record_dewey_import(
        self,
        *,
        artifact_euid: str,
        artifact_type: str,
        storage_uri: str,
        actor_user_id: str,
        metadata=None,
    ):
        return SimpleNamespace(
            import_euid="DI-1",
            artifact_euid=artifact_euid,
            artifact_type=artifact_type,
            storage_uri=storage_uri,
            actor_user_id=actor_user_id,
            created_at="2026-03-25T00:30:00Z",
            metadata=dict(metadata or {}),
        )


class DummyAnalysisStore:
    def list_analyses(self, *, tenant_id=None, workset_euid=None, limit=200):  # pragma: no cover
        _ = (tenant_id, workset_euid, limit)
        return []


class DummyClusterService:
    def get_all_clusters_with_status(
        self, *, force_refresh: bool = False, fetch_ssh_status: bool = False
    ):
        _ = (force_refresh, fetch_ssh_status)
        return []

    def get_region_for_cluster(self, cluster_name: str):
        _ = cluster_name
        return "us-west-2"

    def create_delete_plan(self, cluster_name: str, region: str):
        return {
            "cluster_name": cluster_name,
            "region": region,
            "confirmation_token": "delete-token",
            "dry_run_stdout": "would delete",
            "dry_run_stderr": "",
        }

    def delete_cluster(
        self,
        cluster_name: str,
        region: str,
        *,
        confirmation_token: str,
        confirm_cluster_name: str,
    ):
        if confirmation_token != "delete-token" or confirm_cluster_name != cluster_name:
            raise ValueError("invalid token")
        return {"cluster_name": cluster_name, "region": region, "return_code": 0}


class DummyAnalysisJobManager:
    def __init__(self, resources: MemoryResourceStore) -> None:
        self.resources = resources

    def launch_job(self, job_euid: str, *, actor_user_id: str):
        return self.resources.update_analysis_job_status(
            job_euid=job_euid,
            state="RUNNING",
            created_by=actor_user_id,
            started_at="2026-03-25T00:41:00Z",
            return_code=0,
            output_summary="launched",
            launch={"session_name": "ursa-test", "run_dir": "/home/ubuntu/daylily-runs/ursa-test"},
        )

    def refresh_job(self, job_euid: str, *, actor_user_id: str):
        return self.resources.update_analysis_job_status(
            job_euid=job_euid,
            state="COMPLETED",
            created_by=actor_user_id,
            completed_at="2026-03-25T00:50:00Z",
            return_code=0,
            output_summary="Workflow status: COMPLETED",
            launch={"session_name": "ursa-test", "status": {"exit_code": 0}},
        )

    def logs(self, job_euid: str, *, lines: int = 200):
        return {
            "job_euid": job_euid,
            "session_name": "ursa-test",
            "lines": lines,
            "stdout": "ok\n",
            "stderr": "",
        }


class DummyStagingJobManager:
    def __init__(self, resources: MemoryResourceStore) -> None:
        self.resources = resources

    def run_job(self, job_euid: str, *, actor_user_id: str):
        self.resources.add_staging_job_event(
            job_euid=job_euid,
            event_type="stage",
            status="STAGING",
            summary="Staging stable analysis manifest",
            details={},
            created_by=actor_user_id,
        )
        self.resources.add_staging_job_event(
            job_euid=job_euid,
            event_type="stage",
            status="COMPLETED",
            summary="Staged samples to /fsx/staged_sample_data",
            details={"stage_dir": "/fsx/staged_sample_data"},
            created_by=actor_user_id,
        )
        return self.resources.update_staging_job_status(
            job_euid=job_euid,
            state="COMPLETED",
            created_by=actor_user_id,
            started_at="2026-03-25T00:35:30Z",
            completed_at="2026-03-25T00:36:00Z",
            return_code=0,
            output_summary="Staged samples to /fsx/staged_sample_data",
            stage={
                "stage_dir": "/fsx/staged_sample_data",
                "stdout": "staged\n",
                "stderr": "",
            },
        )

    def logs(self, job_euid: str, *, lines: int = 200):
        _ = lines
        job = self.resources.get_staging_job(job_euid)
        return {
            "job_euid": job_euid,
            "stage_dir": job.stage.get("stage_dir"),
            "lines": lines,
            "stdout": job.stage.get("stdout", ""),
            "stderr": job.stage.get("stderr", ""),
        }


class DummyDeweyClient:
    def __init__(self) -> None:
        self.register_calls: list[dict] = []

    def resolve_artifact_set(self, artifact_set_euid: str):
        members = {
            "AS-1": [
                {"artifact_euid": "AT-1"},
                {"artifact_euid": "AT-2"},
            ],
            "AS-2": [
                {"artifact_euid": "AT-3"},
            ],
        }
        if artifact_set_euid not in members:
            raise DeweyClientError(f"unknown artifact set: {artifact_set_euid}")
        return {
            "artifact_set_euid": artifact_set_euid,
            "members": members[artifact_set_euid],
        }

    def resolve_artifact(self, artifact_euid: str):
        if artifact_euid not in {"AT-1", "AT-2", "AT-3"}:
            raise DeweyClientError(f"unknown artifact: {artifact_euid}")
        return {
            "artifact_euid": artifact_euid,
            "artifact_type": "fastq",
            "storage_uri": f"s3://dewey/{artifact_euid}.bin",
        }

    def register_artifact(self, **kwargs):
        self.register_calls.append(dict(kwargs))
        return "AT-IMPORTED-1"


class DummyS3Client:
    def head_object(self, Bucket: str, Key: str, **kwargs):  # noqa: N803
        _ = (Bucket, Key, kwargs)
        return {"ContentLength": 1, "ContentType": "text/plain"}

    def get_bucket_location(self, Bucket: str, **kwargs):  # noqa: N803
        _ = (Bucket, kwargs)
        return {"LocationConstraint": "us-west-2"}

    def list_objects_v2(self, Bucket: str, **kwargs):  # noqa: N803
        _ = (Bucket, kwargs)
        return {"Contents": [], "CommonPrefixes": []}

    def put_object(self, Bucket: str, Key: str, **kwargs):  # noqa: N803
        _ = (Bucket, Key, kwargs)
        return {}

    def delete_object(self, Bucket: str, Key: str, **kwargs):  # noqa: N803
        _ = (Bucket, Key, kwargs)
        return {}

    def get_object(self, Bucket: str, Key: str, **kwargs):  # noqa: N803
        _ = (Bucket, Key, kwargs)
        return {"Body": io.BytesIO(b"alpha\nbeta\n")}

    def generate_presigned_url(
        self, ClientMethod: str, Params: dict, ExpiresIn: int = 3600, **kwargs
    ):  # noqa: N803
        _ = (ClientMethod, Params, ExpiresIn, kwargs)
        return "https://example.test/download"

    def upload_fileobj(self, Fileobj, Bucket: str, Key: str, **kwargs):  # noqa: N803
        _ = (Bucket, Key, kwargs)
        Fileobj.read()
        return None


def _settings() -> Settings:
    return Settings(
        cors_origins="*",
        ursa_internal_api_key="ursa-test-key",
        bloom_base_url="https://bloom.example",
        atlas_base_url="https://atlas.example",
        ursa_internal_output_bucket="ursa-internal",
        ursa_tapdb_mount_enabled=False,
        session_secret_key="ursa-session-secret",
        cognito_user_pool_id="us-west-2_pool",
        cognito_app_client_id="client-id",
        cognito_region="us-west-2",
        cognito_domain="auth.example.test",
        cognito_callback_url="https://localhost:8913/auth/callback",
        cognito_logout_url="https://localhost:8913/login",
    )


def _create_test_app(
    *,
    resource_store: MemoryResourceStore | None = None,
    dewey_client: DummyDeweyClient | None = None,
):
    resources = resource_store or MemoryResourceStore()
    app = create_app(
        DummyAnalysisStore(),
        bloom_client=object(),
        auth_provider=DummyAuthProvider(),
        resource_store=resources,
        dewey_client=dewey_client,
        cluster_service=DummyClusterService(),
        analysis_job_manager=DummyAnalysisJobManager(resources),
        staging_job_manager=DummyStagingJobManager(resources),
        settings=_settings(),
        s3_client=DummyS3Client(),
    )
    return app


def _auth_headers(token: str | None = None) -> dict[str, str]:
    token = token or "atlas-token"
    return {"Authorization": f"Bearer {token}"}


def _command_payload(*, command_id: str = "illumina_snv_alignstats") -> dict:
    return {
        "command_id": command_id,
        "repository": "daylily-omics-analysis",
        "display_name": "Illumina SNV + Alignstats",
        "description": "Example",
        "datasource": "Illumina",
        "launcher": "workflow_launch",
        "targets": ["produce_alignstats", "produce_snv_concordances"],
        "genome": "hg38",
        "jobs": 10,
        "aligners": ["bwa2a"],
        "dedupers": ["dppl"],
        "snv_callers": ["sentd", "deep19"],
        "sv_callers": [],
        "destination": "dayoa",
        "no_containerized": False,
        "optional_features": {
            "tiddit": {
                "display_name": "Tiddit SV calling",
                "description": "Example",
                "targets": ["produce_tiddit"],
                "sv_callers": ["tiddit"],
            }
        },
    }


def _command_catalog_payload() -> dict:
    return {
        "command_catalog_version": 1,
        "default_repository": "daylily-omics-analysis",
        "repositories": {"daylily-omics-analysis": {"analysis_commands": [_command_payload()]}},
        "commands": [_command_payload()],
    }


def _command_preview_payload() -> dict:
    return {
        "valid": True,
        "command": _command_payload(),
        "argv": [
            "workflow",
            "launch",
            "--repository",
            "daylily-omics-analysis",
            "--destination",
            "dayoa",
            "--genome",
            "hg38",
            "--jobs",
            "10",
            "--aligners",
            "bwa2a",
            "--dedupers",
            "dppl",
            "--snv-callers",
            "sentd,deep19",
            "--target",
            "produce_alignstats produce_snv_concordances",
        ],
        "shell_preview": (
            "daylily-ec workflow launch --repository daylily-omics-analysis "
            "--destination dayoa --genome hg38 --jobs 10 --aligners bwa2a "
            "--dedupers dppl --snv-callers sentd,deep19 --target "
            "'produce_alignstats produce_snv_concordances'"
        ),
    }


def _editor_rows() -> list[dict[str, str]]:
    return [
        {
            "RUN_ID": "R0",
            "SAMPLE_ID": "S1",
            "EXPERIMENTID": "S1",
            "SAMPLE_TYPE": "blood",
            "LIB_PREP": "noampwgs",
            "SEQ_VENDOR": "ILMN",
            "SEQ_PLATFORM": "NOVASEQX",
            "LANE": "1",
            "SEQBC_ID": "S1",
            "PATH_TO_CONCORDANCE_DATA_DIR": "",
            "R1_FQ": "s3://bucket/S1_R1.fastq.gz",
            "R2_FQ": "s3://bucket/S1_R2.fastq.gz",
            "STAGE_DIRECTIVE": "stage_data",
            "STAGE_TARGET": DEFAULT_STAGE_TARGET,
            "SUBSAMPLE_PCT": "na",
            "IS_POS_CTRL": "false",
            "IS_NEG_CTRL": "false",
            "N_X": "1",
            "N_Y": "1",
            "EXTERNAL_SAMPLE_ID": "S1",
        }
    ]


def test_workset_and_manifest_routes_use_versioned_user_api() -> None:
    dewey = DummyDeweyClient()
    app = _create_test_app(resource_store=MemoryResourceStore(), dewey_client=dewey)

    with TestClient(app) as client:
        workset = client.post(
            "/api/v1/worksets",
            headers=_auth_headers(),
            json={"name": "Tumor batch", "artifact_set_euids": ["AS-1"]},
        )
        manifest = client.post(
            "/api/v1/manifests",
            headers=_auth_headers(),
            json={
                "workset_euid": "WS-1",
                "name": "manifest 1",
                "artifact_set_euid": "AS-1",
                "artifact_euids": ["AT-1", "AT-2"],
                "metadata": {"editor_analysis_inputs": _editor_rows()},
            },
        )
        listed_worksets = client.get("/api/v1/worksets", headers=_auth_headers())
        listed_manifests = client.get("/api/v1/manifests", headers=_auth_headers())
        clusters = client.get("/api/v1/clusters", headers=_auth_headers())

    assert workset.status_code == 201, workset.text
    assert workset.json()["tenant_id"] == str(TENANT_ID)
    assert manifest.status_code == 201, manifest.text
    assert manifest.json()["tenant_id"] == str(TENANT_ID)
    manifest_metadata = manifest.json()["metadata"]
    assert "stable_manifest" not in manifest_metadata
    analysis_samples_manifest = manifest_metadata["analysis_samples_manifest"]
    assert analysis_samples_manifest["filename"] == "analysis_samples.tsv"
    assert analysis_samples_manifest["row_count"] == 1
    assert len(analysis_samples_manifest["sha256"]) == 64
    assert analysis_samples_manifest["columns"] == list(ANALYSIS_SAMPLES_COLUMNS)
    assert analysis_samples_manifest["content"].splitlines()[0] == "\t".join(
        ANALYSIS_SAMPLES_COLUMNS
    )
    assert DEFAULT_STAGE_TARGET in analysis_samples_manifest["content"]
    assert listed_worksets.json()[0]["manifests"][0]["manifest_euid"] == "MF-1"
    assert listed_manifests.json()[0]["artifact_set_euid"] == "AS-1"
    assert (
        listed_manifests.json()[0]["input_references"][0]["reference_type"] == "artifact_set_euid"
    )
    assert clusters.status_code == 200
    assert clusters.json() == {"items": []}


def test_manifest_editor_options_api_persists_custom_values_and_scopes_tenants() -> None:
    resources = MemoryResourceStore()
    app = _create_test_app(resource_store=resources, dewey_client=DummyDeweyClient())

    with TestClient(app) as client:
        initial = client.get("/api/v1/manifest-editor/options", headers=_auth_headers())
        created = client.post(
            "/api/v1/manifest-editor/options",
            headers=_auth_headers(),
            json={"option_type": "sample_type", "value": "  nasal   swab  "},
        )
        duplicate = client.post(
            "/api/v1/manifest-editor/options",
            headers=_auth_headers(),
            json={"option_type": "sample_type", "value": "Nasal Swab"},
        )
        builtin = client.post(
            "/api/v1/manifest-editor/options",
            headers=_auth_headers(),
            json={"option_type": "library_prep", "value": "noampwgs"},
        )
        tenant_one = client.get("/api/v1/manifest-editor/options", headers=_auth_headers())
        tenant_two = client.get(
            "/api/v1/manifest-editor/options",
            headers=_auth_headers(token="tenant-two-token"),
        )

    assert initial.status_code == 200, initial.text
    assert "blood" in initial.json()["sample_types"]
    assert "noampwgs" in initial.json()["library_preps"]
    assert "NOVASEQX" in initial.json()["seq_platforms"]
    assert "CG_R1_FQ" in initial.json()["columns"]
    assert created.status_code == 200, created.text
    assert created.json()["value"] == "nasal swab"
    assert created.json()["normalized_value"] == "nasal swab"
    assert created.json()["is_builtin"] is False
    assert duplicate.status_code == 200, duplicate.text
    assert duplicate.json()["option_euid"] == created.json()["option_euid"]
    assert builtin.status_code == 200, builtin.text
    assert builtin.json()["is_builtin"] is True
    assert "nasal swab" in tenant_one.json()["sample_types"]
    assert "nasal swab" not in tenant_two.json()["sample_types"]


def test_manifest_editor_full_daylily_ec_row_keeps_all_columns_and_cg_inputs() -> None:
    resources = MemoryResourceStore()
    app = _create_test_app(resource_store=resources, dewey_client=DummyDeweyClient())
    row = {column: "" for column in ANALYSIS_SAMPLES_COLUMNS}
    row.update(
        {
            "RUN_ID": "R9",
            "SAMPLE_ID": "CGS1",
            "EXPERIMENTID": "EXP-CG",
            "SAMPLE_TYPE": "nasal swab",
            "LIB_PREP": "hybrid-capture",
            "SEQ_VENDOR": "CG",
            "SEQ_PLATFORM": "COMPLETE_GENOMICS",
            "LANE": "1",
            "SEQBC_ID": "CGBC1",
            "CG_R1_FQ": "s3://bucket/cg/CGS1_R1.fastq.gz",
            "CG_R2_FQ": "s3://bucket/cg/CGS1_R2.fastq.gz",
            "STAGE_DIRECTIVE": "stage_data",
            "STAGE_TARGET": "/data/staged_sample_data",
            "SUBSAMPLE_PCT": "na",
            "IS_POS_CTRL": "false",
            "IS_NEG_CTRL": "false",
            "N_X": "1",
            "N_Y": "1",
            "EXTERNAL_SAMPLE_ID": "EXT-CG",
        }
    )

    with TestClient(app) as client:
        workset = client.post(
            "/api/v1/worksets",
            headers=_auth_headers(),
            json={"name": "CG batch", "artifact_set_euids": []},
        )
        manifest = client.post(
            "/api/v1/manifests",
            headers=_auth_headers(),
            json={
                "workset_euid": workset.json()["workset_euid"],
                "name": "cg manifest",
                "metadata": {"editor_analysis_inputs": [row]},
            },
        )

    assert "CG_R1_FQ" in ANALYSIS_SAMPLES_SOURCE_COLUMNS
    assert "CG_R2_FQ" in ANALYSIS_SAMPLES_SOURCE_COLUMNS
    assert manifest.status_code == 201, manifest.text
    analysis_samples_manifest = manifest.json()["metadata"]["analysis_samples_manifest"]
    assert analysis_samples_manifest["columns"] == list(ANALYSIS_SAMPLES_COLUMNS)
    assert set(analysis_samples_manifest["rows"][0]) == set(ANALYSIS_SAMPLES_COLUMNS)
    assert analysis_samples_manifest["rows"][0]["CG_R1_FQ"] == row["CG_R1_FQ"]
    assert analysis_samples_manifest["rows"][0]["CG_R2_FQ"] == row["CG_R2_FQ"]
    option_values = {record.value for record in resources.manifest_options.values()}
    assert {"nasal swab", "hybrid-capture"} <= option_values


def test_analysis_command_catalog_and_preview_routes_use_user_api() -> None:
    app = _create_test_app(resource_store=MemoryResourceStore(), dewey_client=DummyDeweyClient())

    with (
        patch(
            "daylib_ursa.workset_api.command_catalog_payload",
            return_value=_command_catalog_payload(),
        ),
        patch(
            "daylib_ursa.workset_api.analysis_command_payload",
            return_value=_command_payload(),
        ),
        patch(
            "daylib_ursa.workset_api.preview_analysis_command",
            return_value=_command_preview_payload(),
        ),
        TestClient(app) as client,
    ):
        catalog = client.get("/api/v1/analysis-commands", headers=_auth_headers())
        command = client.get(
            "/api/v1/analysis-commands/illumina_snv_alignstats",
            headers=_auth_headers(),
        )
        preview = client.post(
            "/api/v1/analysis-commands/illumina_snv_alignstats/preview",
            headers=_auth_headers(),
            json={
                "optional_features": ["tiddit"],
                "region": "us-west-2",
                "cluster_name": "cluster-1",
            },
        )

    assert catalog.status_code == 200, catalog.text
    assert catalog.json()["command_catalog_version"] == 1
    assert catalog.json()["commands"][0]["command_id"] == "illumina_snv_alignstats"
    assert catalog.json()["commands"][0]["targets"] == [
        "produce_alignstats",
        "produce_snv_concordances",
    ]
    assert command.status_code == 200, command.text
    assert command.json()["command_id"] == "illumina_snv_alignstats"
    assert preview.status_code == 200, preview.text
    assert preview.json()["valid"] is True
    assert preview.json()["argv"][:2] == ["workflow", "launch"]
    assert "daylily-ec workflow launch" in preview.json()["shell_preview"]


def test_analysis_job_routes_define_launch_refresh_and_logs() -> None:
    resources = MemoryResourceStore()
    app = _create_test_app(resource_store=resources, dewey_client=DummyDeweyClient())

    with (
        patch("daylib_ursa.workset_api.analysis_command_payload", return_value=_command_payload()),
        TestClient(app) as client,
    ):
        workset = client.post(
            "/api/v1/worksets",
            headers=_auth_headers(),
            json={"name": "Tumor batch", "artifact_set_euids": ["AS-1"]},
        )
        manifest = client.post(
            "/api/v1/manifests",
            headers=_auth_headers(),
            json={
                "workset_euid": workset.json()["workset_euid"],
                "name": "analysis samples manifest",
                "metadata": {"editor_analysis_inputs": _editor_rows()},
            },
        )
        created = client.post(
            "/api/v1/analysis-jobs",
            headers=_auth_headers(),
            json={
                "workset_euid": workset.json()["workset_euid"],
                "manifest_euid": manifest.json()["manifest_euid"],
                "cluster_name": "cluster-1",
                "region": "us-west-2",
                "reference_bucket": "s3://reference-bucket",
                "analysis_command_id": "illumina_snv_alignstats",
                "optional_features": ["tiddit"],
            },
        )
        listed = client.get("/api/v1/analysis-jobs", headers=_auth_headers())
        detail = client.get(
            f"/api/v1/analysis-jobs/{created.json()['job_euid']}",
            headers=_auth_headers(),
        )
        launched = client.post(
            f"/api/v1/analysis-jobs/{created.json()['job_euid']}/launch",
            headers=_auth_headers(),
            json={},
        )
        refreshed = client.post(
            f"/api/v1/analysis-jobs/{created.json()['job_euid']}/refresh",
            headers=_auth_headers(),
        )
        logs = client.get(
            f"/api/v1/analysis-jobs/{created.json()['job_euid']}/logs",
            headers=_auth_headers(),
        )

    assert created.status_code == 201, created.text
    assert created.json()["state"] == "DEFINED"
    assert created.json()["request"]["analysis_command_id"] == "illumina_snv_alignstats"
    assert listed.status_code == 200, listed.text
    assert listed.json()[0]["job_euid"] == created.json()["job_euid"]
    assert detail.status_code == 200, detail.text
    assert detail.json()["job_euid"] == created.json()["job_euid"]
    assert launched.status_code == 202, launched.text
    assert launched.json()["state"] == "RUNNING"
    assert refreshed.status_code == 200, refreshed.text
    assert refreshed.json()["state"] == "COMPLETED"
    assert logs.status_code == 200, logs.text
    assert logs.json()["stdout"] == "ok\n"


def test_staging_job_routes_define_run_and_logs() -> None:
    resources = MemoryResourceStore()
    app = _create_test_app(resource_store=resources, dewey_client=DummyDeweyClient())

    with TestClient(app) as client:
        workset = client.post(
            "/api/v1/worksets",
            headers=_auth_headers(),
            json={"name": "Tumor batch", "artifact_set_euids": ["AS-1"]},
        )
        manifest = client.post(
            "/api/v1/manifests",
            headers=_auth_headers(),
            json={
                "workset_euid": workset.json()["workset_euid"],
                "name": "analysis samples",
                "metadata": {"editor_analysis_inputs": _editor_rows()},
            },
        )
        created = client.post(
            "/api/v1/staging-jobs",
            headers=_auth_headers(),
            json={
                "workset_euid": workset.json()["workset_euid"],
                "manifest_euid": manifest.json()["manifest_euid"],
                "cluster_name": "cluster-1",
                "region": "us-west-2",
                "reference_bucket": "s3://reference-bucket",
                "stage_target": "/fsx/staged_sample_data",
            },
        )
        listed = client.get("/api/v1/staging-jobs", headers=_auth_headers())
        detail = client.get(
            f"/api/v1/staging-jobs/{created.json()['job_euid']}",
            headers=_auth_headers(),
        )
        run = client.post(
            f"/api/v1/staging-jobs/{created.json()['job_euid']}/run",
            headers=_auth_headers(),
            json={},
        )
        logs = client.get(
            f"/api/v1/staging-jobs/{created.json()['job_euid']}/logs",
            headers=_auth_headers(),
        )

    assert created.status_code == 201, created.text
    assert created.json()["state"] == "DEFINED"
    assert created.json()["request"]["reference_bucket"] == "s3://reference-bucket"
    assert listed.status_code == 200, listed.text
    assert listed.json()[0]["job_euid"] == created.json()["job_euid"]
    assert detail.status_code == 200, detail.text
    assert detail.json()["workset_euid"] == workset.json()["workset_euid"]
    assert run.status_code == 202, run.text
    assert run.json()["state"] == "COMPLETED"
    assert run.json()["stage"]["stage_dir"] == "/fsx/staged_sample_data"
    assert logs.status_code == 200, logs.text
    assert logs.json()["stdout"] == "staged\n"


def test_staging_job_create_rejects_manifest_from_other_workset() -> None:
    resources = MemoryResourceStore()
    app = _create_test_app(resource_store=resources, dewey_client=DummyDeweyClient())

    with TestClient(app) as client:
        first_workset = client.post(
            "/api/v1/worksets",
            headers=_auth_headers(),
            json={"name": "First", "artifact_set_euids": ["AS-1"]},
        )
        second_workset = client.post(
            "/api/v1/worksets",
            headers=_auth_headers(),
            json={"name": "Second", "artifact_set_euids": ["AS-1"]},
        )
        manifest = client.post(
            "/api/v1/manifests",
            headers=_auth_headers(),
            json={
                "workset_euid": first_workset.json()["workset_euid"],
                "name": "analysis samples",
                "metadata": {"editor_analysis_inputs": _editor_rows()},
            },
        )
        created = client.post(
            "/api/v1/staging-jobs",
            headers=_auth_headers(),
            json={
                "workset_euid": second_workset.json()["workset_euid"],
                "manifest_euid": manifest.json()["manifest_euid"],
                "cluster_name": "cluster-1",
                "region": "us-west-2",
                "reference_bucket": "s3://reference-bucket",
            },
        )

    assert created.status_code == 400, created.text
    assert created.json()["detail"] == "Manifest does not belong to workset"


def test_analysis_job_accepts_completed_staging_job_without_reference_bucket() -> None:
    resources = MemoryResourceStore()
    app = _create_test_app(resource_store=resources, dewey_client=DummyDeweyClient())

    with (
        patch("daylib_ursa.workset_api.analysis_command_payload", return_value=_command_payload()),
        TestClient(app) as client,
    ):
        workset = client.post(
            "/api/v1/worksets",
            headers=_auth_headers(),
            json={"name": "Tumor batch", "artifact_set_euids": ["AS-1"]},
        )
        manifest = client.post(
            "/api/v1/manifests",
            headers=_auth_headers(),
            json={
                "workset_euid": workset.json()["workset_euid"],
                "name": "analysis samples",
                "metadata": {"editor_analysis_inputs": _editor_rows()},
            },
        )
        staging = client.post(
            "/api/v1/staging-jobs",
            headers=_auth_headers(),
            json={
                "workset_euid": workset.json()["workset_euid"],
                "manifest_euid": manifest.json()["manifest_euid"],
                "cluster_name": "cluster-1",
                "region": "us-west-2",
                "reference_bucket": "s3://reference-bucket",
                "stage_target": "/fsx/staged_sample_data",
            },
        )
        run = client.post(
            f"/api/v1/staging-jobs/{staging.json()['job_euid']}/run",
            headers=_auth_headers(),
            json={},
        )
        created = client.post(
            "/api/v1/analysis-jobs",
            headers=_auth_headers(),
            json={
                "workset_euid": workset.json()["workset_euid"],
                "manifest_euid": manifest.json()["manifest_euid"],
                "cluster_name": "cluster-1",
                "region": "us-west-2",
                "analysis_command_id": "illumina_snv_alignstats",
                "staging_job_euid": staging.json()["job_euid"],
            },
        )

    assert run.status_code == 202, run.text
    assert run.json()["state"] == "COMPLETED"
    assert created.status_code == 201, created.text
    assert created.json()["request"]["staging_job_euid"] == staging.json()["job_euid"]
    assert created.json()["request"]["reference_bucket"] is None
    assert created.json()["request"]["stage_target"] == "/fsx/staged_sample_data"


def test_analysis_job_rejects_completed_staging_job_manifest_mismatch() -> None:
    resources = MemoryResourceStore()
    app = _create_test_app(resource_store=resources, dewey_client=DummyDeweyClient())

    with (
        patch("daylib_ursa.workset_api.analysis_command_payload", return_value=_command_payload()),
        TestClient(app) as client,
    ):
        first_workset = client.post(
            "/api/v1/worksets",
            headers=_auth_headers(),
            json={"name": "First", "artifact_set_euids": ["AS-1"]},
        )
        first_manifest = client.post(
            "/api/v1/manifests",
            headers=_auth_headers(),
            json={
                "workset_euid": first_workset.json()["workset_euid"],
                "name": "first samples",
                "metadata": {"editor_analysis_inputs": _editor_rows()},
            },
        )
        second_manifest = client.post(
            "/api/v1/manifests",
            headers=_auth_headers(),
            json={
                "workset_euid": first_workset.json()["workset_euid"],
                "name": "second samples",
                "metadata": {"editor_analysis_inputs": _editor_rows()},
            },
        )
        staging = client.post(
            "/api/v1/staging-jobs",
            headers=_auth_headers(),
            json={
                "workset_euid": first_workset.json()["workset_euid"],
                "manifest_euid": first_manifest.json()["manifest_euid"],
                "cluster_name": "cluster-1",
                "region": "us-west-2",
                "reference_bucket": "s3://reference-bucket",
            },
        )
        run = client.post(
            f"/api/v1/staging-jobs/{staging.json()['job_euid']}/run",
            headers=_auth_headers(),
            json={},
        )
        created = client.post(
            "/api/v1/analysis-jobs",
            headers=_auth_headers(),
            json={
                "workset_euid": first_workset.json()["workset_euid"],
                "manifest_euid": second_manifest.json()["manifest_euid"],
                "cluster_name": "cluster-1",
                "region": "us-west-2",
                "analysis_command_id": "illumina_snv_alignstats",
                "staging_job_euid": staging.json()["job_euid"],
            },
        )

    assert run.status_code == 202, run.text
    assert run.json()["state"] == "COMPLETED"
    assert created.status_code == 400, created.text
    assert created.json()["detail"] == "Staging job does not belong to manifest"


def test_cluster_delete_requires_delete_plan_confirmation_token() -> None:
    app = _create_test_app(resource_store=MemoryResourceStore(), dewey_client=DummyDeweyClient())

    with TestClient(app) as client:
        plan = client.post(
            "/api/v1/clusters/cluster-1/delete-plan?region=us-west-2",
            headers=_auth_headers(),
        )
        missing_confirmation = client.delete(
            "/api/v1/clusters/cluster-1?region=us-west-2",
            headers=_auth_headers(),
        )
        deleted = client.delete(
            "/api/v1/clusters/cluster-1"
            "?region=us-west-2&confirmation_token=delete-token&confirm_cluster_name=cluster-1",
            headers=_auth_headers(),
        )

    assert plan.status_code == 200, plan.text
    assert plan.json()["confirmation_token"] == "delete-token"
    assert missing_confirmation.status_code == 422
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["result"]["return_code"] == 0


def test_workset_create_canonicalizes_analysis_command_metadata() -> None:
    app = _create_test_app(resource_store=MemoryResourceStore(), dewey_client=DummyDeweyClient())

    with (
        patch(
            "daylib_ursa.workset_api.analysis_command_payload",
            return_value=_command_payload(),
        ),
        TestClient(app) as client,
    ):
        created = client.post(
            "/api/v1/worksets",
            headers=_auth_headers(),
            json={
                "name": "Tumor batch",
                "artifact_set_euids": ["AS-1"],
                "metadata": {
                    "workset_type": "ruo",
                    "priority": "normal",
                    "analysis_command_id": "illumina_snv_alignstats",
                    "optional_features": ["tiddit"],
                },
            },
        )

    assert created.status_code == 201, created.text
    body = created.json()
    assert body["metadata"]["pipeline_type"] == "Illumina SNV + Alignstats"
    assert body["metadata"]["reference_genome"] == "hg38"
    assert body["metadata"]["analysis_repository"] == "daylily-omics-analysis"
    assert body["metadata"]["analysis_command_id"] == "illumina_snv_alignstats"
    assert body["metadata"]["analysis_command"]["command_id"] == "illumina_snv_alignstats"
    assert body["metadata"]["analysis_command"]["profile"]["aligners"] == ["bwa2a"]
    assert body["metadata"]["analysis_command"]["optional_features"] == ["tiddit"]
    assert body["metadata"]["analysis_command"]["created_at"].endswith("Z")


def test_workset_create_rejects_invalid_analysis_command_metadata() -> None:
    app = _create_test_app(resource_store=MemoryResourceStore(), dewey_client=DummyDeweyClient())

    with (
        patch(
            "daylib_ursa.workset_api.analysis_command_payload",
            side_effect=ValueError("Unknown analysis command: missing"),
        ),
        TestClient(app) as client,
    ):
        created = client.post(
            "/api/v1/worksets",
            headers=_auth_headers(),
            json={
                "name": "Tumor batch",
                "artifact_set_euids": ["AS-1"],
                "metadata": {
                    "analysis_command_id": "missing",
                },
            },
        )

    assert created.status_code == 400
    assert created.json()["detail"] == "Unknown analysis command: missing"


def test_manifest_rejects_artifacts_outside_resolved_dewey_set() -> None:
    app = _create_test_app(resource_store=MemoryResourceStore(), dewey_client=DummyDeweyClient())

    with TestClient(app) as client:
        workset = client.post(
            "/api/v1/worksets",
            headers=_auth_headers(),
            json={"name": "Tumor batch", "artifact_set_euids": ["AS-1"]},
        )
        manifest = client.post(
            "/api/v1/manifests",
            headers=_auth_headers(),
            json={
                "workset_euid": workset.json()["workset_euid"],
                "name": "manifest 1",
                "artifact_set_euid": "AS-1",
                "artifact_euids": ["AT-3"],
                "metadata": {"editor_analysis_inputs": _editor_rows()},
            },
        )

    assert workset.status_code == 201, workset.text
    assert manifest.status_code == 400
    assert "is not a member of artifact set AS-1" in manifest.json()["detail"]


def test_workset_rejects_unknown_dewey_artifact_set() -> None:
    app = _create_test_app(resource_store=MemoryResourceStore(), dewey_client=DummyDeweyClient())

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/worksets",
            headers=_auth_headers(),
            json={"name": "Tumor batch", "artifact_set_euids": ["AS-404"]},
        )

    assert response.status_code == 502
    assert "unknown artifact set: AS-404" in response.json()["detail"]


def test_manifest_accepts_mixed_input_references_and_imports_s3_uris() -> None:
    resources = MemoryResourceStore()
    dewey = DummyDeweyClient()
    app = _create_test_app(resource_store=resources, dewey_client=dewey)
    app.state.s3_client = DummyS3Client()

    with TestClient(app) as client:
        workset = client.post(
            "/api/v1/worksets",
            headers=_auth_headers(),
            json={"name": "Tumor batch", "artifact_set_euids": ["AS-1"]},
        )
        manifest = client.post(
            "/api/v1/manifests",
            headers=_auth_headers(),
            json={
                "workset_euid": workset.json()["workset_euid"],
                "name": "mixed manifest",
                "input_references": [
                    {"reference_type": "artifact_euid", "value": "AT-1"},
                    {"reference_type": "s3_uri", "value": "s3://bucket/sample_R1.fastq.gz"},
                ],
            },
        )

    assert manifest.status_code == 201, manifest.text
    body = manifest.json()
    assert body["artifact_set_euid"] is None
    assert body["artifact_euids"] == ["AT-1", "AT-IMPORTED-1"]
    assert body["metadata"]["input_references"][1]["value"] == "s3://bucket/sample_R1.fastq.gz"
    assert "stable_manifest" not in body["metadata"]
    assert body["metadata"]["analysis_samples_manifest"]["columns"] == list(
        ANALYSIS_SAMPLES_COLUMNS
    )
    assert DEFAULT_STAGE_TARGET in body["metadata"]["analysis_samples_manifest"]["content"]
    assert body["input_references"][1]["reference_type"] == "s3_uri"
    assert dewey.register_calls[0]["artifact_type"] == "fastq"


def test_bucket_routes_create_list_and_delete() -> None:
    app = _create_test_app(resource_store=MemoryResourceStore(), dewey_client=DummyDeweyClient())
    app.state.s3_client = DummyS3Client()

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/buckets",
            headers=_auth_headers(),
            json={"bucket_name": "omics-inputs", "display_name": "Primary Inputs"},
        )
        listed = client.get("/api/v1/buckets", headers=_auth_headers())
        deleted = client.delete(
            f"/api/v1/buckets/{created.json()['bucket_id']}",
            headers=_auth_headers(),
        )
        listed_after = client.get("/api/v1/buckets", headers=_auth_headers())

    assert created.status_code == 201, created.text
    assert created.json()["tenant_id"] == str(TENANT_ID)
    assert listed.json()[0]["bucket_name"] == "omics-inputs"
    assert deleted.status_code == 200
    assert deleted.json()["state"] == "DELETED"
    assert listed_after.json() == []


def test_detail_bucket_and_artifact_routes_cover_user_surface() -> None:
    resources = MemoryResourceStore()
    dewey = DummyDeweyClient()
    app = _create_test_app(resource_store=resources, dewey_client=dewey)
    app.state.s3_client = DummyS3Client()

    with TestClient(app) as client:
        workset = client.post(
            "/api/v1/worksets",
            headers=_auth_headers(),
            json={"name": "Tumor batch", "artifact_set_euids": ["AS-1"]},
        )
        manifest = client.post(
            "/api/v1/manifests",
            headers=_auth_headers(),
            json={
                "workset_euid": workset.json()["workset_euid"],
                "name": "downloadable manifest",
                "artifact_set_euid": "AS-1",
                "artifact_euids": ["AT-1", "AT-2"],
                "metadata": {"editor_analysis_inputs": _editor_rows()},
            },
        )
        validation = client.post(
            "/api/v1/buckets/validate",
            headers=_auth_headers(),
            json={"bucket_name": "omics-inputs", "display_name": "Primary Inputs"},
        )
        created_bucket = client.post(
            "/api/v1/buckets",
            headers=_auth_headers(),
            json={"bucket_name": "omics-inputs", "display_name": "Primary Inputs"},
        )
        workset_detail = client.get(
            f"/api/v1/worksets/{workset.json()['workset_euid']}",
            headers=_auth_headers(),
        )
        manifest_detail = client.get(
            f"/api/v1/manifests/{manifest.json()['manifest_euid']}",
            headers=_auth_headers(),
        )
        manifest_download = client.get(
            f"/api/v1/manifests/{manifest.json()['manifest_euid']}/download",
            headers=_auth_headers(),
        )
        imported = client.post(
            "/api/v1/artifacts/import",
            headers=_auth_headers(),
            json={
                "artifact_type": "fastq",
                "storage_uri": "s3://dewey-imports/sample_R1.fastq.gz",
                "metadata": {"source": "lab"},
            },
        )
        resolved = client.post(
            "/api/v1/artifacts/resolve",
            headers=_auth_headers(),
            json={"artifact_euid": "AT-1"},
        )
        bucket_id = created_bucket.json()["bucket_id"]
        bucket_detail = client.get(f"/api/v1/buckets/{bucket_id}", headers=_auth_headers())
        bucket_update = client.patch(
            f"/api/v1/buckets/{bucket_id}",
            headers=_auth_headers(),
            json={"display_name": "Renamed Inputs", "prefix_restriction": "incoming/"},
        )
        bucket_revalidate = client.post(
            f"/api/v1/buckets/{bucket_id}/revalidate",
            headers=_auth_headers(),
        )
        object_list = client.get(
            f"/api/v1/buckets/{bucket_id}/objects?prefix=incoming/",
            headers=_auth_headers(),
        )
        folder = client.post(
            f"/api/v1/buckets/{bucket_id}/folders?prefix=incoming/",
            headers=_auth_headers(),
            json={"folder_name": "nested"},
        )
        upload = client.post(
            f"/api/v1/buckets/{bucket_id}/upload",
            headers=_auth_headers(),
            data={"prefix": "incoming/"},
            files={"file": ("notes.txt", b"alpha\nbeta\n", "text/plain")},
        )
        download_url = client.get(
            f"/api/v1/buckets/{bucket_id}/objects/download-url?key=incoming/notes.txt",
            headers=_auth_headers(),
        )
        preview = client.get(
            f"/api/v1/buckets/{bucket_id}/objects/preview?key=incoming/notes.txt&lines=2",
            headers=_auth_headers(),
        )
        delete_object = client.delete(
            f"/api/v1/buckets/{bucket_id}/objects?key=incoming/notes.txt",
            headers=_auth_headers(),
        )

    assert workset_detail.status_code == 200, workset_detail.text
    assert workset_detail.json()["workset_euid"] == workset.json()["workset_euid"]
    assert manifest_detail.status_code == 200, manifest_detail.text
    assert manifest_detail.json()["manifest_euid"] == manifest.json()["manifest_euid"]
    assert manifest_download.status_code == 200, manifest_download.text
    assert manifest_download.text.startswith("RUN_ID\tSAMPLE_ID\tEXPERIMENTID")
    assert manifest_download.headers["content-disposition"].endswith('analysis_samples.tsv"')
    assert validation.status_code == 200, validation.text
    assert validation.json()["bucket_name"] == "omics-inputs"
    assert imported.status_code == 201, imported.text
    assert imported.json()["artifact_euid"] == "AT-IMPORTED-1"
    assert dewey.register_calls[0]["metadata"]["tenant_id"] == str(TENANT_ID)
    assert resolved.status_code == 200, resolved.text
    assert resolved.json()["storage_uri"] == "s3://dewey/AT-1.bin"
    assert bucket_detail.status_code == 200, bucket_detail.text
    assert bucket_detail.json()["bucket_name"] == "omics-inputs"
    assert bucket_update.status_code == 200, bucket_update.text
    assert bucket_update.json()["display_name"] == "Renamed Inputs"
    assert bucket_update.json()["prefix_restriction"] == "incoming/"
    assert bucket_revalidate.status_code == 200, bucket_revalidate.text
    assert bucket_revalidate.json()["is_validated"] is True
    assert object_list.status_code == 200, object_list.text
    assert object_list.json()["prefix"] == "incoming/"
    assert folder.status_code == 200, folder.text
    assert folder.json()["folder"] == "incoming/nested/"
    assert upload.status_code == 200, upload.text
    assert upload.json()["key"] == "incoming/notes.txt"
    assert download_url.status_code == 200, download_url.text
    assert download_url.json()["url"] == "https://example.test/download"
    assert preview.status_code == 200, preview.text
    assert preview.json()["lines"] == ["alpha", "beta"]
    assert delete_object.status_code == 200, delete_object.text
    assert delete_object.json()["deleted"] == "incoming/notes.txt"
