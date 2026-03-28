from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid

try:
    from daylib_ursa.tapdb_graph import TapDBBackend, from_json_addl, utc_now_iso
except ImportError:  # pragma: no cover - import-time compatibility for reduced test envs
    TapDBBackend = Any  # type: ignore[assignment]

    def from_json_addl(instance) -> dict[str, Any]:
        return dict(getattr(instance, "json_addl", {}) or {})

    def utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

WORKSET_TEMPLATE = "workflow/workset/gui-ready/1.0/"
MANIFEST_TEMPLATE = "data/manifest/dewey-bound/1.0/"
DEWEY_IMPORT_TEMPLATE = "action/artifact/dewey-import/1.0/"
CLIENT_REGISTRATION_TEMPLATE = "integration/auth/client-registration/1.0/"
CLUSTER_JOB_TEMPLATE = "workflow/cluster/ephemeral-job/1.0/"
CLUSTER_JOB_REVISION_TEMPLATE = "action/cluster/ephemeral-job-revision/1.0/"
CLUSTER_JOB_EVENT_TEMPLATE = "action/cluster/ephemeral-job-event/1.0/"
LINKED_BUCKET_TEMPLATE = "integration/storage/linked-bucket/1.0/"


@dataclass(frozen=True)
class ManifestRecord:
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


@dataclass(frozen=True)
class WorksetRecord:
    workset_euid: str
    name: str
    tenant_id: uuid.UUID
    owner_user_id: str
    state: str
    artifact_set_euids: list[str]
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
    manifests: list[ManifestRecord]
    analysis_euids: list[str]


@dataclass(frozen=True)
class DeweyImportRecord:
    import_euid: str
    artifact_euid: str
    artifact_type: str
    storage_uri: str
    actor_user_id: str
    created_at: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ClientRegistrationRecord:
    client_registration_euid: str
    client_name: str
    owner_user_id: str
    sponsor_user_id: str
    scopes: list[str]
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
    state: str


@dataclass(frozen=True)
class ClusterJobEventRecord:
    event_euid: str
    job_euid: str
    event_type: str
    status: str
    summary: str
    details: dict[str, Any]
    created_by: str | None
    created_at: str


@dataclass(frozen=True)
class ClusterJobRecord:
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
    events: list[ClusterJobEventRecord] = field(default_factory=list)


@dataclass(frozen=True)
class LinkedBucketRecord:
    bucket_id: str
    bucket_name: str
    tenant_id: uuid.UUID
    owner_user_id: str
    display_name: str | None
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
    state: str
    bucket_type: str = "secondary"
    description: str | None = None
    prefix_restriction: str | None = None
    read_only: bool = False
    region: str | None = None
    is_validated: bool = False
    can_read: bool = False
    can_write: bool = False
    can_list: bool = False
    remediation_steps: list[str] = field(default_factory=list)


class ResourceStore:
    def __init__(self, backend: TapDBBackend | None = None) -> None:
        self.backend = backend or TapDBBackend(app_username="ursa")

    def bootstrap(self) -> None:
        with self.backend.session_scope(commit=True) as session:
            self.backend.ensure_templates(session)

    @staticmethod
    def _parse_tenant_uuid(value: Any) -> uuid.UUID:
        return uuid.UUID(str(value or "").strip())

    @staticmethod
    def _manifest_from_instance(instance, *, workset_euid: str) -> ManifestRecord:
        payload = from_json_addl(instance)
        return ManifestRecord(
            manifest_euid=str(instance.euid),
            name=str(instance.name or payload.get("name") or ""),
            workset_euid=workset_euid,
            tenant_id=ResourceStore._parse_tenant_uuid(payload.get("tenant_id")),
            owner_user_id=str(payload.get("owner_user_id") or ""),
            artifact_set_euid=str(payload.get("artifact_set_euid") or "").strip() or None,
            artifact_euids=[str(item) for item in list(payload.get("artifact_euids") or [])],
            input_references=[
                dict(item)
                for item in list(payload.get("input_references") or [])
                if isinstance(item, dict)
            ],
            metadata=dict(payload.get("metadata") or {}),
            created_at=str(payload.get("created_at") or utc_now_iso()),
            updated_at=str(payload.get("updated_at") or payload.get("created_at") or utc_now_iso()),
            state=str(payload.get("state") or instance.bstatus),
        )

    def _workset_from_instance(self, session, instance) -> WorksetRecord:
        payload = from_json_addl(instance)
        manifest_children = self.backend.list_children(
            session,
            parent=instance,
            relationship_type="workset_manifest",
        )
        manifests = [
            self._manifest_from_instance(child, workset_euid=str(instance.euid))
            for child in manifest_children
        ]
        manifests.sort(key=lambda row: row.created_at, reverse=True)
        analysis_euids = [
            str(child.euid)
            for child in self.backend.list_children(
                session,
                parent=instance,
                relationship_type="workset_analysis",
            )
        ]
        return WorksetRecord(
            workset_euid=str(instance.euid),
            name=str(instance.name or payload.get("name") or ""),
            tenant_id=ResourceStore._parse_tenant_uuid(payload.get("tenant_id")),
            owner_user_id=str(payload.get("owner_user_id") or ""),
            state=str(payload.get("state") or instance.bstatus),
            artifact_set_euids=[str(item) for item in list(payload.get("artifact_set_euids") or [])],
            metadata=dict(payload.get("metadata") or {}),
            created_at=str(payload.get("created_at") or utc_now_iso()),
            updated_at=str(payload.get("updated_at") or payload.get("created_at") or utc_now_iso()),
            manifests=manifests,
            analysis_euids=analysis_euids,
        )

    @staticmethod
    def _cluster_job_revision_sort_key(instance: Any) -> int:
        payload = from_json_addl(instance)
        return int(payload.get("revision_no") or 0)

    def _latest_cluster_job_revision(self, session, job_instance) -> Any | None:
        revisions = self.backend.list_children(
            session,
            parent=job_instance,
            relationship_type="revision",
        )
        if not revisions:
            return None
        revisions.sort(key=self._cluster_job_revision_sort_key, reverse=True)
        return revisions[0]

    @staticmethod
    def _cluster_job_event_from_instance(instance) -> ClusterJobEventRecord:
        payload = from_json_addl(instance)
        return ClusterJobEventRecord(
            event_euid=str(instance.euid),
            job_euid=str(payload.get("job_euid") or ""),
            event_type=str(payload.get("event_type") or ""),
            status=str(payload.get("status") or instance.bstatus),
            summary=str(payload.get("summary") or ""),
            details=dict(payload.get("details") or {}),
            created_by=str(payload.get("created_by") or "").strip() or None,
            created_at=str(payload.get("created_at") or utc_now_iso()),
        )

    def _cluster_job_from_instance(self, session, instance) -> ClusterJobRecord:
        payload = from_json_addl(instance)
        latest_revision = self._latest_cluster_job_revision(session, instance)
        revision_payload = from_json_addl(latest_revision) if latest_revision is not None else {}
        events = [
            self._cluster_job_event_from_instance(child)
            for child in self.backend.list_children(
                session,
                parent=instance,
                relationship_type="event",
            )
        ]
        events.sort(key=lambda item: item.created_at, reverse=True)
        state = str(revision_payload.get("state") or payload.get("state") or instance.bstatus)
        return_code_raw = revision_payload.get("return_code")
        try:
            return_code = int(return_code_raw) if return_code_raw is not None else None
        except (TypeError, ValueError):
            return_code = None
        return ClusterJobRecord(
            job_euid=str(instance.euid),
            job_name=str(instance.name or payload.get("job_name") or payload.get("cluster_name") or ""),
            cluster_name=str(payload.get("cluster_name") or ""),
            region=str(payload.get("region") or ""),
            region_az=str(payload.get("region_az") or ""),
            tenant_id=ResourceStore._parse_tenant_uuid(payload.get("tenant_id")),
            owner_user_id=str(payload.get("owner_user_id") or ""),
            sponsor_user_id=str(payload.get("sponsor_user_id") or ""),
            state=state,
            created_at=str(payload.get("created_at") or utc_now_iso()),
            updated_at=str(
                revision_payload.get("created_at")
                or payload.get("updated_at")
                or payload.get("created_at")
                or utc_now_iso()
            ),
            started_at=str(revision_payload.get("started_at") or "").strip() or None,
            completed_at=str(revision_payload.get("completed_at") or "").strip() or None,
            return_code=return_code,
            error=str(revision_payload.get("error") or "").strip() or None,
            output_summary=str(revision_payload.get("output_summary") or "").strip() or None,
            request=dict(payload.get("request") or {}),
            cluster=dict(revision_payload.get("cluster") or {}),
            events=events,
        )

    def list_worksets(self, *, tenant_id: uuid.UUID, limit: int = 100) -> list[WorksetRecord]:
        with self.backend.session_scope(commit=False) as session:
            rows = self.backend.list_instances_by_property(
                session,
                template_code=WORKSET_TEMPLATE,
                key="tenant_id",
                value=str(tenant_id),
                limit=limit,
            )
            return [self._workset_from_instance(session, item) for item in rows]

    def get_workset(self, workset_euid: str) -> WorksetRecord | None:
        with self.backend.session_scope(commit=False) as session:
            workset = self.backend.find_instance_by_euid(
                session,
                template_code=WORKSET_TEMPLATE,
                value=workset_euid,
            )
            if workset is None:
                return None
            return self._workset_from_instance(session, workset)

    def create_workset(
        self,
        *,
        name: str,
        tenant_id: uuid.UUID,
        owner_user_id: str,
        artifact_set_euids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorksetRecord:
        now = utc_now_iso()
        with self.backend.session_scope(commit=True) as session:
            workset = self.backend.create_instance(
                session,
                WORKSET_TEMPLATE,
                name,
                json_addl={
                    "tenant_id": str(tenant_id),
                    "owner_user_id": owner_user_id,
                    "artifact_set_euids": list(artifact_set_euids or []),
                    "metadata": dict(metadata or {}),
                    "created_at": now,
                    "updated_at": now,
                    "state": "ACTIVE",
                },
                bstatus="ACTIVE",
                tenant_id=tenant_id,
            )
            return self._workset_from_instance(session, workset)

    def create_manifest(
        self,
        *,
        workset_euid: str,
        name: str,
        artifact_set_euid: str | None,
        artifact_euids: list[str] | None = None,
        input_references: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ManifestRecord:
        now = utc_now_iso()
        with self.backend.session_scope(commit=True) as session:
            workset = self.backend.find_instance_by_euid(
                session,
                template_code=WORKSET_TEMPLATE,
                value=workset_euid,
                for_update=True,
            )
            if workset is None:
                raise KeyError(f"workset not found: {workset_euid}")
            workset_payload = from_json_addl(workset)
            manifest = self.backend.create_instance(
                session,
                MANIFEST_TEMPLATE,
                name,
                json_addl={
                    "workset_euid": workset_euid,
                    "tenant_id": str(workset_payload.get("tenant_id") or ""),
                    "owner_user_id": str(workset_payload.get("owner_user_id") or ""),
                    "artifact_set_euid": artifact_set_euid,
                    "artifact_euids": list(artifact_euids or []),
                    "input_references": [dict(item) for item in list(input_references or [])],
                    "metadata": dict(metadata or {}),
                    "created_at": now,
                    "updated_at": now,
                    "state": "ACTIVE",
                },
                bstatus="ACTIVE",
                tenant_id=self._parse_tenant_uuid(workset_payload.get("tenant_id")),
            )
            self.backend.create_lineage(
                session,
                parent=workset,
                child=manifest,
                relationship_type="workset_manifest",
            )
            payload = dict(workset.json_addl or {})
            payload["updated_at"] = now
            workset.json_addl = payload
            return self._manifest_from_instance(manifest, workset_euid=workset_euid)

    def list_manifests(self, *, tenant_id: uuid.UUID, limit: int = 200) -> list[ManifestRecord]:
        with self.backend.session_scope(commit=False) as session:
            rows = self.backend.list_instances_by_property(
                session,
                template_code=MANIFEST_TEMPLATE,
                key="tenant_id",
                value=str(tenant_id),
                limit=limit,
            )
            return [
                self._manifest_from_instance(
                    item,
                    workset_euid=str(from_json_addl(item).get("workset_euid") or ""),
                )
                for item in rows
            ]

    def get_manifest(self, manifest_euid: str) -> ManifestRecord | None:
        with self.backend.session_scope(commit=False) as session:
            manifest = self.backend.find_instance_by_euid(
                session,
                template_code=MANIFEST_TEMPLATE,
                value=manifest_euid,
            )
            if manifest is None:
                return None
            return self._manifest_from_instance(
                manifest,
                workset_euid=str(from_json_addl(manifest).get("workset_euid") or ""),
            )

    @staticmethod
    def _linked_bucket_from_instance(instance) -> LinkedBucketRecord:
        payload = from_json_addl(instance)
        return LinkedBucketRecord(
            bucket_id=str(instance.euid),
            bucket_name=str(payload.get("bucket_name") or instance.name or ""),
            tenant_id=ResourceStore._parse_tenant_uuid(payload.get("tenant_id")),
            owner_user_id=str(payload.get("owner_user_id") or ""),
            display_name=str(payload.get("display_name") or "").strip() or None,
            metadata=dict(payload.get("metadata") or {}),
            created_at=str(payload.get("created_at") or utc_now_iso()),
            updated_at=str(payload.get("updated_at") or payload.get("created_at") or utc_now_iso()),
            state=str(payload.get("state") or instance.bstatus),
            bucket_type=str(payload.get("bucket_type") or "secondary"),
            description=str(payload.get("description") or "").strip() or None,
            prefix_restriction=str(payload.get("prefix_restriction") or "").strip() or None,
            read_only=bool(payload.get("read_only")),
            region=str(payload.get("region") or "").strip() or None,
            is_validated=bool(payload.get("is_validated")),
            can_read=bool(payload.get("can_read")),
            can_write=bool(payload.get("can_write")),
            can_list=bool(payload.get("can_list")),
            remediation_steps=[
                str(item)
                for item in list(payload.get("remediation_steps") or [])
                if str(item or "").strip()
            ],
        )

    def list_linked_buckets(self, *, tenant_id: uuid.UUID, limit: int = 200) -> list[LinkedBucketRecord]:
        with self.backend.session_scope(commit=False) as session:
            rows = self.backend.list_instances_by_property(
                session,
                template_code=LINKED_BUCKET_TEMPLATE,
                key="tenant_id",
                value=str(tenant_id),
                limit=limit,
            )
            return [
                self._linked_bucket_from_instance(item)
                for item in rows
                if str(from_json_addl(item).get("state") or item.bstatus) != "DELETED"
            ]

    def get_linked_bucket(self, bucket_id: str) -> LinkedBucketRecord | None:
        with self.backend.session_scope(commit=False) as session:
            bucket = self.backend.find_instance_by_euid(
                session,
                template_code=LINKED_BUCKET_TEMPLATE,
                value=bucket_id,
            )
            if bucket is None:
                return None
            return self._linked_bucket_from_instance(bucket)

    def create_linked_bucket(
        self,
        *,
        bucket_name: str,
        tenant_id: uuid.UUID,
        owner_user_id: str,
        display_name: str | None = None,
        bucket_type: str = "secondary",
        description: str | None = None,
        prefix_restriction: str | None = None,
        read_only: bool = False,
        region: str | None = None,
        is_validated: bool = False,
        can_read: bool = False,
        can_write: bool = False,
        can_list: bool = False,
        remediation_steps: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LinkedBucketRecord:
        now = utc_now_iso()
        normalized_bucket = str(bucket_name or "").strip()
        if not normalized_bucket:
            raise ValueError("bucket_name is required")
        with self.backend.session_scope(commit=True) as session:
            existing = self.backend.find_instance_by_external_id(
                session,
                template_code=LINKED_BUCKET_TEMPLATE,
                key="bucket_name",
                value=normalized_bucket,
            )
            if existing is not None:
                payload = from_json_addl(existing)
                if (
                    str(payload.get("tenant_id") or "") == str(tenant_id)
                    and str(payload.get("state") or existing.bstatus) != "DELETED"
                ):
                    raise ValueError(f"Bucket already linked: {normalized_bucket}")
            bucket = self.backend.create_instance(
                session,
                LINKED_BUCKET_TEMPLATE,
                normalized_bucket,
                json_addl={
                    "bucket_name": normalized_bucket,
                    "tenant_id": str(tenant_id),
                    "owner_user_id": owner_user_id,
                    "display_name": str(display_name or "").strip() or None,
                    "bucket_type": str(bucket_type or "secondary").strip() or "secondary",
                    "description": str(description or "").strip() or None,
                    "prefix_restriction": str(prefix_restriction or "").strip() or None,
                    "read_only": bool(read_only),
                    "region": str(region or "").strip() or None,
                    "is_validated": bool(is_validated),
                    "can_read": bool(can_read),
                    "can_write": bool(can_write),
                    "can_list": bool(can_list),
                    "remediation_steps": [
                        str(item)
                        for item in list(remediation_steps or [])
                        if str(item or "").strip()
                    ],
                    "metadata": dict(metadata or {}),
                    "created_at": now,
                    "updated_at": now,
                    "state": "ACTIVE",
                },
                bstatus="ACTIVE",
                tenant_id=tenant_id,
            )
            return self._linked_bucket_from_instance(bucket)

    def update_linked_bucket(
        self,
        *,
        bucket_id: str,
        display_name: str | None = None,
        bucket_type: str | None = None,
        description: str | None = None,
        prefix_restriction: str | None = None,
        read_only: bool | None = None,
        region: str | None = None,
        is_validated: bool | None = None,
        can_read: bool | None = None,
        can_write: bool | None = None,
        can_list: bool | None = None,
        remediation_steps: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LinkedBucketRecord | None:
        with self.backend.session_scope(commit=True) as session:
            bucket = self.backend.find_instance_by_euid(
                session,
                template_code=LINKED_BUCKET_TEMPLATE,
                value=bucket_id,
                for_update=True,
            )
            if bucket is None:
                return None
            payload = from_json_addl(bucket)
            updates: dict[str, Any] = {
                "updated_at": utc_now_iso(),
            }
            if display_name is not None:
                updates["display_name"] = str(display_name or "").strip() or None
            if bucket_type is not None:
                updates["bucket_type"] = str(bucket_type or "").strip() or "secondary"
            if description is not None:
                updates["description"] = str(description or "").strip() or None
            if prefix_restriction is not None:
                updates["prefix_restriction"] = str(prefix_restriction or "").strip() or None
            if read_only is not None:
                updates["read_only"] = bool(read_only)
            if region is not None:
                updates["region"] = str(region or "").strip() or None
            if is_validated is not None:
                updates["is_validated"] = bool(is_validated)
            if can_read is not None:
                updates["can_read"] = bool(can_read)
            if can_write is not None:
                updates["can_write"] = bool(can_write)
            if can_list is not None:
                updates["can_list"] = bool(can_list)
            if remediation_steps is not None:
                updates["remediation_steps"] = [
                    str(item)
                    for item in list(remediation_steps or [])
                    if str(item or "").strip()
                ]
            if metadata is not None:
                merged_metadata = dict(payload.get("metadata") or {})
                merged_metadata.update(dict(metadata))
                updates["metadata"] = merged_metadata
            self.backend.update_instance_json(session, bucket, updates)
            return self._linked_bucket_from_instance(bucket)

    def delete_linked_bucket(self, *, bucket_id: str) -> LinkedBucketRecord | None:
        with self.backend.session_scope(commit=True) as session:
            bucket = self.backend.find_instance_by_euid(
                session,
                template_code=LINKED_BUCKET_TEMPLATE,
                value=bucket_id,
                for_update=True,
            )
            if bucket is None:
                return None
            self.backend.update_instance_json(
                session,
                bucket,
                {
                    "updated_at": utc_now_iso(),
                    "state": "DELETED",
                },
            )
            bucket.bstatus = "DELETED"
            return self._linked_bucket_from_instance(bucket)

    def link_analysis(self, *, workset_euid: str, analysis_euid: str) -> None:
        with self.backend.session_scope(commit=True) as session:
            workset = self.backend.find_instance_by_euid(
                session,
                template_code=WORKSET_TEMPLATE,
                value=workset_euid,
                for_update=True,
            )
            if workset is None:
                raise KeyError(f"workset not found: {workset_euid}")
            analysis = self.backend.find_instance_by_euid(
                session,
                template_code="workflow/analysis/run-linked/1.0/",
                value=analysis_euid,
                for_update=True,
            )
            if analysis is None:
                raise KeyError(f"analysis not found: {analysis_euid}")
            self.backend.create_lineage(
                session,
                parent=workset,
                child=analysis,
                relationship_type="workset_analysis",
            )
            payload = dict(workset.json_addl or {})
            payload["updated_at"] = utc_now_iso()
            workset.json_addl = payload

    def record_dewey_import(
        self,
        *,
        artifact_euid: str,
        artifact_type: str,
        storage_uri: str,
        actor_user_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> DeweyImportRecord:
        now = utc_now_iso()
        with self.backend.session_scope(commit=True) as session:
            event = self.backend.create_instance(
                session,
                DEWEY_IMPORT_TEMPLATE,
                f"dewey-import:{artifact_euid}",
                json_addl={
                    "artifact_euid": artifact_euid,
                    "artifact_type": artifact_type,
                    "storage_uri": storage_uri,
                    "actor_user_id": actor_user_id,
                    "metadata": dict(metadata or {}),
                    "created_at": now,
                    "updated_at": now,
                    "state": "IMPORTED",
                },
                bstatus="IMPORTED",
            )
            payload = from_json_addl(event)
            return DeweyImportRecord(
                import_euid=str(event.euid),
                artifact_euid=str(payload.get("artifact_euid") or artifact_euid),
                artifact_type=str(payload.get("artifact_type") or artifact_type),
                storage_uri=str(payload.get("storage_uri") or storage_uri),
                actor_user_id=str(payload.get("actor_user_id") or actor_user_id),
                created_at=str(payload.get("created_at") or now),
                metadata=dict(payload.get("metadata") or {}),
            )

    def create_client_registration(
        self,
        *,
        client_name: str,
        owner_user_id: str,
        sponsor_user_id: str,
        scopes: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ClientRegistrationRecord:
        now = utc_now_iso()
        with self.backend.session_scope(commit=True) as session:
            client = self.backend.create_instance(
                session,
                CLIENT_REGISTRATION_TEMPLATE,
                client_name,
                json_addl={
                    "owner_user_id": owner_user_id,
                    "sponsor_user_id": sponsor_user_id,
                    "scopes": list(scopes or []),
                    "metadata": dict(metadata or {}),
                    "created_at": now,
                    "updated_at": now,
                    "state": "ACTIVE",
                },
                bstatus="ACTIVE",
            )
            return self._client_from_instance(client)

    @staticmethod
    def _client_from_instance(instance) -> ClientRegistrationRecord:
        payload = from_json_addl(instance)
        return ClientRegistrationRecord(
            client_registration_euid=str(instance.euid),
            client_name=str(instance.name or payload.get("name") or ""),
            owner_user_id=str(payload.get("owner_user_id") or ""),
            sponsor_user_id=str(payload.get("sponsor_user_id") or ""),
            scopes=[str(item) for item in list(payload.get("scopes") or [])],
            metadata=dict(payload.get("metadata") or {}),
            created_at=str(payload.get("created_at") or utc_now_iso()),
            updated_at=str(payload.get("updated_at") or payload.get("created_at") or utc_now_iso()),
            state=str(payload.get("state") or instance.bstatus),
        )

    def get_client_registration(self, client_registration_euid: str) -> ClientRegistrationRecord | None:
        with self.backend.session_scope(commit=False) as session:
            client = self.backend.find_instance_by_euid(
                session,
                template_code=CLIENT_REGISTRATION_TEMPLATE,
                value=client_registration_euid,
            )
            if client is None:
                return None
            return self._client_from_instance(client)

    def list_client_registrations(
        self,
        *,
        owner_user_id: str | None = None,
        limit: int = 200,
    ) -> list[ClientRegistrationRecord]:
        with self.backend.session_scope(commit=False) as session:
            if owner_user_id:
                rows = self.backend.list_instances_by_property(
                    session,
                    template_code=CLIENT_REGISTRATION_TEMPLATE,
                    key="owner_user_id",
                    value=owner_user_id,
                    limit=limit,
                )
            else:
                rows = self.backend.list_instances_by_template(
                    session,
                    template_code=CLIENT_REGISTRATION_TEMPLATE,
                    limit=limit,
                )
            return [self._client_from_instance(item) for item in rows]

    def create_cluster_job(
        self,
        *,
        cluster_name: str,
        region: str,
        region_az: str,
        tenant_id: uuid.UUID,
        owner_user_id: str,
        sponsor_user_id: str,
        request: dict[str, Any] | None = None,
    ) -> ClusterJobRecord:
        now = utc_now_iso()
        with self.backend.session_scope(commit=True) as session:
            job = self.backend.create_instance(
                session,
                CLUSTER_JOB_TEMPLATE,
                cluster_name,
                json_addl={
                    "cluster_name": cluster_name,
                    "region": region,
                    "region_az": region_az,
                    "tenant_id": str(tenant_id),
                    "owner_user_id": owner_user_id,
                    "sponsor_user_id": sponsor_user_id,
                    "request": dict(request or {}),
                    "created_at": now,
                    "updated_at": now,
                    "state": "QUEUED",
                },
                bstatus="QUEUED",
                tenant_id=tenant_id,
            )
            revision = self.backend.create_instance(
                session,
                CLUSTER_JOB_REVISION_TEMPLATE,
                f"revision:{job.euid}:1",
                json_addl={
                    "job_euid": str(job.euid),
                    "revision_no": 1,
                    "state": "QUEUED",
                    "started_at": None,
                    "completed_at": None,
                    "return_code": None,
                    "error": None,
                    "output_summary": None,
                    "cluster": {},
                    "created_by": sponsor_user_id,
                    "created_at": now,
                },
                bstatus="QUEUED",
            )
            self.backend.create_lineage(
                session,
                parent=job,
                child=revision,
                relationship_type="revision",
            )
            return self._cluster_job_from_instance(session, job)

    def update_cluster_job_status(
        self,
        *,
        job_euid: str,
        state: str,
        created_by: str,
        started_at: str | None = None,
        completed_at: str | None = None,
        return_code: int | None = None,
        error: str | None = None,
        output_summary: str | None = None,
        cluster: dict[str, Any] | None = None,
    ) -> ClusterJobRecord:
        with self.backend.session_scope(commit=True) as session:
            job = self.backend.find_instance_by_euid(
                session,
                template_code=CLUSTER_JOB_TEMPLATE,
                value=job_euid,
                for_update=True,
            )
            if job is None:
                raise KeyError(f"cluster job not found: {job_euid}")
            latest_revision = self._latest_cluster_job_revision(session, job)
            latest_payload = from_json_addl(latest_revision) if latest_revision is not None else {}
            revision_no = int(latest_payload.get("revision_no") or 0) + 1
            created_at = utc_now_iso()
            revision = self.backend.create_instance(
                session,
                CLUSTER_JOB_REVISION_TEMPLATE,
                f"revision:{job.euid}:{revision_no}",
                json_addl={
                    "job_euid": str(job.euid),
                    "revision_no": revision_no,
                    "state": state,
                    "started_at": started_at if started_at is not None else latest_payload.get("started_at"),
                    "completed_at": completed_at if completed_at is not None else latest_payload.get("completed_at"),
                    "return_code": return_code if return_code is not None else latest_payload.get("return_code"),
                    "error": error if error is not None else latest_payload.get("error"),
                    "output_summary": (
                        output_summary if output_summary is not None else latest_payload.get("output_summary")
                    ),
                    "cluster": dict(cluster or latest_payload.get("cluster") or {}),
                    "created_by": created_by,
                    "created_at": created_at,
                },
                bstatus=state,
            )
            self.backend.create_lineage(
                session,
                parent=job,
                child=revision,
                relationship_type="revision",
            )
            self.backend.update_instance_json(
                session,
                job,
                {
                    "updated_at": created_at,
                    "state": state,
                },
            )
            job.bstatus = state
            return self._cluster_job_from_instance(session, job)

    def add_cluster_job_event(
        self,
        *,
        job_euid: str,
        event_type: str,
        status: str,
        summary: str,
        details: dict[str, Any] | None = None,
        created_by: str | None = None,
    ) -> ClusterJobEventRecord:
        created_at = utc_now_iso()
        with self.backend.session_scope(commit=True) as session:
            job = self.backend.find_instance_by_euid(
                session,
                template_code=CLUSTER_JOB_TEMPLATE,
                value=job_euid,
                for_update=True,
            )
            if job is None:
                raise KeyError(f"cluster job not found: {job_euid}")
            event = self.backend.create_instance(
                session,
                CLUSTER_JOB_EVENT_TEMPLATE,
                f"{event_type}:{job.euid}:{created_at}",
                json_addl={
                    "job_euid": str(job.euid),
                    "event_type": event_type,
                    "status": status,
                    "summary": summary,
                    "details": dict(details or {}),
                    "created_by": created_by,
                    "created_at": created_at,
                },
                bstatus=status,
            )
            self.backend.create_lineage(
                session,
                parent=job,
                child=event,
                relationship_type="event",
            )
            self.backend.update_instance_json(
                session,
                job,
                {
                    "updated_at": created_at,
                },
            )
            return self._cluster_job_event_from_instance(event)

    def get_cluster_job(self, job_euid: str) -> ClusterJobRecord | None:
        with self.backend.session_scope(commit=False) as session:
            job = self.backend.find_instance_by_euid(
                session,
                template_code=CLUSTER_JOB_TEMPLATE,
                value=job_euid,
            )
            if job is None:
                return None
            return self._cluster_job_from_instance(session, job)

    def list_cluster_jobs(
        self,
        *,
        tenant_id: uuid.UUID | None = None,
        limit: int = 200,
    ) -> list[ClusterJobRecord]:
        with self.backend.session_scope(commit=False) as session:
            if tenant_id:
                jobs = self.backend.list_instances_by_property(
                    session,
                    template_code=CLUSTER_JOB_TEMPLATE,
                    key="tenant_id",
                    value=str(tenant_id),
                    limit=limit,
                )
            else:
                jobs = self.backend.list_instances_by_template(
                    session,
                    template_code=CLUSTER_JOB_TEMPLATE,
                    limit=limit,
                )
            return [self._cluster_job_from_instance(session, item) for item in jobs]
