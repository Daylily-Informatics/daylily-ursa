from __future__ import annotations

import io

from fastapi.testclient import TestClient

from daylib_ursa.auth import ActorContext
from daylib_ursa.config import Settings
from daylib_ursa.dewey_client import DeweyClientError
from daylib_ursa.resource_store import LinkedBucketRecord, ManifestRecord, WorksetRecord
from daylib_ursa.workset_api import create_app


class DummyIdentityClient:
    def resolve_access_token(self, access_token: str) -> ActorContext:
        assert access_token == "atlas-token"
        return ActorContext(
            user_id="user-1",
            atlas_tenant_id="TEN-1",
            roles=("admin",),
            auth_source="atlas_bearer",
        )

    def resolve_user(self, user_id: str) -> ActorContext:  # pragma: no cover - not used
        return ActorContext(
            user_id=user_id,
            atlas_tenant_id="TEN-1",
            roles=("admin",),
            auth_source="ursa_token",
        )


class MemoryResourceStore:
    def __init__(self) -> None:
        self.worksets: dict[str, WorksetRecord] = {}
        self.manifests: dict[str, ManifestRecord] = {}
        self.buckets: dict[str, LinkedBucketRecord] = {}
        self._workset_seq = 0
        self._manifest_seq = 0
        self._bucket_seq = 0

    def list_worksets(self, *, atlas_tenant_id: str, limit: int = 100):
        _ = limit
        return [item for item in self.worksets.values() if item.atlas_tenant_id == atlas_tenant_id]

    def create_workset(self, *, name: str, atlas_tenant_id: str, owner_user_id: str, artifact_set_euids, metadata):
        self._workset_seq += 1
        record = WorksetRecord(
            workset_euid=f"WS-{self._workset_seq}",
            name=name,
            atlas_tenant_id=atlas_tenant_id,
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

    def list_manifests(self, *, atlas_tenant_id: str, limit: int = 200):
        _ = limit
        return [item for item in self.manifests.values() if item.atlas_tenant_id == atlas_tenant_id]

    def create_manifest(self, *, workset_euid: str, name: str, artifact_set_euid: str | None, artifact_euids, input_references, metadata):
        workset = self.worksets[workset_euid]
        self._manifest_seq += 1
        manifest = ManifestRecord(
            manifest_euid=f"MF-{self._manifest_seq}",
            name=name,
            workset_euid=workset_euid,
            atlas_tenant_id=workset.atlas_tenant_id,
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
            atlas_tenant_id=workset.atlas_tenant_id,
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

    def list_linked_buckets(self, *, atlas_tenant_id: str, limit: int = 200):
        _ = limit
        return [item for item in self.buckets.values() if item.atlas_tenant_id == atlas_tenant_id and item.state != "DELETED"]

    def create_linked_bucket(
        self,
        *,
        bucket_name: str,
        atlas_tenant_id: str,
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
            if item.atlas_tenant_id == atlas_tenant_id and item.bucket_name == bucket_name and item.state != "DELETED":
                raise ValueError(f"Bucket already linked: {bucket_name}")
        self._bucket_seq += 1
        record = LinkedBucketRecord(
            bucket_id=f"BK-{self._bucket_seq}",
            bucket_name=bucket_name,
            atlas_tenant_id=atlas_tenant_id,
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
            atlas_tenant_id=record.atlas_tenant_id,
            owner_user_id=record.owner_user_id,
            display_name=updates.get("display_name", record.display_name),
            metadata=updates.get("metadata", record.metadata),
            created_at=record.created_at,
            updated_at="2026-03-25T00:20:30Z",
            state=record.state,
            bucket_type=updates.get("bucket_type", record.bucket_type),
            description=updates.get("description", record.description),
            prefix_restriction=updates.get("prefix_restriction", record.prefix_restriction),
            read_only=record.read_only if updates.get("read_only") is None else updates.get("read_only"),
            region=updates.get("region", record.region),
            is_validated=record.is_validated if updates.get("is_validated") is None else updates.get("is_validated"),
            can_read=record.can_read if updates.get("can_read") is None else updates.get("can_read"),
            can_write=record.can_write if updates.get("can_write") is None else updates.get("can_write"),
            can_list=record.can_list if updates.get("can_list") is None else updates.get("can_list"),
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
            atlas_tenant_id=record.atlas_tenant_id,
            owner_user_id=record.owner_user_id,
            display_name=record.display_name,
            metadata=record.metadata,
            created_at=record.created_at,
            updated_at="2026-03-25T00:21:00Z",
            state="DELETED",
        )
        self.buckets[bucket_id] = deleted
        return deleted

    def record_dewey_import(self, *, artifact_euid: str, artifact_type: str, storage_uri: str, actor_user_id: str, metadata=None):
        _ = (artifact_type, storage_uri, actor_user_id, metadata)
        return {"artifact_euid": artifact_euid}


class DummyAnalysisStore:
    def list_analyses(self, *, atlas_tenant_id=None, workset_euid=None, limit=200):  # pragma: no cover
        _ = (atlas_tenant_id, workset_euid, limit)
        return []


class DummyClusterService:
    def get_all_clusters_with_status(self, *, force_refresh: bool = False, fetch_ssh_status: bool = False):
        _ = (force_refresh, fetch_ssh_status)
        return []


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

    def generate_presigned_url(self, ClientMethod: str, Params: dict, ExpiresIn: int = 3600, **kwargs):  # noqa: N803
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
    )


def test_workset_and_manifest_routes_use_versioned_user_api() -> None:
    dewey = DummyDeweyClient()
    app = create_app(
        DummyAnalysisStore(),
        bloom_client=object(),
        identity_client=DummyIdentityClient(),
        resource_store=MemoryResourceStore(),
        dewey_client=dewey,
        settings=_settings(),
    )
    app.state.cluster_service = DummyClusterService()

    with TestClient(app) as client:
        workset = client.post(
            "/api/v1/worksets",
            headers={"Authorization": "Bearer atlas-token"},
            json={"name": "Tumor batch", "artifact_set_euids": ["AS-1"]},
        )
        manifest = client.post(
            "/api/v1/manifests",
            headers={"Authorization": "Bearer atlas-token"},
            json={
                "workset_euid": "WS-1",
                "name": "manifest 1",
                "artifact_set_euid": "AS-1",
                "artifact_euids": ["AT-1", "AT-2"],
            },
        )
        listed_worksets = client.get("/api/v1/worksets", headers={"Authorization": "Bearer atlas-token"})
        listed_manifests = client.get("/api/v1/manifests", headers={"Authorization": "Bearer atlas-token"})
        clusters = client.get("/api/v1/clusters", headers={"Authorization": "Bearer atlas-token"})

    assert workset.status_code == 201, workset.text
    assert manifest.status_code == 201, manifest.text
    assert listed_worksets.json()[0]["manifests"][0]["manifest_euid"] == "MF-1"
    assert listed_manifests.json()[0]["artifact_set_euid"] == "AS-1"
    assert listed_manifests.json()[0]["input_references"][0]["reference_type"] == "artifact_set_euid"
    assert clusters.status_code == 200
    assert clusters.json() == {"items": []}


def test_manifest_rejects_artifacts_outside_resolved_dewey_set() -> None:
    app = create_app(
        DummyAnalysisStore(),
        bloom_client=object(),
        identity_client=DummyIdentityClient(),
        resource_store=MemoryResourceStore(),
        dewey_client=DummyDeweyClient(),
        settings=_settings(),
    )
    app.state.cluster_service = DummyClusterService()

    with TestClient(app) as client:
        workset = client.post(
            "/api/v1/worksets",
            headers={"Authorization": "Bearer atlas-token"},
            json={"name": "Tumor batch", "artifact_set_euids": ["AS-1"]},
        )
        manifest = client.post(
            "/api/v1/manifests",
            headers={"Authorization": "Bearer atlas-token"},
            json={
                "workset_euid": workset.json()["workset_euid"],
                "name": "manifest 1",
                "artifact_set_euid": "AS-1",
                "artifact_euids": ["AT-3"],
            },
        )

    assert workset.status_code == 201, workset.text
    assert manifest.status_code == 400
    assert "is not a member of artifact set AS-1" in manifest.json()["detail"]


def test_workset_rejects_unknown_dewey_artifact_set() -> None:
    app = create_app(
        DummyAnalysisStore(),
        bloom_client=object(),
        identity_client=DummyIdentityClient(),
        resource_store=MemoryResourceStore(),
        dewey_client=DummyDeweyClient(),
        settings=_settings(),
    )
    app.state.cluster_service = DummyClusterService()

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/worksets",
            headers={"Authorization": "Bearer atlas-token"},
            json={"name": "Tumor batch", "artifact_set_euids": ["AS-404"]},
        )

    assert response.status_code == 502
    assert "unknown artifact set: AS-404" in response.json()["detail"]


def test_manifest_accepts_mixed_input_references_and_imports_s3_uris() -> None:
    resources = MemoryResourceStore()
    dewey = DummyDeweyClient()
    app = create_app(
        DummyAnalysisStore(),
        bloom_client=object(),
        identity_client=DummyIdentityClient(),
        resource_store=resources,
        dewey_client=dewey,
        settings=_settings(),
    )
    app.state.cluster_service = DummyClusterService()
    app.state.s3_client = DummyS3Client()

    with TestClient(app) as client:
        workset = client.post(
            "/api/v1/worksets",
            headers={"Authorization": "Bearer atlas-token"},
            json={"name": "Tumor batch", "artifact_set_euids": ["AS-1"]},
        )
        manifest = client.post(
            "/api/v1/manifests",
            headers={"Authorization": "Bearer atlas-token"},
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
    assert body["input_references"][1]["reference_type"] == "s3_uri"
    assert dewey.register_calls[0]["artifact_type"] == "fastq"


def test_bucket_routes_create_list_and_delete() -> None:
    app = create_app(
        DummyAnalysisStore(),
        bloom_client=object(),
        identity_client=DummyIdentityClient(),
        resource_store=MemoryResourceStore(),
        dewey_client=DummyDeweyClient(),
        settings=_settings(),
    )
    app.state.cluster_service = DummyClusterService()
    app.state.s3_client = DummyS3Client()

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/buckets",
            headers={"Authorization": "Bearer atlas-token"},
            json={"bucket_name": "omics-inputs", "display_name": "Primary Inputs"},
        )
        listed = client.get("/api/v1/buckets", headers={"Authorization": "Bearer atlas-token"})
        deleted = client.delete(
            f"/api/v1/buckets/{created.json()['bucket_id']}",
            headers={"Authorization": "Bearer atlas-token"},
        )
        listed_after = client.get("/api/v1/buckets", headers={"Authorization": "Bearer atlas-token"})

    assert created.status_code == 201, created.text
    assert listed.json()[0]["bucket_name"] == "omics-inputs"
    assert deleted.status_code == 200
    assert deleted.json()["state"] == "DELETED"
    assert listed_after.json() == []
