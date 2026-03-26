from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from daylib_ursa.auth import (
    ActorContext,
    AtlasUserDirectoryEntry,
    UserTokenService,
)
from daylib_ursa.config import Settings
from daylib_ursa.resource_store import (
    ClientRegistrationRecord,
    ClusterJobEventRecord,
    ClusterJobRecord,
    LinkedBucketRecord,
)
from daylib_ursa.workset_api import create_app


@dataclass
class _Instance:
    uid: int
    euid: str
    name: str
    json_addl: dict
    bstatus: str
    template_code: str
    created_dt: datetime
    modified_dt: datetime
    polymorphic_discriminator: str = "generic_instance"


class MemoryBackend:
    def __init__(self) -> None:
        self._uid = 0
        self.instances: list[_Instance] = []
        self.lineages: list[tuple[_Instance, _Instance, str]] = []

    @contextmanager
    def session_scope(self, commit: bool = False):
        _ = commit
        yield object()

    def create_instance(self, session, template_code: str, name: str, *, json_addl, bstatus, singleton=False):
        _ = (session, singleton)
        self._uid += 1
        prefix = {
            "integration/auth/user-token/1.0/": "UT",
            "integration/auth/user-token-revision/1.0/": "UR",
            "integration/auth/user-token-usage/1.0/": "UG",
        }.get(template_code, "GI")
        now = datetime.now(timezone.utc)
        instance = _Instance(
            uid=self._uid,
            euid=f"{prefix}-{self._uid}",
            name=name,
            json_addl=dict(json_addl),
            bstatus=bstatus,
            template_code=template_code,
            created_dt=now,
            modified_dt=now,
        )
        self.instances.append(instance)
        return instance

    def create_lineage(self, session, *, parent, child, relationship_type, name=None):
        _ = (session, name)
        self.lineages.append((parent, child, relationship_type))

    def list_children(self, session, *, parent, relationship_type=None):
        _ = session
        return [
            child
            for source, child, rel in self.lineages
            if source is parent and (relationship_type is None or rel == relationship_type)
        ]

    def list_parents(self, session, *, child, relationship_type=None):
        _ = session
        return [
            parent
            for parent, target, rel in self.lineages
            if target is child and (relationship_type is None or rel == relationship_type)
        ]

    def find_instance_by_euid(self, session, *, template_code: str, value: str, for_update: bool = False):
        _ = (session, for_update)
        for instance in self.instances:
            if instance.template_code == template_code and instance.euid == value:
                return instance
        return None

    def find_instance_by_external_id(self, session, *, template_code: str, key: str, value: str):
        _ = session
        for instance in self.instances:
            if instance.template_code != template_code:
                continue
            if str(instance.json_addl.get(key) or "") == value:
                return instance
        return None

    def list_instances_by_property(self, session, *, template_code: str, key: str, value: str, limit: int = 200):
        _ = session
        rows = [
            instance
            for instance in self.instances
            if instance.template_code == template_code and str(instance.json_addl.get(key) or "") == value
        ]
        return list(reversed(rows))[:limit]

    def list_instances_by_template(self, session, *, template_code: str, limit: int = 100):
        _ = session
        rows = [instance for instance in self.instances if instance.template_code == template_code]
        return list(reversed(rows))[:limit]


class DummyIdentityClient:
    def __init__(self, *, admin: bool = True) -> None:
        self.admin = admin

    def resolve_access_token(self, access_token: str) -> ActorContext:
        assert access_token == "atlas-token"
        return ActorContext(
            user_id="user-1",
            atlas_tenant_id="TEN-1",
            roles=("admin",) if self.admin else ("internal_ro",),
            email="alice@example.com",
            display_name="Alice Example",
            organization="Atlas Org",
            site="Seattle",
            auth_source="atlas_bearer",
        )

    def resolve_user(self, user_id: str) -> ActorContext:
        return ActorContext(
            user_id=user_id,
            atlas_tenant_id="TEN-1",
            roles=("admin",) if self.admin else ("internal_ro",),
            email="alice@example.com",
            display_name="Alice Example",
            organization="Atlas Org",
            site="Seattle",
            auth_source="ursa_token",
        )

    def list_users(self, **_kwargs):
        return [
            AtlasUserDirectoryEntry(
                user_id="user-2",
                atlas_tenant_id="TEN-1",
                organization_id="ORG-1",
                organization_name="Atlas Org",
                site_id="SITE-1",
                site_name="Seattle",
                roles=("external_user",),
                email="bob@example.com",
                display_name="Bob Example",
                is_active=True,
            )
        ]


class DummyAnalysisStore:
    def list_analyses(self, *, atlas_tenant_id=None, workset_euid=None, limit=200):  # pragma: no cover
        _ = (atlas_tenant_id, workset_euid, limit)
        return []


class MemoryResourceStore:
    def __init__(self) -> None:
        self.client_registrations: dict[str, ClientRegistrationRecord] = {}
        self.cluster_jobs: dict[str, ClusterJobRecord] = {}
        self.buckets: dict[str, LinkedBucketRecord] = {}
        self._client_seq = 0
        self._job_seq = 0
        self._bucket_seq = 0

    def list_worksets(self, *, atlas_tenant_id: str, limit: int = 100):
        _ = (atlas_tenant_id, limit)
        return []

    def list_manifests(self, *, atlas_tenant_id: str, limit: int = 200):
        _ = (atlas_tenant_id, limit)
        return []

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
        self._bucket_seq += 1
        record = LinkedBucketRecord(
            bucket_id=f"BK-{self._bucket_seq}",
            bucket_name=bucket_name,
            atlas_tenant_id=atlas_tenant_id,
            owner_user_id=owner_user_id,
            display_name=display_name,
            metadata=dict(metadata or {}),
            created_at="2026-03-25T00:00:00Z",
            updated_at="2026-03-25T00:00:00Z",
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
            updated_at="2026-03-25T00:00:30Z",
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
            updated_at="2026-03-25T00:01:00Z",
            state="DELETED",
        )
        self.buckets[bucket_id] = deleted
        return deleted

    def create_client_registration(self, *, client_name: str, owner_user_id: str, sponsor_user_id: str, scopes, metadata):
        self._client_seq += 1
        record = ClientRegistrationRecord(
            client_registration_euid=f"UC-{self._client_seq}",
            client_name=client_name,
            owner_user_id=owner_user_id,
            sponsor_user_id=sponsor_user_id,
            scopes=list(scopes or []),
            metadata=dict(metadata or {}),
            created_at="2026-03-25T00:00:00Z",
            updated_at="2026-03-25T00:00:00Z",
            state="ACTIVE",
        )
        self.client_registrations[record.client_registration_euid] = record
        return record

    def get_client_registration(self, client_registration_euid: str):
        return self.client_registrations.get(client_registration_euid)

    def list_client_registrations(self, *, owner_user_id: str | None = None, limit: int = 200):
        _ = limit
        values = list(self.client_registrations.values())
        if owner_user_id:
            values = [item for item in values if item.owner_user_id == owner_user_id]
        return values

    def add_cluster_job(self, *, cluster_name: str, owner_user_id: str, sponsor_user_id: str) -> ClusterJobRecord:
        self._job_seq += 1
        record = ClusterJobRecord(
            job_euid=f"CJ-{self._job_seq}",
            job_name=cluster_name,
            cluster_name=cluster_name,
            region="us-west-2",
            region_az="us-west-2d",
            atlas_tenant_id="TEN-1",
            owner_user_id=owner_user_id,
            sponsor_user_id=sponsor_user_id,
            state="QUEUED",
            created_at="2026-03-25T00:00:00Z",
            updated_at="2026-03-25T00:00:00Z",
            started_at=None,
            completed_at=None,
            return_code=None,
            error=None,
            output_summary=None,
            request={"cluster_name": cluster_name},
            cluster={},
            events=[
                ClusterJobEventRecord(
                    event_euid=f"CE-{self._job_seq}",
                    job_euid=f"CJ-{self._job_seq}",
                    event_type="queued",
                    status="QUEUED",
                    summary="queued",
                    details={},
                    created_by=sponsor_user_id,
                    created_at="2026-03-25T00:00:00Z",
                )
            ],
        )
        self.cluster_jobs[record.job_euid] = record
        return record

    def list_cluster_jobs(self, *, atlas_tenant_id: str | None = None, limit: int = 200):
        _ = (atlas_tenant_id, limit)
        return list(self.cluster_jobs.values())

    def get_cluster_job(self, job_euid: str):
        return self.cluster_jobs.get(job_euid)


class DummyClusterInfo:
    def __init__(self, cluster_name: str, region: str, cluster_status: str = "CREATE_COMPLETE") -> None:
        self.cluster_name = cluster_name
        self.region = region
        self.cluster_status = cluster_status
        self.error_message = None

    def to_dict(self, include_sensitive: bool = True):
        _ = include_sensitive
        return {
            "cluster_name": self.cluster_name,
            "region": self.region,
            "cluster_status": self.cluster_status,
            "compute_fleet_status": "RUNNING",
        }


class DummyClusterService:
    def get_all_clusters_with_status(self, *, force_refresh: bool = False, fetch_ssh_status: bool = False):
        _ = (force_refresh, fetch_ssh_status)
        return [DummyClusterInfo("cluster-1", "us-west-2")]

    def get_region_for_cluster(self, cluster_name: str):
        _ = cluster_name
        return "us-west-2"

    def get_cluster_by_name(self, cluster_name: str, force_refresh: bool = False):
        _ = force_refresh
        return DummyClusterInfo(cluster_name, "us-west-2")

    def describe_cluster(self, cluster_name: str, region: str):
        return DummyClusterInfo(cluster_name, region)

    def fetch_headnode_status(self, cluster):
        return cluster

    def delete_cluster(self, cluster_name: str, region: str):
        return {"cluster_name": cluster_name, "region": region, "status": "DELETE_IN_PROGRESS"}

    def clear_cache(self) -> None:
        return None


class DummyClusterJobManager:
    def __init__(self, resource_store: MemoryResourceStore) -> None:
        self.resource_store = resource_store

    def start_create_job(self, *, cluster_name: str, owner_user_id: str, sponsor_user_id: str, **_kwargs):
        return self.resource_store.add_cluster_job(
            cluster_name=cluster_name,
            owner_user_id=owner_user_id,
            sponsor_user_id=sponsor_user_id,
        )


def _settings() -> Settings:
    return Settings(
        cors_origins="*",
        ursa_internal_api_key="ursa-test-key",
        bloom_base_url="https://bloom.example",
        atlas_base_url="https://atlas.example",
        ursa_internal_output_bucket="ursa-internal",
        ursa_tapdb_mount_enabled=False,
    )


def test_admin_routes_cover_me_user_search_client_tokens_and_clusters() -> None:
    backend = MemoryBackend()
    identity = DummyIdentityClient(admin=True)
    token_service = UserTokenService(backend=backend, identity_client=identity)
    resources = MemoryResourceStore()
    app = create_app(
        DummyAnalysisStore(),
        bloom_client=object(),
        identity_client=identity,
        resource_store=resources,
        token_service=token_service,
        settings=_settings(),
    )
    app.state.cluster_service = DummyClusterService()
    app.state.cluster_job_manager = DummyClusterJobManager(resources)

    with TestClient(app) as client:
        me = client.get("/api/v1/me", headers={"Authorization": "Bearer atlas-token"})
        users = client.get("/api/v1/admin/users", headers={"Authorization": "Bearer atlas-token"})
        registration = client.post(
            "/api/v1/admin/client-registrations",
            headers={"Authorization": "Bearer atlas-token"},
            json={
                "client_name": "dewey-client",
                "owner_user_id": "user-2",
                "scopes": ["internal_rw"],
                "metadata": {"purpose": "integration"},
            },
        )
        registration_euid = registration.json()["client_registration_euid"]
        registration_detail = client.get(
            f"/api/v1/admin/client-registrations/{registration_euid}",
            headers={"Authorization": "Bearer atlas-token"},
        )
        issued_token = client.post(
            f"/api/v1/admin/client-registrations/{registration_euid}/tokens",
            headers={"Authorization": "Bearer atlas-token"},
            json={
                "token_name": "client bootstrap",
                "scope": "internal_rw",
                "expires_in_days": 30,
            },
        )
        token_list = client.get(
            f"/api/v1/admin/client-registrations/{registration_euid}/tokens",
            headers={"Authorization": "Bearer atlas-token"},
        )
        revoked_token = client.post(
            f"/api/v1/admin/user-tokens/{issued_token.json()['token_euid']}/revoke",
            headers={"Authorization": "Bearer atlas-token"},
            json={"note": "revoked in test"},
        )
        cluster_list = client.get("/api/v1/clusters", headers={"Authorization": "Bearer atlas-token"})
        cluster_job = client.post(
            "/api/v1/clusters",
            headers={"Authorization": "Bearer atlas-token"},
            json={
                "cluster_name": "cluster-2",
                "region_az": "us-west-2d",
                "ssh_key_name": "omics-key",
                "s3_bucket_name": "ursa-bucket",
            },
        )
        cluster_jobs = client.get("/api/v1/clusters/jobs", headers={"Authorization": "Bearer atlas-token"})
        cluster_detail = client.get(
            "/api/v1/clusters/cluster-1?region=us-west-2",
            headers={"Authorization": "Bearer atlas-token"},
        )
        cluster_delete = client.delete(
            "/api/v1/clusters/cluster-1?region=us-west-2",
            headers={"Authorization": "Bearer atlas-token"},
        )

    assert me.status_code == 200
    assert me.json()["organization"] == "Atlas Org"
    assert users.status_code == 200
    assert users.json()[0]["user_id"] == "user-2"
    assert registration.status_code == 201
    assert registration_detail.status_code == 200
    assert issued_token.status_code == 201
    assert issued_token.json()["plaintext_token"].startswith("urs_")
    assert token_list.status_code == 200
    assert token_list.json()[0]["client_registration_euid"] == registration_euid
    assert revoked_token.status_code == 200
    assert revoked_token.json()["status"] == "REVOKED"
    assert cluster_list.status_code == 200
    assert cluster_list.json()["items"][0]["cluster_name"] == "cluster-1"
    assert cluster_job.status_code == 202
    assert cluster_jobs.status_code == 200
    assert cluster_jobs.json()[0]["cluster_name"] == "cluster-2"
    assert cluster_detail.status_code == 200
    assert cluster_delete.status_code == 200
    assert cluster_delete.json()["result"]["status"] == "DELETE_IN_PROGRESS"


def test_gui_routes_use_session_auth_and_gate_admin_pages() -> None:
    backend = MemoryBackend()
    identity = DummyIdentityClient(admin=True)
    resources = MemoryResourceStore()
    app = create_app(
        DummyAnalysisStore(),
        bloom_client=object(),
        identity_client=identity,
        resource_store=resources,
        token_service=UserTokenService(backend=backend, identity_client=identity),
        settings=_settings(),
    )
    app.state.cluster_service = DummyClusterService()
    app.state.cluster_job_manager = DummyClusterJobManager(resources)

    with TestClient(app) as client:
        redirect = client.get("/", follow_redirects=False)
        login_page = client.get("/login")
        login = client.post(
            "/login",
            data={"access_token": "atlas-token", "next_path": "/"},
            follow_redirects=False,
        )
        dashboard = client.get("/")
        usage_page = client.get("/usage")
        buckets_page = client.get("/buckets")
        user_token = client.post(
            "/api/v1/user-tokens",
            json={"token_name": "session token", "scope": "internal_rw", "expires_in_days": 30},
        )
        cluster_job = client.post(
            "/api/v1/clusters",
            json={
                "cluster_name": "cluster-2",
                "region_az": "us-west-2d",
                "ssh_key_name": "omics-key",
                "s3_bucket_name": "ursa-bucket",
            },
        )
        session_me = client.get("/api/v1/me")
        token_detail_page = client.get(f"/tokens/{user_token.json()['token_euid']}")
        cluster_job_page = client.get(f"/clusters/jobs/{cluster_job.json()['job_euid']}")
        admin_page = client.get("/admin/tokens")

    assert redirect.status_code == 303
    assert redirect.headers["location"].startswith("/login")
    assert login_page.status_code == 200
    assert login.status_code == 303
    assert dashboard.status_code == 200
    assert "Welcome back" in dashboard.text
    assert usage_page.status_code == 200
    assert "Usage Summary" in usage_page.text
    assert buckets_page.status_code == 200
    assert "S3 Bucket Management" in buckets_page.text
    assert user_token.status_code == 201
    assert cluster_job.status_code == 202
    assert session_me.status_code == 200
    assert session_me.json()["user_id"] == "user-1"
    assert token_detail_page.status_code == 200
    assert "session token" in token_detail_page.text
    assert cluster_job_page.status_code == 200
    assert "Cluster Job" in cluster_job_page.text
    assert admin_page.status_code == 200


def test_gui_admin_pages_reject_non_admin_sessions() -> None:
    app = create_app(
        DummyAnalysisStore(),
        bloom_client=object(),
        identity_client=DummyIdentityClient(admin=False),
        resource_store=MemoryResourceStore(),
        token_service=UserTokenService(backend=MemoryBackend(), identity_client=DummyIdentityClient(admin=False)),
        settings=_settings(),
    )

    with TestClient(app) as client:
        client.post("/login", data={"access_token": "atlas-token", "next_path": "/"})
        response = client.get("/admin/tokens", follow_redirects=False)

    assert response.status_code == 403
