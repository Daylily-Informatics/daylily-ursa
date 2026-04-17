from __future__ import annotations

import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from daylib_ursa import __version__
from daylib_ursa.analysis_store import AnalysisRecord, AnalysisState, ReviewState
from daylib_ursa.auth import (
    AtlasUserDirectoryEntry,
    AuthError,
    CurrentUser,
    Role,
    UserTokenService,
)
from daylib_ursa.config import Settings
from daylib_ursa.resource_store import (
    ClientRegistrationRecord,
    ClusterJobEventRecord,
    ClusterJobRecord,
    LinkedBucketRecord,
    ManifestRecord,
    WorksetRecord,
)
from daylib_ursa.workset_api import create_app

TEST_BASE_URL = "https://testserver"

TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
ADMIN_USER_ID = "00000000-0000-0000-0000-000000000101"
SECONDARY_USER_ID = "00000000-0000-0000-0000-000000000202"


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
    tenant_id: uuid.UUID | None = None
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

    def create_instance(
        self,
        session,
        template_code: str,
        name: str,
        *,
        json_addl,
        bstatus,
        tenant_id: uuid.UUID | None = None,
        singleton: bool = False,
    ):
        _ = (session, singleton)
        self._uid += 1
        prefix = {
            "RGX/auth/user-token/1.0/": "UT",
            "RGX/auth/user-token-revision/1.0/": "UR",
            "RGX/auth/user-token-usage/1.0/": "UG",
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
            tenant_id=tenant_id,
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

    def find_instance_by_euid(
        self, session, *, template_code: str, value: str, for_update: bool = False
    ):
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

    def list_instances_by_property(
        self, session, *, template_code: str, key: str, value: str, limit: int = 200
    ):
        _ = session
        rows = [
            instance
            for instance in self.instances
            if instance.template_code == template_code
            and str(instance.json_addl.get(key) or "") == value
        ]
        return list(reversed(rows))[:limit]

    def list_instances_by_template(self, session, *, template_code: str, limit: int = 100):
        _ = session
        rows = [instance for instance in self.instances if instance.template_code == template_code]
        return list(reversed(rows))[:limit]


class DummyAuthProvider:
    def __init__(self, *, admin: bool = True) -> None:
        self.admin = admin

    def resolve_access_token(self, access_token: str) -> CurrentUser:
        if access_token != "atlas-token":
            raise AuthError("Invalid authentication token")
        return CurrentUser(
            sub=ADMIN_USER_ID,
            email="alice@lsmc.com",
            name="Alice Example",
            tenant_id=TENANT_ID,
            roles=[Role.ADMIN.value] if self.admin else [Role.READ_ONLY.value],
            organization="Atlas Org",
            site="Seattle",
            auth_source="cognito",
        )


class DummyUserDirectory:
    def list_users(self, **_kwargs):
        return [
            AtlasUserDirectoryEntry(
                user_id=SECONDARY_USER_ID,
                tenant_id=TENANT_ID,
                organization_id="ORG-1",
                organization_name="Atlas Org",
                site_id="SITE-1",
                site_name="Seattle",
                roles=(Role.EXTERNAL_USER.value,),
                email="bob@lsmc.bio",
                display_name="Bob Example",
                is_active=True,
            )
        ]

    def get_user(self, user_id: str) -> CurrentUser | None:
        if user_id != SECONDARY_USER_ID:
            return None
        return CurrentUser(
            sub=SECONDARY_USER_ID,
            email="bob@lsmc.bio",
            name="Bob Example",
            tenant_id=TENANT_ID,
            roles=[Role.EXTERNAL_USER.value],
            auth_source="cognito",
            organization="Atlas Org",
            site="Seattle",
        )


class DummyAnalysisStore:
    def __init__(self) -> None:
        self.record = AnalysisRecord(
            analysis_euid="AN-1",
            workset_euid="WS-1",
            run_euid="RUN-1",
            flowcell_id="FLOW-1",
            lane="1",
            library_barcode="LIB-1",
            sequenced_library_assignment_euid="SQA-1",
            tenant_id=TENANT_ID,
            atlas_trf_euid="TRF-1",
            atlas_test_euid="TST-1",
            atlas_test_fulfillment_item_euid="TPC-1",
            analysis_type="somatic",
            state=AnalysisState.REVIEW_PENDING.value,
            review_state=ReviewState.PENDING.value,
            result_status="PENDING",
            run_folder="s3://ursa-internal/RUN-1/",
            internal_bucket="ursa-internal",
            input_references=[],
            result_payload={},
            metadata={},
            created_at="2026-03-25T00:00:00Z",
            updated_at="2026-03-25T00:00:00Z",
            atlas_return={},
            artifacts=[],
        )

    def list_analyses(self, *, tenant_id=None, workset_euid=None, limit=200):
        _ = limit
        if tenant_id is not None and tenant_id != self.record.tenant_id:
            return []
        if workset_euid is not None and workset_euid != self.record.workset_euid:
            return []
        return [self.record]

    def get_analysis(self, analysis_euid: str):
        return self.record if analysis_euid == self.record.analysis_euid else None


class MemoryResourceStore:
    def __init__(self) -> None:
        analysis_command = {
            "repository": "daylily-omics-analysis",
            "repository_ref": "0.7.640.dev0",
            "workflow_id": "germline_wgs_snv",
            "display_name": "Germline WGS SNV",
            "spec": {"genome_build": "hg38", "execution_profile": "slurm"},
            "shell_preview": "source dyoainit && dy-a slurm hg38 && dy-r produce_alignstats produce_snv_concordances -p -k -j 10 --config genome_build=hg38 aligners=['bwa2a'] dedupers=['dppl'] snv_callers=['sentd']",
            "summary": {
                "pipeline_type": "Germline WGS SNV",
                "genome_build": "hg38",
                "execution_profile": "slurm",
            },
        }
        self.worksets: dict[str, WorksetRecord] = {
            "WS-1": WorksetRecord(
                workset_euid="WS-1",
                name="Tumor Batch",
                tenant_id=TENANT_ID,
                owner_user_id=ADMIN_USER_ID,
                state="ACTIVE",
                artifact_set_euids=["AS-1"],
                metadata={"analysis_command": analysis_command},
                created_at="2026-03-25T00:00:00Z",
                updated_at="2026-03-25T00:00:00Z",
                manifests=[],
                analysis_euids=["AN-1"],
            )
        }
        self.manifests: dict[str, ManifestRecord] = {
            "MF-1": ManifestRecord(
                manifest_euid="MF-1",
                name="Manifest One",
                workset_euid="WS-1",
                tenant_id=TENANT_ID,
                owner_user_id=ADMIN_USER_ID,
                artifact_set_euid="AS-1",
                artifact_euids=["AT-1"],
                input_references=[{"reference_type": "artifact_set_euid", "value": "AS-1"}],
                metadata={"editor_manifest_tsv": "sample\tartifact\nS1\tAT-1\n"},
                created_at="2026-03-25T00:05:00Z",
                updated_at="2026-03-25T00:05:00Z",
                state="ACTIVE",
            )
        }
        self.buckets: dict[str, LinkedBucketRecord] = {
            "BK-1": LinkedBucketRecord(
                bucket_id="BK-1",
                bucket_name="omics-inputs",
                tenant_id=TENANT_ID,
                owner_user_id=ADMIN_USER_ID,
                display_name="Primary Inputs",
                metadata={},
                created_at="2026-03-25T00:10:00Z",
                updated_at="2026-03-25T00:10:00Z",
                state="ACTIVE",
                bucket_type="secondary",
                description=None,
                prefix_restriction="incoming/",
                read_only=False,
                region="us-west-2",
                is_validated=True,
                can_read=True,
                can_write=True,
                can_list=True,
                remediation_steps=[],
            )
        }
        self.client_registrations: dict[str, ClientRegistrationRecord] = {}
        self.cluster_jobs: dict[str, ClusterJobRecord] = {}
        self._client_seq = 0
        self._job_seq = 0
        self.worksets["WS-1"] = WorksetRecord(
            workset_euid="WS-1",
            name="Tumor Batch",
            tenant_id=TENANT_ID,
            owner_user_id=ADMIN_USER_ID,
            state="ACTIVE",
            artifact_set_euids=["AS-1"],
            metadata={"analysis_command": analysis_command},
            created_at="2026-03-25T00:00:00Z",
            updated_at="2026-03-25T00:05:00Z",
            manifests=[self.manifests["MF-1"]],
            analysis_euids=["AN-1"],
        )

    def list_worksets(self, *, tenant_id: uuid.UUID, limit: int = 100):
        _ = limit
        return [item for item in self.worksets.values() if item.tenant_id == tenant_id]

    def get_workset(self, workset_euid: str):
        return self.worksets.get(workset_euid)

    def list_manifests(self, *, tenant_id: uuid.UUID, limit: int = 200):
        _ = limit
        return [item for item in self.manifests.values() if item.tenant_id == tenant_id]

    def get_manifest(self, manifest_euid: str):
        return self.manifests.get(manifest_euid)

    def list_linked_buckets(self, *, tenant_id: uuid.UUID, limit: int = 200):
        _ = limit
        return [
            item
            for item in self.buckets.values()
            if item.tenant_id == tenant_id and item.state != "DELETED"
        ]

    def get_linked_bucket(self, bucket_id: str):
        return self.buckets.get(bucket_id)

    def create_client_registration(
        self, *, client_name: str, owner_user_id: str, sponsor_user_id: str, scopes, metadata
    ):
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

    def add_cluster_job(
        self, *, cluster_name: str, owner_user_id: str, sponsor_user_id: str
    ) -> ClusterJobRecord:
        self._job_seq += 1
        record = ClusterJobRecord(
            job_euid=f"CJ-{self._job_seq}",
            job_name=cluster_name,
            cluster_name=cluster_name,
            region="us-west-2",
            region_az="us-west-2d",
            tenant_id=TENANT_ID,
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

    def list_cluster_jobs(self, *, tenant_id: uuid.UUID | None = None, limit: int = 200):
        _ = (tenant_id, limit)
        return list(self.cluster_jobs.values())

    def get_cluster_job(self, job_euid: str):
        return self.cluster_jobs.get(job_euid)


class DummyClusterInfo:
    def __init__(
        self, cluster_name: str, region: str, cluster_status: str = "CREATE_COMPLETE"
    ) -> None:
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
    def __init__(
        self,
        clusters: list[DummyClusterInfo] | None = None,
        regions: list[str] | None = None,
    ) -> None:
        self._clusters = list(clusters or [DummyClusterInfo("cluster-1", "us-west-2")])
        derived_regions = [cluster.region for cluster in self._clusters if cluster.region]
        self.regions = list(regions or derived_regions or ["us-west-2"])

    def get_all_clusters_with_status(
        self, *, force_refresh: bool = False, fetch_ssh_status: bool = False
    ):
        _ = (force_refresh, fetch_ssh_status)
        return list(self._clusters)

    def get_region_for_cluster(self, cluster_name: str):
        for cluster in self._clusters:
            if cluster.cluster_name == cluster_name:
                return cluster.region
        return self.regions[0] if self.regions else "us-west-2"

    def get_cluster_by_name(self, cluster_name: str, force_refresh: bool = False):
        _ = force_refresh
        for cluster in self._clusters:
            if cluster.cluster_name == cluster_name:
                return cluster
        return DummyClusterInfo(cluster_name, self.get_region_for_cluster(cluster_name))

    def describe_cluster(self, cluster_name: str, region: str):
        for cluster in self._clusters:
            if cluster.cluster_name == cluster_name and cluster.region == region:
                return cluster
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

    def start_create_job(
        self, *, cluster_name: str, owner_user_id: str, sponsor_user_id: str, **_kwargs
    ):
        return self.resource_store.add_cluster_job(
            cluster_name=cluster_name,
            owner_user_id=owner_user_id,
            sponsor_user_id=sponsor_user_id,
        )


class DummyS3Client:
    def list_objects_v2(self, Bucket: str, **kwargs):  # noqa: N803
        _ = (Bucket, kwargs)
        return {
            "Contents": [
                {
                    "Key": "incoming/data.txt",
                    "Size": 5,
                    "LastModified": datetime(2026, 3, 25, tzinfo=timezone.utc),
                }
            ],
            "CommonPrefixes": [{"Prefix": "incoming/subdir/"}],
        }

    def get_object(self, Bucket: str, Key: str, **kwargs):  # noqa: N803
        _ = (Bucket, Key, kwargs)
        return {"Body": io.BytesIO(b"alpha\nbeta\n")}


class DummyAdminBucketSession:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def client(self, service_name: str, region_name: str | None = None):
        _ = region_name
        if service_name == "ec2":
            return SimpleNamespace(
                describe_key_pairs=lambda: {"KeyPairs": [{"KeyName": "omics-key"}]},
                describe_availability_zones=lambda Filters=None: {
                    "AvailabilityZones": [
                        {"ZoneName": "us-west-2a"},
                        {"ZoneName": "us-west-2b"},
                    ]
                },
            )
        assert service_name == "s3"
        return SimpleNamespace(
            list_buckets=lambda: {
                "Buckets": [
                    {
                        "Name": "omics-secondary",
                        "CreationDate": datetime(2026, 3, 26, 12, 30, tzinfo=timezone.utc),
                    },
                    {
                        "Name": "omics-primary",
                        "CreationDate": datetime(2026, 3, 25, 9, 15, tzinfo=timezone.utc),
                    },
                ]
            }
        )


def _workflow_catalog_payload() -> dict[str, object]:
    return {
        "schema_version": "1",
        "catalog_version": "1.0.0",
        "repository": "daylily-omics-analysis",
        "repository_ref": "0.7.640.dev0",
        "display_name": "Daylily Omics Analysis",
        "workflows": [
            {
                "workflow_id": "germline_wgs_snv",
                "display_name": "Germline WGS SNV",
                "description": "Example",
                "targets": ["produce_alignstats"],
                "supported_genome_builds": ["hg38"],
                "default_genome_build": "hg38",
                "supported_execution_profiles": ["slurm", "local"],
                "default_execution_profile": "slurm",
                "required_inputs": [
                    {
                        "input_id": "workset_manifest",
                        "label": "Workset manifest",
                        "description": "Manifest",
                        "required": True,
                    }
                ],
                "options": [],
            }
        ],
    }


def _settings() -> Settings:
    return Settings(
        aws_profile="",
        cors_origins="*",
        ursa_internal_api_key="ursa-test-key",
        session_secret_key="ursa-session-secret",
        cognito_domain="auth.example.test",
        cognito_app_client_id="client-123",
        cognito_app_client_secret="ursa-cognito-secret",
        cognito_callback_url="https://testserver/auth/callback",
        cognito_logout_url="https://testserver/auth/logout",
        bloom_base_url="https://bloom.example",
        atlas_base_url="https://atlas.example",
        ursa_internal_output_bucket="ursa-internal",
        ursa_tapdb_mount_enabled=False,
        deployment_name="inflec3",
        day_aws_region="us-west-2",
        ui_show_environment_chrome=True,
    )


def _create_test_app(*, admin: bool = True):
    backend = MemoryBackend()
    auth_provider = DummyAuthProvider(admin=admin)
    user_directory = DummyUserDirectory()
    resources = MemoryResourceStore()
    with (
        patch("daylib_ursa.workset_api.RegionAwareS3Client", return_value=object()),
        patch(
            "daylib_ursa.gui_app.get_ursa_config",
            return_value=SimpleNamespace(
                config_path=Path("/opt/dayhoff/repo/.dayhoff/deployments/inflec3/config/ursa.yaml")
            ),
        ),
    ):
        app = create_app(
            DummyAnalysisStore(),
            bloom_client=object(),
            auth_provider=auth_provider,
            user_directory=user_directory,
            resource_store=resources,
            token_service=UserTokenService(backend=backend, user_lookup=user_directory.get_user),
            settings=_settings(),
        )
    app.state.ursa_config = SimpleNamespace(
        config_path=Path("/opt/dayhoff/repo/.dayhoff/deployments/inflec3/config/ursa.yaml")
    )
    app.state.cluster_service = DummyClusterService()
    app.state.cluster_job_manager = DummyClusterJobManager(resources)
    app.state.s3_client = DummyS3Client()
    return app, resources


def _verification_result(
    *, has_failures: bool = False, failing_partitions: list[str] | None = None
):
    partitions = [
        SimpleNamespace(partition=partition, status="FAIL")
        for partition in list(failing_partitions or [])
    ]
    return SimpleNamespace(has_failures=has_failures, partitions=partitions)


def test_create_app_uses_package_version() -> None:
    app, _resources = _create_test_app(admin=True)

    assert app.version == __version__


def test_admin_routes_cover_me_user_search_client_tokens_and_clusters() -> None:
    app, resources = _create_test_app(admin=True)
    app.state.settings.aws_profile = "lsmc"

    with (
        TestClient(app, base_url=TEST_BASE_URL) as client,
        patch("daylib_ursa.workset_api.boto3.Session", DummyAdminBucketSession),
        patch(
            "daylib_ursa.workset_api.run_cluster_partition_verification",
            return_value=_verification_result(),
        ),
    ):
        me = client.get("/api/v1/me", headers={"Authorization": "Bearer atlas-token"})
        users = client.get("/api/v1/admin/users", headers={"Authorization": "Bearer atlas-token"})
        registration = client.post(
            "/api/v1/admin/client-registrations",
            headers={"Authorization": "Bearer atlas-token"},
            json={
                "client_name": "dewey-client",
                "owner_user_id": SECONDARY_USER_ID,
                "scopes": ["internal_rw"],
                "metadata": {"purpose": "integration"},
            },
        )
        registration_euid = registration.json()["client_registration_euid"]
        registration_list = client.get(
            "/api/v1/admin/client-registrations",
            headers={"Authorization": "Bearer atlas-token"},
        )
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
        admin_token = client.post(
            "/api/v1/admin/user-tokens",
            headers={"Authorization": "Bearer atlas-token"},
            json={
                "owner_user_id": SECONDARY_USER_ID,
                "token_name": "secondary user token",
                "scope": "internal_rw",
                "expires_in_days": 30,
            },
        )
        admin_token_list = client.get(
            "/api/v1/admin/user-tokens",
            headers={"Authorization": "Bearer atlas-token"},
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
        cluster_list = client.get(
            "/api/v1/clusters", headers={"Authorization": "Bearer atlas-token"}
        )
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
        cluster_jobs = client.get(
            "/api/v1/clusters/jobs", headers={"Authorization": "Bearer atlas-token"}
        )
        cluster_job_detail = client.get(
            f"/api/v1/clusters/jobs/{cluster_job.json()['job_euid']}",
            headers={"Authorization": "Bearer atlas-token"},
        )
        cluster_create_options = client.get(
            "/api/v1/clusters/create-options?region=us-west-2",
            headers={"Authorization": "Bearer atlas-token"},
        )
        admin_bucket_list = client.get(
            "/api/v1/admin/s3-buckets", headers={"Authorization": "Bearer atlas-token"}
        )
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
    assert me.json()["tenant_id"] == str(TENANT_ID)
    assert users.status_code == 200
    assert users.json()[0]["user_id"] == SECONDARY_USER_ID
    assert users.json()[0]["tenant_id"] == str(TENANT_ID)
    assert registration.status_code == 201
    assert registration_list.status_code == 200
    assert registration_list.json()[0]["client_registration_euid"] == registration_euid
    assert registration_detail.status_code == 200
    assert issued_token.status_code == 201
    assert issued_token.json()["plaintext_token"].startswith("urs_")
    assert admin_token.status_code == 201
    assert admin_token_list.status_code == 200
    assert any(item["token_name"] == "secondary user token" for item in admin_token_list.json())
    assert token_list.status_code == 200
    assert token_list.json()[0]["client_registration_euid"] == registration_euid
    assert revoked_token.status_code == 200
    assert revoked_token.json()["status"] == "REVOKED"
    assert cluster_list.status_code == 200
    assert cluster_list.json()["items"][0]["cluster_name"] == "cluster-1"
    assert cluster_job.status_code == 202
    assert cluster_jobs.status_code == 200
    assert cluster_jobs.json()[0]["cluster_name"] == "cluster-2"
    assert cluster_jobs.json()[0]["tenant_id"] == str(TENANT_ID)
    assert cluster_job_detail.status_code == 200
    assert cluster_job_detail.json()["job_euid"] == cluster_job.json()["job_euid"]
    assert cluster_create_options.status_code == 200
    assert sorted(cluster_create_options.json().keys()) == [
        "availability_zones",
        "buckets",
        "keypairs",
    ]
    assert cluster_create_options.json()["availability_zones"] == ["us-west-2a", "us-west-2b"]
    assert admin_bucket_list.status_code == 200
    assert admin_bucket_list.json()["profile"] == "lsmc"
    assert [item["bucket_name"] for item in admin_bucket_list.json()["buckets"]] == [
        "omics-primary",
        "omics-secondary",
    ]
    assert cluster_detail.status_code == 200
    assert cluster_delete.status_code == 200
    assert cluster_delete.json()["result"]["status"] == "DELETE_IN_PROGRESS"
    assert resources.get_cluster_job(cluster_job.json()["job_euid"]) is not None


def test_gui_routes_cover_remaining_pages_and_logout() -> None:
    app, _resources = _create_test_app(admin=True)

    with (
        TestClient(app, base_url=TEST_BASE_URL) as client,
        patch(
            "daylib_ursa.workset_api.run_cluster_partition_verification",
            return_value=_verification_result(),
        ),
        patch(
            "daylib_ursa.gui_app.load_workflow_catalog_snapshot",
            return_value=_workflow_catalog_payload(),
        ),
    ):
        client.post(
            "/login",
            data={"access_token": "atlas-token", "next_path": "/"},
            follow_redirects=False,
        )
        user_token = client.post(
            "/api/v1/user-tokens",
            json={"token_name": "session token", "scope": "internal_rw", "expires_in_days": 30},
        )
        registration = client.post(
            "/api/v1/admin/client-registrations",
            headers={"Authorization": "Bearer atlas-token"},
            json={
                "client_name": "dewey-client",
                "owner_user_id": SECONDARY_USER_ID,
                "scopes": ["internal_rw"],
                "metadata": {"purpose": "integration"},
            },
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
        worksets_page = client.get("/worksets")
        worksets_new_page = client.get("/worksets/new")
        workset_detail_page = client.get("/worksets/WS-1")
        manifests_page = client.get("/manifests")
        manifest_detail_page = client.get("/manifests/MF-1")
        bucket_detail_page = client.get("/buckets/BK-1")
        analyses_page = client.get("/analyses")
        analysis_detail_page = client.get("/analyses/AN-1")
        artifacts_page = client.get("/artifacts")
        tokens_page = client.get("/tokens")
        token_detail_page = client.get(f"/tokens/{user_token.json()['token_euid']}")
        clusters_page = client.get("/clusters")
        cluster_detail_page = client.get("/clusters/cluster-1")
        cluster_job_page = client.get(f"/clusters/jobs/{cluster_job.json()['job_euid']}")
        admin_clients_page = client.get("/admin/clients")
        admin_client_detail_page = client.get(
            f"/admin/clients/{registration.json()['client_registration_euid']}"
        )
        admin_config_page = client.get("/admin/config")
        logout = client.get("/auth/logout", follow_redirects=False)

    assert worksets_page.status_code == 200
    assert "Tumor Batch" in worksets_page.text
    assert worksets_new_page.status_code == 200
    assert "Create Workset" in worksets_new_page.text
    assert "Analysis Repository" in worksets_new_page.text
    assert "Execution Profile" in worksets_new_page.text
    assert "Command Preview" in worksets_new_page.text
    assert workset_detail_page.status_code == 200
    assert "Workset Tumor Batch" in workset_detail_page.text
    assert "Analysis Command" in workset_detail_page.text
    assert "Germline WGS SNV" in workset_detail_page.text
    assert manifests_page.status_code == 200
    assert "Manifest One" in manifests_page.text
    assert manifest_detail_page.status_code == 200
    assert "Manifest Manifest One" in manifest_detail_page.text
    assert bucket_detail_page.status_code == 200
    assert "Browse Bucket" in bucket_detail_page.text
    assert analyses_page.status_code == 200
    assert "AN-1" in analyses_page.text
    assert analysis_detail_page.status_code == 200
    assert "Analysis AN-1" in analysis_detail_page.text
    assert artifacts_page.status_code == 200
    assert "Artifact Tools" in artifacts_page.text
    assert tokens_page.status_code == 200
    assert "session token" in tokens_page.text
    assert token_detail_page.status_code == 200
    assert "session token" in token_detail_page.text
    assert clusters_page.status_code == 200
    assert "cluster-1" in clusters_page.text
    assert 'list="cc-region-options"' in clusters_page.text
    assert 'value="us-east-1"' in clusters_page.text
    assert 'value="us-east-2"' in clusters_page.text
    assert 'value="ap-south-1"' in clusters_page.text
    assert 'value="eu-central-1"' in clusters_page.text
    assert "Availability Zone" in clusters_page.text
    assert "Verify Partition Instances" in clusters_page.text
    assert "Calculate Spot Pricing Per Partition" in clusters_page.text
    assert cluster_detail_page.status_code == 200
    assert "Cluster cluster-1" in cluster_detail_page.text
    assert cluster_job_page.status_code == 200
    assert "Cluster Job" in cluster_job_page.text
    assert admin_clients_page.status_code == 200
    assert "dewey-client" in admin_clients_page.text
    assert admin_client_detail_page.status_code == 200
    assert "Client dewey-client" in admin_client_detail_page.text
    assert admin_config_page.status_code == 200
    assert "Configuration" in admin_config_page.text
    assert (
        "/opt/dayhoff/repo/.dayhoff/deployments/inflec3/config/ursa.yaml" in admin_config_page.text
    )
    assert "<redacted>" in admin_config_page.text
    assert logout.status_code == 303
    assert logout.headers["location"].startswith("https://auth.example.test/logout?")


def test_cluster_scan_regions_update_refreshes_runtime_service() -> None:
    app, _resources = _create_test_app(admin=True)
    app.state.settings.aws_profile = "lsmc"

    updated_config = SimpleNamespace(
        get_allowed_regions=lambda: ["us-west-2", "us-east-1", "eu-central-1"],
        config_path=Path("/opt/dayhoff/repo/.dayhoff/deployments/inflec3/config/ursa.yaml"),
        aws_profile="lsmc",
    )

    with (
        TestClient(app, base_url=TEST_BASE_URL) as client,
        patch("daylib_ursa.workset_api.update_config_regions", return_value=updated_config),
    ):
        response = client.post(
            "/api/v1/clusters/scan-regions",
            headers={"Authorization": "Bearer atlas-token"},
            json={"regions_csv": "us-west-2, us-east-1, eu-central-1"},
        )

    assert response.status_code == 200, response.text
    assert response.json()["regions"] == ["us-west-2", "us-east-1", "eu-central-1"]
    assert response.json()["regions_csv"] == "us-west-2,us-east-1,eu-central-1"
    assert app.state.settings.ursa_allowed_regions == "us-west-2,us-east-1,eu-central-1"
    assert app.state.cluster_service.regions == ["us-west-2", "us-east-1", "eu-central-1"]
    assert app.state.cluster_job_manager.cluster_service is app.state.cluster_service


def test_clusters_page_groups_regions_and_surfaces_pending_create_jobs() -> None:
    app, resources = _create_test_app(admin=True)
    app.state.cluster_service = DummyClusterService(
        clusters=[
            DummyClusterInfo("lsmc-20260413", "us-west-2"),
            DummyClusterInfo("cluster-east", "us-east-1"),
        ],
        regions=["us-west-2", "us-east-1", "eu-central-1"],
    )
    app.state.cluster_job_manager.cluster_service = app.state.cluster_service
    resources.add_cluster_job(
        cluster_name="test-del-me",
        owner_user_id=ADMIN_USER_ID,
        sponsor_user_id=ADMIN_USER_ID,
    )

    with TestClient(app, base_url=TEST_BASE_URL) as client:
        client.post(
            "/login",
            data={"access_token": "atlas-token", "next_path": "/clusters"},
            follow_redirects=False,
        )
        response = client.get("/clusters")

    assert response.status_code == 200
    assert "Cluster Scan Regions" in response.text
    assert 'value="us-west-2,us-east-1,eu-central-1"' in response.text
    assert "Scanning regions: <strong>us-west-2, us-east-1, eu-central-1</strong>" in response.text
    assert "Create Jobs In Flight" in response.text
    assert "test-del-me" in response.text
    assert "lsmc-20260413" in response.text
    assert "cluster-east" in response.text
    assert "us-west-2" in response.text
    assert "us-east-1" in response.text
    assert "eu-central-1" in response.text
    assert "1 create job in flight." in response.text


def test_gui_routes_use_session_auth_and_gate_admin_pages() -> None:
    app, _resources = _create_test_app(admin=True)

    with (
        TestClient(app, base_url=TEST_BASE_URL) as client,
        patch(
            "daylib_ursa.workset_api.run_cluster_partition_verification",
            return_value=_verification_result(),
        ),
    ):
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
        admin_home = client.get("/admin", follow_redirects=False)
        admin_page = client.get("/admin/tokens")
        admin_config_page = client.get("/admin/config")

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
    assert 'href="/admin"' in dashboard.text
    assert "Admin Access" in dashboard.text
    assert "List Buckets" in buckets_page.text
    assert user_token.status_code == 201
    assert cluster_job.status_code == 202
    assert session_me.status_code == 200
    assert session_me.json()["user_id"] == ADMIN_USER_ID
    assert token_detail_page.status_code == 200
    assert "session token" in token_detail_page.text
    assert cluster_job_page.status_code == 200
    assert "Cluster Job" in cluster_job_page.text
    assert admin_home.status_code == 303
    assert admin_home.headers["location"] == "/admin/tokens"
    assert admin_page.status_code == 200
    assert admin_config_page.status_code == 200
    assert "Configuration" in admin_config_page.text


def test_cluster_create_blocks_when_partition_verification_fails() -> None:
    app, resources = _create_test_app(admin=True)

    with (
        TestClient(app, base_url=TEST_BASE_URL) as client,
        patch(
            "daylib_ursa.workset_api.run_cluster_partition_verification",
            return_value=_verification_result(
                has_failures=True,
                failing_partitions=["i192bigmem"],
            ),
        ),
    ):
        response = client.post(
            "/api/v1/clusters",
            headers={"Authorization": "Bearer atlas-token"},
            json={
                "cluster_name": "cluster-2",
                "region": "us-west-2",
                "region_az": "us-west-2d",
                "ssh_key_name": "omics-key",
                "s3_bucket_name": "ursa-bucket",
            },
        )

    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "Create blocked because partition verification found no current Spot availability for: i192bigmem."
    )
    assert resources.cluster_jobs == {}


def test_cluster_partition_prechecks_cover_pass_warn_fail_and_pricing() -> None:
    app, _resources = _create_test_app(admin=True)
    snapshot = {
        "captured_at": "2026-04-15T18:00:00Z",
        "cluster_config_path": "/tmp/prod_cluster.yaml",
        "points": [
            {
                "region": "us-west-2",
                "availability_zone": "us-west-2a",
                "partition": "i8",
                "instance_type": "c7i.2xlarge",
                "hourly_spot_price": 0.42,
            },
            {
                "region": "us-west-2",
                "availability_zone": "us-west-2a",
                "partition": "i192",
                "instance_type": "c7i.48xlarge",
                "hourly_spot_price": 8.4,
            },
            {
                "region": "us-west-2",
                "availability_zone": "us-west-2a",
                "partition": "i192",
                "instance_type": "m7i.48xlarge",
                "hourly_spot_price": 9.6,
            },
            {
                "region": "us-west-2",
                "availability_zone": "us-west-2b",
                "partition": "i192bigmem",
                "instance_type": "r7i.48xlarge",
                "hourly_spot_price": 12.1,
            },
        ],
    }

    with (
        TestClient(app, base_url=TEST_BASE_URL) as client,
        patch(
            "daylib_ursa.workset_api.load_daylily_partition_instance_types",
            return_value=(
                Path("/tmp/prod_cluster.yaml"),
                {
                    "i8": ["c7i.2xlarge", "m7i.2xlarge"],
                    "i192": ["c7i.48xlarge", "m7i.48xlarge"],
                    "i192bigmem": ["r7i.48xlarge"],
                },
            ),
        ),
        patch(
            "daylib_ursa.workset_api.collect_daylily_cluster_pricing_snapshot",
            return_value=snapshot,
        ),
    ):
        verification = client.post(
            "/api/v1/clusters/verify-partitions",
            headers={"Authorization": "Bearer atlas-token"},
            json={"region": "us-west-2", "region_az": "us-west-2a"},
        )
        pricing = client.post(
            "/api/v1/clusters/partition-pricing",
            headers={"Authorization": "Bearer atlas-token"},
            json={"region": "us-west-2"},
        )

    assert verification.status_code == 200, verification.text
    assert verification.json()["has_failures"] is True
    assert [item["status"] for item in verification.json()["partitions"]] == [
        "WARN",
        "PASS",
        "FAIL",
    ]
    assert verification.json()["partitions"][0]["missing_instance_types"] == ["m7i.2xlarge"]
    assert verification.json()["partitions"][1]["spot_available_instance_types"] == [
        "c7i.48xlarge",
        "m7i.48xlarge",
    ]
    assert verification.json()["partitions"][2]["summary"].startswith(
        "No configured instance types"
    )

    assert pricing.status_code == 200, pricing.text
    assert pricing.json()["captured_at"] == "2026-04-15T18:00:00Z"
    assert pricing.json()["availability_zones"] == ["us-west-2a", "us-west-2b"]
    assert [item["partition"] for item in pricing.json()["partitions"]] == [
        "i8",
        "i192",
        "i192bigmem",
    ]
    assert pricing.json()["partitions"][0]["availability_zones"][0]["count"] == 1
    assert pricing.json()["partitions"][0]["availability_zones"][0]["mean"] == 0.42
    assert pricing.json()["partitions"][0]["availability_zones"][1]["count"] == 0
    assert pricing.json()["partitions"][1]["availability_zones"][0]["count"] == 2
    assert pricing.json()["partitions"][1]["availability_zones"][0]["median"] == 9.0
    assert pricing.json()["partitions"][1]["availability_zones"][0]["min"] == 8.4
    assert pricing.json()["partitions"][1]["availability_zones"][0]["max"] == 9.6
    assert pricing.json()["partitions"][2]["availability_zones"][0]["count"] == 0
    assert pricing.json()["partitions"][2]["availability_zones"][1]["mean"] == 12.1


def test_gui_admin_pages_reject_non_admin_sessions() -> None:
    app, _resources = _create_test_app(admin=False)

    with TestClient(app, base_url=TEST_BASE_URL) as client:
        client.post("/login", data={"access_token": "atlas-token", "next_path": "/"})
        dashboard = client.get("/")
        buckets_page = client.get("/buckets")
        admin_home = client.get("/admin", follow_redirects=False)
        response = client.get("/admin/tokens", follow_redirects=False)

    assert dashboard.status_code == 200
    assert 'href="/admin"' not in dashboard.text
    assert "Admin Access" not in dashboard.text
    assert buckets_page.status_code == 200
    assert "List Buckets" not in buckets_page.text
    assert admin_home.status_code == 403
    assert response.status_code == 403
