from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient

from daylib_ursa.analysis_store import (
    AnalysisArtifact,
    AnalysisRecord,
    AnalysisState,
    ReviewState,
    RunResolution,
)
from daylib_ursa.cluster_service import ClusterInfo
from daylib_ursa.config import Settings
from daylib_ursa.portal_auth import PORTAL_SESSION_COOKIE_NAME, encode_portal_session
from daylib_ursa.s3_bucket_validator import BucketValidationResult
from daylib_ursa.workset_api import create_app


class DummyStore:
    def __init__(self) -> None:
        self.record = AnalysisRecord(
            analysis_euid="AN-1",
            run_euid="RUN-1",
            flowcell_id="FLOW-1",
            lane="1",
            library_barcode="LIB-1",
            sequenced_library_assignment_euid="SQA-1",
            atlas_tenant_id="TEN-1",
            atlas_trf_euid="TRF-1",
            atlas_test_euid="TST-1",
            atlas_test_process_item_euid="TPC-1",
            analysis_type="beta-default",
            state=AnalysisState.INGESTED.value,
            review_state=ReviewState.PENDING.value,
            result_status="PENDING",
            run_folder="s3://analysis-bucket/RUN-1/",
            artifact_bucket="analysis-bucket",
            result_payload={},
            metadata={},
            created_at="2026-03-07T00:00:00Z",
            updated_at="2026-03-07T00:00:00Z",
            atlas_return={},
            artifacts=[],
        )

    def ingest_analysis(self, **kwargs):
        resolution = kwargs["resolution"]
        self.record = replace(
            self.record,
            run_euid=resolution.run_euid,
            flowcell_id=resolution.flowcell_id,
            lane=resolution.lane,
            library_barcode=resolution.library_barcode,
            analysis_type=kwargs["analysis_type"],
            artifact_bucket=kwargs["artifact_bucket"],
        )
        return self.record

    def get_analysis(self, analysis_euid: str):
        return self.record if analysis_euid == self.record.analysis_euid else None

    def update_analysis_state(self, analysis_euid: str, **kwargs):
        assert analysis_euid == self.record.analysis_euid
        self.record = replace(
            self.record,
            state=kwargs["state"].value,
            result_status=kwargs.get("result_status") or self.record.result_status,
            result_payload=kwargs.get("result_payload") or self.record.result_payload,
            metadata={**self.record.metadata, **kwargs.get("metadata", {})},
        )
        return self.record

    def add_artifact(self, analysis_euid: str, **kwargs):
        assert analysis_euid == self.record.analysis_euid
        artifact = AnalysisArtifact(
            artifact_euid="AF-1",
            artifact_type=kwargs["artifact_type"],
            storage_uri=kwargs["storage_uri"],
            filename=kwargs["filename"],
            mime_type=kwargs.get("mime_type"),
            checksum_sha256=kwargs.get("checksum_sha256"),
            size_bytes=kwargs.get("size_bytes"),
            created_at="2026-03-07T02:00:00Z",
            metadata=kwargs.get("metadata") or {},
        )
        self.record = replace(self.record, artifacts=[artifact])
        return artifact

    def set_review_state(self, analysis_euid: str, **kwargs):
        assert analysis_euid == self.record.analysis_euid
        self.record = replace(
            self.record,
            review_state=kwargs["review_state"].value,
            state=AnalysisState.REVIEWED.value,
        )
        return self.record

    def mark_returned(self, analysis_euid: str, **kwargs):
        assert analysis_euid == self.record.analysis_euid
        self.record = replace(
            self.record,
            state=AnalysisState.RETURNED.value,
            atlas_return=kwargs["atlas_return"],
        )
        return self.record


class DummyBloomClient:
    def resolve_run_assignment(self, run_euid: str, flowcell_id: str, lane: str, library_barcode: str):
        return RunResolution(
            run_euid=run_euid,
            flowcell_id=flowcell_id,
            lane=lane,
            library_barcode=library_barcode,
            sequenced_library_assignment_euid="SQA-1",
            atlas_tenant_id="TEN-1",
            atlas_trf_euid="TRF-1",
            atlas_test_euid="TST-1",
            atlas_test_process_item_euid="TPC-1",
        )


class DummyAtlasClient:
    def return_analysis_result(self, **kwargs):
        return {"status": "accepted", "atlas_result_id": "AR-1"}


class FakeClusterService:
    def __init__(self) -> None:
        self._clusters = [
            ClusterInfo(
                cluster_name="cluster-1",
                region="us-west-2",
                cluster_status="CREATE_COMPLETE",
                compute_fleet_status="RUNNING",
            )
        ]

    def get_all_clusters(self, force_refresh: bool = False):
        return list(self._clusters)

    def delete_cluster(self, cluster_name: str, region: str):
        return {"cluster_name": cluster_name, "region": region}


class FakeRegionAwareS3Client:
    def __init__(self, default_region: str, profile: str | None = None) -> None:
        self.default_region = default_region
        self.profile = profile

    def get_bucket_region(self, bucket_name: str) -> str:
        return "us-west-2"


class FakePresignClient:
    def generate_presigned_url(self, operation_name: str, Params, ExpiresIn: int) -> str:  # noqa: N803
        return "https://download.example/presigned"


class FakeS3Client:
    def list_objects_v2(self, **kwargs):
        prefix = str(kwargs.get("Prefix") or "")
        return {
            "CommonPrefixes": [{"Prefix": f"{prefix}folder/"}],
            "Contents": [
                {
                    "Key": f"{prefix}reads_R1.fastq.gz",
                    "Size": 123,
                    "LastModified": "2026-03-08T12:00:00Z",
                }
            ],
            "IsTruncated": False,
        }

    def put_object(self, **kwargs):
        return {}

    def delete_object(self, **kwargs):
        return {}

    def get_client_for_bucket(self, bucket: str) -> FakePresignClient:
        return FakePresignClient()


def _build_app(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    fake_service = FakeClusterService()
    monkeypatch.setattr("daylib_ursa.portal.get_cluster_service", lambda *args, **kwargs: fake_service)
    monkeypatch.setattr(
        "daylib_ursa.portal._load_create_options",
        lambda settings, region: {"keypairs": ["key-a"], "buckets": ["bucket-a"]},
    )
    monkeypatch.setattr(
        "daylib_ursa.portal.start_create_job",
        lambda **kwargs: SimpleNamespace(
            job_id="job-1",
            cluster_name=kwargs["cluster_name"],
            region_az=kwargs["region_az"],
            status="running",
        ),
    )
    monkeypatch.setattr(
        "daylib_ursa.portal._validate_cluster_create_identity",
        lambda settings, *, region_az: {"account_id": "000000000000", "arn": "arn:aws:iam::000000000000:user/test"},
    )
    monkeypatch.setattr("daylib_ursa.portal.list_cluster_create_jobs", lambda limit=20: [{"job_id": "job-1"}])
    monkeypatch.setattr("daylib_ursa.portal.tail_job_log", lambda job_id, lines=200: "log")
    monkeypatch.setattr("daylib_ursa.portal.PricingMonitor.start", lambda self: None)
    monkeypatch.setattr("daylib_ursa.portal.RegionAwareS3Client", FakeRegionAwareS3Client)

    settings = Settings(
        cors_origins="*",
        ursa_internal_api_key="test-key",
        bloom_base_url="https://bloom.example",
        atlas_base_url="https://atlas.example",
        atlas_internal_api_key="atlas-key",
        enable_auth=True,
        cognito_domain="daylily-ursa-5r8giqv5p.auth.us-west-2.amazoncognito.com",
        cognito_app_client_id="34g35v8tpurbe309a8e5t5ot7i",
        cognito_user_pool_id="us-west-2_5r8gIqV5P",
        cognito_region="us-west-2",
        ursa_tapdb_mount_enabled=False,
    )
    app = create_app(
        DummyStore(),
        bloom_client=DummyBloomClient(),
        atlas_client=DummyAtlasClient(),
        settings=settings,
    )

    app.state.pricing_monitor.get_snapshot_payload = lambda **kwargs: {"snapshots": []}
    app.state.pricing_monitor.queue_capture = lambda **kwargs: {"run_id": "PC-1", "status": "queued"}

    portal_state = app.state.portal_state
    portal_state.s3 = FakeS3Client()

    def _fake_validate(bucket_name: str, test_prefix: str = "daylily-validation-test/"):
        return BucketValidationResult(
            bucket_name=bucket_name,
            exists=True,
            accessible=True,
            can_read=True,
            can_write=True,
            can_list=True,
            region="us-west-2",
        )

    portal_state.bucket_manager.validator.validate_bucket = _fake_validate
    portal_state.bucket_manager.validator.get_setup_instructions = lambda bucket_name, result: []

    return app, settings


def _discover_routes(app) -> set[tuple[str, str]]:
    discovered: set[tuple[str, str]] = set()
    for route in app.routes:
        path = getattr(route, "path", "")
        methods = getattr(route, "methods", set())
        if not path or not methods:
            continue
        if path.startswith("/openapi") or path.startswith("/docs") or path.startswith("/redoc"):
            continue
        if path == "/static":
            continue
        for method in methods:
            if method in {"HEAD", "OPTIONS"}:
                continue
            discovered.add((method, path))
    return discovered


def test_every_api_and_gui_route_is_exercised(monkeypatch, tmp_path):
    app, settings = _build_app(monkeypatch, tmp_path)
    customer_id = "default-customer"
    admin_headers = {"X-API-Key": "test-key", "X-Ursa-Admin": "true"}
    covered_routes: set[tuple[str, str]] = set()

    with TestClient(app) as client:
        assert client.get("/", follow_redirects=False).status_code == 307
        covered_routes.add(("GET", "/"))

        assert client.get("/portal/login").status_code == 200
        covered_routes.add(("GET", "/portal/login"))

        assert client.get("/auth/callback", follow_redirects=False).status_code == 307
        covered_routes.add(("GET", "/auth/callback"))

        assert client.get("/healthz").status_code == 200
        covered_routes.add(("GET", "/healthz"))

        client.cookies.set(
            PORTAL_SESSION_COOKIE_NAME,
            encode_portal_session(
                settings.session_secret_key,
                {
                    "logged_in": True,
                    "is_admin": True,
                    "user_email": "route-coverage@lsmc.bio",
                    "customer_id": customer_id,
                    "customer_name": "Route Coverage",
                    "s3_bucket": "bucket-a",
                },
            ),
        )

        link_bucket_response = client.post(
            f"/api/files/buckets/link?customer_id={customer_id}",
            json={"bucket_name": "bucket-a", "bucket_type": "primary"},
        )
        assert link_bucket_response.status_code == 200
        bucket_id = link_bucket_response.json()["bucket_id"]
        covered_routes.add(("POST", "/api/files/buckets/link"))

        register_file_response = client.post(
            f"/api/files/register?customer_id={customer_id}",
            json={
                "file_metadata": {
                    "s3_uri": "s3://bucket-a/reads_R1.fastq.gz",
                    "file_size_bytes": 123,
                    "file_format": "fastq.gz",
                },
                "biosample_metadata": {"biosample_id": "BS-1", "subject_id": "SUBJ-1"},
                "sequencing_metadata": {"platform": "NOVASEQX"},
            },
        )
        assert register_file_response.status_code == 200
        file_id = register_file_response.json()["file_id"]
        covered_routes.add(("POST", "/api/files/register"))

        create_fileset_response = client.post(
            f"/api/files/filesets?customer_id={customer_id}",
            json={"name": "set-1", "description": "route-coverage", "tags": ["qc"], "file_ids": [file_id]},
        )
        assert create_fileset_response.status_code == 200
        fileset_id = create_fileset_response.json()["fileset_id"]
        covered_routes.add(("POST", "/api/files/filesets"))

        create_manifest_response = client.post(
            f"/api/customers/{customer_id}/manifests",
            json={
                "name": "manifest-1",
                "description": "route-coverage",
                "tsv_content": "RUN_ID\tSAMPLE_ID\nRUN-1\tSAMPLE-1\n",
            },
        )
        assert create_manifest_response.status_code == 201
        manifest_id = create_manifest_response.json()["manifest"]["manifest_id"]
        covered_routes.add(("POST", "/api/customers/{customer_id}/manifests"))

        create_workset_response = client.post(
            f"/api/customers/{customer_id}/worksets",
            headers=admin_headers,
            json={
                "workset_name": "route-coverage",
                "pipeline_type": "germline_wgs_snv",
                "reference_genome": "GRCh38",
                "preferred_cluster": "cluster-1",
                "target_region": "us-west-2",
            },
        )
        assert create_workset_response.status_code == 200
        workset_id = create_workset_response.json()["workset_id"]
        covered_routes.add(("POST", "/api/customers/{customer_id}/worksets"))

        assert client.get("/portal").status_code == 200
        covered_routes.add(("GET", "/portal"))
        assert client.get("/portal/manifest-generator").status_code == 200
        covered_routes.add(("GET", "/portal/manifest-generator"))
        assert client.get("/portal/files").status_code == 200
        covered_routes.add(("GET", "/portal/files"))
        assert client.get("/portal/files/register").status_code == 200
        covered_routes.add(("GET", "/portal/files/register"))
        assert (
            client.post(
                "/portal/files/register",
                json={
                    "bucket_id": bucket_id,
                    "prefix": "",
                    "selected_keys": ["reads_R1.fastq.gz"],
                    "biosample_id": "BS-2",
                    "subject_id": "SUBJ-2",
                    "sequencing_platform": "NOVASEQX",
                },
            ).status_code
            == 200
        )
        covered_routes.add(("POST", "/portal/files/register"))
        assert client.get("/portal/files/upload").status_code == 200
        covered_routes.add(("GET", "/portal/files/upload"))
        assert (
            client.post(
                "/portal/files/upload",
                data={"bucket_id": bucket_id, "prefix": "uploads/", "auto_register": "false"},
                files={"file": ("coverage.txt", b"coverage", "text/plain")},
            ).status_code
            == 200
        )
        covered_routes.add(("POST", "/portal/files/upload"))
        assert client.get("/portal/files/buckets").status_code == 200
        covered_routes.add(("GET", "/portal/files/buckets"))
        assert client.get(f"/portal/files/browse/{bucket_id}", params={"prefix": ""}).status_code == 200
        covered_routes.add(("GET", "/portal/files/browse/{bucket_id}"))
        assert client.get("/portal/files/browser", params={"bucket_id": bucket_id, "prefix": ""}).status_code == 200
        covered_routes.add(("GET", "/portal/files/browser"))
        assert client.get("/portal/files/filesets").status_code == 404
        covered_routes.add(("GET", "/portal/files/filesets"))
        assert client.get(f"/portal/files/filesets/{fileset_id}").status_code == 404
        covered_routes.add(("GET", "/portal/files/filesets/{fileset_id}"))
        assert client.get(f"/portal/files/{file_id}").status_code == 200
        covered_routes.add(("GET", "/portal/files/{file_id}"))
        assert client.get(f"/portal/files/{file_id}/edit").status_code == 200
        covered_routes.add(("GET", "/portal/files/{file_id}/edit"))
        assert client.get("/portal/usage").status_code == 200
        covered_routes.add(("GET", "/portal/usage"))
        assert client.get("/portal/usage/export").status_code == 200
        covered_routes.add(("GET", "/portal/usage/export"))
        assert client.get("/portal/clusters").status_code == 200
        covered_routes.add(("GET", "/portal/clusters"))
        assert client.get("/portal/worksets/new").status_code == 200
        covered_routes.add(("GET", "/portal/worksets/new"))
        assert client.get("/portal/worksets").status_code == 200
        covered_routes.add(("GET", "/portal/worksets"))
        assert client.get(f"/portal/worksets/{workset_id}").status_code == 200
        covered_routes.add(("GET", "/portal/worksets/{workset_id}"))
        assert client.get("/portal/account").status_code == 200
        covered_routes.add(("GET", "/portal/account"))
        assert (
            client.post(
                "/api/account/preferences",
                json={"display_timezone": "America/Los_Angeles"},
            ).status_code
            == 200
        )
        covered_routes.add(("POST", "/api/account/preferences"))
        assert client.get("/portal/admin/users", headers=admin_headers).status_code == 200
        covered_routes.add(("GET", "/portal/admin/users"))
        assert client.get("/portal/monitor", headers=admin_headers).status_code == 200
        covered_routes.add(("GET", "/portal/monitor"))
        assert client.get("/portal/biospecimen").status_code == 404

        assert client.get("/api/clusters", headers=admin_headers).status_code == 200
        covered_routes.add(("GET", "/api/clusters"))
        assert client.get("/api/monitor/status", headers=admin_headers).status_code == 200
        covered_routes.add(("GET", "/api/monitor/status"))
        assert client.get("/api/monitor/logs", headers=admin_headers).status_code == 200
        covered_routes.add(("GET", "/api/monitor/logs"))
        assert (
            client.delete(
                "/api/clusters/cluster-1",
                params={"region": "us-west-2"},
                headers=admin_headers,
            ).status_code
            == 200
        )
        covered_routes.add(("DELETE", "/api/clusters/{cluster_name}"))
        assert (
            client.get(
                "/api/clusters/create/options",
                params={"region": "us-west-2"},
                headers=admin_headers,
            ).status_code
            == 200
        )
        covered_routes.add(("GET", "/api/clusters/create/options"))
        assert (
            client.post(
                "/api/clusters/create",
                headers=admin_headers,
                json={
                    "region_az": "us-west-2a",
                    "cluster_name": "cluster-2",
                    "ssh_key_name": "key-a",
                    "s3_bucket_name": "bucket-a",
                },
            ).status_code
            == 200
        )
        covered_routes.add(("POST", "/api/clusters/create"))
        assert client.get("/api/clusters/create/jobs", headers=admin_headers).status_code == 200
        covered_routes.add(("GET", "/api/clusters/create/jobs"))
        assert (
            client.get(
                "/api/clusters/create/jobs/job-1/logs",
                params={"lines": 10},
                headers=admin_headers,
            ).status_code
            == 200
        )
        covered_routes.add(("GET", "/api/clusters/create/jobs/{job_id}/logs"))
        assert client.get("/api/pricing-snapshots", headers=admin_headers).status_code == 200
        covered_routes.add(("GET", "/api/pricing-snapshots"))
        assert client.post("/api/pricing-snapshots/run", headers=admin_headers).status_code == 200
        covered_routes.add(("POST", "/api/pricing-snapshots/run"))

        assert client.get(f"/api/customers/{customer_id}/dashboard/stats").status_code == 200
        covered_routes.add(("GET", "/api/customers/{customer_id}/dashboard/stats"))
        assert client.get(f"/api/customers/{customer_id}/dashboard/activity").status_code == 200
        covered_routes.add(("GET", "/api/customers/{customer_id}/dashboard/activity"))
        assert client.get(f"/api/customers/{customer_id}/dashboard/cost-history").status_code == 200
        covered_routes.add(("GET", "/api/customers/{customer_id}/dashboard/cost-history"))
        assert client.get(f"/api/customers/{customer_id}/dashboard/cost-breakdown").status_code == 200
        covered_routes.add(("GET", "/api/customers/{customer_id}/dashboard/cost-breakdown"))
        assert client.get(f"/api/customers/{customer_id}/usage").status_code == 200
        covered_routes.add(("GET", "/api/customers/{customer_id}/usage"))
        assert client.get(f"/api/customers/{customer_id}/usage/details").status_code == 200
        covered_routes.add(("GET", "/api/customers/{customer_id}/usage/details"))
        assert client.get(f"/api/customers/{customer_id}/manifests").status_code == 200
        covered_routes.add(("GET", "/api/customers/{customer_id}/manifests"))
        assert client.get(f"/api/customers/{customer_id}/manifests/{manifest_id}").status_code == 200
        covered_routes.add(("GET", "/api/customers/{customer_id}/manifests/{manifest_id}"))
        assert client.get(f"/api/customers/{customer_id}/manifests/{manifest_id}/download").status_code == 200
        covered_routes.add(("GET", "/api/customers/{customer_id}/manifests/{manifest_id}/download"))
        assert client.get(f"/api/customers/{customer_id}/worksets").status_code == 200
        covered_routes.add(("GET", "/api/customers/{customer_id}/worksets"))
        assert client.get(f"/api/customers/{customer_id}/worksets/{workset_id}").status_code == 200
        covered_routes.add(("GET", "/api/customers/{customer_id}/worksets/{workset_id}"))
        assert client.get(f"/api/customers/{customer_id}/worksets/{workset_id}/logs").status_code == 200
        covered_routes.add(("GET", "/api/customers/{customer_id}/worksets/{workset_id}/logs"))

        assert client.get("/api/files").status_code == 200
        covered_routes.add(("GET", "/api/files"))
        assert client.get(f"/api/files/list?customer_id={customer_id}").status_code == 200
        covered_routes.add(("GET", "/api/files/list"))
        assert client.post("/api/files/search", json={"subject_id": "SUBJ-1"}).status_code == 200
        covered_routes.add(("POST", "/api/files/search"))
        assert (
            client.post(
                f"/api/files/bulk-import?customer_id={customer_id}",
                json={
                    "files": [
                        {
                            "file_metadata": {
                                "s3_uri": "s3://bucket-a/reads_R2.fastq.gz",
                                "file_size_bytes": 125,
                                "file_format": "fastq.gz",
                            },
                            "biosample_metadata": {"biosample_id": "BS-2", "subject_id": "SUBJ-2"},
                            "sequencing_metadata": {"platform": "NOVASEQX"},
                        }
                    ],
                    "fileset_name": "import-set",
                },
            ).status_code
            == 200
        )
        covered_routes.add(("POST", "/api/files/bulk-import"))
        assert client.get("/api/files/manifest/template").status_code == 200
        covered_routes.add(("GET", "/api/files/manifest/template"))
        assert client.get(f"/api/files/buckets/list?customer_id={customer_id}").status_code == 200
        covered_routes.add(("GET", "/api/files/buckets/list"))
        assert client.post("/api/files/buckets/validate?bucket_name=bucket-a").status_code == 200
        covered_routes.add(("POST", "/api/files/buckets/validate"))
        assert client.get(f"/api/files/buckets/{bucket_id}").status_code == 200
        covered_routes.add(("GET", "/api/files/buckets/{bucket_id}"))
        assert (
            client.patch(
                f"/api/files/buckets/{bucket_id}",
                json={"description": "updated"},
            ).status_code
            == 200
        )
        covered_routes.add(("PATCH", "/api/files/buckets/{bucket_id}"))
        assert client.post(f"/api/files/buckets/{bucket_id}/revalidate").status_code == 200
        covered_routes.add(("POST", "/api/files/buckets/{bucket_id}/revalidate"))
        assert client.get(f"/api/files/buckets/{bucket_id}/browse", params={"prefix": ""}).status_code == 200
        covered_routes.add(("GET", "/api/files/buckets/{bucket_id}/browse"))
        assert (
            client.post(
                f"/api/files/buckets/{bucket_id}/discover?customer_id={customer_id}&prefix=&max_files=10",
            ).status_code
            == 200
        )
        covered_routes.add(("POST", "/api/files/buckets/{bucket_id}/discover"))
        assert (
            client.post(
                f"/api/files/buckets/{bucket_id}/folders?customer_id={customer_id}&prefix=tmp/",
                json={"folder_name": "new-folder"},
            ).status_code
            == 200
        )
        covered_routes.add(("POST", "/api/files/buckets/{bucket_id}/folders"))
        assert (
            client.delete(
                f"/api/files/buckets/{bucket_id}/files?customer_id={customer_id}&file_key=tmp/new-folder/file.txt",
            ).status_code
            == 200
        )
        covered_routes.add(("DELETE", "/api/files/buckets/{bucket_id}/files"))
        assert client.get("/api/files/filesets", params={"customer_id": customer_id}).status_code == 200
        covered_routes.add(("GET", "/api/files/filesets"))
        assert client.patch(f"/api/files/filesets/{fileset_id}", json={"description": "patched"}).status_code == 200
        covered_routes.add(("PATCH", "/api/files/filesets/{fileset_id}"))
        assert client.post(f"/api/files/filesets/{fileset_id}/add-files", json=[file_id]).status_code == 200
        covered_routes.add(("POST", "/api/files/filesets/{fileset_id}/add-files"))
        assert client.post(f"/api/files/filesets/{fileset_id}/remove-files", json=[file_id]).status_code == 200
        covered_routes.add(("POST", "/api/files/filesets/{fileset_id}/remove-files"))
        clone_response = client.post(
            f"/api/files/filesets/{fileset_id}/clone",
            json={"new_name": "set-1-copy"},
        )
        assert clone_response.status_code == 200
        clone_fileset_id = clone_response.json()["fileset_id"]
        covered_routes.add(("POST", "/api/files/filesets/{fileset_id}/clone"))
        assert client.post(f"/api/files/filesets/{fileset_id}/manifest").status_code == 200
        covered_routes.add(("POST", "/api/files/filesets/{fileset_id}/manifest"))
        assert client.post(f"/api/files/{file_id}/manifest").status_code == 200
        covered_routes.add(("POST", "/api/files/{file_id}/manifest"))
        assert client.delete(f"/api/files/filesets/{clone_fileset_id}").status_code == 200
        covered_routes.add(("DELETE", "/api/files/filesets/{fileset_id}"))
        assert client.get(f"/api/files/{file_id}/download").status_code == 200
        covered_routes.add(("GET", "/api/files/{file_id}/download"))
        assert client.post(f"/api/files/{file_id}/tags", json={"tag": "release"}).status_code == 200
        covered_routes.add(("POST", "/api/files/{file_id}/tags"))
        assert client.put(f"/api/files/{file_id}/tags", json={"tags": ["release", "verified"]}).status_code == 200
        covered_routes.add(("PUT", "/api/files/{file_id}/tags"))
        assert client.delete(f"/api/files/{file_id}/tags/release").status_code == 200
        covered_routes.add(("DELETE", "/api/files/{file_id}/tags/{tag}"))
        assert (
            client.patch(
                f"/api/files/{file_id}",
                json={"file_metadata": {"file_format": "bam"}},
            ).status_code
            == 200
        )
        covered_routes.add(("PATCH", "/api/files/{file_id}"))
        assert (
            client.patch(
                f"/api/v1/files/{file_id}",
                json={"file_metadata": {"file_format": "bam"}},
            ).status_code
            == 200
        )
        covered_routes.add(("PATCH", "/api/v1/files/{file_id}"))
        assert client.get("/api/s3/discover-samples", params={"customer_id": customer_id}).status_code == 200
        covered_routes.add(("GET", "/api/s3/discover-samples"))
        assert client.get("/api/s3/bucket-region/bucket-a").status_code == 200
        covered_routes.add(("GET", "/api/s3/bucket-region/{bucket_name}"))
        assert client.post(f"/api/files/buckets/{bucket_id}/unlink").status_code == 200
        covered_routes.add(("POST", "/api/files/buckets/{bucket_id}/unlink"))

        ingest_response = client.post(
            "/api/analyses/ingest",
            headers={
                "X-API-Key": "test-key",
                "Idempotency-Key": "idem-ingest-1",
            },
            json={
                "run_euid": "RUN-1",
                "flowcell_id": "FLOW-1",
                "lane": "1",
                "library_barcode": "LIB-1",
                "analysis_type": "germline",
                "artifact_bucket": "analysis-bucket",
                "input_files": ["s3://analysis-bucket/RUN-1/read1.fastq.gz"],
                "metadata": {"pipeline": "beta"},
            },
        )
        assert ingest_response.status_code == 201
        analysis_euid = ingest_response.json()["analysis_euid"]
        covered_routes.add(("POST", "/api/analyses/ingest"))
        assert client.get(f"/api/analyses/{analysis_euid}").status_code == 200
        covered_routes.add(("GET", "/api/analyses/{analysis_euid}"))
        assert (
            client.post(
                f"/api/analyses/{analysis_euid}/status",
                headers={"X-API-Key": "test-key"},
                json={"state": "RUNNING", "metadata": {}},
            ).status_code
            == 200
        )
        covered_routes.add(("POST", "/api/analyses/{analysis_euid}/status"))
        assert (
            client.post(
                f"/api/analyses/{analysis_euid}/artifacts",
                headers={"X-API-Key": "test-key"},
                json={
                    "artifact_type": "REPORT",
                    "storage_uri": "s3://analysis-bucket/RUN-1/report.json",
                    "filename": "report.json",
                    "metadata": {},
                },
            ).status_code
            == 201
        )
        covered_routes.add(("POST", "/api/analyses/{analysis_euid}/artifacts"))
        assert (
            client.post(
                f"/api/analyses/{analysis_euid}/review",
                headers={"X-API-Key": "test-key"},
                json={"review_state": "APPROVED", "notes": "ok"},
            ).status_code
            == 200
        )
        covered_routes.add(("POST", "/api/analyses/{analysis_euid}/review"))
        assert (
            client.post(
                f"/api/analyses/{analysis_euid}/return",
                headers={"X-API-Key": "test-key", "Idempotency-Key": "idem-return-1"},
                json={"result_payload": {"status": "ok"}, "result_status": "COMPLETED"},
            ).status_code
            == 200
        )
        covered_routes.add(("POST", "/api/analyses/{analysis_euid}/return"))

        logout_response = client.get("/portal/logout", follow_redirects=False)
        assert logout_response.status_code == 307
        parsed_logout = urlparse(logout_response.headers["location"])
        assert parsed_logout.path.endswith("/logout")
        logout_query = parse_qs(parsed_logout.query)
        assert logout_query["client_id"][0] == "34g35v8tpurbe309a8e5t5ot7i"
        assert logout_query["logout_uri"][0].endswith("://testserver/")
        covered_routes.add(("GET", "/portal/logout"))

    discovered_routes = _discover_routes(app)
    missing = discovered_routes - covered_routes
    assert not missing, f"Uncovered routes: {sorted(missing)}"
