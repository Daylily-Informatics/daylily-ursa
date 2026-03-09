from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient

from daylib_ursa.cluster_service import ClusterInfo
from daylib_ursa.config import get_settings_for_testing
from daylib_ursa.s3_bucket_validator import BucketValidationResult
from daylib_ursa.workset_api import create_app


class DummyStore:
    pass


class DummyBloomClient:
    pass


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
        _ = force_refresh
        return list(self._clusters)

    def delete_cluster(self, cluster_name: str, region: str):
        return {"cluster_name": cluster_name, "region": region}


class FakePresignClient:
    def generate_presigned_url(self, operation_name: str, Params, ExpiresIn: int):  # noqa: N803
        _ = (operation_name, Params, ExpiresIn)
        return "https://download.example/presigned"


class FakeS3Client:
    def __init__(self) -> None:
        self._objects: dict[str, bytes] = {}

    def list_objects_v2(self, **kwargs):
        prefix = str(kwargs.get("Prefix") or "")
        delimiter = kwargs.get("Delimiter")

        keys = sorted([key for key in self._objects if key.startswith(prefix)])
        if delimiter == "/":
            common: set[str] = set()
            contents = []
            for key in keys:
                rest = key[len(prefix) :]
                if "/" in rest:
                    common.add(prefix + rest.split("/", 1)[0] + "/")
                else:
                    contents.append({"Key": key, "Size": len(self._objects[key]), "LastModified": "2026-03-09T00:00:00Z"})
            return {
                "CommonPrefixes": [{"Prefix": value} for value in sorted(common)],
                "Contents": contents,
                "IsTruncated": False,
            }

        return {
            "Contents": [
                {"Key": key, "Size": len(self._objects[key]), "LastModified": "2026-03-09T00:00:00Z"}
                for key in keys
            ],
            "IsTruncated": False,
        }

    def put_object(self, **kwargs):
        key = str(kwargs.get("Key") or "")
        body = kwargs.get("Body") or b""
        if isinstance(body, str):
            body = body.encode("utf-8")
        self._objects[key] = body
        return {}

    def delete_object(self, **kwargs):
        key = str(kwargs.get("Key") or "")
        self._objects.pop(key, None)
        return {}

    def get_client_for_bucket(self, bucket: str) -> FakePresignClient:
        _ = bucket
        return FakePresignClient()


def _build_app(monkeypatch, tmp_path, *, customer_id: str):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("daylib_ursa.portal.get_cluster_service", lambda *args, **kwargs: FakeClusterService())
    monkeypatch.setattr("daylib_ursa.portal.PricingMonitor.start", lambda self: None)

    settings = get_settings_for_testing(
        enable_auth=False,
        ursa_internal_api_key="test-key",
        ursa_tapdb_mount_enabled=False,
        ursa_portal_default_customer_id=customer_id,
    )
    app = create_app(DummyStore(), bloom_client=DummyBloomClient(), settings=settings)

    portal_state = app.state.portal_state
    portal_state.s3 = FakeS3Client()

    def _fake_validate(bucket_name: str, test_prefix: str = "daylily-validation-test/"):
        _ = test_prefix
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
    return app


def test_bucket_management_endpoints_round_trip(monkeypatch, tmp_path):
    customer_id = f"cust-{uuid4().hex[:8]}"
    bucket_name = f"bucket-{uuid4().hex[:10]}"
    app = _build_app(monkeypatch, tmp_path, customer_id=customer_id)

    with TestClient(app) as client:
        link = client.post(
            f"/api/files/buckets/link?customer_id={customer_id}",
            json={"bucket_name": bucket_name, "bucket_type": "primary"},
        )
        assert link.status_code == 404, link.text

        listed = client.get(f"/api/files/buckets/list?customer_id={customer_id}")
        assert listed.status_code == 404

        validate = client.post(f"/api/files/buckets/validate?bucket_name={bucket_name}")
        assert validate.status_code == 404


def test_legacy_file_surfaces_are_removed(monkeypatch, tmp_path):
    customer_id = f"cust-{uuid4().hex[:8]}"
    bucket_name = f"bucket-{uuid4().hex[:10]}"
    app = _build_app(monkeypatch, tmp_path, customer_id=customer_id)

    with TestClient(app) as client:
        list_compat = client.get(f"/api/files/list?customer_id={customer_id}")
        assert list_compat.status_code == 404

        register = client.post(
            f"/api/files/register?customer_id={customer_id}",
            json={
                "file_metadata": {"s3_uri": f"s3://{bucket_name}/reads_R1.fastq.gz"},
                "biosample_metadata": {"biosample_id": "BS-1", "subject_id": "SUBJ-A"},
            },
        )
        assert register.status_code == 404

        search = client.post(f"/api/files/search?customer_id={customer_id}", json={"subject_id": "SUBJ-B"})
        assert search.status_code == 404

        template = client.get("/api/files/manifest/template")
        assert template.status_code == 404

        create_manifest = client.post(
            f"/api/customers/{customer_id}/manifests",
            json={"name": "m1", "tsv_content": "RUN_ID\tSAMPLE_ID\nR1\tS1\n"},
        )
        assert create_manifest.status_code == 201
        manifest_payload = create_manifest.json()
        assert "download_url" in manifest_payload
        manifest_id = manifest_payload["manifest"]["manifest_id"]

        get_manifest = client.get(f"/api/customers/{customer_id}/manifests/{manifest_id}")
        assert get_manifest.status_code == 200
        download_manifest = client.get(f"/api/customers/{customer_id}/manifests/{manifest_id}/download")
        assert download_manifest.status_code == 200
        assert download_manifest.text.startswith("RUN_ID\tSAMPLE_ID")


def test_workset_creation_from_manifest_id_and_raw_manifest_tsv(monkeypatch, tmp_path):
    customer_id = f"cust-{uuid4().hex[:8]}"
    app = _build_app(monkeypatch, tmp_path, customer_id=customer_id)

    manifest_tsv = "RUN_ID\tSAMPLE_ID\tR1_FQ\tR2_FQ\nR1\tS1\ts3://bucket/r1.fq.gz\ts3://bucket/r2.fq.gz\n"

    with TestClient(app) as client:
        create_manifest = client.post(
            f"/api/customers/{customer_id}/manifests",
            json={"name": "saved-manifest", "tsv_content": manifest_tsv},
        )
        assert create_manifest.status_code == 201
        manifest_id = create_manifest.json()["manifest"]["manifest_id"]

        from_saved = client.post(
            f"/api/customers/{customer_id}/worksets",
            headers={"X-Ursa-Admin": "true"},
            json={
                "workset_name": "ws-saved",
                "pipeline_type": "germline_wgs_snv",
                "reference_genome": "GRCh38",
                "preferred_cluster": "cluster-1",
                "manifest_id": manifest_id,
            },
        )
        assert from_saved.status_code == 200, from_saved.text
        saved_workset_id = from_saved.json()["workset_id"]
        saved_workset = client.get(f"/api/customers/{customer_id}/worksets/{saved_workset_id}")
        assert saved_workset.status_code == 200
        assert saved_workset.json()["manifest_id"] == manifest_id

        invalid_manifest = client.post(
            f"/api/customers/{customer_id}/worksets",
            headers={"X-Ursa-Admin": "true"},
            json={
                "workset_name": "ws-invalid-manifest",
                "pipeline_type": "germline_wgs_snv",
                "reference_genome": "GRCh38",
                "preferred_cluster": "cluster-1",
                "manifest_id": "missing-manifest-euid",
            },
        )
        assert invalid_manifest.status_code == 400
        assert "Manifest not found for customer" in invalid_manifest.json()["detail"]

        from_raw = client.post(
            f"/api/customers/{customer_id}/worksets",
            headers={"X-Ursa-Admin": "true"},
            json={
                "workset_name": "ws-raw",
                "pipeline_type": "germline_wgs_snv",
                "reference_genome": "GRCh38",
                "preferred_cluster": "cluster-1",
                "manifest_tsv_content": manifest_tsv,
            },
        )
        assert from_raw.status_code == 200, from_raw.text
        raw_workset_id = from_raw.json()["workset_id"]
        raw_workset = client.get(f"/api/customers/{customer_id}/worksets/{raw_workset_id}")
        assert raw_workset.status_code == 200
        raw_manifest_id = raw_workset.json()["manifest_id"]
        assert raw_manifest_id

        raw_manifest = client.get(f"/api/customers/{customer_id}/manifests/{raw_manifest_id}")
        assert raw_manifest.status_code == 200
        assert raw_manifest.json()["tsv_content"].strip() == manifest_tsv.strip()


def test_workset_new_page_includes_manifest_euid_field(monkeypatch, tmp_path):
    customer_id = f"cust-{uuid4().hex[:8]}"
    app = _build_app(monkeypatch, tmp_path, customer_id=customer_id)

    with TestClient(app) as client:
        response = client.get("/portal/worksets/new")
        assert response.status_code == 200
        assert "Manifest EUID" in response.text
        assert 'id="saved-manifest-id"' in response.text
        assert 'id="create-cluster-modal" class="workset-modal-overlay d-none"' in response.text


def test_bucket_mutation_guardrails_for_upload_folder_and_delete(monkeypatch, tmp_path):
    customer_id = f"cust-{uuid4().hex[:8]}"
    bucket_name = f"bucket-{uuid4().hex[:10]}"
    app = _build_app(monkeypatch, tmp_path, customer_id=customer_id)

    with TestClient(app) as client:
        link = client.post(
            f"/api/files/buckets/link?customer_id={customer_id}",
            json={
                "bucket_name": bucket_name,
                "bucket_type": "primary",
                "prefix_restriction": "allowed/",
            },
        )
        assert link.status_code == 404

        discover = client.post(
            "/api/files/buckets/does-not-exist/discover",
            params={"customer_id": customer_id, "prefix": "", "max_files": 10},
        )
        assert discover.status_code == 404

        delete_file = client.delete(
            "/api/files/buckets/does-not-exist/files",
            params={"customer_id": customer_id, "file_key": "allowed/registered.fastq.gz"},
        )
        assert delete_file.status_code == 404
