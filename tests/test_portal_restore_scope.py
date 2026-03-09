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


def _register_file(client: TestClient, *, customer_id: str, s3_uri: str, subject_id: str = "SUBJ-1") -> str:
    response = client.post(
        f"/api/files/register?customer_id={customer_id}",
        json={
            "file_metadata": {
                "s3_uri": s3_uri,
                "file_size_bytes": 123,
                "file_format": "fastq.gz",
            },
            "biosample_metadata": {"biosample_id": "BS-1", "subject_id": subject_id},
            "sequencing_metadata": {"platform": "NOVASEQX"},
        },
    )
    assert response.status_code == 200, response.text
    return str(response.json()["file_id"])


def test_bucket_management_endpoints_round_trip(monkeypatch, tmp_path):
    customer_id = f"cust-{uuid4().hex[:8]}"
    bucket_name = f"bucket-{uuid4().hex[:10]}"
    app = _build_app(monkeypatch, tmp_path, customer_id=customer_id)

    with TestClient(app) as client:
        link = client.post(
            f"/api/files/buckets/link?customer_id={customer_id}",
            json={"bucket_name": bucket_name, "bucket_type": "primary"},
        )
        assert link.status_code == 200, link.text
        bucket_id = str(link.json()["bucket_id"])

        listed = client.get(f"/api/files/buckets/list?customer_id={customer_id}")
        assert listed.status_code == 200
        assert any(item["bucket_id"] == bucket_id for item in listed.json()["buckets"])

        detail = client.get(f"/api/files/buckets/{bucket_id}")
        assert detail.status_code == 200
        assert detail.json()["bucket_name"] == bucket_name

        patched = client.patch(f"/api/files/buckets/{bucket_id}", json={"description": "updated"})
        assert patched.status_code == 200
        assert patched.json()["description"] == "updated"

        revalidated = client.post(f"/api/files/buckets/{bucket_id}/revalidate")
        assert revalidated.status_code == 200
        assert revalidated.json()["is_valid"] is True

        unlinked = client.post(f"/api/files/buckets/{bucket_id}/unlink")
        assert unlinked.status_code == 200
        assert unlinked.json()["success"] is True


def test_file_compat_aliases_bulk_import_search_and_manifest_surfaces(monkeypatch, tmp_path):
    customer_id = f"cust-{uuid4().hex[:8]}"
    bucket_name = f"bucket-{uuid4().hex[:10]}"
    app = _build_app(monkeypatch, tmp_path, customer_id=customer_id)

    with TestClient(app) as client:
        link = client.post(
            f"/api/files/buckets/link?customer_id={customer_id}",
            json={"bucket_name": bucket_name, "bucket_type": "primary"},
        )
        assert link.status_code == 200

        file_id = _register_file(
            client,
            customer_id=customer_id,
            s3_uri=f"s3://{bucket_name}/reads_R1.fastq.gz",
            subject_id="SUBJ-A",
        )

        list_compat = client.get(f"/api/files/list?customer_id={customer_id}")
        assert list_compat.status_code == 200
        assert list_compat.json()["file_count"] >= 1

        patch_compat = client.patch(
            f"/api/files/{file_id}?customer_id={customer_id}",
            json={"file_metadata": {"file_format": "bam"}},
        )
        assert patch_compat.status_code == 200
        assert patch_compat.json()["file_format"] == "bam"

        replace_tags = client.put(
            f"/api/files/{file_id}/tags?customer_id={customer_id}",
            json={"tags": ["release", "verified"]},
        )
        assert replace_tags.status_code == 200
        assert replace_tags.json()["tags"] == ["release", "verified"]

        file_manifest = client.post(f"/api/files/{file_id}/manifest")
        assert file_manifest.status_code == 200
        assert file_manifest.text.startswith("RUN_ID\tSAMPLE_ID")

        create_fileset = client.post(
            f"/api/files/filesets?customer_id={customer_id}",
            json={"name": "set-1", "file_ids": [file_id], "tags": []},
        )
        assert create_fileset.status_code == 200
        fileset_id = create_fileset.json()["fileset_id"]

        fileset_manifest = client.post(f"/api/files/filesets/{fileset_id}/manifest")
        assert fileset_manifest.status_code == 200
        assert fileset_manifest.text.startswith("RUN_ID\tSAMPLE_ID")

        bulk = client.post(
            f"/api/files/bulk-import?customer_id={customer_id}",
            json={
                "files": [
                    {
                        "file_metadata": {
                            "s3_uri": f"s3://{bucket_name}/reads_R2.fastq.gz",
                            "file_size_bytes": 124,
                            "file_format": "fastq.gz",
                        },
                        "biosample_metadata": {"biosample_id": "BS-2", "subject_id": "SUBJ-B"},
                        "sequencing_metadata": {"platform": "NOVASEQX"},
                    }
                ],
                "fileset_name": "imported",
            },
        )
        assert bulk.status_code == 200
        assert bulk.json()["imported_count"] == 1

        search = client.post(f"/api/files/search?customer_id={customer_id}", json={"subject_id": "SUBJ-B"})
        assert search.status_code == 200
        assert search.json()["file_count"] >= 1

        template = client.get("/api/files/manifest/template")
        assert template.status_code == 200
        assert template.text.startswith("RUN_ID\tSAMPLE_ID")

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
        assert link.status_code == 200
        bucket_id = str(link.json()["bucket_id"])

        make_read_only = client.patch(f"/api/files/buckets/{bucket_id}", json={"read_only": True})
        assert make_read_only.status_code == 200
        read_only_folder = client.post(
            f"/api/files/buckets/{bucket_id}/folders?customer_id={customer_id}&prefix=allowed/",
            json={"folder_name": "x"},
        )
        assert read_only_folder.status_code == 403

        writable = client.patch(f"/api/files/buckets/{bucket_id}", json={"read_only": False, "can_write": True})
        assert writable.status_code == 200

        blocked_prefix_upload = client.post(
            "/portal/files/upload",
            data={"bucket_id": bucket_id, "prefix": "blocked/", "auto_register": "false"},
            files={"file": ("reads.fastq.gz", b"data", "application/gzip")},
        )
        assert blocked_prefix_upload.status_code == 403

        allowed_upload = client.post(
            "/portal/files/upload",
            data={"bucket_id": bucket_id, "prefix": "allowed/", "auto_register": "false"},
            files={"file": ("reads.fastq.gz", b"data", "application/gzip")},
        )
        assert allowed_upload.status_code == 200

        _register_file(
            client,
            customer_id=customer_id,
            s3_uri=f"s3://{bucket_name}/allowed/registered.fastq.gz",
            subject_id="SUBJ-X",
        )

        delete_registered = client.delete(
            f"/api/files/buckets/{bucket_id}/files?customer_id={customer_id}&file_key=allowed/registered.fastq.gz"
        )
        assert delete_registered.status_code == 409

        delete_outside_restriction = client.delete(
            f"/api/files/buckets/{bucket_id}/files?customer_id={customer_id}&file_key=blocked/file.fastq.gz"
        )
        assert delete_outside_restriction.status_code == 403

        blocked_folder = client.post(
            f"/api/files/buckets/{bucket_id}/folders?customer_id={customer_id}&prefix=blocked/",
            json={"folder_name": "new"},
        )
        assert blocked_folder.status_code == 403
