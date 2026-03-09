from __future__ import annotations

from dataclasses import replace

from fastapi.testclient import TestClient

from daylib_ursa.analysis_store import (
    AnalysisArtifact,
    AnalysisRecord,
    AnalysisState,
    ReviewState,
    RunResolution,
)
from daylib_ursa.config import Settings
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
        self.last_ingest = None

    def ingest_analysis(self, **kwargs):
        self.last_ingest = kwargs
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
            updated_at="2026-03-07T01:00:00Z",
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
            updated_at="2026-03-07T03:00:00Z",
        )
        return self.record

    def mark_returned(self, analysis_euid: str, **kwargs):
        assert analysis_euid == self.record.analysis_euid
        self.record = replace(
            self.record,
            state=AnalysisState.RETURNED.value,
            atlas_return=kwargs["atlas_return"],
            updated_at="2026-03-07T04:00:00Z",
        )
        return self.record


class DummyBloomClient:
    def __init__(self) -> None:
        self.calls = []

    def resolve_run_assignment(
        self, run_euid: str, flowcell_id: str, lane: str, library_barcode: str
    ) -> RunResolution:
        self.calls.append((run_euid, flowcell_id, lane, library_barcode))
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


def _settings() -> Settings:
    return Settings(
        cors_origins="*",
        ursa_internal_api_key="ursa-test-key",
        bloom_base_url="https://bloom.example",
        atlas_base_url="https://atlas.example",
        ursa_tapdb_mount_enabled=False,
    )


def test_ingest_analysis_resolves_bloom_and_persists():
    store = DummyStore()
    bloom = DummyBloomClient()
    app = create_app(store, bloom_client=bloom, settings=_settings())

    with TestClient(app) as client:
        response = client.post(
            "/api/analyses/ingest",
            headers={
                "X-API-Key": "ursa-test-key",
                "Idempotency-Key": "idem-1",
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

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["analysis_euid"] == "AN-1"
    assert body["atlas_test_process_item_euid"] == "TPC-1"
    assert bloom.calls == [("RUN-1", "FLOW-1", "1", "LIB-1")]
    assert store.last_ingest["idempotency_key"] == "idem-1"
    assert store.last_ingest["analysis_type"] == "germline"


def test_ingest_analysis_requires_idempotency_key():
    store = DummyStore()
    bloom = DummyBloomClient()
    app = create_app(store, bloom_client=bloom, settings=_settings())

    with TestClient(app) as client:
        response = client.post(
            "/api/analyses/ingest",
            headers={"X-API-Key": "ursa-test-key"},
            json={
                "run_euid": "RUN-1",
                "flowcell_id": "FLOW-1",
                "lane": "1",
                "library_barcode": "LIB-1",
                "analysis_type": "germline",
                "artifact_bucket": "analysis-bucket",
            },
        )

    assert response.status_code == 400
    assert "Idempotency-Key" in response.text


def test_legacy_workset_route_is_not_registered():
    app = create_app(DummyStore(), bloom_client=DummyBloomClient(), settings=_settings())

    with TestClient(app) as client:
        response = client.get("/worksets")

    assert response.status_code == 404


def test_root_redirects_to_login_and_health_moves_to_healthz():
    app = create_app(DummyStore(), bloom_client=DummyBloomClient(), settings=_settings())

    with TestClient(app) as client:
        root_response = client.get("/", follow_redirects=False)
        health_response = client.get("/healthz")

    assert root_response.status_code == 307
    assert root_response.headers["location"] == "/portal/login"
    assert health_response.status_code == 200
    assert health_response.json()["status"] == "healthy"


def test_settings_force_auth_enabled_even_when_disabled_in_override():
    settings = Settings(enable_auth=False)
    assert settings.enable_auth is True


def test_login_uses_non_portal_callback_uri():
    settings = Settings(
        cors_origins="*",
        ursa_internal_api_key="ursa-test-key",
        bloom_base_url="https://bloom.example",
        atlas_base_url="https://atlas.example",
        enable_auth=True,
        cognito_domain="daylily-ursa-5r8giqv5p.auth.us-west-2.amazoncognito.com",
        cognito_app_client_id="34g35v8tpurbe309a8e5t5ot7i",
        ursa_tapdb_mount_enabled=False,
    )
    app = create_app(DummyStore(), bloom_client=DummyBloomClient(), settings=settings)

    with TestClient(app) as client:
        response = client.get("/portal/login")

    assert response.status_code == 200
    assert "redirect_uri=" in response.text
    assert "auth%2Fcallback" in response.text
    assert "portal%2Fauth%2Fcallback" not in response.text


def test_auth_callback_redirect_target_exists():
    app = create_app(DummyStore(), bloom_client=DummyBloomClient(), settings=_settings())

    with TestClient(app) as client:
        callback = client.get("/auth/callback", follow_redirects=False)
        portal = client.get("/portal")

    assert callback.status_code == 307
    assert callback.headers["location"].startswith("/portal/login")
    assert portal.status_code == 200
