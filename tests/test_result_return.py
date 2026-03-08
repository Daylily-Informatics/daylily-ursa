from __future__ import annotations

from dataclasses import replace

from fastapi.testclient import TestClient

from daylib_ursa.analysis_store import AnalysisArtifact, AnalysisRecord, AnalysisState, ReviewState
from daylib_ursa.atlas_result_client import AtlasResultArtifact
from daylib_ursa.config import Settings
from daylib_ursa.workset_api import create_app


class DummyStore:
    def __init__(self) -> None:
        self.record = AnalysisRecord(
            analysis_euid="AN-1",
            run_euid="RUN-1",
            index_string="IDX-01",
            atlas_tenant_id="TEN-1",
            atlas_order_euid="ORD-1",
            atlas_test_order_euid="TST-1",
            source_euid="LIB-1",
            analysis_type="somatic",
            state=AnalysisState.REVIEW_PENDING.value,
            review_state=ReviewState.PENDING.value,
            result_status="PENDING",
            run_folder="s3://analysis-bucket/RUN-1/",
            artifact_bucket="analysis-bucket",
            result_payload={},
            metadata={},
            created_at="2026-03-07T00:00:00Z",
            updated_at="2026-03-07T00:00:00Z",
            atlas_return={},
            artifacts=[
                AnalysisArtifact(
                    artifact_euid="AF-1",
                    artifact_type="vcf",
                    storage_uri="s3://analysis-bucket/RUN-1/sample.vcf.gz",
                    filename="sample.vcf.gz",
                    mime_type="application/gzip",
                    checksum_sha256="abc123",
                    size_bytes=100,
                    created_at="2026-03-07T00:10:00Z",
                    metadata={"index_string": "IDX-01"},
                )
            ],
        )
        self.mark_returned_calls = []

    def get_analysis(self, analysis_euid: str):
        return self.record if analysis_euid == self.record.analysis_euid else None

    def mark_returned(self, analysis_euid: str, **kwargs):
        assert analysis_euid == self.record.analysis_euid
        self.mark_returned_calls.append(kwargs)
        self.record = replace(
            self.record,
            state=AnalysisState.RETURNED.value,
            atlas_return=kwargs["atlas_return"],
            updated_at="2026-03-07T04:00:00Z",
        )
        return self.record


class DummyBloomClient:
    def resolve_run_index(self, run_euid: str, index_string: str):
        raise AssertionError("Bloom resolver should not be called during result return")


class DummyAtlasClient:
    def __init__(self) -> None:
        self.calls = []

    def return_analysis_result(self, **kwargs):
        self.calls.append(kwargs)
        artifacts = kwargs["artifacts"]
        assert artifacts == [
            AtlasResultArtifact(
                artifact_type="vcf",
                storage_uri="s3://analysis-bucket/RUN-1/sample.vcf.gz",
                filename="sample.vcf.gz",
                mime_type="application/gzip",
                checksum_sha256="abc123",
                size_bytes=100,
                metadata={"index_string": "IDX-01"},
            )
        ]
        return {
            "assay_run_euid": "ASR-1",
            "assay_result_euid": "RES-1",
            "artifact_euids": ["AT-1"],
        }


def _settings() -> Settings:
    return Settings(
        cors_origins="*",
        ursa_internal_api_key="ursa-test-key",
        bloom_base_url="https://bloom.example",
        atlas_base_url="https://atlas.example",
    )


def test_return_analysis_result_calls_atlas_and_marks_returned():
    store = DummyStore()
    atlas = DummyAtlasClient()
    app = create_app(
        store,
        bloom_client=DummyBloomClient(),
        atlas_client=atlas,
        settings=_settings(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/analyses/AN-1/return",
            headers={
                "X-API-Key": "ursa-test-key",
                "Idempotency-Key": "return-1",
            },
            json={
                "result_payload": {"calls": []},
                "result_status": "COMPLETED",
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["state"] == "RETURNED"
    assert body["atlas_return"]["assay_run_euid"] == "ASR-1"
    assert atlas.calls[0]["atlas_test_order_euid"] == "TST-1"
    assert store.mark_returned_calls[0]["idempotency_key"] == "return-1"


def test_review_analysis_updates_review_state():
    store = DummyStore()
    atlas = DummyAtlasClient()
    app = create_app(
        store,
        bloom_client=DummyBloomClient(),
        atlas_client=atlas,
        settings=_settings(),
        require_api_key=False,
    )

    def _set_review_state(analysis_euid: str, **kwargs):
        assert analysis_euid == "AN-1"
        store.record = replace(
            store.record,
            review_state=kwargs["review_state"].value,
            state=AnalysisState.REVIEWED.value,
            updated_at="2026-03-07T03:00:00Z",
        )
        return store.record

    store.set_review_state = _set_review_state  # type: ignore[method-assign]

    with TestClient(app) as client:
        response = client.post(
            "/api/analyses/AN-1/review",
            json={"review_state": "APPROVED", "reviewer": "qa@example.com"},
        )

    assert response.status_code == 200, response.text
    assert response.json()["review_state"] == "APPROVED"
