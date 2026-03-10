from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from daylib_ursa.analysis_store import AnalysisArtifact, AnalysisRecord, AnalysisState, ReviewState
from daylib_ursa.atlas_result_client import AtlasResultArtifact, AtlasResultClient, AtlasResultClientError
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
            atlas_test_fulfillment_item_euid="TPC-1",
            analysis_type="somatic",
            state=AnalysisState.REVIEW_PENDING.value,
            review_state=ReviewState.PENDING.value,
            result_status="PENDING",
            run_folder="s3://analysis-bucket/RUN-1/",
            internal_bucket="analysis-bucket",
            input_references=[],
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
                    metadata={"index_string": "IDX-01", "dewey_artifact_euid": "AT-1"},
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
    def resolve_run_assignment(
        self, run_euid: str, flowcell_id: str, lane: str, library_barcode: str
    ):
        raise AssertionError("Bloom resolver should not be called during result return")


class DummyAtlasClient:
    def __init__(self) -> None:
        self.calls = []

    def return_analysis_result(self, **kwargs):
        self.calls.append(kwargs)
        artifacts = kwargs["artifacts"]
        assert artifacts == [
            AtlasResultArtifact(
                artifact_euid="AT-1",
                artifact_type="vcf",
                storage_uri="s3://analysis-bucket/RUN-1/sample.vcf.gz",
                filename="sample.vcf.gz",
                mime_type="application/gzip",
                checksum_sha256="abc123",
                size_bytes=100,
                metadata={"index_string": "IDX-01", "dewey_artifact_euid": "AT-1"},
            )
        ]
        return {
            "fulfillment_run_euid": "ASR-1",
            "fulfillment_output_euid": "RES-1",
            "artifact_euids": ["AT-1"],
        }


def _settings() -> Settings:
    return Settings(
        cors_origins="*",
        ursa_internal_api_key="ursa-test-key",
        bloom_base_url="https://bloom.example",
        atlas_base_url="https://atlas.example",
        ursa_internal_output_bucket="analysis-bucket",
        ursa_tapdb_mount_enabled=False,
    )


def test_return_analysis_result_calls_atlas_and_marks_returned():
    store = DummyStore()
    store.record = replace(
        store.record,
        review_state=ReviewState.APPROVED.value,
        state=AnalysisState.REVIEWED.value,
    )
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
    assert body["atlas_return"]["fulfillment_run_euid"] == "ASR-1"
    assert atlas.calls[0]["atlas_test_fulfillment_item_euid"] == "TPC-1"
    assert store.mark_returned_calls[0]["idempotency_key"] == "return-1"


def test_return_analysis_result_requires_manual_approval():
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

    assert response.status_code == 409, response.text
    assert "manual approval" in response.text
    assert atlas.calls == []


def test_review_analysis_updates_review_state():
    store = DummyStore()
    atlas = DummyAtlasClient()
    app = create_app(
        store,
        bloom_client=DummyBloomClient(),
        atlas_client=atlas,
        settings=_settings(),
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
            headers={"X-API-Key": "ursa-test-key"},
            json={"review_state": "APPROVED", "reviewer": "qa@example.com"},
        )

    assert response.status_code == 200, response.text
    assert response.json()["review_state"] == "APPROVED"


def test_atlas_result_client_rejects_non_https_base_url():
    client = AtlasResultClient(base_url="http://atlas.example", api_key="atlas-test-key")

    with pytest.raises(AtlasResultClientError, match="absolute https:// URL"):
        client.return_analysis_result(
            atlas_tenant_id="TEN-1",
            atlas_trf_euid="TRF-1",
            atlas_test_euid="TST-1",
            atlas_test_fulfillment_item_euid="TPC-1",
            analysis_euid="AN-1",
            run_euid="RUN-1",
            sequenced_library_assignment_euid="SQA-1",
            flowcell_id="FLOW-1",
            lane="1",
            library_barcode="LIB-1",
            analysis_type="somatic",
            result_status="COMPLETED",
            review_state="APPROVED",
            result_payload={},
            artifacts=[],
            idempotency_key="idem-1",
        )


def test_create_app_rejects_no_auth_write_mode():
    with pytest.raises(ValueError, match="cannot be disabled"):
        create_app(
            DummyStore(),
            bloom_client=DummyBloomClient(),
            atlas_client=DummyAtlasClient(),
            settings=_settings(),
            require_api_key=False,
        )
