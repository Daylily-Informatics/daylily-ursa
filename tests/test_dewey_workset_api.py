from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from daylib_ursa.analysis_store import AnalysisArtifact
from daylib_ursa.config import Settings
from daylib_ursa.workset_api import create_app


class _FakeRegionAwareS3Client:
    def __init__(self, default_region: str, profile: str | None = None) -> None:
        self.default_region = default_region
        self.profile = profile

    def head_object(self, Bucket: str, Key: str, **kwargs):  # noqa: N803
        _ = (Bucket, Key, kwargs)
        return {"ContentLength": 1}


class _DummyBloomClient:
    def resolve_run_assignment(self, *_args, **_kwargs):  # pragma: no cover - not used in these tests
        raise AssertionError("Bloom resolver should not be called in Dewey artifact tests")


@dataclass
class _DummyStore:
    calls: list[dict] = field(default_factory=list)
    artifacts_by_analysis: dict[str, list[AnalysisArtifact]] = field(default_factory=dict)

    def get_analysis(self, analysis_euid: str):
        artifacts = self.artifacts_by_analysis.get(analysis_euid, [])
        return SimpleNamespace(artifacts=artifacts)

    def add_artifact(self, analysis_euid: str, **kwargs):
        for existing in self.artifacts_by_analysis.get(analysis_euid, []):
            if existing.storage_uri == str(kwargs["storage_uri"]):
                self.calls.append({"analysis_euid": analysis_euid, **kwargs})
                return existing
        payload = {"analysis_euid": analysis_euid, **kwargs}
        self.calls.append(payload)
        created = AnalysisArtifact(
            artifact_euid="AF-LOCAL-1",
            artifact_type=str(kwargs["artifact_type"]),
            storage_uri=str(kwargs["storage_uri"]),
            filename=str(kwargs["filename"]),
            mime_type=kwargs.get("mime_type"),
            checksum_sha256=kwargs.get("checksum_sha256"),
            size_bytes=kwargs.get("size_bytes"),
            created_at="2026-03-09T00:00:00Z",
            metadata=dict(kwargs.get("metadata") or {}),
        )
        self.artifacts_by_analysis.setdefault(analysis_euid, []).append(created)
        return created


def _settings() -> Settings:
    return Settings(
        cors_origins="*",
        ursa_internal_api_key="ursa-test-key",
        bloom_base_url="https://bloom.example",
        atlas_base_url="https://atlas.example",
        ursa_internal_output_bucket="ursa-internal",
        ursa_tapdb_mount_enabled=False,
    )


def test_add_artifact_resolves_dewey_reference():
    store = _DummyStore()

    class _FakeDeweyClient:
        def resolve_artifact(self, artifact_euid: str):
            assert artifact_euid == "AT-1"
            return {
                "artifact_euid": "AT-1",
                "artifact_type": "fastq",
                "storage_uri": "s3://bucket/RUN-1/read1.fastq.gz",
                "metadata": {"producer_system": "bloom"},
            }

        def register_artifact(self, **kwargs):  # pragma: no cover - should not be called in this path
            raise AssertionError("register_artifact should not be called for artifact_euid inputs")

    app = create_app(
        store,
        bloom_client=_DummyBloomClient(),
        atlas_client=None,
        dewey_client=_FakeDeweyClient(),
        settings=_settings(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/analyses/AN-1/artifacts",
            headers={"X-API-Key": "ursa-test-key"},
            json={
                "artifact_euid": "AT-1",
                "metadata": {"lane": "1"},
            },
        )

    assert response.status_code == 201, response.text
    call = store.calls[0]
    assert call["artifact_type"] == "fastq"
    assert call["storage_uri"] == "s3://bucket/RUN-1/read1.fastq.gz"
    assert call["filename"] == "read1.fastq.gz"
    assert call["metadata"]["dewey_artifact_euid"] == "AT-1"
    assert call["metadata"]["dewey_resolved"] is True
    assert call["metadata"]["lane"] == "1"


def test_add_artifact_registers_to_dewey_for_raw_storage_uri(monkeypatch):
    monkeypatch.setattr("daylib_ursa.workset_api.RegionAwareS3Client", _FakeRegionAwareS3Client)
    store = _DummyStore()
    captured: dict[str, str] = {}

    class _FakeDeweyClient:
        def resolve_artifact(self, _artifact_euid: str):  # pragma: no cover - not used in this path
            raise AssertionError("resolve_artifact should not be called for raw storage_uri inputs")

        def register_artifact(self, *, artifact_type: str, storage_uri: str, metadata, idempotency_key: str):
            captured["artifact_type"] = artifact_type
            captured["storage_uri"] = storage_uri
            captured["idempotency_key"] = idempotency_key
            captured["producer_system"] = metadata["producer_system"]
            captured["producer_object_euid"] = metadata["producer_object_euid"]
            return "AT-REG-1"

    app = create_app(
        store,
        bloom_client=_DummyBloomClient(),
        atlas_client=None,
        dewey_client=_FakeDeweyClient(),
        settings=_settings(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/analyses/AN-2/artifacts",
            headers={"X-API-Key": "ursa-test-key"},
            json={
                "artifact_type": "vcf",
                "storage_uri": "s3://ursa-internal/RUN-2/sample.vcf.gz",
                "filename": "sample.vcf.gz",
            },
        )

    assert response.status_code == 201, response.text
    assert captured["artifact_type"] == "vcf"
    assert captured["storage_uri"] == "s3://ursa-internal/RUN-2/sample.vcf.gz"
    assert captured["idempotency_key"] == "AN-2:s3://ursa-internal/RUN-2/sample.vcf.gz"
    assert captured["producer_system"] == "ursa"
    assert captured["producer_object_euid"] == "AN-2"
    assert store.calls[0]["metadata"]["dewey_artifact_euid"] == "AT-REG-1"


def test_add_artifact_with_reference_requires_dewey_client():
    store = _DummyStore()
    app = create_app(
        store,
        bloom_client=_DummyBloomClient(),
        atlas_client=None,
        dewey_client=None,
        settings=_settings(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/analyses/AN-3/artifacts",
            headers={"X-API-Key": "ursa-test-key"},
            json={"artifact_euid": "AT-1"},
        )
    assert response.status_code == 400
    assert "Dewey integration configuration is required" in response.json()["detail"]


def test_add_artifact_reuses_existing_dewey_id_without_reregistering(monkeypatch):
    monkeypatch.setattr("daylib_ursa.workset_api.RegionAwareS3Client", _FakeRegionAwareS3Client)
    store = _DummyStore()
    register_calls: list[str] = []

    class _FakeDeweyClient:
        def resolve_artifact(self, _artifact_euid: str):  # pragma: no cover - not used
            raise AssertionError("resolve_artifact should not be called")

        def register_artifact(self, *, artifact_type: str, storage_uri: str, metadata, idempotency_key: str):
            register_calls.append(storage_uri)
            return "AT-REG-1"

    app = create_app(
        store,
        bloom_client=_DummyBloomClient(),
        atlas_client=None,
        dewey_client=_FakeDeweyClient(),
        settings=_settings(),
    )
    with TestClient(app) as client:
        first = client.post(
            "/api/analyses/AN-4/artifacts",
            headers={"X-API-Key": "ursa-test-key"},
            json={
                "artifact_type": "vcf",
                "storage_uri": "s3://ursa-internal/RUN-4/sample.vcf.gz",
                "filename": "sample.vcf.gz",
            },
        )
        second = client.post(
            "/api/analyses/AN-4/artifacts",
            headers={"X-API-Key": "ursa-test-key"},
            json={
                "artifact_type": "vcf",
                "storage_uri": "s3://ursa-internal/RUN-4/sample.vcf.gz",
                "filename": "sample.vcf.gz",
            },
        )

    assert first.status_code == 201, first.text
    assert second.status_code == 201, second.text
    assert register_calls == [
        "s3://ursa-internal/RUN-4/sample.vcf.gz",
        "s3://ursa-internal/RUN-4/sample.vcf.gz",
    ]
