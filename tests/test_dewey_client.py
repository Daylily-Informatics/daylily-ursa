from __future__ import annotations

import httpx
import pytest

from daylib_ursa.integrations.dewey_client import DeweyClient, DeweyClientError


def test_resolve_rejects_non_https_base_url():
    client = DeweyClient(base_url="http://dewey.example", token="token-1")
    with pytest.raises(DeweyClientError, match="absolute https:// URL"):
        client.resolve_artifact("AT-1")


def test_resolve_artifact_uses_bearer_auth():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("Authorization")
        return httpx.Response(
            200,
            json={
                "artifact_euid": "AT-1",
                "artifact_type": "fastq",
                "storage_uri": "s3://bucket/RUN-1/read1.fastq.gz",
                "metadata": {"producer_system": "bloom"},
            },
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as http_client:
        client = DeweyClient(
            base_url="https://dewey.example",
            token="token-1",
            client=http_client,
        )
        payload = client.resolve_artifact("AT-1")

    assert payload["artifact_euid"] == "AT-1"
    assert captured["url"] == "https://dewey.example/api/v1/resolve/artifact"
    assert captured["authorization"] == "Bearer token-1"


def test_register_artifact_uses_bearer_auth_and_idempotency_key():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("Authorization")
        captured["idempotency"] = request.headers.get("Idempotency-Key")
        captured["body"] = request.read().decode("utf-8")
        return httpx.Response(200, json={"artifact_euid": "AT-2"})

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as http_client:
        client = DeweyClient(
            base_url="https://dewey.example",
            token="token-2",
            client=http_client,
        )
        artifact_euid = client.register_artifact(
            artifact_type="vcf",
            storage_uri="s3://bucket/RUN-1/sample.vcf.gz",
            metadata={"producer_system": "ursa"},
            idempotency_key="idem-1",
        )

    assert artifact_euid == "AT-2"
    assert captured["url"] == "https://dewey.example/api/v1/artifacts/import"
    assert captured["authorization"] == "Bearer token-2"
    assert captured["idempotency"] == "idem-1"
