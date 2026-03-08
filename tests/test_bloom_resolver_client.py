from __future__ import annotations

import httpx
import pytest

from daylib.bloom_resolver_client import BloomResolverClient, BloomResolverError


def test_resolve_run_index_returns_resolution():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/api/v1/external/atlas/beta/runs/RUN-1/resolve")
        assert request.url.params["index_string"] == "IDX-01"
        assert request.headers["Authorization"] == "Bearer bloom-token"
        return httpx.Response(
            200,
            json={
                "run_euid": "RUN-1",
                "index_string": "IDX-01",
                "atlas_tenant_id": "TEN-1",
                "atlas_order_euid": "ORD-1",
                "atlas_test_order_euid": "TST-1",
                "source_euid": "LIB-1",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    resolver = BloomResolverClient(
        base_url="https://bloom.example",
        token="bloom-token",
        client=client,
    )

    resolved = resolver.resolve_run_index("RUN-1", "IDX-01")

    assert resolved.atlas_test_order_euid == "TST-1"
    assert resolved.source_euid == "LIB-1"


def test_resolve_run_index_raises_for_bad_response():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "not found"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    resolver = BloomResolverClient(base_url="https://bloom.example", client=client)

    with pytest.raises(BloomResolverError, match="404"):
        resolver.resolve_run_index("RUN-1", "IDX-01")


def test_resolve_run_index_raises_for_missing_fields():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "run_euid": "RUN-1",
                "index_string": "IDX-01",
                "atlas_tenant_id": "TEN-1",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    resolver = BloomResolverClient(base_url="https://bloom.example", client=client)

    with pytest.raises(BloomResolverError, match="missing required fields"):
        resolver.resolve_run_index("RUN-1", "IDX-01")
