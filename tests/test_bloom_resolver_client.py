from __future__ import annotations

import httpx
import pytest

from daylib_ursa.bloom_resolver_client import BloomResolverClient, BloomResolverError


def test_resolve_run_assignment_returns_resolution():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/api/v1/external/atlas/beta/runs/RUN-1/resolve")
        assert request.url.params["flowcell_id"] == "FLOW-1"
        assert request.url.params["lane"] == "1"
        assert request.url.params["library_barcode"] == "LIB-1"
        assert request.headers["Authorization"] == "Bearer bloom-token"
        return httpx.Response(
            200,
            json={
                "run_euid": "RUN-1",
                "flowcell_id": "FLOW-1",
                "lane": "1",
                "library_barcode": "LIB-1",
                "sequenced_library_assignment_euid": "SQA-1",
                "atlas_tenant_id": "TEN-1",
                "atlas_trf_euid": "TRF-1",
                "atlas_test_euid": "TST-1",
                "atlas_test_fulfillment_item_euid": "TPC-1",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    resolver = BloomResolverClient(
        base_url="https://bloom.example",
        token="bloom-token",
        client=client,
    )

    resolved = resolver.resolve_run_assignment("RUN-1", "FLOW-1", "1", "LIB-1")

    assert resolved.atlas_test_fulfillment_item_euid == "TPC-1"
    assert resolved.sequenced_library_assignment_euid == "SQA-1"


def test_resolve_run_assignment_raises_for_bad_response():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "not found"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    resolver = BloomResolverClient(base_url="https://bloom.example", client=client)

    with pytest.raises(BloomResolverError, match="404"):
        resolver.resolve_run_assignment("RUN-1", "FLOW-1", "1", "LIB-1")


def test_resolve_run_assignment_raises_for_missing_fields():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "run_euid": "RUN-1",
                "flowcell_id": "FLOW-1",
                "atlas_tenant_id": "TEN-1",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    resolver = BloomResolverClient(base_url="https://bloom.example", client=client)

    with pytest.raises(BloomResolverError, match="missing required fields"):
        resolver.resolve_run_assignment("RUN-1", "FLOW-1", "1", "LIB-1")
