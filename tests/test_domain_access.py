from __future__ import annotations

from fastapi.testclient import TestClient

from daylib_ursa.config import Settings
from daylib_ursa.workset_api import create_app


class DummyStore:
    def ingest_analysis(self, **kwargs):
        raise RuntimeError("not used")

    def get_analysis(self, analysis_euid: str):
        return None

    def update_analysis_state(self, analysis_euid: str, **kwargs):
        raise KeyError("not used")

    def add_artifact(self, analysis_euid: str, **kwargs):
        raise KeyError("not used")

    def set_review_state(self, analysis_euid: str, **kwargs):
        raise KeyError("not used")

    def mark_returned(self, analysis_euid: str, **kwargs):
        raise KeyError("not used")


class DummyBloomClient:
    def resolve_run_assignment(
        self,
        run_euid: str,
        flowcell_id: str,
        lane: str,
        library_barcode: str,
    ):
        raise RuntimeError("not used")


def _settings() -> Settings:
    return Settings(
        bloom_base_url="https://bloom.example",
        bloom_api_token="test-bloom-token",
        atlas_base_url="https://atlas.example",
        atlas_internal_api_key="atlas-internal-key",
        ursa_internal_output_bucket="ursa-internal",
        ursa_internal_api_key="ursa-internal-key",
    )


def test_ursa_allows_approved_origin_preflight() -> None:
    app = create_app(DummyStore(), bloom_client=DummyBloomClient(), settings=_settings())

    with TestClient(app) as client:
        response = client.options(
            "/healthz",
            headers={
                "Origin": "https://portal.lsmc.bio",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://portal.lsmc.bio"


def test_ursa_rejects_disallowed_origin() -> None:
    app = create_app(DummyStore(), bloom_client=DummyBloomClient(), settings=_settings())

    with TestClient(app) as client:
        response = client.options(
            "/healthz",
            headers={
                "Origin": "https://evil.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert response.status_code == 403
    assert response.text == "Origin not allowed"


def test_ursa_allows_arbitrary_host_when_host_filtering_is_delegated_upstream() -> None:
    app = create_app(DummyStore(), bloom_client=DummyBloomClient(), settings=_settings())

    with TestClient(app) as client:
        response = client.get("/healthz", headers={"host": "evil.example.com"})

    assert response.status_code == 200
