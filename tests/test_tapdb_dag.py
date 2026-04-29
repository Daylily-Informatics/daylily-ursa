from __future__ import annotations

from fastapi import APIRouter
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
        self, run_euid: str, flowcell_id: str, lane: str, library_barcode: str
    ):
        raise RuntimeError("not used")


class DummyS3Client:
    pass


def _settings(*, tapdb_config_path: str = "/tmp/ursa-tapdb-config.yaml") -> Settings:
    return Settings(
        cors_origins="*",
        aws_profile=None,
        ursa_internal_api_key="test-key",
        bloom_base_url="https://bloom.example",
        atlas_base_url="https://atlas.example",
        ursa_internal_output_bucket="ursa-internal",
        cognito_domain="auth.example.com",
        cognito_app_client_id="client-1",
        cognito_callback_url="https://localhost:8913/auth/callback",
        cognito_logout_url="https://localhost:8913/login",
        tapdb_config_path=tapdb_config_path,
        ursa_tapdb_mount_enabled=False,
        enable_auth=True,
    )


def _dummy_dag_router() -> APIRouter:
    router = APIRouter()

    @router.get("/api/dag/object/{euid}")
    async def dag_object(euid: str) -> dict[str, str]:
        return {"euid": euid, "system": "ursa"}

    @router.get("/api/dag/data")
    async def dag_data() -> dict[str, object]:
        return {"elements": {"nodes": [], "edges": []}, "system": "ursa"}

    @router.get("/api/dag/search")
    async def dag_search() -> dict[str, object]:
        return {"items": [], "system": "ursa"}

    @router.get("/api/dag/external")
    async def dag_external() -> dict[str, object]:
        return {"elements": {"nodes": [], "edges": []}, "system": "ursa"}

    @router.get("/api/dag/external/object")
    async def dag_external_object() -> dict[str, str]:
        return {"system": "ursa"}

    return router


def _app(monkeypatch, *, tapdb_config_path: str = "/tmp/ursa-tapdb-config.yaml"):
    monkeypatch.setattr(
        "daylib_ursa.tapdb_dag.create_tapdb_dag_router",
        lambda **_kwargs: _dummy_dag_router(),
    )
    return create_app(
        DummyStore(),
        bloom_client=DummyBloomClient(),
        settings=_settings(tapdb_config_path=tapdb_config_path),
        s3_client=DummyS3Client(),
    )


def test_tapdb_dag_routes_use_observability_auth(monkeypatch) -> None:
    app = _app(monkeypatch)

    with TestClient(app) as client:
        denied = client.get("/api/dag/search")
        allowed = client.get("/api/dag/search", headers={"Authorization": "Bearer test-key"})

    assert denied.status_code == 401
    assert allowed.status_code == 200
    assert allowed.json() == {"items": [], "system": "ursa"}


def test_obs_services_advertises_tapdb_dag_search_when_configured(monkeypatch) -> None:
    app = _app(monkeypatch)

    with TestClient(app) as client:
        response = client.get("/obs_services", headers={"Authorization": "Bearer test-key"})

    assert response.status_code == 200
    body = response.json()
    paths = {item["path"]: item["kind"] for item in body["endpoints"]}
    assert paths["/api/dag/object/{euid}"] == "dag_exact_lookup"
    assert paths["/api/dag/data"] == "dag_native_graph"
    assert paths["/api/dag/search"] == "dag_object_search"
    assert paths["/api/dag/external"] == "dag_external_graph"
    assert paths["/api/dag/external/object"] == "dag_external_object"
    assert "tapdb.dag_v1" in body["extensions"]
    assert "object_search" in body["capabilities"]
    assert "typed_external_identifier" in body["external_ref_models"]
    assert body["tapdb_dag_contract_version"] == "dag:v1"


def test_tapdb_dag_is_not_advertised_without_explicit_config(monkeypatch) -> None:
    app = _app(monkeypatch, tapdb_config_path="")

    with TestClient(app) as client:
        route = client.get("/api/dag/search", headers={"Authorization": "Bearer test-key"})
        obs = client.get("/obs_services", headers={"Authorization": "Bearer test-key"})

    assert route.status_code == 404
    assert "/api/dag/search" not in {item["path"] for item in obs.json()["endpoints"]}
    assert "tapdb.dag_v1" not in obs.json()["extensions"]
