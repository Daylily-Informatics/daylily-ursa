from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from fastapi.testclient import TestClient

from daylib_ursa import __version__
from daylib_ursa import observability as observability_module
from tests.test_admin_gui_and_cluster_routes import ADMIN_USER_ID, _create_test_app


def _schema_root() -> Path:
    root = os.environ.get("DAYHOFF_PROJECT_ROOT")
    if not root:
        raise RuntimeError("DAYHOFF_PROJECT_ROOT must point at the canonical Dayhoff repo root")
    return Path(root) / "contracts" / "observability"


def _load_schema(name: str) -> dict[str, Any]:
    return json.loads((_schema_root() / name).read_text(encoding="utf-8"))


def _validate(name: str, payload: dict[str, Any]) -> None:
    jsonschema.validate(payload, _load_schema(name))


@pytest.fixture(autouse=True)
def _stub_schema_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    observability_module._SCHEMA_DRIFT_CACHE.clear()
    monkeypatch.setenv(
        "DAYHOFF_PROJECT_ROOT",
        str(Path(__file__).resolve().parents[6]),
    )
    monkeypatch.setattr(
        observability_module,
        "run_tapdb_schema_drift_check",
        lambda **_kwargs: {
            "status": "clean",
            "checked_at": "2026-03-29T00:00:00Z",
            "environment": "dev",
            "tool_version": "5.1.0",
            "summary": "no schema drift reported",
            "report": {"counts": {"expected": 10, "live": 10}},
            "strict": False,
        },
    )


def test_observability_contract_endpoints_match_shared_frame() -> None:
    app, _resources = _create_test_app(admin=True)
    service_headers = {"Authorization": f"Bearer {app.state.api_key}"}
    admin_headers = {"Authorization": "Bearer atlas-token"}

    app.state.observability.record_db_query(
        statement="SELECT workset_euid FROM generic_instance",
        duration_ms=5.0,
        success=True,
    )

    with TestClient(app) as client:
        healthz = client.get("/healthz")
        readyz = client.get("/readyz")
        client.get("/api/v1/analyses/AN-1", headers=admin_headers)
        client.get("/api/v1/worksets", headers=admin_headers)
        health_response = client.get("/health", headers=service_headers)
        obs_services_response = client.get("/obs_services", headers=service_headers)
        api_health_response = client.get("/api_health", headers=service_headers)
        endpoint_health_response = client.get("/endpoint_health", headers=service_headers)
        db_health_response = client.get("/db_health", headers=service_headers)
        auth_health_response = client.get("/auth_health", headers=service_headers)

        responses = {
            "/healthz": (healthz, "healthz.schema.json"),
            "/readyz": (readyz, "readyz.schema.json"),
            "/health": (health_response, "health.schema.json"),
            "/obs_services": (obs_services_response, "obs_services.schema.json"),
            "/api_health": (api_health_response, "api_health.schema.json"),
            "/endpoint_health": (endpoint_health_response, "endpoint_health.schema.json"),
            "/db_health": (db_health_response, "db_health.schema.json"),
            "/auth_health": (auth_health_response, "auth_health.schema.json"),
        }

    assert healthz.status_code == 200
    assert readyz.status_code in {200, 503}
    assert isinstance(readyz.json()["ready"], bool)
    assert readyz.json()["checks"]["database"]["detail"]
    configured_services = set(obs_services_response.json()["dependencies"]["configured_services"])
    assert {"atlas", "bloom"}.issubset(configured_services)
    assert obs_services_response.json()["dependencies"]["observed_services"] == []
    assert db_health_response.json()["database"]["schema_drift"]["status"] == "clean"
    assert auth_health_response.json()["auth"]["sessions"]["supported"] is False

    for path, (response, schema_name) in responses.items():
        assert response.status_code in {200, 503}, f"{path} returned {response.status_code}"
        payload = response.json()
        _validate(schema_name, payload)
        assert payload["service"] == "ursa"
        assert payload["contract_version"] == "v3"
        assert payload["build"]["version"] == __version__


def test_my_health_matches_shared_schema_for_authenticated_user() -> None:
    app, _resources = _create_test_app(admin=True)
    with TestClient(app) as client:
        response = client.get("/my_health", headers={"Authorization": "Bearer atlas-token"})

    assert response.status_code == 200
    payload = response.json()
    _validate("my_health.schema.json", payload)
    assert payload["service"] == "ursa"
    assert payload["principal"]["auth_mode"] == "cognito"
    assert payload["principal"]["service_principal"] is False
    assert payload["principal"]["subject"] == ADMIN_USER_ID


def test_endpoint_health_uses_route_templates_not_raw_instances() -> None:
    app, _resources = _create_test_app(admin=True)
    service_headers = {"Authorization": f"Bearer {app.state.api_key}"}

    with TestClient(app) as client:
        client.get("/healthz")
        client.get("/readyz")
        client.get("/api/v1/analyses/AN-1", headers={"Authorization": "Bearer atlas-token"})
        response = client.get("/endpoint_health", headers=service_headers)

    assert response.status_code == 200
    route_templates = {item["route_template"] for item in response.json()["items"]}
    assert "/healthz" in route_templates
    assert "/readyz" in route_templates
    assert "/api/v1/analyses/{analysis_euid}" in route_templates
    assert all("AN-1" not in item for item in route_templates)


def test_obs_services_advertises_canonical_capabilities() -> None:
    app, _resources = _create_test_app(admin=True)
    service_headers = {"Authorization": f"Bearer {app.state.api_key}"}

    with TestClient(app) as client:
        response = client.get("/obs_services", headers=service_headers)

    assert response.status_code == 200
    advertised = {
        item["path"]: {"auth": item["auth"], "kind": item["kind"]}
        for item in response.json()["endpoints"]
    }
    assert advertised == {
        "/healthz": {"auth": "none", "kind": "liveness"},
        "/readyz": {"auth": "none", "kind": "readiness"},
        "/health": {"auth": "operator_or_service_token", "kind": "summary"},
        "/obs_services": {"auth": "operator_or_service_token", "kind": "discovery"},
        "/api_health": {"auth": "operator_or_service_token", "kind": "api_rollup"},
        "/endpoint_health": {"auth": "operator_or_service_token", "kind": "endpoint_rollup"},
        "/db_health": {"auth": "operator_or_service_token", "kind": "database"},
        "/api/anomalies": {"auth": "operator_or_service_token", "kind": "anomaly_list"},
        "/api/anomalies/{anomaly_id}": {
            "auth": "operator_or_service_token",
            "kind": "anomaly_detail",
        },
        "/my_health": {"auth": "authenticated_self", "kind": "self"},
        "/auth_health": {"auth": "operator_or_service_token", "kind": "auth"},
    }
    configured_services = set(response.json()["dependencies"]["configured_services"])
    assert {"atlas", "bloom"}.issubset(configured_services)
    assert response.json()["dependencies"]["observed_services"] == []


def test_obs_services_reports_observed_dependencies_when_recorded() -> None:
    app, _resources = _create_test_app(admin=True)
    headers = {"Authorization": f"Bearer {app.state.api_key}"}
    app.state.observability.record_observed_dependency("dewey")

    with TestClient(app) as client:
        response = client.get("/obs_services", headers=headers)

    assert response.status_code == 200
    assert response.json()["dependencies"]["observed_services"] == ["dewey"]


def test_my_health_rejects_internal_service_token() -> None:
    app, _resources = _create_test_app(admin=True)
    with TestClient(app) as client:
        response = client.get(
            "/my_health",
            headers={"Authorization": f"Bearer {app.state.api_key}"},
        )

    assert response.status_code == 401


def test_admin_observability_page_renders() -> None:
    app, _resources = _create_test_app(admin=True)

    with TestClient(app) as client:
        login = client.post(
            "/login",
            data={"access_token": "atlas-token", "next_path": "/admin/observability"},
            follow_redirects=False,
        )
        response = client.get("/admin/observability")

    assert login.status_code == 303
    assert response.status_code == 200
    assert "Observability" in response.text
    assert "/db_health" in response.text
    assert "Configured dependencies:" in response.text
    assert "Schema drift:" in response.text
    assert "Session summary:" in response.text


def test_schema_drift_check_is_not_run_per_db_health_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observability_module._SCHEMA_DRIFT_CACHE.clear()
    calls = {"count": 0}

    def _fake_drift(**_kwargs):
        calls["count"] += 1
        return {
            "status": "drift",
            "checked_at": "2026-03-29T00:00:00Z",
            "environment": "dev",
            "tool_version": "5.1.0",
            "summary": "schema drift detected",
            "report": {},
            "strict": False,
        }

    monkeypatch.setattr(observability_module, "run_tapdb_schema_drift_check", _fake_drift)
    app, _resources = _create_test_app(admin=True)
    headers = {"Authorization": f"Bearer {app.state.api_key}"}

    with TestClient(app) as client:
        first = client.get("/db_health", headers=headers)
        second = client.get("/db_health", headers=headers)

    assert first.status_code == 200
    assert second.status_code == 200
    assert calls["count"] == 1
    assert first.json()["database"]["schema_drift"]["status"] == "drift"
