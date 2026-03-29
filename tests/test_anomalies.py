from __future__ import annotations

from fastapi.testclient import TestClient

from daylib_ursa.anomalies import open_anomaly_repository
from tests.test_admin_gui_and_cluster_routes import _create_test_app


def test_anomaly_repository_persists_and_redacts_records() -> None:
    app, _resources = _create_test_app(admin=True)
    repository = open_anomaly_repository(
        resource_store=app.state.resource_store,
        settings=app.state.settings,
        backend=app.state.token_service.backend,
    )

    first = repository.record(
        category="database",
        severity="error",
        fingerprint="db-probe-failed",
        summary="Database probe failed",
        redacted_context={
            "token": "secret-token",
            "sql": "SELECT secret FROM patient",
            "nested": {"password": "secret"},
        },
    )
    second = repository.record(
        category="database",
        severity="error",
        fingerprint="db-probe-failed",
        summary="Database probe failed again",
        redacted_context={"authorization": "Bearer xyz"},
    )

    assert first.id == second.id
    assert second.occurrence_count == 2
    assert second.redacted_context["authorization"] == "[redacted]"


def test_anomaly_api_routes_return_records_for_observability_auth() -> None:
    app, _resources = _create_test_app(admin=True)
    repository = open_anomaly_repository(
        resource_store=app.state.resource_store,
        settings=app.state.settings,
        backend=app.state.token_service.backend,
    )
    created = repository.record(
        category="database",
        severity="error",
        fingerprint="db-probe-failed",
        summary="Database probe failed",
        redacted_context={"token": "secret-token"},
    )
    headers = {"Authorization": f"Bearer {app.state.api_key}"}

    with TestClient(app) as client:
        list_response = client.get("/api/anomalies", headers=headers)
        detail_response = client.get(f"/api/anomalies/{created.id}", headers=headers)
        missing_response = client.get("/api/anomalies/ANM-DOES-NOT-EXIST", headers=headers)

    assert list_response.status_code == 200
    list_payload = list_response.json()
    assert list_payload["service"] == "ursa"
    assert list_payload["count"] == 1
    assert list_payload["items"][0]["id"] == created.id
    assert list_payload["items"][0]["redacted_context"]["token"] == "[redacted]"

    assert detail_response.status_code == 200
    assert detail_response.json()["item"]["id"] == created.id
    assert missing_response.status_code == 404


def test_obs_services_advertises_anomaly_capability() -> None:
    app, _resources = _create_test_app(admin=True)
    headers = {"Authorization": f"Bearer {app.state.api_key}"}

    with TestClient(app) as client:
        response = client.get("/obs_services", headers=headers)

    assert response.status_code == 200
    payload = response.json()
    endpoints = {item["path"]: item for item in payload["endpoints"]}
    assert endpoints["/api/anomalies"]["kind"] == "anomaly_list"
    assert endpoints["/api/anomalies/{anomaly_id}"]["kind"] == "anomaly_detail"
    assert "ursa.anomalies_v1" in payload["extensions"]


def test_admin_anomalies_page_requires_session_and_renders() -> None:
    app, _resources = _create_test_app(admin=True)
    repository = open_anomaly_repository(
        resource_store=app.state.resource_store,
        settings=app.state.settings,
        backend=app.state.token_service.backend,
    )
    created = repository.record(
        category="database",
        severity="error",
        fingerprint="db-probe-failed",
        summary="Database probe failed",
        redacted_context={"token": "secret-token"},
    )

    with TestClient(app) as client:
        redirect = client.get("/admin/anomalies", follow_redirects=False)
        client.post(
            "/login",
            data={"access_token": "atlas-token", "next_path": "/admin/anomalies"},
            follow_redirects=False,
        )
        list_response = client.get("/admin/anomalies")
        detail_response = client.get(f"/admin/anomalies/{created.id}")

    assert redirect.status_code == 303
    assert list_response.status_code == 200
    assert "Anomalies" in list_response.text
    assert created.summary in list_response.text
    assert detail_response.status_code == 200
    assert created.id in detail_response.text
