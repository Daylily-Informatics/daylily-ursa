"""Route-level coverage smoke tests.

These tests exist to ensure key GUI/API route modules have at least one
integration-style request test, so refactors don't leave endpoints unexercised.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware


def _make_app_with_session(*, secret: str = "test-secret") -> FastAPI:
    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key=secret)
    return app


def test_clusters_list_returns_actionable_error_when_no_regions_configured():
    """GET /api/v2/clusters should not crash when no regions are configured."""
    from daylily_ursa.routes.clusters import ClusterDependencies, create_clusters_router

    settings = MagicMock()
    settings.aws_profile = None
    settings.get_allowed_regions.return_value = []

    def get_current_user():
        return None

    deps = ClusterDependencies(settings=settings, get_current_user=get_current_user)
    app = FastAPI()
    app.include_router(create_clusters_router(deps))

    # Ensure we don't depend on a developer's local ~/.config/ursa/ursa-config.yaml
    # being present (if it is, get_ursa_config() can return is_configured=True).
    with patch(
        "daylily_ursa.ursa_config.get_ursa_config",
        return_value=SimpleNamespace(is_configured=False, aws_profile=None),
    ):
        with TestClient(app, base_url="https://testserver") as client:
            resp = client.get("/api/v2/clusters")
            assert resp.status_code == 200
            payload = resp.json()
            assert payload["clusters"] == []
            assert payload["regions"] == []
            assert "No regions configured" in payload.get("error", "")


def test_monitoring_command_log_endpoint_admin_success_parses_entries():
    """GET /api/v2/admin/worksets/{id}/command-log should return parsed entries for admins."""
    from daylily_ursa.routes.monitoring import MonitoringDependencies, create_monitoring_router

    state_db = MagicMock()
    state_db.get_workset.return_value = {
        "workset_id": "ws-123",
        "bucket": "test-bucket",
        "prefix": "worksets/ws-123/",
    }

    app = _make_app_with_session()
    # route reads request.app.state.settings for region/profile
    app.state.settings = SimpleNamespace(get_effective_region=lambda: "us-west-2", aws_profile=None)

    @app.get("/__test__/login-admin")
    async def _login_admin(request: Request):
        request.session["is_admin"] = True
        request.session["user_email"] = "admin@example.com"
        return {"ok": True}

    deps = MonitoringDependencies(state_db=state_db, settings=MagicMock())
    app.include_router(create_monitoring_router(deps))

    log_content = (
        "[2025-01-20T10:30:00Z] ============================================================\n"
        "LABEL: stage_samples\n"
        "COMMAND: ssh stage.sh\n"
        "EXIT_CODE: 0\n"
        "STDOUT:\n"
        "Staging complete\n"
        "\n"
        "[2025-01-20T10:35:00Z] ============================================================\n"
        "LABEL: clone_pipeline\n"
        "COMMAND: git clone repo\n"
        "EXIT_CODE: 0\n"
    )

    mock_body = MagicMock()
    mock_body.read.return_value = log_content.encode("utf-8")

    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": mock_body}
    mock_s3.exceptions = SimpleNamespace(NoSuchKey=type("NoSuchKey", (Exception,), {}))

    mock_session = MagicMock()
    mock_session.client.return_value = mock_s3

    with patch("daylily_ursa.routes.monitoring.boto3.Session", return_value=mock_session):
        with TestClient(app, base_url="https://testserver") as client:
            assert client.get("/__test__/login-admin").status_code == 200
            resp = client.get("/api/v2/admin/worksets/ws-123/command-log")
            assert resp.status_code == 200
            payload = resp.json()
            assert payload["log_available"] is True
            assert payload["entry_count"] >= 1
            assert payload["entries_returned"] >= 1
            assert any("stage_samples" in e for e in payload["entries"])


def test_customers_list_endpoint_returns_customer_rows():
    """GET /customers should serialize CustomerResponse rows."""
    from daylily_ursa.routes.customers import CustomerDependencies, create_customers_router

    customer_manager = MagicMock()
    customer_manager.list_customers.return_value = [
        SimpleNamespace(
            customer_id="cust-001",
            customer_name="Acme",
            email="acme@example.com",
            s3_bucket="acme-bucket",
            max_concurrent_worksets=10,
            max_storage_gb=500,
            billing_account_id=None,
            cost_center=None,
        ),
    ]

    def get_current_user():
        return None

    deps = CustomerDependencies(customer_manager=customer_manager, get_current_user=get_current_user)
    app = FastAPI()
    app.include_router(create_customers_router(deps))

    with TestClient(app, base_url="https://testserver") as client:
        resp = client.get("/api/v2/customers")
        assert resp.status_code == 200
        rows = resp.json()
        assert len(rows) == 1
        assert rows[0]["customer_id"] == "cust-001"
        assert rows[0]["s3_bucket"] == "acme-bucket"


def test_dashboard_cost_breakdown_endpoint_returns_categories_and_total():
    """GET /api/v2/customers/{id}/dashboard/cost-breakdown should render a stable schema."""
    from daylily_ursa.routes.dashboard import DashboardDependencies, create_dashboard_router

    state_db = MagicMock()
    customer_manager = MagicMock()
    customer_manager.get_customer_config.return_value = SimpleNamespace(customer_id="cust-001")

    deps = DashboardDependencies(state_db=state_db, settings=MagicMock(), customer_manager=customer_manager)
    app = FastAPI()
    app.include_router(create_dashboard_router(deps))

    breakdown = {
        "total": 12.34,
        "has_actual_costs": False,
        "compute_cost_usd": 1.0,
        "storage_cost_usd": 2.0,
        "transfer_cost_usd": 3.0,
        "transfer_intra_region_cost_usd": 0.0,
        "transfer_cross_region_cost_usd": 0.5,
        "transfer_internet_cost_usd": 2.5,
        "transfer_intra_region_gb": 0.0,
        "transfer_cross_region_gb": 1.0,
        "transfer_internet_gb": 5.0,
        "storage_gb": 10.0,
        "rates": {},
    }
    with patch("daylily_ursa.billing.calculate_customer_cost_breakdown", return_value=breakdown):
        with TestClient(app, base_url="https://testserver") as client:
            resp = client.get("/api/v2/customers/cust-001/dashboard/cost-breakdown")
            assert resp.status_code == 200
            payload = resp.json()
            assert payload["total"] == 12.34
            assert isinstance(payload["categories"], list)
            assert isinstance(payload["values"], list)
            assert len(payload["categories"]) == len(payload["values"])


def test_files_list_endpoint_returns_folders_and_files():
    """GET /api/v2/customers/{id}/files should return both folder and file rows."""
    from daylily_ursa.routes.files import FileDependencies, create_files_router

    customer_manager = MagicMock()
    customer_manager.get_customer_config.return_value = SimpleNamespace(s3_bucket="cust-bucket")

    mock_s3 = MagicMock()
    mock_s3.list_objects_v2.return_value = {
        "CommonPrefixes": [{"Prefix": "data/folder-a/"}],
        "Contents": [
            {"Key": "data/", "Size": 0, "LastModified": dt.datetime.now(dt.timezone.utc)},
            {"Key": "data/readme.txt", "Size": 12, "LastModified": dt.datetime.now(dt.timezone.utc)},
        ],
    }

    app = FastAPI()
    app.include_router(create_files_router(FileDependencies(customer_manager=customer_manager)))

    with patch("daylily_ursa.routes.files.boto3.client", return_value=mock_s3):
        with TestClient(app, base_url="https://testserver") as client:
            resp = client.get("/api/v2/customers/cust-001/files", params={"prefix": "data/"})
            assert resp.status_code == 200
            payload = resp.json()
            assert payload["bucket"] == "cust-bucket"
            types = {row["type"] for row in payload["files"]}
            assert "folder" in types
            assert "file" in types


def test_customer_worksets_list_filters_to_customer_id_ownership():
    """GET /api/v2/customers/{id}/worksets should enforce customer_id ownership filtering."""
    from daylily_ursa.routes.customer_worksets import CustomerWorksetDependencies, create_customer_worksets_router

    customer_manager = MagicMock()
    customer_manager.get_customer_config.return_value = SimpleNamespace(customer_id="cust-001")

    state_db = MagicMock()
    state_db.list_worksets_by_state.return_value = [
        {"workset_id": "ws-1", "customer_id": "cust-001"},
        {"workset_id": "ws-2", "customer_id": "cust-999"},
    ]

    deps = CustomerWorksetDependencies(
        state_db=state_db,
        settings=MagicMock(),
        customer_manager=customer_manager,
        integration=None,
        manifest_registry=None,
        get_current_user=None,
    )
    app = FastAPI()
    app.include_router(create_customer_worksets_router(deps))

    with TestClient(app, base_url="https://testserver") as client:
        resp = client.get("/api/v2/customers/cust-001/worksets", params={"state": "ready", "limit": 20})
        assert resp.status_code == 200
        payload = resp.json()
        assert [w["workset_id"] for w in payload["worksets"]] == ["ws-1"]


def test_app_inline_utility_endpoints_have_request_level_coverage():
    """Ensure inline endpoints defined in create_app are exercised at request level."""
    from types import SimpleNamespace

    from daylily_ursa.config import get_settings_for_testing
    from daylily_ursa.workset_api import create_app

    mock_state_db = MagicMock()
    mock_validator = MagicMock()
    mock_validator.validate_workset.return_value = SimpleNamespace(
        is_valid=True,
        errors=[],
        warnings=[],
        estimated_cost_usd=0.0,
        estimated_duration_minutes=0,
        estimated_vcpu_hours=0.0,
        estimated_storage_gb=0.0,
    )

    app = create_app(
        state_db=mock_state_db,
        enable_auth=False,
        customer_manager=None,
        validator=mock_validator,
        settings=get_settings_for_testing(),
    )

    with TestClient(app, base_url="https://testserver") as client:
        assert client.post("/api/v2/estimate-cost", json={"pipeline_type": "germline"}).status_code != 404
        assert client.post(
            "/api/v2/worksets/generate-yaml",
            json={"samples": [], "reference_genome": "GRCh38"},
        ).status_code != 404
        assert client.post("/api/v2/worksets/validate?bucket=b&prefix=p").status_code != 404


def test_dashboard_activity_and_cost_history_have_request_level_coverage():
    """GET /api/v2/customers/{id}/dashboard/* should be reachable."""
    from daylily_ursa.routes.dashboard import DashboardDependencies, create_dashboard_router

    state_db = MagicMock()
    state_db.list_worksets_by_state.return_value = []

    customer_manager = MagicMock()
    customer_manager.get_customer_config.return_value = SimpleNamespace(customer_id="cust-001")

    deps = DashboardDependencies(state_db=state_db, settings=MagicMock(), customer_manager=customer_manager)
    app = FastAPI()
    app.include_router(create_dashboard_router(deps))

    with TestClient(app, base_url="https://testserver") as client:
        assert client.get("/api/v2/customers/cust-001/dashboard/activity").status_code != 404
        assert client.get("/api/v2/customers/cust-001/dashboard/cost-history").status_code != 404


def test_manifest_metadata_endpoint_is_registered_even_when_storage_not_configured():
    """GET /api/v2/customers/{id}/manifests/{id} should exist (503 when registry is missing)."""
    from daylily_ursa.routes.manifests import ManifestDependencies, create_manifests_router

    customer_manager = MagicMock()
    deps = ManifestDependencies(customer_manager=customer_manager, manifest_registry=None)
    app = FastAPI()
    app.include_router(create_manifests_router(deps))

    with TestClient(app, base_url="https://testserver") as client:
        resp = client.get("/api/v2/customers/cust-001/manifests/manifest-001")
        assert resp.status_code == 503
