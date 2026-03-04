"""Route-level coverage smoke tests.

These tests exist to ensure key GUI/API route modules have at least one
integration-style request test, so refactors don't leave endpoints unexercised.
"""

from __future__ import annotations

import datetime as dt
import io
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
    """GET /api/clusters should not crash when no regions are configured."""
    from daylib.routes.clusters import ClusterDependencies, create_clusters_router

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
        "daylib.ursa_config.get_ursa_config",
        return_value=SimpleNamespace(is_configured=False, aws_profile=None),
    ):
        with TestClient(app) as client:
            resp = client.get("/api/clusters")
            assert resp.status_code == 200
            payload = resp.json()
            assert payload["clusters"] == []
            assert payload["regions"] == []
            assert "No regions configured" in payload.get("error", "")


def test_monitoring_command_log_endpoint_admin_success_parses_entries():
    """GET /api/admin/worksets/{id}/command-log should return parsed entries for admins."""
    from daylib.routes.monitoring import MonitoringDependencies, create_monitoring_router

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

    with patch("daylib.routes.monitoring.boto3.Session", return_value=mock_session):
        with TestClient(app) as client:
            assert client.get("/__test__/login-admin").status_code == 200
            resp = client.get("/api/admin/worksets/ws-123/command-log")
            assert resp.status_code == 200
            payload = resp.json()
            assert payload["log_available"] is True
            assert payload["entry_count"] >= 1
            assert payload["entries_returned"] >= 1
            assert any("stage_samples" in e for e in payload["entries"])


def test_customers_list_endpoint_returns_customer_rows():
    """GET /customers should serialize CustomerResponse rows."""
    from daylib.routes.customers import CustomerDependencies, create_customers_router

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

    with TestClient(app) as client:
        resp = client.get("/customers")
        assert resp.status_code == 200
        rows = resp.json()
        assert len(rows) == 1
        assert rows[0]["customer_id"] == "cust-001"
        assert rows[0]["s3_bucket"] == "acme-bucket"


def test_dashboard_cost_breakdown_endpoint_returns_categories_and_total():
    """GET /api/customers/{id}/dashboard/cost-breakdown should render a stable schema."""
    from daylib.routes.dashboard import DashboardDependencies, create_dashboard_router

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
    with patch("daylib.billing.calculate_customer_cost_breakdown", return_value=breakdown):
        with TestClient(app) as client:
            resp = client.get("/api/customers/cust-001/dashboard/cost-breakdown")
            assert resp.status_code == 200
            payload = resp.json()
            assert payload["total"] == 12.34
            assert isinstance(payload["categories"], list)
            assert isinstance(payload["values"], list)
            assert len(payload["categories"]) == len(payload["values"])


def test_files_list_endpoint_returns_folders_and_files():
    """GET /api/customers/{id}/files should return both folder and file rows."""
    from daylib.routes.files import FileDependencies, create_files_router

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

    with patch("daylib.routes.files.boto3.client", return_value=mock_s3):
        with TestClient(app) as client:
            resp = client.get("/api/customers/cust-001/files", params={"prefix": "data/"})
            assert resp.status_code == 200
            payload = resp.json()
            assert payload["bucket"] == "cust-bucket"
            types = {row["type"] for row in payload["files"]}
            assert "folder" in types
            assert "file" in types


def test_customer_worksets_list_filters_to_customer_id_ownership():
    """GET /api/customers/{id}/worksets should enforce customer_id ownership filtering."""
    from daylib.routes.customer_worksets import CustomerWorksetDependencies, create_customer_worksets_router

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

    with TestClient(app) as client:
        resp = client.get("/api/customers/cust-001/worksets", params={"state": "ready", "limit": 20})
        assert resp.status_code == 200
        payload = resp.json()
        assert [w["workset_id"] for w in payload["worksets"]] == ["ws-1"]
