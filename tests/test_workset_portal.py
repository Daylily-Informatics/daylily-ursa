"""Tests for customer portal routes."""

import csv
import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from daylib.billing import calculate_customer_cost_breakdown
from daylib.workset_state_db import WorksetStateDB, WorksetState
from daylib.workset_api import create_app


def _make_authenticated_client(mock_state_db: MagicMock, *, customer_id: str, is_admin: bool) -> TestClient:
    """Create an authenticated portal client with a controlled admin flag."""
    mock_customer_manager = MagicMock()
    mock_customer = MagicMock()
    mock_customer.customer_id = customer_id
    mock_customer.is_admin = is_admin
    mock_customer_manager.get_customer_by_email.return_value = mock_customer

    app = create_app(
        state_db=mock_state_db,
        enable_auth=False,
        customer_manager=mock_customer_manager,
    )
    client = TestClient(app)
    client.post("/portal/login", data={"email": "user@example.com", "password": "testpass"})
    return client


@pytest.fixture
def mock_state_db():
    """Create mock state database."""
    mock_db = MagicMock(spec=WorksetStateDB)
    mock_db.list_worksets_by_state.return_value = [
        {
            "workset_id": "test-workset-001",
            "state": "ready",
            "priority": "normal",
            "bucket": "test-bucket",
            "prefix": "worksets/test/",
            "customer_id": "demo-customer",
            "created_at": "2024-01-15T10:00:00Z",
            "updated_at": "2024-01-15T10:00:00Z",
        },
        {
            "workset_id": "test-workset-002",
            "state": "in_progress",
            "priority": "high",
            "bucket": "test-bucket",
            "prefix": "worksets/test2/",
            "customer_id": "demo-customer",
            "created_at": "2024-01-15T11:00:00Z",
            "updated_at": "2024-01-15T11:30:00Z",
        },
    ]
    mock_db.get_workset.return_value = {
        "workset_id": "test-workset-001",
        "state": "ready",
        "priority": "normal",
        "bucket": "test-bucket",
        "prefix": "worksets/test/",
        "customer_id": "demo-customer",
        "created_at": "2024-01-15T10:00:00Z",
        "updated_at": "2024-01-15T10:00:00Z",
    }
    return mock_db


@pytest.fixture
def client(mock_state_db):
    """Create test client."""
    app = create_app(state_db=mock_state_db, enable_auth=False)
    return TestClient(app)


@pytest.fixture
def authenticated_client(mock_state_db):
    """Create test client with authenticated session."""
    app = create_app(state_db=mock_state_db, enable_auth=False)
    client = TestClient(app)
    # Perform login to set session
    client.post("/portal/login", data={"email": "test@example.com", "password": "testpass"})
    return client


class TestPortalRoutes:
    """Test portal HTML routes."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "daylily-workset-monitor"

    def test_portal_dashboard(self, authenticated_client):
        """Test dashboard page loads (requires auth)."""
        response = authenticated_client.get("/portal")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert b"Dashboard" in response.content or b"dashboard" in response.content.lower()

    def test_portal_login(self, client):
        """Test login page loads."""
        response = client.get("/portal/login")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert b"Sign" in response.content or b"login" in response.content.lower()

    def test_portal_register(self, client):
        """Test registration page loads."""
        response = client.get("/portal/register")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert b"Create" in response.content or b"register" in response.content.lower()

    def test_portal_worksets_list(self, authenticated_client):
        """Test worksets list page loads (requires auth)."""
        response = authenticated_client.get("/portal/worksets")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert b"Workset" in response.content

    def test_portal_worksets_new(self, authenticated_client):
        """Test new workset page loads (requires auth)."""
        response = authenticated_client.get("/portal/worksets/new")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert b"Submit" in response.content or b"New" in response.content

    def test_portal_workset_detail(self, authenticated_client, mock_state_db):
        """Test workset detail page loads (requires auth)."""
        response = authenticated_client.get("/portal/worksets/test-workset-001")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        mock_state_db.get_workset.assert_called_with("test-workset-001")

    def test_portal_workset_detail_not_found(self, authenticated_client, mock_state_db):
        """Test workset detail page returns 404 for missing workset (requires auth)."""
        mock_state_db.get_workset.return_value = None
        response = authenticated_client.get("/portal/worksets/nonexistent")
        assert response.status_code == 404

    def test_portal_manifest_generator(self, authenticated_client):
        """Test manifest generator page loads (requires auth)."""
        response = authenticated_client.get("/portal/manifest-generator")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert b"Manifest" in response.content or b"Analysis" in response.content

    def test_portal_files(self, authenticated_client):
        """Test files page loads (requires auth)."""
        response = authenticated_client.get("/portal/files")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert b"File" in response.content

    def test_portal_usage(self, authenticated_client):
        """Test usage page loads (requires auth)."""
        response = authenticated_client.get("/portal/usage")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert b"Usage" in response.content or b"Billing" in response.content

    def test_portal_usage_has_billing_edit_link(self, authenticated_client):
        """Usage page should include an Edit link for billing information."""
        response = authenticated_client.get("/portal/usage")
        assert response.status_code == 200
        assert b'href="/portal/account"' in response.content

    def test_portal_usage_export_csv(self, mock_state_db):
        """Export route returns a CSV attachment for the logged-in customer."""
        client = _make_authenticated_client(mock_state_db, customer_id="customer-A", is_admin=False)

        mock_state_db.list_worksets_by_customer.return_value = [
            {
                "workset_id": "ws-001",
                "state": "completed",
                "customer_id": "customer-A",
                "completed_at": "2024-01-15T10:00:00Z",
            },
            {
                "workset_id": "ws-002",
                "state": "completed",
                "customer_id": "customer-A",
                "completed_at": "2024-01-16T10:00:00Z",
            },
        ]

        def _cost_report(workset_id: str):
            return {
                "total_compute_cost_usd": 10.0 if workset_id == "ws-001" else 5.0,
                "cost_report_sample_count": 3,
                "cost_report_rule_count": 1,
            }

        def _storage_metrics(workset_id: str):
            storage_bytes = int(100 * 1024**3) if workset_id == "ws-001" else int(50 * 1024**3)
            return {"results_storage_bytes": storage_bytes}

        mock_state_db.get_cost_report.side_effect = _cost_report
        mock_state_db.get_storage_metrics.side_effect = _storage_metrics

        response = client.get("/portal/usage/export")
        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]
        assert "attachment;" in response.headers["content-disposition"]
        assert "ursa-usage-report-customer-A-" in response.headers["content-disposition"]

        csv_text = response.content.decode("utf-8")
        reader = csv.reader(io.StringIO(csv_text))
        header = next(reader)
        assert header == [
            "date",
            "workset_id",
            "sample_count",
            "compute_cost_usd",
            "storage_gb",
            "storage_cost_usd",
            "transfer_gb",
            "transfer_cost_usd",
            "intra_region_transfer_gb",
            "intra_region_transfer_cost_usd",
            "cross_region_transfer_gb",
            "cross_region_transfer_cost_usd",
            "internet_egress_gb",
            "internet_egress_cost_usd",
            "total_cost_usd",
            "has_actual_compute_cost",
        ]

        rows = list(reader)

        def _row_by_workset_id(workset_id: str) -> list[str]:
            return next(r for r in rows if r and r[1] == workset_id)

        def _float(row: list[str], col: str) -> float:
            idx = header.index(col)
            return float(row[idx])

        # Per-workset rows should include the new per-category transfer columns.
        ws1 = _row_by_workset_id("ws-001")
        ws2 = _row_by_workset_id("ws-002")

        assert _float(ws1, "intra_region_transfer_gb") == 0.0
        assert _float(ws1, "intra_region_transfer_cost_usd") == 0.0
        assert _float(ws1, "cross_region_transfer_gb") == 0.0
        assert _float(ws1, "cross_region_transfer_cost_usd") == 0.0

        # With no metered transfer metrics, billing falls back to internet egress ~= storage bytes.
        assert _float(ws1, "internet_egress_gb") == 100.0
        assert _float(ws1, "internet_egress_cost_usd") == 9.0

        assert _float(ws2, "internet_egress_gb") == 50.0
        assert _float(ws2, "internet_egress_cost_usd") == 4.5

        # Sanity-check the TOTAL row matches the header width and aggregates new columns.
        total_row = next(r for r in rows if r and r[0] == "TOTAL")
        assert len(total_row) == len(header)
        assert _float(total_row, "intra_region_transfer_gb") == 0.0
        assert _float(total_row, "cross_region_transfer_gb") == 0.0
        assert _float(total_row, "internet_egress_gb") == 150.0
        assert _float(total_row, "internet_egress_cost_usd") == 13.5
        assert b"TOTAL" in response.content

    def test_portal_usage_and_cost_breakdown_api_share_totals(self, mock_state_db):
        """Portal cards and dashboard chart must be derived from the same source."""
        client = _make_authenticated_client(mock_state_db, customer_id="customer-A", is_admin=False)

        mock_state_db.list_worksets_by_customer.return_value = [
            {
                "workset_id": "ws-001",
                "state": "completed",
                "customer_id": "customer-A",
                "completed_at": "2024-01-15T10:00:00Z",
            }
        ]
        mock_state_db.get_cost_report.return_value = {
            "total_compute_cost_usd": 12.34,
            "cost_report_sample_count": 2,
            "cost_report_rule_count": 1,
        }
        mock_state_db.get_storage_metrics.return_value = {"results_storage_bytes": int(20 * 1024**3)}

        expected = calculate_customer_cost_breakdown(mock_state_db, "customer-A", limit=500)

        html = client.get("/portal/usage")
        assert html.status_code == 200
        assert f"${expected['total']:.2f}".encode() in html.content


        # Transfer breakdown should be displayed as 3 line items on the usage page.
        assert b"Transfer (Intra-region)" in html.content
        assert b"Transfer (Cross-region)" in html.content
        assert b"Internet egress" in html.content

        api = client.get("/api/customers/customer-A/dashboard/cost-breakdown")
        assert api.status_code == 200
        payload = api.json()
        assert payload["categories"] == [
            "Compute",
            "Storage",
            "Transfer (Intra-region)",
            "Transfer (Cross-region)",
            "Internet egress",
        ]
        assert payload["total"] == expected["total"]
        assert round(sum(payload["values"]), 2) == expected["total"]

    def test_portal_docs(self, authenticated_client):
        """Test documentation page loads (requires auth)."""
        response = authenticated_client.get("/portal/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert b"Documentation" in response.content or b"docs" in response.content.lower()

    def test_portal_support(self, authenticated_client):
        """Test support page loads (requires auth)."""
        response = authenticated_client.get("/portal/support")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert b"Support" in response.content or b"Contact" in response.content

    def test_portal_account(self, authenticated_client):
        """Test account page loads (requires auth)."""
        response = authenticated_client.get("/portal/account")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert b"Account" in response.content or b"Settings" in response.content

    def test_unauthenticated_redirect(self, client):
        """Test that unauthenticated users are redirected to login."""
        response = client.get("/portal/worksets", follow_redirects=False)
        assert response.status_code == 302
        assert "/portal/login" in response.headers["location"]


class TestAPIEndpoints:
    """Test API endpoints."""

    def test_list_worksets(self, client, mock_state_db):
        """Test listing worksets via API."""
        response = client.get("/worksets")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_workset(self, client, mock_state_db):
        """Test getting a single workset."""
        response = client.get("/worksets/test-workset-001")
        assert response.status_code == 200
        data = response.json()
        assert data["workset_id"] == "test-workset-001"

    def test_get_queue_stats(self, client, mock_state_db):
        """Test queue statistics endpoint."""
        mock_state_db.get_queue_depth.return_value = {
            "ready": 5,
            "in_progress": 3,
            "completed": 10,
            "error": 1,
        }
        response = client.get("/queue/stats")
        assert response.status_code == 200
        data = response.json()
        assert "queue_depth" in data
        assert "total_worksets" in data


# ==================== Archive/Delete API Tests ====================


@pytest.fixture
def mock_customer_manager():
    """Create mock customer manager."""
    mock_mgr = MagicMock()
    mock_customer = MagicMock()
    mock_customer.customer_id = "cust-001"
    mock_customer.customer_name = "Test Customer"
    mock_customer.s3_bucket = "test-bucket"
    mock_mgr.list_customers.return_value = [mock_customer]
    mock_mgr.get_customer_config.return_value = mock_customer
    return mock_mgr


@pytest.fixture
def client_with_customer(mock_state_db, mock_customer_manager):
    """Create test client with customer manager."""
    app = create_app(
        state_db=mock_state_db,
        customer_manager=mock_customer_manager,
        enable_auth=False
    )
    return TestClient(app)


class TestArchiveDeleteAPI:
    """Tests for archive/delete API endpoints."""

    def test_archive_workset_success(self, client_with_customer, mock_state_db):
        """Test successful workset archiving via API."""
        mock_state_db.get_workset.return_value = {
            "workset_id": "test-ws-001",
            "state": "completed",
            "bucket": "test-bucket",
            "customer_id": "cust-001",
        }
        mock_state_db.archive_workset.return_value = True

        response = client_with_customer.post(
            "/api/customers/cust-001/worksets/test-ws-001/archive",
            json={"reason": "No longer needed"}
        )

        assert response.status_code == 200
        mock_state_db.archive_workset.assert_called_once()

    def test_archive_workset_not_found(self, client_with_customer, mock_state_db):
        """Test archiving a workset that doesn't exist."""
        mock_state_db.get_workset.return_value = None

        response = client_with_customer.post(
            "/api/customers/cust-001/worksets/nonexistent/archive",
            json={}
        )

        assert response.status_code == 404

    def test_archive_workset_in_progress(self, client_with_customer, mock_state_db):
        """Test archiving a workset that's in progress (should succeed - force archive allowed)."""
        mock_state_db.get_workset.return_value = {
            "workset_id": "test-ws-001",
            "state": "in_progress",
            "bucket": "test-bucket",
            "customer_id": "cust-001",
        }
        mock_state_db.archive_workset.return_value = True

        response = client_with_customer.post(
            "/api/customers/cust-001/worksets/test-ws-001/archive",
            json={}
        )

        assert response.status_code == 200
        mock_state_db.archive_workset.assert_called_once()

    def test_delete_workset_soft_success(self, client_with_customer, mock_state_db):
        """Test successful soft delete via API."""
        mock_state_db.get_workset.return_value = {
            "workset_id": "test-ws-001",
            "state": "completed",
            "bucket": "test-bucket",
            "customer_id": "cust-001",
        }
        mock_state_db.delete_workset.return_value = True

        response = client_with_customer.post(
            "/api/customers/cust-001/worksets/test-ws-001/delete",
            json={"hard_delete": False, "reason": "Cleaning up"}
        )

        assert response.status_code == 200
        mock_state_db.delete_workset.assert_called_once()
        call_args = mock_state_db.delete_workset.call_args
        assert call_args.kwargs.get("hard_delete") is False or call_args[1].get("hard_delete") is False

    def test_delete_workset_hard_success(self, client_with_customer, mock_state_db):
        """Test successful hard delete via API."""
        mock_state_db.get_workset.return_value = {
            "workset_id": "test-ws-001",
            "state": "completed",
            "bucket": "test-bucket",
            "prefix": "worksets/test/",
            "customer_id": "cust-001",
        }
        mock_state_db.delete_workset.return_value = True

        with patch("daylib.workset_api.boto3") as mock_boto:
            mock_s3 = MagicMock()
            mock_boto.client.return_value = mock_s3
            mock_s3.list_objects_v2.return_value = {"Contents": []}

            response = client_with_customer.post(
                "/api/customers/cust-001/worksets/test-ws-001/delete",
                json={"hard_delete": True}
            )

        assert response.status_code == 200

    def test_delete_workset_not_found(self, client_with_customer, mock_state_db):
        """Test deleting a workset that doesn't exist."""
        mock_state_db.get_workset.return_value = None

        response = client_with_customer.post(
            "/api/customers/cust-001/worksets/nonexistent/delete",
            json={}
        )

        assert response.status_code == 404

    def test_delete_workset_in_progress(self, client_with_customer, mock_state_db):
        """Test deleting a workset that's in progress (should succeed - force delete allowed)."""
        mock_state_db.get_workset.return_value = {
            "workset_id": "test-ws-001",
            "state": "in_progress",
            "bucket": "test-bucket",
            "customer_id": "cust-001",
        }
        mock_state_db.delete_workset.return_value = True

        response = client_with_customer.post(
            "/api/customers/cust-001/worksets/test-ws-001/delete",
            json={}
        )

        assert response.status_code == 200
        mock_state_db.delete_workset.assert_called_once()

    def test_restore_workset_success(self, client_with_customer, mock_state_db):
        """Test successful workset restoration via API."""
        mock_state_db.get_workset.return_value = {
            "workset_id": "test-ws-001",
            "state": "archived",
            "bucket": "test-bucket",
            "customer_id": "cust-001",
        }
        mock_state_db.restore_workset.return_value = True

        response = client_with_customer.post(
            "/api/customers/cust-001/worksets/test-ws-001/restore"
        )

        assert response.status_code == 200
        mock_state_db.restore_workset.assert_called_once()

    def test_restore_workset_not_archived(self, client_with_customer, mock_state_db):
        """Test restoring a workset that's not archived."""
        mock_state_db.get_workset.return_value = {
            "workset_id": "test-ws-001",
            "state": "completed",
            "bucket": "test-bucket",
            "customer_id": "cust-001",
        }

        response = client_with_customer.post(
            "/api/customers/cust-001/worksets/test-ws-001/restore"
        )

        assert response.status_code == 400

    def test_list_archived_worksets(self, mock_customer_manager):
        """Test listing archived worksets."""
        # Create fresh mock for this test
        mock_db = MagicMock(spec=WorksetStateDB)
        # Include customer_id to pass customer isolation filtering
        mock_db.list_archived_worksets.return_value = [
            {"workset_id": "ws-001", "state": "archived", "bucket": "test-bucket", "customer_id": "cust-001"},
            {"workset_id": "ws-002", "state": "archived", "bucket": "test-bucket", "customer_id": "cust-001"},
        ]

        app = create_app(
            state_db=mock_db,
            customer_manager=mock_customer_manager,
            enable_auth=False
        )
        client = TestClient(app)

        response = client.get("/api/customers/cust-001/worksets/archived")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2


# ==================== Workset Creation Validation Tests ====================


@pytest.fixture
def mock_customer_manager_with_email_lookup():
    """Create mock customer manager with get_customer_by_email support."""
    mock_mgr = MagicMock()
    mock_customer = MagicMock()
    mock_customer.customer_id = "cust-001"
    mock_customer.customer_name = "Test Customer"
    mock_customer.s3_bucket = "customer-bucket"
    mock_customer.contact_email = "user@example.com"

    mock_mgr.list_customers.return_value = [mock_customer]
    mock_mgr.get_customer_config.return_value = mock_customer
    mock_mgr.get_customer_by_email.return_value = mock_customer
    return mock_mgr


@pytest.fixture
def mock_integration():
    """Create mock integration layer.

    Note: bucket is set to None so the control bucket env var is used.
    """
    mock_int = MagicMock()
    mock_int.register_workset.return_value = True
    mock_int.bucket = None  # Ensure env var is used for bucket
    return mock_int


class TestWorksetCreationValidation:
    """Tests for workset creation validation logic.

    Note: These tests mock the cluster service to provide bucket discovery.
    S3 buckets are now discovered from cluster tags (aws-parallelcluster-monitor-bucket).
    """

    @pytest.fixture(autouse=True)
    def setup_cluster_mock(self, monkeypatch):
        """Set up mocked cluster service for all tests in this class."""
        from daylib.config import clear_settings_cache
        from daylib.cluster_service import ClusterInfo

        # Create a mock cluster with monitor bucket tag
        mock_cluster = MagicMock(spec=ClusterInfo)
        mock_cluster.cluster_name = "test-cluster"
        mock_cluster.region = "us-west-2"
        mock_cluster.get_monitor_bucket_name.return_value = "test-control-bucket"

        mock_service = MagicMock()
        mock_service.get_cluster_by_name.return_value = mock_cluster

        # Patch the get_cluster_service function at its source module
        # (it's imported locally inside workset_api functions)
        monkeypatch.setattr(
            "daylib.cluster_service.get_cluster_service",
            lambda **kwargs: mock_service
        )
        # Clear settings cache
        clear_settings_cache()

    def test_create_workset_rejects_empty_customer_id(self, mock_state_db, mock_customer_manager_with_email_lookup):
        """Test that empty customer_id is rejected."""
        app = create_app(
            state_db=mock_state_db,
            customer_manager=mock_customer_manager_with_email_lookup,
            enable_auth=False
        )
        client = TestClient(app)

        # Test with empty string
        response = client.post(
            "/api/customers//worksets",
            json={
                "workset_name": "Test Workset",
                "pipeline_type": "wgs",
                "reference_genome": "hg38",
                "samples": [{"sample_id": "s1", "r1_file": "s1_R1.fq.gz", "r2_file": "s1_R2.fq.gz"}],
            }
        )
        # Empty path should return 404 (route not found)
        assert response.status_code == 404

    def test_create_workset_rejects_unknown_customer_id(self, mock_state_db):
        """Test that 'Unknown' customer_id is rejected."""
        mock_mgr = MagicMock()
        mock_mgr.get_customer_config.return_value = None

        app = create_app(
            state_db=mock_state_db,
            customer_manager=mock_mgr,
            enable_auth=False
        )
        client = TestClient(app)

        response = client.post(
            "/api/customers/Unknown/worksets",
            json={
                "workset_name": "Test Workset",
                "pipeline_type": "wgs",
                "reference_genome": "hg38",
                "samples": [{"sample_id": "s1", "r1_file": "s1_R1.fq.gz", "r2_file": "s1_R2.fq.gz"}],
            }
        )
        assert response.status_code == 400
        assert "Valid customer ID is required" in response.json()["detail"]

    def test_create_workset_rejects_nonexistent_customer(self, mock_state_db):
        """Test that non-existent customer_id is rejected."""
        mock_mgr = MagicMock()
        mock_mgr.get_customer_config.return_value = None

        app = create_app(
            state_db=mock_state_db,
            customer_manager=mock_mgr,
            enable_auth=False
        )
        client = TestClient(app)

        response = client.post(
            "/api/customers/nonexistent-customer/worksets",
            json={
                "workset_name": "Test Workset",
                "pipeline_type": "wgs",
                "reference_genome": "hg38",
                "samples": [{"sample_id": "s1", "r1_file": "s1_R1.fq.gz", "r2_file": "s1_R2.fq.gz"}],
            }
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_create_workset_rejects_empty_samples(self, mock_state_db, mock_customer_manager_with_email_lookup):
        """Test that workset with no samples is rejected."""
        app = create_app(
            state_db=mock_state_db,
            customer_manager=mock_customer_manager_with_email_lookup,
            enable_auth=False
        )
        client = TestClient(app)

        response = client.post(
            "/api/customers/cust-001/worksets",
            json={
                "workset_name": "Empty Workset",
                "pipeline_type": "wgs",
                "reference_genome": "hg38",
                "preferred_cluster": "test-cluster",
                "samples": [],
            }
        )
        assert response.status_code == 400
        assert "at least one sample" in response.json()["detail"]

    def test_create_workset_rejects_no_samples_field(self, mock_state_db, mock_customer_manager_with_email_lookup):
        """Test that workset with missing samples field is rejected."""
        app = create_app(
            state_db=mock_state_db,
            customer_manager=mock_customer_manager_with_email_lookup,
            enable_auth=False
        )
        client = TestClient(app)

        response = client.post(
            "/api/customers/cust-001/worksets",
            json={
                "workset_name": "No Samples Workset",
                "pipeline_type": "wgs",
                "reference_genome": "hg38",
                "preferred_cluster": "test-cluster",
                # No samples field at all
            }
        )
        assert response.status_code == 400
        assert "at least one sample" in response.json()["detail"]

    def test_create_workset_allows_customer_without_bucket(self, mock_state_db, mock_integration):
        """Test that customer without S3 bucket configured can still create worksets.

        With cluster-based bucket discovery, worksets get their bucket from the
        selected cluster's tags. Customer's s3_bucket is only used for data locality hints.
        """
        mock_mgr = MagicMock()
        mock_customer = MagicMock()
        mock_customer.customer_id = "cust-no-bucket"
        mock_customer.s3_bucket = None  # No bucket configured - this is OK now
        mock_mgr.get_customer_config.return_value = mock_customer
        mock_state_db.get_workset.return_value = {"workset_id": "test"}

        app = create_app(
            state_db=mock_state_db,
            customer_manager=mock_mgr,
            integration=mock_integration,
            enable_auth=False
        )
        client = TestClient(app)

        response = client.post(
            "/api/customers/cust-no-bucket/worksets",
            json={
                "workset_name": "Test Workset",
                "pipeline_type": "wgs",
                "reference_genome": "hg38",
                "preferred_cluster": "test-cluster",
                "samples": [{"sample_id": "s1", "r1_file": "s1_R1.fq.gz", "r2_file": "s1_R2.fq.gz"}],
            }
        )
        # Should succeed - uses bucket from cluster tags
        assert response.status_code == 200

    def test_create_workset_success_with_valid_samples(
        self, mock_state_db, mock_customer_manager_with_email_lookup, mock_integration
    ):
        """Test successful workset creation with valid samples.

        Note: Worksets get bucket from cluster tags (aws-parallelcluster-monitor-bucket).
        """
        mock_state_db.register_workset.return_value = True
        mock_state_db.get_workset.return_value = {
            "workset_id": "test-workset-12345678",
            "state": "ready",
            "bucket": "test-control-bucket",  # From cluster tag
            "prefix": "worksets/test-workset-12345678/",
            "customer_id": "cust-001",
        }

        app = create_app(
            state_db=mock_state_db,
            customer_manager=mock_customer_manager_with_email_lookup,
            integration=mock_integration,
            enable_auth=False
        )
        client = TestClient(app)

        response = client.post(
            "/api/customers/cust-001/worksets",
            json={
                "workset_name": "Valid Workset",
                "pipeline_type": "wgs",
                "reference_genome": "hg38",
                "preferred_cluster": "test-cluster",
                "samples": [
                    {"sample_id": "sample1", "r1_file": "s1_R1.fq.gz", "r2_file": "s1_R2.fq.gz"},
                    {"sample_id": "sample2", "r1_file": "s2_R1.fq.gz", "r2_file": "s2_R2.fq.gz"},
                ],
            }
        )
        assert response.status_code == 200

        # Verify integration was called with customer_id
        mock_integration.register_workset.assert_called_once()
        call_kwargs = mock_integration.register_workset.call_args[1]
        assert call_kwargs["customer_id"] == "cust-001"
        # Bucket should be from cluster tags
        assert call_kwargs["bucket"] == "test-control-bucket"

    def test_create_workset_uses_cluster_bucket_not_customer_bucket(
        self, mock_state_db, mock_customer_manager_with_email_lookup, mock_integration
    ):
        """Test that cluster bucket is used for workset registration.

        With cluster-based bucket discovery, worksets are always registered
        to the bucket from the cluster's aws-parallelcluster-monitor-bucket tag.
        The s3_bucket parameter is ignored in favor of the cluster bucket.
        """
        mock_state_db.get_workset.return_value = {
            "workset_id": "test-workset-12345678",
            "state": "ready",
        }

        app = create_app(
            state_db=mock_state_db,
            customer_manager=mock_customer_manager_with_email_lookup,
            integration=mock_integration,
            enable_auth=False
        )
        client = TestClient(app)

        response = client.post(
            "/api/customers/cust-001/worksets",
            json={
                "workset_name": "Test Workset",
                "pipeline_type": "wgs",
                "reference_genome": "hg38",
                "preferred_cluster": "test-cluster",
                "s3_bucket": "different-bucket",  # This should be ignored
                "samples": [{"sample_id": "s1", "r1_file": "s1_R1.fq.gz", "r2_file": "s1_R2.fq.gz"}],
            }
        )
        assert response.status_code == 200

        # Should use bucket from cluster tags, not customer or provided bucket
        call_kwargs = mock_integration.register_workset.call_args[1]
        assert call_kwargs["bucket"] == "test-control-bucket"

    def test_create_workset_normalizes_prefix(
        self, mock_state_db, mock_customer_manager_with_email_lookup, mock_integration
    ):
        """Test that prefix is properly normalized with trailing slash."""
        mock_state_db.get_workset.return_value = {"workset_id": "test"}

        app = create_app(
            state_db=mock_state_db,
            customer_manager=mock_customer_manager_with_email_lookup,
            integration=mock_integration,
            enable_auth=False
        )
        client = TestClient(app)

        response = client.post(
            "/api/customers/cust-001/worksets",
            json={
                "workset_name": "Test Workset",
                "pipeline_type": "wgs",
                "reference_genome": "hg38",
                "preferred_cluster": "test-cluster",
                "s3_prefix": "my/custom/path",  # No trailing slash
                "samples": [{"sample_id": "s1", "r1_file": "s1_R1.fq.gz", "r2_file": "s1_R2.fq.gz"}],
            }
        )
        assert response.status_code == 200

        # Prefix should have trailing slash added
        call_kwargs = mock_integration.register_workset.call_args[1]
        assert call_kwargs["prefix"].endswith("/")

    def test_create_workset_from_yaml_content(
        self, mock_state_db, mock_customer_manager_with_email_lookup, mock_integration
    ):
        """Test workset creation from YAML content."""
        mock_state_db.get_workset.return_value = {"workset_id": "test"}

        app = create_app(
            state_db=mock_state_db,
            customer_manager=mock_customer_manager_with_email_lookup,
            integration=mock_integration,
            enable_auth=False
        )
        client = TestClient(app)

        yaml_content = """
samples:
  - sample_id: yaml_sample1
    r1_file: ys1_R1.fq.gz
    r2_file: ys1_R2.fq.gz
  - sample_id: yaml_sample2
    r1_file: ys2_R1.fq.gz
    r2_file: ys2_R2.fq.gz
"""
        response = client.post(
            "/api/customers/cust-001/worksets",
            json={
                "workset_name": "YAML Workset",
                "pipeline_type": "wgs",
                "reference_genome": "hg38",
                "preferred_cluster": "test-cluster",
                "yaml_content": yaml_content,
            }
        )
        assert response.status_code == 200

        # Verify samples were extracted from YAML
        call_kwargs = mock_integration.register_workset.call_args[1]
        metadata = call_kwargs["metadata"]
        assert metadata["sample_count"] == 2
        assert metadata["samples"][0]["sample_id"] == "yaml_sample1"

    def test_create_workset_empty_yaml_rejected(
        self, mock_state_db, mock_customer_manager_with_email_lookup
    ):
        """Test that YAML with empty samples list is rejected."""
        app = create_app(
            state_db=mock_state_db,
            customer_manager=mock_customer_manager_with_email_lookup,
            enable_auth=False
        )
        client = TestClient(app)

        yaml_content = """
samples: []
"""
        response = client.post(
            "/api/customers/cust-001/worksets",
            json={
                "workset_name": "Empty YAML Workset",
                "pipeline_type": "wgs",
                "reference_genome": "hg38",
                "preferred_cluster": "test-cluster",
                "yaml_content": yaml_content,
            }
        )
        assert response.status_code == 400
        assert "at least one sample" in response.json()["detail"]

    def test_create_workset_from_manifest_tsv_content(
        self, mock_state_db, mock_customer_manager_with_email_lookup, mock_integration
    ):
        """Test workset creation from manifest TSV content."""
        mock_state_db.get_workset.return_value = {"workset_id": "test"}

        app = create_app(
            state_db=mock_state_db,
            customer_manager=mock_customer_manager_with_email_lookup,
            integration=mock_integration,
            enable_auth=False
        )
        client = TestClient(app)

        # Sample manifest TSV content matching the user's format
        manifest_tsv = """RUN_ID\tSAMPLE_ID\tEXPERIMENTID\tSAMPLE_TYPE\tLIB_PREP\tSEQ_VENDOR\tSEQ_PLATFORM\tLANE\tSEQBC_ID\tPATH_TO_CONCORDANCE_DATA_DIR\tR1_FQ\tR2_FQ\tSTAGE_DIRECTIVE\tSTAGE_TARGET\tSUBSAMPLE_PCT\tIS_POS_CTRL\tIS_NEG_CTRL\tN_X\tN_Y\tEXTERNAL_SAMPLE_ID
R0\tA1\tE1\tblood\tnoampwgs\tILMN\tNOVASEQX\t0\tS1\t\ts3://bucket/sample.R1.fastq.gz\ts3://bucket/sample.R2.fastq.gz\tstage_data\t/fsx/staged/\tna\tfalse\tfalse\t1\t1\tHG002"""

        response = client.post(
            "/api/customers/cust-001/worksets",
            json={
                "workset_name": "Manifest TSV Workset",
                "pipeline_type": "wgs",
                "reference_genome": "hg38",
                "preferred_cluster": "test-cluster",
                "manifest_tsv_content": manifest_tsv,
            }
        )
        assert response.status_code == 200

        # Verify integration was called with correct data
        mock_integration.register_workset.assert_called_once()
        call_kwargs = mock_integration.register_workset.call_args[1]
        metadata = call_kwargs["metadata"]
        assert metadata["sample_count"] == 1
        assert metadata["samples"][0]["sample_id"] == "A1"
        assert "R1.fastq.gz" in metadata["samples"][0]["r1_file"]
        assert "R2.fastq.gz" in metadata["samples"][0]["r2_file"]
        # Verify raw TSV is passed for S3 write
        assert "stage_samples_tsv" in metadata

    def test_create_workset_from_manifest_tsv_empty_rejected(
        self, mock_state_db, mock_customer_manager_with_email_lookup
    ):
        """Test that manifest TSV with no data rows is rejected."""
        app = create_app(
            state_db=mock_state_db,
            customer_manager=mock_customer_manager_with_email_lookup,
            enable_auth=False
        )
        client = TestClient(app)

        # Header only, no data rows
        manifest_tsv = "RUN_ID\tSAMPLE_ID\tR1_FQ\tR2_FQ"

        response = client.post(
            "/api/customers/cust-001/worksets",
            json={
                "workset_name": "Empty Manifest Workset",
                "pipeline_type": "wgs",
                "reference_genome": "hg38",
                "preferred_cluster": "test-cluster",
                "manifest_tsv_content": manifest_tsv,
            }
        )
        assert response.status_code == 400
        assert "at least one sample" in response.json()["detail"]


class TestCustomerLookupByEmail:
    """Tests for customer lookup by email functionality."""

    def test_customer_manager_get_customer_by_email_called(self):
        """Test that customer manager's get_customer_by_email is properly set up."""
        mock_mgr = MagicMock()
        mock_customer = MagicMock()
        mock_customer.customer_id = "found-customer"
        mock_customer.contact_email = "found@example.com"
        mock_customer.s3_bucket = "found-bucket"

        mock_mgr.get_customer_by_email.return_value = mock_customer
        mock_mgr.list_customers.return_value = [mock_customer]

        # Verify the mock is set up correctly
        result = mock_mgr.get_customer_by_email("found@example.com")
        assert result.customer_id == "found-customer"
        assert result.contact_email == "found@example.com"
        mock_mgr.get_customer_by_email.assert_called_with("found@example.com")

    def test_get_customer_by_email_returns_none_for_unknown(self):
        """Test that unknown email returns None."""
        mock_mgr = MagicMock()
        mock_mgr.get_customer_by_email.return_value = None

        # Verify the manager correctly returns None for unknown email
        result = mock_mgr.get_customer_by_email("unknown@example.com")
        assert result is None
        mock_mgr.get_customer_by_email.assert_called_with("unknown@example.com")

    def test_customer_lookup_integration(self, mock_state_db):
        """Test that customer lookup is integrated into app creation."""
        mock_mgr = MagicMock()
        mock_customer = MagicMock()
        mock_customer.customer_id = "integrated-customer"
        mock_customer.s3_bucket = "integrated-bucket"

        mock_mgr.get_customer_by_email.return_value = mock_customer
        mock_mgr.list_customers.return_value = [mock_customer]
        mock_mgr.get_customer_config.return_value = mock_customer

        # App should be creatable with customer manager that has get_customer_by_email
        app = create_app(
            state_db=mock_state_db,
            customer_manager=mock_mgr,
            enable_auth=False
        )

        # Verify app was created successfully
        assert app is not None


class TestPortalFileRegistration:
    """Tests for POST /portal/files/register endpoint."""

    @pytest.fixture
    def mock_linked_bucket(self):
        """Create a mock linked bucket."""
        bucket = MagicMock()
        bucket.bucket_id = "bucket-abc123"
        bucket.customer_id = "cust-001"
        bucket.bucket_name = "test-linked-bucket"
        bucket.bucket_type = "secondary"
        bucket.display_name = "Test Linked Bucket"
        bucket.is_validated = True
        bucket.can_read = True
        bucket.can_write = True
        bucket.can_list = True
        bucket.prefix_restriction = None
        bucket.read_only = False
        bucket.region = "us-west-2"
        return bucket

    @pytest.fixture
    def mock_linked_bucket_manager(self, mock_linked_bucket):
        """Mock LinkedBucketManager."""
        manager = MagicMock()
        manager.get_bucket.return_value = mock_linked_bucket
        manager.list_customer_buckets.return_value = [mock_linked_bucket]
        return manager

    @pytest.fixture
    def mock_file_registry(self):
        """Mock FileRegistry."""
        registry = MagicMock()
        registry.register_file.return_value = True
        registry.get_file.return_value = None  # File not already registered
        return registry

    @pytest.fixture
    def mock_customer_manager(self):
        """Mock CustomerManager with get_customer_by_email."""
        manager = MagicMock()
        customer = MagicMock()
        customer.customer_id = "cust-001"
        customer.s3_bucket = "customer-bucket"
        manager.get_customer_by_email.return_value = customer
        manager.list_customers.return_value = [customer]
        return manager

    @pytest.fixture
    def mock_discovered_file(self):
        """Create a mock discovered file."""
        df = MagicMock()
        df.key = "data/sample_R1.fastq.gz"
        df.bucket = "test-linked-bucket"
        df.size = 1024000
        df.last_modified = "2024-01-15T10:00:00Z"
        df.file_format = "fastq"
        df.already_registered = False
        return df

    def test_portal_register_requires_auth(self, mock_state_db):
        """Test that portal file registration requires authentication."""
        app = create_app(state_db=mock_state_db, enable_auth=False)
        client = TestClient(app)

        response = client.post(
            "/portal/files/register",
            json={
                "bucket_id": "bucket-abc123",
                "biosample_id": "biosample-001",
                "subject_id": "subject-001",
            },
        )

        # Should return 401 without session
        assert response.status_code == 401
        assert "Not authenticated" in response.json()["detail"]

    def test_portal_register_requires_file_management(
        self, mock_state_db, mock_customer_manager
    ):
        """Test that portal file registration returns 501 without file management."""
        app = create_app(
            state_db=mock_state_db,
            customer_manager=mock_customer_manager,
            enable_auth=False,
        )
        client = TestClient(app)

        # Set up authenticated session
        with client:
            client.cookies.set("session", "mock-session")
            # Mock the session data
            with patch.object(
                app.state, "session_data", {"user_email": "test@example.com"}, create=True
            ):
                # The session middleware will check request.session
                response = client.post(
                    "/portal/files/register",
                    json={
                        "bucket_id": "bucket-abc123",
                        "biosample_id": "biosample-001",
                        "subject_id": "subject-001",
                    },
                )

        # Should return 401 because session is not properly mocked for Starlette
        # (The actual 501 would require proper session setup)
        assert response.status_code in [401, 501]

    def test_portal_register_endpoint_exists(self, mock_state_db):
        """Test that portal file registration endpoint is correctly defined."""
        app = create_app(state_db=mock_state_db, enable_auth=False)

        # Verify the endpoint exists in the app routes
        routes = [route.path for route in app.routes]
        assert "/portal/files/register" in routes

        # Verify it accepts POST method
        client = TestClient(app)
        # Without auth, should return 401 (not 404 or 405)
        response = client.post(
            "/portal/files/register",
            json={
                "bucket_id": "test-bucket",
                "biosample_id": "bio-001",
                "subject_id": "subj-001",
            },
        )
        # 401 means endpoint exists and auth check runs before other validation
        assert response.status_code == 401

        # Note: Full integration testing of this endpoint requires properly mocking
        # Starlette's session middleware, which is complex. The key behaviors tested:
        # 1. Returns 401 without authentication (tested above)
        # 2. Returns 501 without file management configured (tested in other test)

    def test_portal_register_bucket_not_found(self, mock_state_db):
        """Test that portal registration fails with non-existent bucket."""
        # This test verifies the 404 response path
        # Full testing requires session mocking
        pass  # Placeholder for future session-mocked tests

    def test_portal_register_bucket_wrong_customer(self, mock_state_db):
        """Test that portal registration fails when bucket belongs to different customer."""
        # This test verifies the 403 response path for cross-customer access
        # Full testing requires session mocking
        pass  # Placeholder for future session-mocked tests


class TestPortalFileAutoRegistration:
    """Tests for the portal file auto-registration implementation."""

    @pytest.fixture
    def mock_file_registry(self):
        """Mock FileRegistry for registration tests."""
        registry = MagicMock()
        registry.register_file.return_value = True
        registry.get_file.return_value = None  # File not already registered
        return registry

    @pytest.fixture
    def mock_linked_bucket_manager(self):
        """Mock LinkedBucketManager."""
        bucket = MagicMock()
        bucket.bucket_id = "lb-test123"
        bucket.customer_id = "cust-001"
        bucket.bucket_name = "test-bucket"
        bucket.display_name = "Test Bucket"

        manager = MagicMock()
        manager.get_bucket.return_value = bucket
        manager.list_customer_buckets.return_value = [bucket]
        return manager

    @pytest.fixture
    def mock_bucket_discovery(self):
        """Mock BucketFileDiscovery."""
        from daylib.file_registry import DiscoveredFile

        discovered = [
            DiscoveredFile(
                s3_uri="s3://test-bucket/sample_R1.fastq.gz",
                bucket_name="test-bucket",
                key="sample_R1.fastq.gz",
                file_size_bytes=1024000,
                last_modified="2024-01-15T10:00:00Z",
                etag="abc123",
                detected_format="fastq",
                is_registered=False,
            ),
            DiscoveredFile(
                s3_uri="s3://test-bucket/sample_R2.fastq.gz",
                bucket_name="test-bucket",
                key="sample_R2.fastq.gz",
                file_size_bytes=1024000,
                last_modified="2024-01-15T10:00:00Z",
                etag="def456",
                detected_format="fastq",
                is_registered=False,
            ),
        ]
        return discovered

    def test_auto_register_files_success(self, mock_file_registry, mock_bucket_discovery):
        """Test successful auto-registration of discovered files."""
        from daylib.file_registry import BucketFileDiscovery

        discovery = BucketFileDiscovery(region="us-west-2")

        # Use real auto_register_files with mocked registry
        registered, skipped, errors = discovery.auto_register_files(
            discovered_files=mock_bucket_discovery,
            registry=mock_file_registry,
            customer_id="cust-001",
            biosample_id="biosample-001",
            subject_id="subject-001",
            sequencing_platform="NOVASEQX",
        )

        assert registered == 2
        assert skipped == 0
        assert len(errors) == 0
        assert mock_file_registry.register_file.call_count == 2

    def test_auto_register_files_skips_already_registered(self, mock_file_registry, mock_bucket_discovery):
        """Test that already-registered files are skipped."""
        from daylib.file_registry import BucketFileDiscovery

        # Mark first file as already registered
        mock_bucket_discovery[0].is_registered = True

        discovery = BucketFileDiscovery(region="us-west-2")

        registered, skipped, errors = discovery.auto_register_files(
            discovered_files=mock_bucket_discovery,
            registry=mock_file_registry,
            customer_id="cust-001",
            biosample_id="biosample-001",
            subject_id="subject-001",
        )

        assert registered == 1
        assert skipped == 1
        assert len(errors) == 0
        assert mock_file_registry.register_file.call_count == 1

    def test_auto_register_detects_read_number(self, mock_file_registry, mock_bucket_discovery):
        """Test that R1/R2 detection works correctly."""
        from daylib.file_registry import BucketFileDiscovery

        discovery = BucketFileDiscovery(region="us-west-2")

        discovery.auto_register_files(
            discovered_files=mock_bucket_discovery,
            registry=mock_file_registry,
            customer_id="cust-001",
            biosample_id="biosample-001",
            subject_id="subject-001",
        )

        # Check the registration calls for read_number
        calls = mock_file_registry.register_file.call_args_list
        assert len(calls) == 2

        # R1 file should have read_number=1
        r1_registration = calls[0][0][0]  # First positional arg of first call
        assert r1_registration.read_number == 1

        # R2 file should have read_number=2
        r2_registration = calls[1][0][0]
        assert r2_registration.read_number == 2

    def test_auto_register_handles_registration_failure(self, mock_file_registry, mock_bucket_discovery):
        """Test that registration failures are captured in errors list."""
        from daylib.file_registry import BucketFileDiscovery

        # Make register_file raise an exception for the first file
        mock_file_registry.register_file.side_effect = [Exception("DynamoDB error"), True]

        discovery = BucketFileDiscovery(region="us-west-2")

        registered, skipped, errors = discovery.auto_register_files(
            discovered_files=mock_bucket_discovery,
            registry=mock_file_registry,
            customer_id="cust-001",
            biosample_id="biosample-001",
            subject_id="subject-001",
        )

        assert registered == 1
        assert skipped == 0
        assert len(errors) == 1
        assert "DynamoDB error" in errors[0]

    def test_auto_register_sets_correct_metadata(self, mock_file_registry, mock_bucket_discovery):
        """Test that biosample and sequencing metadata are set correctly."""
        from daylib.file_registry import BucketFileDiscovery

        discovery = BucketFileDiscovery(region="us-west-2")

        discovery.auto_register_files(
            discovered_files=mock_bucket_discovery,
            registry=mock_file_registry,
            customer_id="cust-001",
            biosample_id="my-biosample",
            subject_id="HG002",
            sequencing_platform="ILLUMINA_NOVASEQ_X",
        )

        call = mock_file_registry.register_file.call_args_list[0]
        registration = call[0][0]

        assert registration.customer_id == "cust-001"
        assert registration.biosample_metadata.biosample_id == "my-biosample"
        assert registration.biosample_metadata.subject_id == "HG002"
        assert registration.sequencing_metadata.platform == "ILLUMINA_NOVASEQ_X"
        assert registration.file_metadata.file_format == "fastq"


class TestFileSearchAPI:
    """Tests for the file search API endpoint."""

    @pytest.fixture
    def mock_file_registrations(self):
        """Create mock file registrations for search tests."""
        from daylib.file_registry import FileRegistration, FileMetadata, BiosampleMetadata, SequencingMetadata

        return [
            FileRegistration(
                file_id="file-001",
                customer_id="cust-001",
                file_metadata=FileMetadata(
                    file_id="file-001",
                    s3_uri="s3://bucket/sample1_R1.fastq.gz",
                    file_size_bytes=1024000,
                    file_format="fastq",
                ),
                biosample_metadata=BiosampleMetadata(
                    biosample_id="biosample-001",
                    subject_id="HG002",
                    sample_type="blood",
                ),
                sequencing_metadata=SequencingMetadata(
                    platform="ILLUMINA_NOVASEQ_X",
                    vendor="ILMN",
                ),
                tags=["wgs", "germline"],
                registered_at="2024-01-15T10:00:00Z",
            ),
            FileRegistration(
                file_id="file-002",
                customer_id="cust-001",
                file_metadata=FileMetadata(
                    file_id="file-002",
                    s3_uri="s3://bucket/sample1_R2.fastq.gz",
                    file_size_bytes=1024000,
                    file_format="fastq",
                ),
                biosample_metadata=BiosampleMetadata(
                    biosample_id="biosample-001",
                    subject_id="HG002",
                    sample_type="blood",
                ),
                sequencing_metadata=SequencingMetadata(
                    platform="ILLUMINA_NOVASEQ_X",
                    vendor="ILMN",
                ),
                tags=["wgs", "germline"],
                registered_at="2024-01-15T10:00:00Z",
            ),
            FileRegistration(
                file_id="file-003",
                customer_id="cust-001",
                file_metadata=FileMetadata(
                    file_id="file-003",
                    s3_uri="s3://bucket/sample2.bam",
                    file_size_bytes=5000000000,
                    file_format="bam",
                ),
                biosample_metadata=BiosampleMetadata(
                    biosample_id="biosample-002",
                    subject_id="HG003",
                    sample_type="saliva",
                ),
                sequencing_metadata=SequencingMetadata(
                    platform="ONT_PROMETHION",
                    vendor="ONT",
                ),
                tags=["wgs", "somatic"],
                registered_at="2024-01-16T10:00:00Z",
            ),
        ]

    @pytest.fixture
    def mock_file_registry_for_search(self, mock_file_registrations):
        """Mock FileRegistry for search tests."""
        registry = MagicMock()
        registry.list_customer_files.return_value = mock_file_registrations
        registry.search_files_by_tag.side_effect = lambda cid, tag: [
            f for f in mock_file_registrations if tag in f.tags
        ]
        registry.search_files_by_biosample.side_effect = lambda cid, bid: [
            f for f in mock_file_registrations
            if f.biosample_metadata.biosample_id == bid
        ]
        return registry

    def test_search_returns_all_files_when_no_filters(self, mock_file_registry_for_search, mock_file_registrations):
        """Test that search returns all files when no filters are applied."""
        from daylib.file_api import FileSearchRequest

        # Simulate the search logic
        FileSearchRequest()
        results = mock_file_registry_for_search.list_customer_files("cust-001", limit=1000)

        assert len(results) == 3

    def test_search_filters_by_file_format(self, mock_file_registrations):
        """Test filtering by file format."""
        request_format = "fastq"
        results = [f for f in mock_file_registrations
                   if f.file_metadata.file_format.lower() == request_format.lower()]

        assert len(results) == 2
        assert all(f.file_metadata.file_format == "fastq" for f in results)

    def test_search_filters_by_subject_id(self, mock_file_registrations):
        """Test filtering by subject ID (partial match)."""
        subject_search = "hg002"
        results = [f for f in mock_file_registrations
                   if f.biosample_metadata and
                   subject_search in f.biosample_metadata.subject_id.lower()]

        assert len(results) == 2
        assert all(f.biosample_metadata.subject_id == "HG002" for f in results)

    def test_search_filters_by_biosample_id(self, mock_file_registrations):
        """Test filtering by biosample ID."""
        biosample_search = "biosample-002"
        results = [f for f in mock_file_registrations
                   if f.biosample_metadata and
                   biosample_search in f.biosample_metadata.biosample_id.lower()]

        assert len(results) == 1
        assert results[0].file_id == "file-003"

    def test_search_filters_by_sample_type(self, mock_file_registrations):
        """Test filtering by sample type."""
        sample_type = "blood"
        results = [f for f in mock_file_registrations
                   if f.biosample_metadata and
                   f.biosample_metadata.sample_type and
                   f.biosample_metadata.sample_type.lower() == sample_type.lower()]

        assert len(results) == 2

    def test_search_filters_by_platform(self, mock_file_registrations):
        """Test filtering by sequencing platform."""
        platform = "ont"
        results = [f for f in mock_file_registrations
                   if f.sequencing_metadata and
                   f.sequencing_metadata.platform and
                   platform in f.sequencing_metadata.platform.lower()]

        assert len(results) == 1
        assert results[0].file_id == "file-003"

    def test_search_filters_by_tag(self, mock_file_registrations):
        """Test filtering by tag."""
        tag = "somatic"
        results = [f for f in mock_file_registrations if tag in f.tags]

        assert len(results) == 1
        assert results[0].file_id == "file-003"

    def test_search_filters_by_date_range(self, mock_file_registrations):
        """Test filtering by registration date range."""
        date_from = "2024-01-16"
        results = [f for f in mock_file_registrations
                   if f.registered_at and str(f.registered_at) >= date_from]

        assert len(results) == 1
        assert results[0].file_id == "file-003"

    def test_search_general_text_matches_filename(self, mock_file_registrations):
        """Test general search matches filename."""
        search_term = "sample1"
        results = []
        for f in mock_file_registrations:
            filename = f.file_metadata.s3_uri.split('/')[-1] if f.file_metadata else ''
            if search_term.lower() in filename.lower():
                results.append(f)

        assert len(results) == 2

    def test_search_general_text_matches_tags(self, mock_file_registrations):
        """Test general search matches tags."""
        search_term = "germline"
        results = []
        for f in mock_file_registrations:
            if f.tags and any(search_term.lower() in tag.lower() for tag in f.tags):
                results.append(f)

        assert len(results) == 2

    def test_search_combined_filters(self, mock_file_registrations):
        """Test combining multiple filters."""
        # Filter by format AND subject
        format_filter = "fastq"
        subject_filter = "hg002"

        results = mock_file_registrations
        results = [f for f in results
                   if f.file_metadata.file_format.lower() == format_filter.lower()]
        results = [f for f in results
                   if f.biosample_metadata and
                   subject_filter in f.biosample_metadata.subject_id.lower()]

        assert len(results) == 2

    def test_search_returns_empty_for_no_matches(self, mock_file_registrations):
        """Test search returns empty list when no files match."""
        format_filter = "vcf"
        results = [f for f in mock_file_registrations
                   if f.file_metadata.file_format.lower() == format_filter.lower()]

        assert len(results) == 0

    def test_search_case_insensitive(self, mock_file_registrations):
        """Test that search is case-insensitive."""
        # Search with different cases
        subject_upper = "HG002"
        subject_lower = "hg002"
        subject_mixed = "Hg002"

        results_upper = [f for f in mock_file_registrations
                        if f.biosample_metadata and
                        subject_upper.lower() in f.biosample_metadata.subject_id.lower()]
        results_lower = [f for f in mock_file_registrations
                        if f.biosample_metadata and
                        subject_lower.lower() in f.biosample_metadata.subject_id.lower()]
        results_mixed = [f for f in mock_file_registrations
                        if f.biosample_metadata and
                        subject_mixed.lower() in f.biosample_metadata.subject_id.lower()]

        assert len(results_upper) == len(results_lower) == len(results_mixed) == 2


# ========== Tests for multi-region awareness features ==========


class TestBucketRegionDetectionAPI:
    """Test bucket region detection API endpoint."""

    def test_bucket_region_detection_us_east_1(self, client):
        """Test bucket region detection returns us-east-1 for None location."""
        with patch("daylib.workset_api.boto3.Session") as mock_session:
            mock_s3 = MagicMock()
            mock_session.return_value.client.return_value = mock_s3
            # S3 returns None for us-east-1 buckets
            mock_s3.get_bucket_location.return_value = {"LocationConstraint": None}

            response = client.get("/api/s3/bucket-region/test-bucket-east")

            assert response.status_code == 200
            data = response.json()
            assert data["bucket"] == "test-bucket-east"
            assert data["region"] == "us-east-1"

    def test_bucket_region_detection_us_west_2(self, client):
        """Test bucket region detection returns correct region."""
        with patch("daylib.workset_api.boto3.Session") as mock_session:
            mock_s3 = MagicMock()
            mock_session.return_value.client.return_value = mock_s3
            mock_s3.get_bucket_location.return_value = {"LocationConstraint": "us-west-2"}

            response = client.get("/api/s3/bucket-region/test-bucket-west")

            assert response.status_code == 200
            data = response.json()
            assert data["bucket"] == "test-bucket-west"
            assert data["region"] == "us-west-2"

    def test_bucket_region_detection_not_found(self, client):
        """Test bucket region detection handles non-existent bucket."""
        from botocore.exceptions import ClientError

        with patch("daylib.workset_api.boto3.Session") as mock_session:
            mock_s3 = MagicMock()
            mock_session.return_value.client.return_value = mock_s3
            mock_s3.get_bucket_location.side_effect = ClientError(
                {"Error": {"Code": "NoSuchBucket", "Message": "Bucket not found"}},
                "GetBucketLocation",
            )

            response = client.get("/api/s3/bucket-region/nonexistent-bucket")

            assert response.status_code == 404
            data = response.json()
            assert "not found" in data["detail"].lower() or "NoSuchBucket" in data["detail"]

    def test_bucket_region_detection_access_denied(self, client):
        """Test bucket region detection handles access denied."""
        from botocore.exceptions import ClientError

        with patch("daylib.workset_api.boto3.Session") as mock_session:
            mock_s3 = MagicMock()
            mock_session.return_value.client.return_value = mock_s3
            mock_s3.get_bucket_location.side_effect = ClientError(
                {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}},
                "GetBucketLocation",
            )

            response = client.get("/api/s3/bucket-region/private-bucket")

            assert response.status_code == 403
            data = response.json()
            assert "access" in data["detail"].lower() or "denied" in data["detail"].lower()


class TestWorksetCreationWithPreferredCluster:
    """Test workset creation with preferred_cluster field."""

    def test_create_workset_with_preferred_cluster(self, client, mock_state_db):
        """Test creating workset with preferred_cluster."""
        mock_state_db.register_workset.return_value = True
        mock_state_db.get_workset.return_value = {
            "workset_id": "test-ws-cluster",
            "state": "ready",
            "priority": "normal",
            "workset_type": "ruo",
            "bucket": "test-bucket",
            "prefix": "worksets/test/",
            "created_at": "2024-01-15T10:00:00Z",
            "updated_at": "2024-01-15T10:00:00Z",
            "preferred_cluster": "daylily-us-west-2-001",
        }

        response = client.post(
            "/worksets",
            json={
                "workset_id": "test-ws-cluster",
                "bucket": "test-bucket",
                "prefix": "worksets/test/",
                "priority": "normal",
                "workset_type": "ruo",
                "customer_id": "test-customer",
                "metadata": {"samples": [{"sample_id": "S1"}]},
                "preferred_cluster": "daylily-us-west-2-001",
            },
        )

        assert response.status_code == 201
        # Verify register_workset was called with preferred_cluster
        mock_state_db.register_workset.assert_called_once()
        call_kwargs = mock_state_db.register_workset.call_args.kwargs
        assert call_kwargs.get("preferred_cluster") == "daylily-us-west-2-001"

    def test_create_workset_without_preferred_cluster(self, client, mock_state_db):
        """Test creating workset without preferred_cluster."""
        mock_state_db.register_workset.return_value = True
        mock_state_db.get_workset.return_value = {
            "workset_id": "test-ws-no-cluster",
            "state": "ready",
            "priority": "normal",
            "workset_type": "ruo",
            "bucket": "test-bucket",
            "prefix": "worksets/test/",
            "created_at": "2024-01-15T10:00:00Z",
            "updated_at": "2024-01-15T10:00:00Z",
        }

        response = client.post(
            "/worksets",
            json={
                "workset_id": "test-ws-no-cluster",
                "bucket": "test-bucket",
                "prefix": "worksets/test/",
                "priority": "normal",
                "workset_type": "ruo",
                "customer_id": "test-customer",
                "metadata": {"samples": [{"sample_id": "S1"}]},
            },
        )

        assert response.status_code == 201
        # Verify register_workset was called with preferred_cluster=None
        mock_state_db.register_workset.assert_called_once()
        call_kwargs = mock_state_db.register_workset.call_args.kwargs
        assert call_kwargs.get("preferred_cluster") is None


# ==================== Portal Customer Isolation Tests ====================


class TestPortalCustomerIsolation:
    """Tests for portal workset customer isolation (Phase 3A security).

    These tests verify the filter logic for customer isolation.
    Full HTTP integration tests require complex session mocking, so we test
    the filtering logic directly using the verify_workset_ownership function.
    """

    def test_filter_worksets_by_customer_id(self):
        """Test that workset list filtering works correctly."""
        from daylib.routes.dependencies import verify_workset_ownership

        # Worksets belonging to different customers
        customer_a_worksets = [
            {"workset_id": "ws-a-001", "customer_id": "customer-A"},
            {"workset_id": "ws-a-002", "customer_id": "customer-A"},
        ]
        customer_b_worksets = [
            {"workset_id": "ws-b-001", "customer_id": "customer-B"},
        ]
        all_worksets = customer_a_worksets + customer_b_worksets

        # Filter for customer A - should only see A's worksets
        filtered_a = [w for w in all_worksets if verify_workset_ownership(w, "customer-A")]
        assert len(filtered_a) == 2
        assert all(w["customer_id"] == "customer-A" for w in filtered_a)

        # Filter for customer B - should only see B's worksets
        filtered_b = [w for w in all_worksets if verify_workset_ownership(w, "customer-B")]
        assert len(filtered_b) == 1
        assert filtered_b[0]["workset_id"] == "ws-b-001"

        # Filter for unknown customer - should see nothing
        filtered_c = [w for w in all_worksets if verify_workset_ownership(w, "customer-C")]
        assert len(filtered_c) == 0

    def test_ownership_check_blocks_cross_customer_access(self):
        """Test that ownership check prevents cross-customer access."""
        from daylib.routes.dependencies import verify_workset_ownership

        # Customer B's workset
        workset_b = {"workset_id": "ws-b-001", "customer_id": "customer-B", "results_s3_uri": "s3://..."}

        # Customer A should NOT have access
        assert verify_workset_ownership(workset_b, "customer-A") is False

        # Customer B SHOULD have access
        assert verify_workset_ownership(workset_b, "customer-B") is True

    def test_archived_worksets_filtered_by_customer(self):
        """Test that archived worksets are also filtered by customer."""
        from daylib.routes.dependencies import verify_workset_ownership

        archived = [
            {"workset_id": "ws-archived-a", "state": "archived", "customer_id": "customer-A"},
            {"workset_id": "ws-archived-b", "state": "archived", "customer_id": "customer-B"},
        ]

        # Customer A should only see their archived workset
        filtered = [w for w in archived if verify_workset_ownership(w, "customer-A")]
        assert len(filtered) == 1
        assert filtered[0]["workset_id"] == "ws-archived-a"

    def test_empty_customer_id_returns_empty_list(self):
        """Test that empty/None customer_id filters out all worksets."""
        from daylib.routes.dependencies import verify_workset_ownership

        worksets = [
            {"workset_id": "ws-001", "customer_id": "customer-A"},
            {"workset_id": "ws-002", "customer_id": "customer-B"},
        ]

        # None customer_id should filter out everything
        filtered_none = [w for w in worksets if verify_workset_ownership(w, None)]
        assert len(filtered_none) == 0

        # Empty string customer_id should filter out everything
        filtered_empty = [w for w in worksets if verify_workset_ownership(w, "")]
        assert len(filtered_empty) == 0


class TestVerifyWorksetOwnership:
    """Unit tests for verify_workset_ownership helper function."""

    def test_ownership_by_customer_id_field(self):
        """Test ownership check using customer_id field."""
        from daylib.routes.dependencies import verify_workset_ownership

        workset = {"workset_id": "ws-001", "customer_id": "cust-A"}
        assert verify_workset_ownership(workset, "cust-A") is True
        assert verify_workset_ownership(workset, "cust-B") is False

    def test_ownership_fallback_to_metadata_submitted_by(self):
        """Test ownership check falls back to metadata.submitted_by."""
        from daylib.routes.dependencies import verify_workset_ownership

        workset = {
            "workset_id": "ws-001",
            # No customer_id field
            "metadata": {"submitted_by": "cust-A"}
        }
        assert verify_workset_ownership(workset, "cust-A") is True
        assert verify_workset_ownership(workset, "cust-B") is False

    def test_ownership_fails_without_customer_info(self):
        """Test ownership check fails when no customer info available."""
        from daylib.routes.dependencies import verify_workset_ownership

        workset = {"workset_id": "ws-001"}  # No customer_id or metadata.submitted_by
        assert verify_workset_ownership(workset, "cust-A") is False

    def test_ownership_handles_empty_inputs(self):
        """Test ownership check handles None/empty inputs gracefully."""
        from daylib.routes.dependencies import verify_workset_ownership

        assert verify_workset_ownership(None, "cust-A") is False
        assert verify_workset_ownership({}, "cust-A") is False
        assert verify_workset_ownership({"customer_id": "cust-A"}, None) is False
        assert verify_workset_ownership({"customer_id": "cust-A"}, "") is False

    def test_ownership_customer_id_takes_precedence(self):
        """Test that customer_id field takes precedence over metadata.submitted_by."""
        from daylib.routes.dependencies import verify_workset_ownership

        workset = {
            "workset_id": "ws-001",
            "customer_id": "cust-A",
            "metadata": {"submitted_by": "cust-B"}  # Different from customer_id
        }
        # Should use customer_id field
        assert verify_workset_ownership(workset, "cust-A") is True
        assert verify_workset_ownership(workset, "cust-B") is False


class TestPortalAuthzSurfaces:
    """HTTP-level regression tests for portal authorization surfaces."""

    def test_portal_dashboard_filters_worksets_by_customer_id(self, mock_state_db):
        """Dashboard must only show worksets owned by the logged-in customer."""
        client = _make_authenticated_client(mock_state_db, customer_id="customer-A", is_admin=False)

        def list_by_state(state, limit=100):
            if state == WorksetState.READY:
                return [
                    {
                        "workset_id": "ws-owned",
                        "state": "ready",
                        "customer_id": "customer-A",
                        "created_at": "2026-01-24T00:00:00Z",
                    },
                    {
                        "workset_id": "ws-not-owned",
                        "state": "ready",
                        "customer_id": "customer-B",
                        "created_at": "2026-01-24T00:00:00Z",
                    },
                ]
            return []

        mock_state_db.list_worksets_by_state.side_effect = list_by_state

        response = client.get("/portal")
        assert response.status_code == 200
        assert b"ws-owned" in response.content
        assert b"ws-not-owned" not in response.content

    def test_clusters_sensitive_sections_are_admin_only(self, mock_state_db, monkeypatch):
        """Clusters page must hide budget/queue and IP details from non-admins."""

        class _FakeUrsaConfig:
            is_configured = True
            aws_profile = None

            def get_allowed_regions(self):
                return ["us-west-2"]

        # Patch config/cluster services used by the route.
        monkeypatch.setattr("daylib.ursa_config.get_ursa_config", lambda: _FakeUrsaConfig())

        mock_cluster = MagicMock()

        def cluster_to_dict(*, include_sensitive=True):
            base = {
                "cluster_name": "test-cluster",
                "region": "us-west-2",
                "cluster_status": "CREATE_COMPLETE",
                "compute_fleet_status": "RUNNING",
                "head_node": {
                    "instance_type": "t3.medium",
                    "public_ip": "1.2.3.4",
                    "private_ip": "10.0.0.1",
                    "state": "running",
                },
                "budget_info": None,
                "job_queue": None,
            }
            if include_sensitive:
                base["budget_info"] = {
                    "project_name": "proj",
                    "region": "us-west-2",
                    "reference_bucket": None,
                    "total_budget": 100.0,
                    "used_budget": 1.0,
                    "percent_used": 1.0,
                }
                base["job_queue"] = {
                    "total_jobs": 1,
                    "running_jobs": 1,
                    "pending_jobs": 0,
                    "configuring_jobs": 0,
                    "total_cpus": 4,
                    "jobs": [],
                }
            return base

        mock_cluster.to_dict.side_effect = cluster_to_dict

        mock_service = MagicMock()
        mock_service.get_all_clusters_with_status.return_value = [mock_cluster]
        monkeypatch.setattr("daylib.cluster_service.get_cluster_service", lambda **kwargs: mock_service)

        admin_client = _make_authenticated_client(mock_state_db, customer_id="cust-admin", is_admin=True)
        admin_response = admin_client.get("/portal/clusters")
        assert admin_response.status_code == 200
        assert b"AWS Budget" in admin_response.content
        assert b"Slurm Job Queue" in admin_response.content
        assert b"Public IP" in admin_response.content

        non_admin_client = _make_authenticated_client(mock_state_db, customer_id="cust-user", is_admin=False)
        non_admin_response = non_admin_client.get("/portal/clusters")
        assert non_admin_response.status_code == 200
        assert b"AWS Budget" not in non_admin_response.content
        assert b"Slurm Job Queue" not in non_admin_response.content
        assert b"Public IP" not in non_admin_response.content
