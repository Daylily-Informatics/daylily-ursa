"""Tests for Phase 5D: Per-Sample/Per-GB Billing Model."""

import datetime as dt
from unittest.mock import MagicMock, patch

import pytest

from daylib.billing import (
    BillingCalculator,
    BillingRates,
    CustomerBillingSummary,
    WorksetBillingItem,
)


@pytest.fixture
def mock_state_db():
    """Create a mock WorksetStateDB."""
    mock_db = MagicMock()
    mock_db.get_cost_report.return_value = None
    mock_db.get_storage_metrics.return_value = None
    mock_db.list_worksets_by_customer.return_value = []
    return mock_db


@pytest.fixture
def billing_calculator(mock_state_db):
    """Create a BillingCalculator with mocked state_db."""
    return BillingCalculator(state_db=mock_state_db)


class TestBillingRates:
    """Tests for BillingRates dataclass."""

    def test_default_rates(self):
        """Test default billing rates are set correctly."""
        rates = BillingRates()
        assert rates.s3_storage_per_gb_month == 0.023
        assert rates.data_egress_per_gb == 0.09
        assert rates.platform_fee_per_sample == 0.0
        assert rates.platform_fee_percentage == 0.0

    def test_custom_rates(self):
        """Test custom billing rates can be set."""
        rates = BillingRates(
            s3_storage_per_gb_month=0.05,
            data_egress_per_gb=0.15,
            platform_fee_per_sample=1.0,
            platform_fee_percentage=0.10,
        )
        assert rates.s3_storage_per_gb_month == 0.05
        assert rates.data_egress_per_gb == 0.15
        assert rates.platform_fee_per_sample == 1.0
        assert rates.platform_fee_percentage == 0.10


class TestWorksetBillingCalculation:
    """Tests for calculate_workset_billing method."""

    def test_workset_with_actual_cost_report(self, billing_calculator, mock_state_db):
        """Test billing calculation with actual cost report data."""
        mock_state_db.get_cost_report.return_value = {
            "total_compute_cost_usd": 25.50,
            "cost_report_sample_count": 2,
            "cost_report_rule_count": 15,
        }
        mock_state_db.get_storage_metrics.return_value = {
            "results_storage_bytes": 5 * 1024**3,  # 5 GB
        }

        workset = {
            "workset_id": "test-ws-001",
            "customer_id": "cust-001",
            "completed_at": "2026-01-24T10:00:00Z",
        }

        item = billing_calculator.calculate_workset_billing(workset)

        assert item.workset_id == "test-ws-001"
        assert item.customer_id == "cust-001"
        assert item.compute_cost_usd == 25.50
        assert item.sample_count == 2
        assert item.rule_count == 15
        assert item.has_actual_compute_cost is True
        assert item.storage_bytes == 5 * 1024**3
        assert item.has_actual_storage is True
        assert item.storage_gb == pytest.approx(5.0, rel=0.01)
        # Storage cost: 5 GB * $0.023 = $0.115
        assert item.storage_cost_usd == pytest.approx(0.115, rel=0.01)
        # Transfer cost: 5 GB * $0.09 = $0.45
        assert item.transfer_cost_usd == pytest.approx(0.45, rel=0.01)
        # Total: 25.50 + 0.115 + 0.45 = 26.065
        assert item.total_cost_usd == pytest.approx(26.065, rel=0.01)

    def test_workset_with_estimated_cost(self, billing_calculator, mock_state_db):
        """Test billing calculation falls back to estimates."""
        mock_state_db.get_cost_report.return_value = None
        mock_state_db.get_storage_metrics.return_value = None

        workset = {
            "workset_id": "test-ws-002",
            "customer_id": "cust-001",
            "metadata": {
                "estimated_cost_usd": 10.0,
                "sample_count": 1,
            },
        }

        item = billing_calculator.calculate_workset_billing(workset)

        assert item.compute_cost_usd == 10.0
        assert item.sample_count == 1
        assert item.has_actual_compute_cost is False
        assert item.has_actual_storage is False

    def test_workset_with_performance_metrics_fallback(self, billing_calculator, mock_state_db):
        """Test billing calculation falls back to performance_metrics."""
        mock_state_db.get_cost_report.return_value = None
        mock_state_db.get_storage_metrics.return_value = None

        workset = {
            "workset_id": "test-ws-003",
            "customer_id": "cust-001",
            "performance_metrics": {
                "cost_summary": {
                    "total_cost": 15.0,
                    "sample_count": 1,
                },
                "post_export_metrics": {
                    "analysis_directory_size_bytes": 2 * 1024**3,  # 2 GB
                },
            },
        }

        item = billing_calculator.calculate_workset_billing(workset)

        assert item.compute_cost_usd == 15.0
        assert item.has_actual_compute_cost is True
        assert item.storage_bytes == 2 * 1024**3
        assert item.has_actual_storage is True

    def test_workset_with_platform_fees(self, mock_state_db):
        """Test billing calculation with platform fees."""
        rates = BillingRates(
            platform_fee_per_sample=2.0,
            platform_fee_percentage=0.10,  # 10%
        )
        calculator = BillingCalculator(state_db=mock_state_db, rates=rates)

        mock_state_db.get_cost_report.return_value = {
            "total_compute_cost_usd": 100.0,
            "cost_report_sample_count": 5,
        }
        mock_state_db.get_storage_metrics.return_value = None

        workset = {
            "workset_id": "test-ws-004",
            "customer_id": "cust-001",
        }

        item = calculator.calculate_workset_billing(workset)

        # Per-sample fee: 5 * $2.0 = $10.0
        # Percentage fee: (100 + 0 + 0) * 0.10 = $10.0
        # Total platform fee: $20.0
        assert item.platform_fee_usd == pytest.approx(20.0, rel=0.01)


class TestCustomerBillingCalculation:
    """Tests for calculate_customer_billing method."""

    def test_customer_billing_aggregates_worksets(self, billing_calculator, mock_state_db):
        """Test customer billing aggregates multiple worksets."""
        mock_state_db.list_worksets_by_customer.return_value = [
            {
                "workset_id": "ws-001",
                "customer_id": "cust-001",
                "state": "complete",
                "completed_at": "2026-01-20T10:00:00Z",
            },
            {
                "workset_id": "ws-002",
                "customer_id": "cust-001",
                "state": "complete",
                "completed_at": "2026-01-21T10:00:00Z",
            },
        ]

        # Set up cost reports for each workset
        def get_cost_report(workset_id):
            if workset_id == "ws-001":
                return {"total_compute_cost_usd": 10.0, "cost_report_sample_count": 1}
            elif workset_id == "ws-002":
                return {"total_compute_cost_usd": 20.0, "cost_report_sample_count": 2}
            return None

        mock_state_db.get_cost_report.side_effect = get_cost_report
        mock_state_db.get_storage_metrics.return_value = None

        summary = billing_calculator.calculate_customer_billing("cust-001")

        assert summary.customer_id == "cust-001"
        assert summary.total_worksets == 2
        assert summary.billable_worksets == 2
        assert summary.total_samples == 3
        assert summary.total_compute_cost_usd == pytest.approx(30.0, rel=0.01)
        assert summary.has_actual_costs is True
        assert len(summary.workset_items) == 2

    def test_customer_billing_filters_by_state(self, billing_calculator, mock_state_db):
        """Test customer billing only includes specified states."""
        mock_state_db.list_worksets_by_customer.return_value = [
            {"workset_id": "ws-001", "customer_id": "cust-001", "state": "complete"},
            {"workset_id": "ws-002", "customer_id": "cust-001", "state": "error"},
            {"workset_id": "ws-003", "customer_id": "cust-001", "state": "in_progress"},
        ]
        mock_state_db.get_cost_report.return_value = {"total_compute_cost_usd": 10.0}

        summary = billing_calculator.calculate_customer_billing("cust-001")

        # Only "complete" state should be included by default
        assert summary.total_worksets == 1

    def test_customer_billing_filters_by_period(self, billing_calculator, mock_state_db):
        """Test customer billing filters by date period."""
        mock_state_db.list_worksets_by_customer.return_value = [
            {
                "workset_id": "ws-001",
                "customer_id": "cust-001",
                "state": "complete",
                "completed_at": "2026-01-15T10:00:00Z",  # Within period
            },
            {
                "workset_id": "ws-002",
                "customer_id": "cust-001",
                "state": "complete",
                "completed_at": "2025-12-01T10:00:00Z",  # Outside period
            },
        ]
        mock_state_db.get_cost_report.return_value = {"total_compute_cost_usd": 10.0}

        period_start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
        period_end = dt.datetime(2026, 1, 31, tzinfo=dt.timezone.utc)

        summary = billing_calculator.calculate_customer_billing(
            "cust-001", period_start=period_start, period_end=period_end
        )

        assert summary.total_worksets == 1

    def test_customer_billing_empty_worksets(self, billing_calculator, mock_state_db):
        """Test customer billing with no worksets."""
        mock_state_db.list_worksets_by_customer.return_value = []

        summary = billing_calculator.calculate_customer_billing("cust-001")

        assert summary.total_worksets == 0
        assert summary.billable_worksets == 0
        assert summary.grand_total_usd == 0.0


class TestInvoiceGeneration:
    """Tests for generate_invoice_data method."""

    def test_generate_invoice_data(self, billing_calculator, mock_state_db):
        """Test invoice data generation."""
        mock_state_db.list_worksets_by_customer.return_value = [
            {
                "workset_id": "ws-001",
                "customer_id": "cust-001",
                "state": "complete",
                "completed_at": "2026-01-20T10:00:00Z",
            },
        ]
        mock_state_db.get_cost_report.return_value = {
            "total_compute_cost_usd": 50.0,
            "cost_report_sample_count": 3,
            "cost_report_rule_count": 20,
        }
        mock_state_db.get_storage_metrics.return_value = {
            "results_storage_bytes": 10 * 1024**3,  # 10 GB
        }

        invoice = billing_calculator.generate_invoice_data("cust-001")

        assert invoice["customer_id"] == "cust-001"
        assert "invoice_date" in invoice
        assert "period_start" in invoice
        assert "period_end" in invoice
        assert invoice["summary"]["total_worksets"] == 1
        assert invoice["summary"]["total_samples"] == 3
        assert invoice["summary"]["compute_cost_usd"] == 50.0
        assert len(invoice["line_items"]) == 1
        assert invoice["line_items"][0]["workset_id"] == "ws-001"
        assert invoice["rates"]["s3_storage_per_gb_month"] == 0.023
        assert invoice["accuracy"]["has_actual_costs"] is True


class TestBillingAPIEndpoints:
    """Tests for billing API endpoints."""

    @pytest.fixture
    def mock_app_dependencies(self):
        """Create mock dependencies for API testing."""
        mock_state_db = MagicMock()
        mock_customer_manager = MagicMock()
        mock_customer_config = MagicMock()
        mock_customer_config.customer_id = "cust-001"
        mock_customer_config.customer_name = "Test Customer"
        mock_customer_config.email = "test@example.com"
        mock_customer_config.billing_account_id = "BA-001"
        mock_customer_config.cost_center = "CC-001"
        mock_customer_manager.get_customer_config.return_value = mock_customer_config
        return mock_state_db, mock_customer_manager

    def test_billing_summary_endpoint(self, mock_app_dependencies):
        """Test /api/customers/{customer_id}/billing/summary endpoint."""
        from fastapi.testclient import TestClient
        from daylib.workset_api import create_app

        mock_state_db, mock_customer_manager = mock_app_dependencies
        mock_state_db.list_worksets_by_customer.return_value = []
        mock_state_db.get_cost_report.return_value = None
        mock_state_db.get_storage_metrics.return_value = None

        app = create_app(
            state_db=mock_state_db,
            customer_manager=mock_customer_manager,
        )
        client = TestClient(app)

        response = client.get("/api/customers/cust-001/billing/summary?days=30")

        assert response.status_code == 200
        data = response.json()
        assert data["customer_id"] == "cust-001"
        assert "period_start" in data
        assert "period_end" in data
        assert "costs" in data
        assert "grand_total_usd" in data["costs"]

    def test_billing_invoice_endpoint(self, mock_app_dependencies):
        """Test /api/customers/{customer_id}/billing/invoice endpoint."""
        from fastapi.testclient import TestClient
        from daylib.workset_api import create_app

        mock_state_db, mock_customer_manager = mock_app_dependencies
        mock_state_db.list_worksets_by_customer.return_value = []
        mock_state_db.get_cost_report.return_value = None
        mock_state_db.get_storage_metrics.return_value = None

        app = create_app(
            state_db=mock_state_db,
            customer_manager=mock_customer_manager,
        )
        client = TestClient(app)

        response = client.get("/api/customers/cust-001/billing/invoice?days=30")

        assert response.status_code == 200
        data = response.json()
        assert data["customer_id"] == "cust-001"
        assert data["customer_name"] == "Test Customer"
        assert "summary" in data
        assert "line_items" in data
        assert "rates" in data

    def test_billing_workset_endpoint(self, mock_app_dependencies):
        """Test /api/customers/{customer_id}/billing/workset/{workset_id} endpoint."""
        from fastapi.testclient import TestClient
        from daylib.workset_api import create_app

        mock_state_db, mock_customer_manager = mock_app_dependencies
        mock_state_db.get_workset.return_value = {
            "workset_id": "ws-001",
            "customer_id": "cust-001",
            "state": "complete",
        }
        mock_state_db.get_cost_report.return_value = {
            "total_compute_cost_usd": 25.0,
            "cost_report_sample_count": 2,
        }
        mock_state_db.get_storage_metrics.return_value = {
            "results_storage_bytes": 1024**3,  # 1 GB
        }

        app = create_app(
            state_db=mock_state_db,
            customer_manager=mock_customer_manager,
        )
        client = TestClient(app)

        response = client.get("/api/customers/cust-001/billing/workset/ws-001")

        assert response.status_code == 200
        data = response.json()
        assert data["workset_id"] == "ws-001"
        assert data["samples"] == 2
        assert data["costs"]["compute_usd"] == 25.0
        assert data["accuracy"]["has_actual_compute_cost"] is True

    def test_billing_workset_not_found(self, mock_app_dependencies):
        """Test billing endpoint returns 404 for non-existent workset."""
        from fastapi.testclient import TestClient
        from daylib.workset_api import create_app

        mock_state_db, mock_customer_manager = mock_app_dependencies
        mock_state_db.get_workset.return_value = None

        app = create_app(
            state_db=mock_state_db,
            customer_manager=mock_customer_manager,
        )
        client = TestClient(app)

        response = client.get("/api/customers/cust-001/billing/workset/nonexistent")

        assert response.status_code == 404

    def test_billing_workset_wrong_customer(self, mock_app_dependencies):
        """Test billing endpoint returns 403 for wrong customer."""
        from fastapi.testclient import TestClient
        from daylib.workset_api import create_app

        mock_state_db, mock_customer_manager = mock_app_dependencies
        mock_state_db.get_workset.return_value = {
            "workset_id": "ws-001",
            "customer_id": "other-customer",  # Different customer
            "state": "complete",
        }

        app = create_app(
            state_db=mock_state_db,
            customer_manager=mock_customer_manager,
        )
        client = TestClient(app)

        response = client.get("/api/customers/cust-001/billing/workset/ws-001")

        assert response.status_code == 403

