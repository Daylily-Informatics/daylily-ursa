"""Tests for customer management."""

from unittest.mock import MagicMock, patch

import pytest

from daylib.workset_customer import CustomerManager, CustomerConfig
from daylib.routes.dependencies import verify_workset_ownership


@pytest.fixture
def mock_aws():
    """Mock AWS services."""
    with patch("daylib.workset_customer.boto3.Session") as mock_session:
        mock_s3 = MagicMock()
        mock_dynamodb = MagicMock()
        mock_table = MagicMock()

        mock_session.return_value.client.return_value = mock_s3
        mock_session.return_value.resource.return_value = mock_dynamodb
        mock_dynamodb.Table.return_value = mock_table

        yield {
            "session": mock_session,
            "s3": mock_s3,
            "dynamodb": mock_dynamodb,
            "table": mock_table,
        }


@pytest.fixture
def customer_manager(mock_aws):
    """Create CustomerManager instance."""
    return CustomerManager(
        region="us-west-2",
        profile=None,
        bucket_prefix="test-customer",
    )


def test_generate_customer_id(customer_manager):
    """Test customer ID generation."""
    customer_id = customer_manager._generate_customer_id("Test Customer")

    assert customer_id.startswith("test-customer-")
    assert len(customer_id) > len("test-customer-")


def test_onboard_customer(customer_manager, mock_aws):
    """Test customer onboarding."""
    mock_table = mock_aws["table"]

    config = customer_manager.onboard_customer(
        customer_name="Test Customer",
        email="test@example.com",
        max_concurrent_worksets=10,
        max_storage_gb=2000,
        cost_center="CC-123",
    )

    assert isinstance(config, CustomerConfig)
    assert config.customer_name == "Test Customer"
    assert config.email == "test@example.com"
    assert config.max_concurrent_worksets == 10
    assert config.max_storage_gb == 2000
    assert config.cost_center == "CC-123"
    assert config.s3_bucket.startswith("test-customer-")

    # Verify S3 bucket creation was called
    mock_aws["s3"].create_bucket.assert_called_once()

    # Verify DynamoDB put_item was called
    mock_table.put_item.assert_called_once()


def test_save_customer_config(customer_manager, mock_aws):
    """Test saving customer configuration."""
    mock_table = mock_aws["table"]

    config = CustomerConfig(
        customer_id="test-123",
        customer_name="Test Customer",
        email="test@example.com",
        s3_bucket="test-bucket",
        max_concurrent_worksets=5,
        max_storage_gb=1000,
        billing_account_id="BA-456",
        cost_center="CC-789",
    )

    customer_manager._save_customer_config(config)

    mock_table.put_item.assert_called_once()
    call_args = mock_table.put_item.call_args
    item = call_args[1]["Item"]

    assert item["customer_id"] == "test-123"
    assert item["customer_name"] == "Test Customer"
    assert item["email"] == "test@example.com"
    assert item["billing_account_id"] == "BA-456"
    assert item["cost_center"] == "CC-789"


def test_get_customer_config(customer_manager, mock_aws):
    """Test getting customer configuration."""
    mock_table = mock_aws["table"]
    mock_table.get_item.return_value = {
        "Item": {
            "customer_id": "test-123",
            "customer_name": "Test Customer",
            "email": "test@example.com",
            "s3_bucket": "test-bucket",
            "max_concurrent_worksets": 5,
            "max_storage_gb": 1000,
        }
    }

    config = customer_manager.get_customer_config("test-123")

    assert config is not None
    assert config.customer_id == "test-123"
    assert config.customer_name == "Test Customer"
    assert config.email == "test@example.com"


def test_get_customer_config_not_found(customer_manager, mock_aws):
    """Test getting non-existent customer."""
    mock_table = mock_aws["table"]
    mock_table.get_item.return_value = {}

    config = customer_manager.get_customer_config("nonexistent")

    assert config is None


def test_list_customers(customer_manager, mock_aws):
    """Test listing customers."""
    mock_table = mock_aws["table"]
    mock_table.scan.return_value = {
        "Items": [
            {
                "customer_id": "test-1",
                "customer_name": "Customer 1",
                "email": "customer1@example.com",
                "s3_bucket": "bucket-1",
                "max_concurrent_worksets": 5,
                "max_storage_gb": 1000,
            },
            {
                "customer_id": "test-2",
                "customer_name": "Customer 2",
                "email": "customer2@example.com",
                "s3_bucket": "bucket-2",
                "max_concurrent_worksets": 10,
                "max_storage_gb": 2000,
            },
        ]
    }

    customers = customer_manager.list_customers()

    assert len(customers) == 2
    assert customers[0].customer_id == "test-1"
    assert customers[1].customer_id == "test-2"


# =============================================================================
# API Customer Isolation Tests
# =============================================================================


class TestAPICustomerIsolation:
    """Tests for API customer isolation (Phase 3B).

    Verifies that customer workset endpoints filter by customer_id
    ownership rather than bucket, preventing cross-customer data leakage.
    """

    def test_public_worksets_endpoint_filters_by_customer_id(self):
        """Test that /worksets?customer_id=X filters to customer X's worksets."""
        # Simulate worksets from multiple customers
        worksets = [
            {"workset_id": "ws-1", "customer_id": "customer-alpha", "state": "ready"},
            {"workset_id": "ws-2", "customer_id": "customer-beta", "state": "ready"},
            {"workset_id": "ws-3", "customer_id": "customer-alpha", "state": "complete"},
            {"workset_id": "ws-4", "customer_id": "customer-beta", "state": "complete"},
        ]

        # Filter by customer_id using verify_workset_ownership (same logic as API)
        customer_id = "customer-alpha"
        filtered = [w for w in worksets if verify_workset_ownership(w, customer_id)]

        assert len(filtered) == 2
        assert all(w["customer_id"] == "customer-alpha" for w in filtered)
        assert "ws-1" in [w["workset_id"] for w in filtered]
        assert "ws-3" in [w["workset_id"] for w in filtered]

    def test_customer_worksets_endpoint_uses_ownership_not_bucket(self):
        """Test that /api/customers/{id}/worksets filters by customer_id, not bucket."""
        # Worksets with same bucket but different customer_ids
        worksets = [
            {"workset_id": "ws-1", "customer_id": "customer-a", "bucket": "shared-bucket"},
            {"workset_id": "ws-2", "customer_id": "customer-b", "bucket": "shared-bucket"},
            {"workset_id": "ws-3", "customer_id": "customer-a", "bucket": "other-bucket"},
        ]

        # Filter by customer_id (not bucket)
        customer_id = "customer-a"
        filtered = [w for w in worksets if verify_workset_ownership(w, customer_id)]

        # Should get both of customer-a's worksets, regardless of bucket
        assert len(filtered) == 2
        assert all(w["customer_id"] == "customer-a" for w in filtered)

        # Old bucket-based filtering would have given wrong results
        bucket = "shared-bucket"
        bucket_filtered = [w for w in worksets if w.get("bucket") == bucket]
        assert len(bucket_filtered) == 2  # Would include customer-b's workset!

    def test_archived_worksets_endpoint_uses_ownership_not_bucket(self):
        """Test that /api/customers/{id}/worksets/archived filters by customer_id."""
        archived = [
            {"workset_id": "ws-1", "customer_id": "cust-x", "state": "archived", "bucket": "bucket-x"},
            {"workset_id": "ws-2", "customer_id": "cust-y", "state": "archived", "bucket": "bucket-x"},
            {"workset_id": "ws-3", "customer_id": "cust-x", "state": "archived", "bucket": "bucket-y"},
        ]

        # Filter by customer_id ownership
        customer_id = "cust-x"
        filtered = [w for w in archived if verify_workset_ownership(w, customer_id)]

        assert len(filtered) == 2
        assert {"ws-1", "ws-3"} == {w["workset_id"] for w in filtered}

    def test_cross_customer_access_blocked(self):
        """Test that customer A cannot access customer B's worksets via API filtering."""
        customer_a_workset = {
            "workset_id": "ws-secret-a",
            "customer_id": "customer-a",
            "bucket": "bucket-a",
        }
        customer_b_workset = {
            "workset_id": "ws-secret-b",
            "customer_id": "customer-b",
            "bucket": "bucket-b",
        }

        # Customer A trying to access customer B's workset
        assert verify_workset_ownership(customer_a_workset, "customer-a") is True
        assert verify_workset_ownership(customer_b_workset, "customer-a") is False

        # Customer B trying to access customer A's workset
        assert verify_workset_ownership(customer_b_workset, "customer-b") is True
        assert verify_workset_ownership(customer_a_workset, "customer-b") is False

    def test_ownership_fallback_to_metadata_submitted_by(self):
        """Test API filtering works for legacy worksets using metadata.submitted_by."""
        legacy_workset = {
            "workset_id": "legacy-ws-1",
            # No customer_id field (legacy)
            "metadata": {"submitted_by": "legacy-customer"},
            "bucket": "some-bucket",
        }

        # Should match via fallback
        assert verify_workset_ownership(legacy_workset, "legacy-customer") is True
        assert verify_workset_ownership(legacy_workset, "other-customer") is False

    def test_no_customer_id_returns_empty_filter(self):
        """Test that worksets without customer_id are filtered out for security."""
        workset_no_owner = {
            "workset_id": "orphan-ws",
            "bucket": "some-bucket",
            # No customer_id, no metadata.submitted_by
        }

        # Should not match any customer
        assert verify_workset_ownership(workset_no_owner, "any-customer") is False

