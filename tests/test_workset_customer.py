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


def test_update_customer(customer_manager, mock_aws):
    """Test updating customer configuration."""
    mock_table = mock_aws["table"]
    # Simulate existing customer in DB
    mock_table.get_item.return_value = {
        "Item": {
            "customer_id": "test-123",
            "customer_name": "Old Name",
            "email": "old@example.com",
            "s3_bucket": "test-bucket",
            "max_concurrent_worksets": 5,
            "max_storage_gb": 1000,
            "billing_account_id": "OLD-BA",
            "cost_center": "OLD-CC",
        }
    }

    # Update only customer_name and cost_center
    updated = customer_manager.update_customer(
        customer_id="test-123",
        customer_name="New Name",
        cost_center="NEW-CC",
    )

    assert updated is not None
    assert updated.customer_name == "New Name"
    assert updated.cost_center == "NEW-CC"
    # Fields not updated should remain the same
    assert updated.email == "old@example.com"
    assert updated.billing_account_id == "OLD-BA"

    # Verify put_item was called to save updates
    mock_table.put_item.assert_called_once()


def test_update_customer_not_found(customer_manager, mock_aws):
    """Test updating non-existent customer returns None."""
    mock_table = mock_aws["table"]
    mock_table.get_item.return_value = {}

    result = customer_manager.update_customer(
        customer_id="nonexistent",
        customer_name="New Name",
    )

    assert result is None
    # Should not attempt to save if customer not found
    mock_table.put_item.assert_not_called()


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


class TestBillingAccountIdIsMetadataOnly:
    """Tests documenting that billing_account_id is metadata-only.

    Per Task 27 (NEW_FINAL_LINGERING_TASKS.md):
    - billing_account_id visibility is already implemented (captured at register +
      displayed on /portal/account) => 'visible' is STALE (already done).
    - For 'usable': Only add tests to detect whether any existing code path uses
      billing_account_id (e.g., affects billing, workset submissions, tags, AWS ops).
      If this cannot be reliably tested in-unit (or only exists in external infra),
      do not implement new behavior; document result in test comments and treat as
      skipped per instruction.

    INVESTIGATION RESULTS (2026-01-25):
    -----------------------------------
    1. billing_account_id is defined in CustomerConfig dataclass (workset_customer.py)
    2. It is stored in DynamoDB via _save_customer_config() (conditionally, only if not None)
    3. It is returned in API responses via CustomerResponse model (workset_api.py)
    4. It is displayed on the /portal/account page (templates/account.html)
    5. It is NOT used in daylib/billing.py - BillingCalculator calculates costs from:
       - compute_cost_usd: From Snakemake benchmark data (actual spot instance pricing)
       - storage_bytes: From S3 storage metrics
       - sample_count: From workset metadata
       - platform_fee_usd: Configurable markup based on rates
    6. It is NOT used in workset registration (workset_api.py does not tag worksets with it)
    7. It is NOT used in AWS operations or resource tagging

    CONCLUSION: billing_account_id is metadata-only. It is captured and displayed but
    NOT operationally used in any code path that affects billing, workset processing,
    or AWS resource management. This is intentional - billing is calculated from actual
    workset data, not routed to external billing accounts.

    These tests document this finding for future reference.
    """

    def test_billing_account_id_stored_and_retrieved(self, customer_manager, mock_aws):
        """Verify billing_account_id is correctly stored and retrieved.

        This confirms the 'visible' part is working - the field is persisted.
        """
        mock_table = mock_aws["table"]
        mock_table.get_item.return_value = {}

        # Create customer with billing_account_id
        config = CustomerConfig(
            customer_id="test-ba-123",
            customer_name="Test BA Customer",
            email="testba@example.com",
            s3_bucket="test-ba-bucket",
            billing_account_id="BA-ACME-001",
        )
        customer_manager._save_customer_config(config)

        # Verify billing_account_id was saved to DynamoDB
        mock_table.put_item.assert_called_once()
        call_args = mock_table.put_item.call_args
        item = call_args[1]["Item"]
        assert item["billing_account_id"] == "BA-ACME-001"

    def test_billing_calculator_does_not_use_billing_account_id(self):
        """Document that BillingCalculator does NOT use billing_account_id.

        This test verifies the 'usable' investigation finding: billing calculations
        are based on actual workset data (compute costs, storage, samples), NOT on
        any billing_account_id routing.

        SKIPPED IMPLEMENTATION per task instruction: Since billing_account_id is not
        used operationally, there is no new behavior to implement. This test documents
        the architecture decision that billing is calculated from workset data.
        """
        from daylib.billing import BillingCalculator

        # BillingCalculator signature and methods do not accept billing_account_id
        # This is by design - billing is calculated from workset data

        # Verify BillingCalculator.__init__ signature has no billing_account_id param
        import inspect
        init_sig = inspect.signature(BillingCalculator.__init__)
        init_params = list(init_sig.parameters.keys())
        assert "billing_account_id" not in init_params, (
            "BillingCalculator should not accept billing_account_id - "
            "billing is calculated from workset data, not routed to accounts"
        )

        # Verify calculate_workset_billing does not accept billing_account_id
        calc_sig = inspect.signature(BillingCalculator.calculate_workset_billing)
        calc_params = list(calc_sig.parameters.keys())
        assert "billing_account_id" not in calc_params, (
            "calculate_workset_billing should not use billing_account_id"
        )

        # Verify calculate_customer_billing does not accept billing_account_id
        cust_sig = inspect.signature(BillingCalculator.calculate_customer_billing)
        cust_params = list(cust_sig.parameters.keys())
        assert "billing_account_id" not in cust_params, (
            "calculate_customer_billing should not use billing_account_id"
        )

    def test_billing_account_id_not_in_billing_module_source(self):
        """Verify billing_account_id is not referenced anywhere in billing module.

        This is a regression test to ensure billing_account_id remains metadata-only.
        If this test fails in the future, it means someone has started using
        billing_account_id operationally and the architecture has changed.
        """
        import daylib.billing as billing_module
        import inspect

        # Get the source code of the entire billing module
        source = inspect.getsource(billing_module)

        # billing_account_id should NOT appear anywhere in the billing module
        assert "billing_account_id" not in source, (
            "billing_account_id should not be referenced in daylib/billing.py - "
            "billing is calculated from workset data (compute costs, storage, samples), "
            "not routed to external billing accounts. If this test fails, the "
            "architecture has changed and this test should be updated."
        )

