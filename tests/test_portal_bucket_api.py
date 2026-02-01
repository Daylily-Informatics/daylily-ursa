"""Tests for portal S3 bucket API endpoints.

Tests cover:
- GET /api/v1/account/bucket-status
- POST /api/v1/account/provision-bucket

These endpoints were added to allow customers to check their S3 bucket
status and provision buckets that don't exist yet.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from daylib.workset_api import create_app
from daylib.workset_state_db import WorksetStateDB
from daylib.s3_bucket_validator import BucketValidationResult


def _make_state_db() -> MagicMock:
    """Create a mock WorksetStateDB."""
    return MagicMock(spec=WorksetStateDB)


def _make_customer(
    customer_id: str = "cust-001",
    email: str = "user@example.com",
    s3_bucket: str = "daylily-customer-cust-001",
    is_admin: bool = False,
    cost_center: str = "default",
    bucket_region: str = "us-west-2",
) -> MagicMock:
    """Create a mock customer object."""
    customer = MagicMock()
    customer.customer_id = customer_id
    customer.email = email
    customer.s3_bucket = s3_bucket
    customer.is_admin = is_admin
    customer.cost_center = cost_center
    customer.bucket_region = bucket_region
    return customer


def _login_client(client: TestClient, email: str = "user@example.com") -> None:
    """Perform login to establish session."""
    client.post(
        "/portal/login",
        data={"email": email, "password": "password"},
        follow_redirects=False,
    )


class TestBucketStatusEndpoint:
    """Tests for GET /api/v1/account/bucket-status endpoint."""

    def test_bucket_status_not_authenticated(self) -> None:
        """Test bucket status returns 401 when user not authenticated."""
        customer_manager = MagicMock()
        app = create_app(
            state_db=_make_state_db(),
            enable_auth=False,
            customer_manager=customer_manager,
        )
        client = TestClient(app)

        # No login - should get 401
        response = client.get("/api/v1/account/bucket-status")
        assert response.status_code == 401
        assert response.json()["detail"] == "Not authenticated"

    def test_bucket_status_fully_configured(self) -> None:
        """Test bucket status returns fully_configured when bucket exists with all permissions."""
        customer_manager = MagicMock()
        customer = _make_customer()
        customer_manager.get_customer_by_email.return_value = customer
        customer_manager.get_customer_config.return_value = customer

        app = create_app(
            state_db=_make_state_db(),
            enable_auth=False,
            customer_manager=customer_manager,
        )
        client = TestClient(app)
        _login_client(client)

        # Mock S3BucketValidator
        with patch("daylib.s3_bucket_validator.S3BucketValidator") as MockValidator:
            mock_result = BucketValidationResult(
                bucket_name="daylily-customer-cust-001",
                exists=True,
                accessible=True,
                can_read=True,
                can_write=True,
                can_list=True,
                region="us-west-2",
                errors=[],
                warnings=[],
            )
            MockValidator.return_value.validate_bucket.return_value = mock_result

            response = client.get("/api/v1/account/bucket-status")

        assert response.status_code == 200
        data = response.json()
        assert data["bucket"] == "daylily-customer-cust-001"
        assert data["status"] == "fully_configured"
        assert data["exists"] is True
        assert data["accessible"] is True
        assert data["can_read"] is True
        assert data["can_write"] is True
        assert data["can_list"] is True
        assert data["region"] == "us-west-2"
        assert data["errors"] == []

    def test_bucket_status_missing(self) -> None:
        """Test bucket status returns missing when bucket doesn't exist."""
        customer_manager = MagicMock()
        customer = _make_customer()
        customer_manager.get_customer_by_email.return_value = customer
        customer_manager.get_customer_config.return_value = customer

        app = create_app(
            state_db=_make_state_db(),
            enable_auth=False,
            customer_manager=customer_manager,
        )
        client = TestClient(app)
        _login_client(client)

        with patch("daylib.s3_bucket_validator.S3BucketValidator") as MockValidator:
            mock_result = BucketValidationResult(
                bucket_name="daylily-customer-cust-001",
                exists=False,
                accessible=False,
                can_read=False,
                can_write=False,
                can_list=False,
                errors=["Bucket does not exist"],
            )
            MockValidator.return_value.validate_bucket.return_value = mock_result

            response = client.get("/api/v1/account/bucket-status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "missing"
        assert data["exists"] is False
        assert "Bucket does not exist" in data["errors"]

    def test_bucket_status_no_bucket_configured(self) -> None:
        """Test bucket status returns not_configured when customer has no bucket."""
        customer_manager = MagicMock()
        customer = _make_customer(s3_bucket=None)
        customer_manager.get_customer_by_email.return_value = customer
        customer_manager.get_customer_config.return_value = customer

        app = create_app(
            state_db=_make_state_db(),
            enable_auth=False,
            customer_manager=customer_manager,
        )
        client = TestClient(app)
        _login_client(client)

        response = client.get("/api/v1/account/bucket-status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "not_configured"
        assert data["bucket"] is None

    def test_bucket_status_permission_denied(self) -> None:
        """Test bucket status returns permission_denied when bucket exists but not accessible."""
        customer_manager = MagicMock()
        customer = _make_customer()
        customer_manager.get_customer_by_email.return_value = customer
        customer_manager.get_customer_config.return_value = customer

        app = create_app(
            state_db=_make_state_db(),
            enable_auth=False,
            customer_manager=customer_manager,
        )
        client = TestClient(app)
        _login_client(client)

        with patch("daylib.s3_bucket_validator.S3BucketValidator") as MockValidator:
            mock_result = BucketValidationResult(
                bucket_name="daylily-customer-cust-001",
                exists=True,
                accessible=False,
                can_read=False,
                can_write=False,
                can_list=False,
                errors=["Access denied"],
            )
            MockValidator.return_value.validate_bucket.return_value = mock_result

            response = client.get("/api/v1/account/bucket-status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "permission_denied"
        assert data["exists"] is True
        assert data["accessible"] is False

    def test_bucket_status_customer_not_found(self) -> None:
        """Test bucket status returns 404 when customer not found."""
        customer_manager = MagicMock()
        customer = _make_customer()
        customer_manager.get_customer_by_email.return_value = customer
        customer_manager.get_customer_config.return_value = None  # Customer not found

        app = create_app(
            state_db=_make_state_db(),
            enable_auth=False,
            customer_manager=customer_manager,
        )
        client = TestClient(app)
        _login_client(client)

        response = client.get("/api/v1/account/bucket-status")

        assert response.status_code == 404
        assert response.json()["detail"] == "Customer not found"

    def test_bucket_status_validation_error(self) -> None:
        """Test bucket status handles validator exceptions gracefully."""
        customer_manager = MagicMock()
        customer = _make_customer()
        customer_manager.get_customer_by_email.return_value = customer
        customer_manager.get_customer_config.return_value = customer

        app = create_app(
            state_db=_make_state_db(),
            enable_auth=False,
            customer_manager=customer_manager,
        )
        client = TestClient(app)
        _login_client(client)

        with patch("daylib.s3_bucket_validator.S3BucketValidator") as MockValidator:
            MockValidator.return_value.validate_bucket.side_effect = Exception("S3 API error")

            response = client.get("/api/v1/account/bucket-status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert "Failed to check bucket status" in data["errors"][0]


class TestBucketProvisionEndpoint:
    """Tests for POST /api/v1/account/provision-bucket endpoint."""

    def test_provision_bucket_not_authenticated(self) -> None:
        """Test bucket provisioning returns 401 when user not authenticated."""
        customer_manager = MagicMock()
        app = create_app(
            state_db=_make_state_db(),
            enable_auth=False,
            customer_manager=customer_manager,
        )
        client = TestClient(app)

        response = client.post("/api/v1/account/provision-bucket")
        assert response.status_code == 401
        assert response.json()["detail"] == "Not authenticated"

    def test_provision_bucket_success(self) -> None:
        """Test successful bucket provisioning when bucket doesn't exist."""
        customer_manager = MagicMock()
        customer = _make_customer()
        customer_manager.get_customer_by_email.return_value = customer
        customer_manager.get_customer_config.return_value = customer

        app = create_app(
            state_db=_make_state_db(),
            enable_auth=False,
            customer_manager=customer_manager,
        )
        client = TestClient(app)
        _login_client(client)

        with patch("daylib.s3_bucket_validator.S3BucketValidator") as MockValidator:
            # First call: bucket doesn't exist
            mock_result_missing = BucketValidationResult(
                bucket_name="daylily-customer-cust-001",
                exists=False,
                accessible=False,
                can_read=False,
                can_write=False,
                can_list=False,
                errors=["Bucket does not exist"],
            )
            # Second call: bucket now exists after creation
            mock_result_created = BucketValidationResult(
                bucket_name="daylily-customer-cust-001",
                exists=True,
                accessible=True,
                can_read=True,
                can_write=True,
                can_list=True,
                region="us-west-2",
                errors=[],
            )
            MockValidator.return_value.validate_bucket.side_effect = [
                mock_result_missing,
                mock_result_created,
            ]

            response = client.post("/api/v1/account/provision-bucket")

        assert response.status_code == 200
        data = response.json()
        assert data["provisioned"] is True
        assert data["bucket"] == "daylily-customer-cust-001"
        assert data["message"] == "Bucket created successfully"
        customer_manager._create_customer_bucket.assert_called_once()

    def test_provision_bucket_already_exists(self) -> None:
        """Test provisioning returns appropriate message when bucket already exists."""
        customer_manager = MagicMock()
        customer = _make_customer()
        customer_manager.get_customer_by_email.return_value = customer
        customer_manager.get_customer_config.return_value = customer

        app = create_app(
            state_db=_make_state_db(),
            enable_auth=False,
            customer_manager=customer_manager,
        )
        client = TestClient(app)
        _login_client(client)

        with patch("daylib.s3_bucket_validator.S3BucketValidator") as MockValidator:
            mock_result = BucketValidationResult(
                bucket_name="daylily-customer-cust-001",
                exists=True,
                accessible=True,
                can_read=True,
                can_write=True,
                can_list=True,
            )
            MockValidator.return_value.validate_bucket.return_value = mock_result

            response = client.post("/api/v1/account/provision-bucket")

        assert response.status_code == 200
        data = response.json()
        assert data["provisioned"] is False
        assert data["message"] == "Bucket already exists"
        customer_manager._create_customer_bucket.assert_not_called()

    def test_provision_bucket_no_bucket_configured(self) -> None:
        """Test provisioning returns 400 when customer has no bucket configured."""
        customer_manager = MagicMock()
        customer = _make_customer(s3_bucket=None)
        customer_manager.get_customer_by_email.return_value = customer
        customer_manager.get_customer_config.return_value = customer

        app = create_app(
            state_db=_make_state_db(),
            enable_auth=False,
            customer_manager=customer_manager,
        )
        client = TestClient(app)
        _login_client(client)

        response = client.post("/api/v1/account/provision-bucket")

        assert response.status_code == 400
        assert response.json()["detail"] == "No S3 bucket configured for this customer"

    def test_provision_bucket_creation_failure(self) -> None:
        """Test provisioning returns 500 when bucket creation fails."""
        customer_manager = MagicMock()
        customer = _make_customer()
        customer_manager.get_customer_by_email.return_value = customer
        customer_manager.get_customer_config.return_value = customer
        customer_manager._create_customer_bucket.side_effect = Exception("S3 CreateBucket failed")

        app = create_app(
            state_db=_make_state_db(),
            enable_auth=False,
            customer_manager=customer_manager,
        )
        client = TestClient(app)
        _login_client(client)

        with patch("daylib.s3_bucket_validator.S3BucketValidator") as MockValidator:
            mock_result = BucketValidationResult(
                bucket_name="daylily-customer-cust-001",
                exists=False,
                accessible=False,
                can_read=False,
                can_write=False,
                can_list=False,
            )
            MockValidator.return_value.validate_bucket.return_value = mock_result

            response = client.post("/api/v1/account/provision-bucket")

        assert response.status_code == 500
        assert "Failed to create bucket" in response.json()["detail"]

    def test_provision_bucket_customer_not_found(self) -> None:
        """Test provisioning returns 404 when customer not found."""
        customer_manager = MagicMock()
        customer = _make_customer()
        customer_manager.get_customer_by_email.return_value = customer
        customer_manager.get_customer_config.return_value = None

        app = create_app(
            state_db=_make_state_db(),
            enable_auth=False,
            customer_manager=customer_manager,
        )
        client = TestClient(app)
        _login_client(client)

        response = client.post("/api/v1/account/provision-bucket")

        assert response.status_code == 404
        assert response.json()["detail"] == "Customer not found"

