"""Graph-native tests for customer management and ownership filtering."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from daylib.routes.dependencies import verify_workset_ownership
from daylib.workset_customer import CustomerConfig, CustomerManager


class _SessionCtx:
    def __init__(self, session: MagicMock):
        self._session = session

    def __enter__(self):
        return self._session

    def __exit__(self, exc_type, exc, tb):
        return False


def _instance(payload: dict, *, euid: str = "cust-euid"):
    row = MagicMock()
    row.json_addl = dict(payload)
    row.euid = euid
    row.name = payload.get("customer_name") or payload.get("customer_id") or "customer"
    row.created_dt = None
    row.modified_dt = None
    row.bstatus = "active"
    row.uuid = hash(euid) & 0xFFFFFFFF
    row.is_deleted = False
    return row


@pytest.fixture
def customer_manager() -> CustomerManager:
    mgr = CustomerManager.__new__(CustomerManager)
    mgr._session = MagicMock()
    mgr.s3 = MagicMock()
    mgr.region = "us-west-2"
    mgr.bucket_prefix = "test-customer"
    mgr.profile = None
    mgr.customer_table_name = "tapdb-customer-graph"
    mgr.backend = MagicMock()
    mgr._backend_session = MagicMock()
    mgr.backend.session_scope.return_value = _SessionCtx(mgr._backend_session)
    return mgr


def test_generate_customer_id(customer_manager: CustomerManager):
    customer_id = customer_manager._generate_customer_id("Test Customer")
    assert customer_id.startswith("test-customer-")


def test_create_customer_bucket_handles_us_east_1(customer_manager: CustomerManager):
    east = MagicMock()
    customer_manager._session.client.side_effect = lambda service, region_name=None: east

    customer_manager._create_customer_bucket(
        bucket_name="daylily-customer-test",
        customer_id="cust-001",
        cost_center="CC-1",
        bucket_region="us-east-1",
    )

    east.create_bucket.assert_called_once_with(Bucket="daylily-customer-test")
    east.put_bucket_tagging.assert_called_once()


def test_create_customer_bucket_uses_location_constraint_for_non_east(customer_manager: CustomerManager):
    west = MagicMock()
    customer_manager._session.client.side_effect = lambda service, region_name=None: west

    customer_manager._create_customer_bucket(
        bucket_name="daylily-customer-test",
        customer_id="cust-001",
        cost_center="CC-1",
        bucket_region="eu-central-1",
    )

    west.create_bucket.assert_called_once_with(
        Bucket="daylily-customer-test",
        CreateBucketConfiguration={"LocationConstraint": "eu-central-1"},
    )


def test_onboard_customer_creates_graph_record(customer_manager: CustomerManager):
    customer_manager.get_customer_by_email = MagicMock(return_value=None)
    customer_manager._create_customer_bucket = MagicMock()

    cfg = customer_manager.onboard_customer(
        customer_name="Acme",
        email="acme@example.com",
        max_concurrent_worksets=10,
        max_storage_gb=2000,
        cost_center="CC-123",
    )

    assert isinstance(cfg, CustomerConfig)
    assert cfg.customer_name == "Acme"
    assert cfg.email == "acme@example.com"
    assert cfg.max_concurrent_worksets == 10
    assert cfg.max_storage_gb == 2000
    customer_manager.backend.create_instance.assert_called_once()


def test_onboard_customer_returns_existing(customer_manager: CustomerManager):
    existing = CustomerConfig(
        customer_id="cust-001",
        customer_name="Acme",
        email="acme@example.com",
        s3_bucket="bucket",
    )
    customer_manager.get_customer_by_email = MagicMock(return_value=existing)

    cfg = customer_manager.onboard_customer(customer_name="Acme", email="acme@example.com")

    assert cfg is existing
    customer_manager.backend.create_instance.assert_not_called()


def test_get_customer_config_and_list(customer_manager: CustomerManager):
    payload = {
        "customer_id": "cust-001",
        "customer_name": "Acme",
        "email": "acme@example.com",
        "s3_bucket": "bucket-1",
        "max_concurrent_worksets": 5,
        "max_storage_gb": 1000,
    }
    row = _instance(payload)

    customer_manager.backend.find_instance_by_external_id.return_value = row
    cfg = customer_manager.get_customer_config("cust-001")
    assert cfg is not None
    assert cfg.customer_name == "Acme"

    customer_manager.backend.list_instances_by_template.return_value = [row]
    listed = customer_manager.list_customers()
    assert len(listed) == 1
    assert listed[0].customer_id == "cust-001"


def test_update_customer_returns_none_when_missing(customer_manager: CustomerManager):
    customer_manager.backend.find_instance_by_external_id.return_value = None
    assert customer_manager.update_customer("missing", customer_name="new") is None


def test_update_customer_writes_payload(customer_manager: CustomerManager):
    row = _instance(
        {
            "customer_id": "cust-001",
            "customer_name": "Old",
            "email": "old@example.com",
            "s3_bucket": "bucket",
            "max_concurrent_worksets": 5,
            "max_storage_gb": 1000,
            "api_tokens": [],
        }
    )
    customer_manager.backend.find_instance_by_external_id.return_value = row

    updated = customer_manager.update_customer("cust-001", customer_name="New", cost_center="CC-NEW")

    assert updated is not None
    assert updated.customer_name == "New"
    assert updated.cost_center == "CC-NEW"
    customer_manager.backend.update_instance_json.assert_called_once()


def test_api_token_views_and_revoke(customer_manager: CustomerManager):
    cfg = CustomerConfig(
        customer_id="cust-001",
        customer_name="Acme",
        email="acme@example.com",
        s3_bucket="bucket",
        api_tokens=[
            {
                "id": "tok-1",
                "name": "primary",
                "token_hash": "hash",
                "created_at": "2026-01-01T00:00:00Z",
                "expires_at": "2099-01-01T00:00:00Z",
                "revoked": False,
            }
        ],
    )
    customer_manager.get_customer_config = MagicMock(return_value=cfg)
    customer_manager.update_customer = MagicMock(return_value=cfg)

    listed = customer_manager.list_api_tokens("cust-001")
    assert "token_hash" not in listed[0]

    assert customer_manager.revoke_api_token("cust-001", "tok-1") is True
    customer_manager.update_customer.assert_called_once()


def test_verify_workset_ownership_filters_by_customer_id():
    worksets = [
        {"workset_id": "ws-1", "customer_id": "customer-a"},
        {"workset_id": "ws-2", "customer_id": "customer-b"},
    ]
    filtered = [w for w in worksets if verify_workset_ownership(w, "customer-a")]
    assert [w["workset_id"] for w in filtered] == ["ws-1"]


def test_verify_workset_ownership_fallback_to_metadata_submitted_by():
    legacy = {"workset_id": "legacy", "metadata": {"submitted_by": "cust-legacy"}}
    assert verify_workset_ownership(legacy, "cust-legacy") is True
    assert verify_workset_ownership(legacy, "other") is False


def test_verify_workset_ownership_missing_owner_is_denied():
    assert verify_workset_ownership({"workset_id": "orphan"}, "cust-any") is False
