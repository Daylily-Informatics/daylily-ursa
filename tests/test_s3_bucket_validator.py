"""Unit tests for LinkedBucketManager bucket-link behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

from daylily_ursa.s3_bucket_validator import BucketValidationResult, LinkedBucketManager


class _SessionCtx:
    def __init__(self, session: MagicMock):
        self._session = session

    def __enter__(self):
        return self._session

    def __exit__(self, exc_type, exc, tb):
        return False


def _build_manager() -> LinkedBucketManager:
    mgr = LinkedBucketManager.__new__(LinkedBucketManager)
    mgr.region = "us-west-2"
    mgr.profile = None
    mgr.validator = MagicMock()
    mgr.backend = MagicMock()
    session = MagicMock()
    mgr.backend.session_scope.return_value = _SessionCtx(session)
    mgr.backend.find_instance_by_external_id.return_value = None
    mgr.backend.create_instance.return_value = MagicMock(euid="BK-12345")
    mgr.backend.create_lineage.return_value = None
    mgr._ensure_customer = MagicMock(return_value=MagicMock())
    return mgr


def test_link_bucket_validates_access_when_requested():
    mgr = _build_manager()
    validation = BucketValidationResult(
        bucket_name="my-bucket",
        exists=True,
        accessible=True,
        can_read=True,
        can_write=True,
        can_list=True,
        region="us-west-2",
    )
    mgr.validator.validate_bucket.return_value = validation

    bucket, result = mgr.link_bucket(
        customer_id="cust-001",
        bucket_name="my-bucket",
        validate=True,
    )

    assert result is validation
    assert bucket.is_validated is True
    assert bucket.can_read is True
    assert bucket.can_write is True
    assert bucket.can_list is True
    mgr.validator.validate_bucket.assert_called_once_with("my-bucket")


def test_link_bucket_skips_validation_when_disabled():
    mgr = _build_manager()

    bucket, result = mgr.link_bucket(
        customer_id="cust-001",
        bucket_name="my-bucket",
        validate=False,
    )

    assert result is None
    assert bucket.is_validated is False
    assert bucket.can_read is False
    assert bucket.can_write is False
    assert bucket.can_list is False
    mgr.validator.validate_bucket.assert_not_called()

