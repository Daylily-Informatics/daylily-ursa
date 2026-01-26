"""Tests for DynamoDB-based workset state management."""

import datetime as dt
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from daylib.workset_state_db import (
    ErrorCategory,
    WorksetPriority,
    WorksetState,
    WorksetStateDB,
)


@pytest.fixture
def mock_dynamodb():
    """Mock DynamoDB resource."""
    with patch("daylib.workset_state_db.boto3.Session") as mock_session:
        mock_resource = MagicMock()
        mock_table = MagicMock()
        mock_client = MagicMock()
        
        mock_session.return_value.resource.return_value = mock_resource
        mock_session.return_value.client.return_value = mock_client
        mock_resource.Table.return_value = mock_table
        
        yield {
            "session": mock_session,
            "resource": mock_resource,
            "table": mock_table,
            "client": mock_client,
        }


@pytest.fixture
def state_db(mock_dynamodb):
    """Create WorksetStateDB instance with mocked DynamoDB."""
    db = WorksetStateDB(
        table_name="test-worksets",
        region="us-west-2",
        profile=None,
        lock_timeout_seconds=300,
    )
    return db


def test_register_workset_success(state_db, mock_dynamodb):
    """Test successful workset registration."""
    mock_table = mock_dynamodb["table"]
    mock_table.put_item.return_value = {}

    result = state_db.register_workset(
        workset_id="test-workset-001",
        bucket="test-bucket",
        prefix="worksets/test-001/",
        priority=WorksetPriority.NORMAL,
        metadata={"samples": [{"sample_id": "S1"}], "sample_count": 1, "estimated_cost": 50.0},
        customer_id="test-customer",
    )

    assert result is True
    mock_table.put_item.assert_called_once()

    call_args = mock_table.put_item.call_args
    item = call_args.kwargs["Item"]

    assert item["workset_id"] == "test-workset-001"
    assert item["state"] == WorksetState.READY.value
    assert item["priority"] == WorksetPriority.NORMAL.value
    assert item["bucket"] == "test-bucket"
    assert item["prefix"] == "worksets/test-001/"
    assert item["customer_id"] == "test-customer"
    assert "created_at" in item
    assert "state_history" in item


def test_register_workset_already_exists(state_db, mock_dynamodb):
    """Test registering a workset that already exists."""
    from botocore.exceptions import ClientError

    mock_table = mock_dynamodb["table"]
    mock_table.put_item.side_effect = ClientError(
        {"Error": {"Code": "ConditionalCheckFailedException"}},
        "PutItem",
    )

    result = state_db.register_workset(
        workset_id="existing-workset",
        bucket="test-bucket",
        prefix="worksets/existing/",
        metadata={"samples": [{"sample_id": "S1"}]},
        customer_id="test-customer",
    )

    assert result is False


def test_register_workset_rejects_missing_customer_id(state_db, mock_dynamodb):
    """Test that registering a workset without customer_id is rejected."""
    import pytest

    with pytest.raises(ValueError, match="customer_id is required"):
        state_db.register_workset(
            workset_id="test-workset",
            bucket="test-bucket",
            prefix="worksets/test/",
            metadata={"samples": [{"sample_id": "S1"}]},
            customer_id=None,
        )


def test_register_workset_rejects_empty_customer_id(state_db, mock_dynamodb):
    """Test that registering a workset with empty customer_id is rejected."""
    import pytest

    with pytest.raises(ValueError, match="customer_id cannot be empty"):
        state_db.register_workset(
            workset_id="test-workset",
            bucket="test-bucket",
            prefix="worksets/test/",
            metadata={"samples": [{"sample_id": "S1"}]},
            customer_id="  ",
        )


def test_register_workset_rejects_unknown_customer_id(state_db, mock_dynamodb):
    """Test that registering a workset with 'Unknown' customer_id is rejected."""
    import pytest

    with pytest.raises(ValueError, match="customer_id cannot be 'Unknown'"):
        state_db.register_workset(
            workset_id="test-workset",
            bucket="test-bucket",
            prefix="worksets/test/",
            metadata={"samples": [{"sample_id": "S1"}]},
            customer_id="Unknown",
        )


def test_register_workset_rejects_no_samples(state_db, mock_dynamodb):
    """Test that registering a workset without samples is rejected."""
    import pytest

    with pytest.raises(ValueError, match="Workset must have at least one sample"):
        state_db.register_workset(
            workset_id="test-workset",
            bucket="test-bucket",
            prefix="worksets/test/",
            metadata={"other_field": "value"},
            customer_id="test-customer",
        )


def test_register_workset_skip_validation(state_db, mock_dynamodb):
    """Test that skip_validation=True bypasses customer_id and sample validation."""
    mock_table = mock_dynamodb["table"]
    mock_table.put_item.return_value = {}

    # This should succeed without customer_id or samples when skip_validation=True
    result = state_db.register_workset(
        workset_id="monitor-discovered-workset",
        bucket="test-bucket",
        prefix="worksets/test/",
        metadata={"source": "monitor_discovery"},
        customer_id=None,
        skip_validation=True,
    )

    assert result is True


def test_acquire_lock_success(state_db, mock_dynamodb):
    """Test successful lock acquisition."""
    mock_table = mock_dynamodb["table"]
    
    # Mock get_item to return a ready workset
    mock_table.get_item.return_value = {
        "Item": {
            "workset_id": "test-workset",
            "state": WorksetState.READY.value,
            "priority": WorksetPriority.NORMAL.value,
        }
    }
    
    # Mock update_item to succeed
    mock_table.update_item.return_value = {}
    
    result = state_db.acquire_lock(
        workset_id="test-workset",
        owner_id="monitor-instance-1",
    )
    
    assert result is True
    mock_table.update_item.assert_called_once()


def test_acquire_lock_already_locked(state_db, mock_dynamodb):
    """Test lock acquisition when workset is already locked."""
    mock_table = mock_dynamodb["table"]

    # Mock get_item to return a locked workset (recent lock)
    now = dt.datetime.now(dt.timezone.utc)
    mock_table.get_item.return_value = {
        "Item": {
            "workset_id": "test-workset",
            "state": WorksetState.LOCKED.value,
            "lock_owner": "other-monitor",
            "lock_acquired_at": now.isoformat() + "Z",
        }
    }
    
    result = state_db.acquire_lock(
        workset_id="test-workset",
        owner_id="monitor-instance-1",
    )
    
    assert result is False
    mock_table.update_item.assert_not_called()


def test_acquire_lock_stale_lock(state_db, mock_dynamodb):
    """Test lock acquisition with stale lock (auto-release).

    Note: Locking is now separate from state. A workset with a stale lock
    is identified by lock_owner/lock_acquired_at fields, not by state.
    The state remains READY or RETRYING (lockable states).
    """
    mock_table = mock_dynamodb["table"]

    # Mock get_item to return a READY workset with stale lock fields
    stale_time = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=400)
    stale_expires = (stale_time + dt.timedelta(seconds=300)).isoformat().replace("+00:00", "Z")
    mock_table.get_item.return_value = {
        "Item": {
            "workset_id": "test-workset",
            "state": WorksetState.READY.value,  # State is still lockable
            "lock_owner": "dead-monitor",
            "lock_acquired_at": stale_time.isoformat() + "Z",
            "lock_expires_at": stale_expires,  # Already expired
        }
    }

    mock_table.update_item.return_value = {}

    result = state_db.acquire_lock(
        workset_id="test-workset",
        owner_id="monitor-instance-1",
    )

    assert result is True
    mock_table.update_item.assert_called_once()


def test_release_lock_success(state_db, mock_dynamodb):
    """Test successful lock release."""
    mock_table = mock_dynamodb["table"]
    mock_table.update_item.return_value = {}

    result = state_db.release_lock(
        workset_id="test-workset",
        owner_id="monitor-instance-1",
    )

    assert result is True
    mock_table.update_item.assert_called_once()


def test_update_state(state_db, mock_dynamodb):
    """Test state update with audit trail."""
    mock_table = mock_dynamodb["table"]
    mock_table.update_item.return_value = {}

    state_db.update_state(
        workset_id="test-workset",
        new_state=WorksetState.IN_PROGRESS,
        reason="Pipeline started",
        cluster_name="test-cluster",
        metrics={"vcpus": 32, "cost": 10.5},
    )

    mock_table.update_item.assert_called_once()
    call_args = mock_table.update_item.call_args

    assert call_args.kwargs["ExpressionAttributeValues"][":state"] == WorksetState.IN_PROGRESS.value
    assert call_args.kwargs["ExpressionAttributeValues"][":cluster"] == "test-cluster"


def test_get_ready_worksets_prioritized(state_db, mock_dynamodb):
    """Test getting ready worksets ordered by priority."""
    mock_table = mock_dynamodb["table"]

    # Mock query to return worksets for each priority
    urgent_worksets = [
        {"workset_id": "urgent-1", "priority": "urgent", "state": "ready"}
    ]
    normal_worksets = [
        {"workset_id": "normal-1", "priority": "normal", "state": "ready"},
        {"workset_id": "normal-2", "priority": "normal", "state": "ready"},
    ]

    mock_table.query.side_effect = [
        {"Items": urgent_worksets},
        {"Items": normal_worksets},
        {"Items": []},  # low priority
    ]

    worksets = state_db.get_ready_worksets_prioritized(limit=10)

    assert len(worksets) == 3
    assert worksets[0]["workset_id"] == "urgent-1"
    assert worksets[1]["workset_id"] == "normal-1"


def test_serialize_metadata(state_db):
    """Test metadata serialization for DynamoDB."""
    data = {
        "cost": 10.5,
        "samples": 5,
        "nested": {
            "value": 3.14,
            "list": [1.1, 2.2, 3.3],
        },
    }

    serialized = state_db._serialize_metadata(data)

    assert isinstance(serialized["cost"], Decimal)
    assert serialized["samples"] == 5
    assert isinstance(serialized["nested"]["value"], Decimal)
    assert all(isinstance(v, Decimal) for v in serialized["nested"]["list"])


def test_deserialize_item(state_db):
    """Test item deserialization from DynamoDB."""
    item = {
        "workset_id": "test",
        "cost": Decimal("10.5"),
        "metrics": {
            "vcpus": Decimal("32"),
            "values": [Decimal("1.1"), Decimal("2.2")],
        },
    }

    deserialized = state_db._deserialize_item(item)

    assert deserialized["cost"] == 10.5
    assert deserialized["metrics"]["vcpus"] == 32.0
    assert deserialized["metrics"]["values"] == [1.1, 2.2]


def test_deserialize_item_promotes_sample_count_from_metadata(state_db):
    """Test that sample_count is promoted from metadata to top level.

    This ensures templates can use ws.sample_count instead of ws.metadata.sample_count.
    Regression test for: "Samples" column in Recent Worksets always showing 0.
    """
    item = {
        "workset_id": "test-ws",
        "state": "ready",
        "metadata": {
            "sample_count": 5,
            "pipeline_type": "germline",
            "samples": [{"sample_id": f"S{i}"} for i in range(5)],
        },
    }

    deserialized = state_db._deserialize_item(item)

    # sample_count should be promoted to top level
    assert deserialized["sample_count"] == 5
    # pipeline_type should also be promoted
    assert deserialized["pipeline_type"] == "germline"
    # Original metadata should still be intact
    assert deserialized["metadata"]["sample_count"] == 5
    assert deserialized["metadata"]["pipeline_type"] == "germline"


def test_deserialize_item_does_not_overwrite_top_level_sample_count(state_db):
    """Test that top-level sample_count is not overwritten by metadata."""
    item = {
        "workset_id": "test-ws",
        "sample_count": 10,  # Top-level takes precedence
        "metadata": {
            "sample_count": 5,  # Should not overwrite
        },
    }

    deserialized = state_db._deserialize_item(item)

    # Top-level value should be preserved
    assert deserialized["sample_count"] == 10


def test_record_failure_transient(state_db, mock_dynamodb):
    """Test recording a transient failure."""
    mock_table = mock_dynamodb["table"]
    mock_table.get_item.return_value = {
        "Item": {
            "workset_id": "ws-001",
            "state": WorksetState.IN_PROGRESS.value,
            "retry_count": 0,
            "max_retries": 3,
        }
    }
    mock_table.update_item.return_value = {}

    should_retry = state_db.record_failure(
        "ws-001",
        "Network timeout",
        ErrorCategory.TRANSIENT,
    )

    assert should_retry is True
    mock_table.update_item.assert_called_once()


def test_record_failure_permanent(state_db, mock_dynamodb):
    """Test recording a permanent failure."""
    mock_table = mock_dynamodb["table"]
    mock_table.get_item.return_value = {
        "Item": {
            "workset_id": "ws-001",
            "state": WorksetState.IN_PROGRESS.value,
            "retry_count": 0,
            "max_retries": 3,
        }
    }
    mock_table.update_item.return_value = {}

    should_retry = state_db.record_failure(
        "ws-001",
        "Invalid configuration",
        ErrorCategory.CONFIGURATION,
    )

    assert should_retry is False
    mock_table.update_item.assert_called_once()


def test_record_failure_max_retries_exceeded(state_db, mock_dynamodb):
    """Test recording failure when max retries exceeded."""
    mock_table = mock_dynamodb["table"]
    mock_table.get_item.return_value = {
        "Item": {
            "workset_id": "ws-001",
            "state": WorksetState.RETRYING.value,
            "retry_count": 3,
            "max_retries": 3,
        }
    }
    mock_table.update_item.return_value = {}

    should_retry = state_db.record_failure(
        "ws-001",
        "Still failing",
        ErrorCategory.TRANSIENT,
    )

    assert should_retry is False


def test_get_retryable_worksets(state_db, mock_dynamodb):
    """Test getting retryable worksets."""
    mock_table = mock_dynamodb["table"]
    past_time = "2024-01-01T00:00:00Z"
    future_time = "2099-01-01T00:00:00Z"

    mock_table.query.return_value = {
        "Items": [
            {
                "workset_id": "ws-001",
                "state": WorksetState.RETRYING.value,
                "retry_after": past_time,
            },
            {
                "workset_id": "ws-002",
                "state": WorksetState.RETRYING.value,
                "retry_after": future_time,
            },
        ]
    }

    retryable = state_db.get_retryable_worksets()

    # Only ws-001 should be retryable (past time)
    assert len(retryable) == 1
    assert retryable[0]["workset_id"] == "ws-001"


def test_set_cluster_affinity(state_db, mock_dynamodb):
    """Test setting cluster affinity."""
    mock_table = mock_dynamodb["table"]
    mock_table.update_item.return_value = {}

    success = state_db.set_cluster_affinity(
        "ws-001",
        "cluster-us-west-2a",
        "data_locality",
    )

    assert success is True
    mock_table.update_item.assert_called_once()


def test_get_concurrent_worksets_count(state_db, mock_dynamodb):
    """Test getting concurrent worksets count.

    Note: Concurrency count now includes:
    - IN_PROGRESS worksets (via state GSI query)
    - Worksets with lock_owner set (via scan) - these are locked but may not have
      transitioned state yet
    """
    mock_table = mock_dynamodb["table"]
    # list_worksets_by_state uses query on state GSI
    mock_table.query.return_value = {
        "Items": [{"workset_id": "ws-001"}, {"workset_id": "ws-002"}]  # IN_PROGRESS
    }
    # list_locked_worksets uses scan with filter on lock_owner
    mock_table.scan.return_value = {
        "Items": [{"workset_id": "ws-003", "lock_owner": "monitor-1"}]  # locked
    }

    count = state_db.get_concurrent_worksets_count()

    # 2 IN_PROGRESS + 1 locked
    assert count == 3


def test_can_start_new_workset(state_db, mock_dynamodb):
    """Test checking if new workset can start."""
    mock_table = mock_dynamodb["table"]
    mock_table.query.return_value = {
        "Items": [{"workset_id": "ws-001"}]  # IN_PROGRESS
    }
    mock_table.scan.return_value = {
        "Items": []  # No locked worksets
    }

    can_start = state_db.can_start_new_workset(max_concurrent=5)

    assert can_start is True


def test_can_start_new_workset_at_limit(state_db, mock_dynamodb):
    """Test checking if new workset can start when at limit."""
    mock_table = mock_dynamodb["table"]
    mock_table.query.return_value = {
        "Items": [{"workset_id": f"ws-{i}"} for i in range(5)]  # IN_PROGRESS
    }
    mock_table.scan.return_value = {
        "Items": []  # No locked worksets
    }

    can_start = state_db.can_start_new_workset(max_concurrent=5)

    assert can_start is False


# ==================== Archive/Delete/Restore Tests ====================


def test_archive_workset_success(state_db, mock_dynamodb):
    """Test successful workset archiving."""
    mock_table = mock_dynamodb["table"]
    mock_table.get_item.return_value = {
        "Item": {
            "workset_id": "test-ws-001",
            "state": WorksetState.COMPLETE.value,
            "state_history": [],
        }
    }
    mock_table.update_item.return_value = {}

    result = state_db.archive_workset(
        workset_id="test-ws-001",
        archived_by="test-user",
        archive_reason="No longer needed",
    )

    assert result is True
    mock_table.update_item.assert_called_once()

    call_args = mock_table.update_item.call_args
    assert call_args.kwargs["Key"] == {"workset_id": "test-ws-001"}
    # Check that state is set to ARCHIVED
    expr_values = call_args.kwargs["ExpressionAttributeValues"]
    assert expr_values[":state"] == WorksetState.ARCHIVED.value
    assert expr_values[":archived_by"] == "test-user"
    assert expr_values[":reason"] == "No longer needed"


def test_archive_workset_dynamodb_error(state_db, mock_dynamodb):
    """Test archiving fails on DynamoDB error."""
    mock_table = mock_dynamodb["table"]
    mock_table.update_item.side_effect = ClientError(
        {"Error": {"Code": "ConditionalCheckFailedException"}},
        "UpdateItem"
    )

    result = state_db.archive_workset(workset_id="nonexistent-ws")

    assert result is False


def test_delete_workset_soft_delete_success(state_db, mock_dynamodb):
    """Test successful soft delete of a workset."""
    mock_table = mock_dynamodb["table"]
    mock_table.get_item.return_value = {
        "Item": {
            "workset_id": "test-ws-001",
            "state": WorksetState.COMPLETE.value,
            "state_history": [],
        }
    }
    mock_table.update_item.return_value = {}

    result = state_db.delete_workset(
        workset_id="test-ws-001",
        deleted_by="test-user",
        delete_reason="Cleaning up old data",
        hard_delete=False,
    )

    assert result is True
    mock_table.update_item.assert_called_once()
    mock_table.delete_item.assert_not_called()

    call_args = mock_table.update_item.call_args
    expr_values = call_args.kwargs["ExpressionAttributeValues"]
    assert expr_values[":state"] == WorksetState.DELETED.value


def test_delete_workset_hard_delete_success(state_db, mock_dynamodb):
    """Test successful hard delete of a workset."""
    mock_table = mock_dynamodb["table"]
    mock_table.get_item.return_value = {
        "Item": {
            "workset_id": "test-ws-001",
            "state": WorksetState.COMPLETE.value,
        }
    }
    mock_table.delete_item.return_value = {}

    result = state_db.delete_workset(
        workset_id="test-ws-001",
        deleted_by="test-user",
        hard_delete=True,
    )

    assert result is True
    mock_table.delete_item.assert_called_once()
    call_args = mock_table.delete_item.call_args
    assert call_args.kwargs["Key"] == {"workset_id": "test-ws-001"}


def test_delete_workset_dynamodb_error(state_db, mock_dynamodb):
    """Test deleting fails on DynamoDB error."""
    mock_table = mock_dynamodb["table"]
    mock_table.update_item.side_effect = ClientError(
        {"Error": {"Code": "InternalServerError"}},
        "UpdateItem"
    )

    result = state_db.delete_workset(workset_id="test-ws-001", hard_delete=False)

    assert result is False


def test_delete_workset_hard_delete_error(state_db, mock_dynamodb):
    """Test hard delete fails on DynamoDB error."""
    mock_table = mock_dynamodb["table"]
    mock_table.delete_item.side_effect = ClientError(
        {"Error": {"Code": "InternalServerError"}},
        "DeleteItem"
    )

    result = state_db.delete_workset(workset_id="test-ws-001", hard_delete=True)

    assert result is False


def test_restore_workset_success(state_db, mock_dynamodb):
    """Test successful restoration of an archived workset."""
    mock_table = mock_dynamodb["table"]
    mock_table.update_item.return_value = {}

    result = state_db.restore_workset(
        workset_id="test-ws-001",
        restored_by="test-user",
    )

    assert result is True
    mock_table.update_item.assert_called_once()

    call_args = mock_table.update_item.call_args
    expr_values = call_args.kwargs["ExpressionAttributeValues"]
    assert expr_values[":state"] == WorksetState.READY.value
    # restore_workset has a ConditionExpression that requires state == archived
    assert expr_values[":archived"] == WorksetState.ARCHIVED.value


def test_restore_workset_not_archived_fails(state_db, mock_dynamodb):
    """Test restoring a non-archived workset fails due to condition check."""
    mock_table = mock_dynamodb["table"]
    # DynamoDB returns ConditionalCheckFailedException when condition not met
    mock_table.update_item.side_effect = ClientError(
        {"Error": {"Code": "ConditionalCheckFailedException"}},
        "UpdateItem"
    )

    result = state_db.restore_workset(workset_id="test-ws-001")

    assert result is False


def test_list_archived_worksets(state_db, mock_dynamodb):
    """Test listing archived worksets."""
    mock_table = mock_dynamodb["table"]
    mock_table.query.return_value = {
        "Items": [
            {"workset_id": "ws-001", "state": WorksetState.ARCHIVED.value},
            {"workset_id": "ws-002", "state": WorksetState.ARCHIVED.value},
        ]
    }

    result = state_db.list_archived_worksets(limit=50)

    assert len(result) == 2
    assert result[0]["workset_id"] == "ws-001"
    mock_table.query.assert_called_once()


def test_archive_workset_with_reason(state_db, mock_dynamodb):
    """Test archiving a workset with a reason."""
    mock_table = mock_dynamodb["table"]
    mock_table.update_item.return_value = {}
    # Mock get_workset to return a workset (archive_workset looks it up first)
    mock_table.get_item.return_value = {
        "Item": {"workset_id": "test-ws-001", "state": "complete"}
    }

    result = state_db.archive_workset(
        workset_id="test-ws-001",
        archived_by="admin",
        archive_reason="Data retention policy"
    )

    assert result is True
    call_args = mock_table.update_item.call_args
    expr_values = call_args.kwargs["ExpressionAttributeValues"]
    assert expr_values[":reason"] == "Data retention policy"


def test_delete_workset_with_reason(state_db, mock_dynamodb):
    """Test soft deleting a workset with a reason."""
    mock_table = mock_dynamodb["table"]
    mock_table.update_item.return_value = {}

    result = state_db.delete_workset(
        workset_id="test-ws-001",
        deleted_by="admin",
        delete_reason="Customer request",
        hard_delete=False
    )

    assert result is True
    call_args = mock_table.update_item.call_args
    expr_values = call_args.kwargs["ExpressionAttributeValues"]
    assert expr_values[":reason"] == "Customer request"


# ========== Tests for Phase 1 bucket normalization fixes ==========


class TestBucketNormalizationInStateDB:
    """Test bucket names are normalized in WorksetStateDB.register_workset."""

    def test_register_workset_normalizes_s3_prefix(self, state_db, mock_dynamodb):
        """Test register_workset strips s3:// prefix from bucket."""
        mock_table = mock_dynamodb["table"]
        mock_table.put_item.return_value = {}

        result = state_db.register_workset(
            workset_id="test-ws-norm",
            bucket="s3://my-bucket-with-prefix",
            prefix="worksets/test/",
            priority=WorksetPriority.NORMAL,
            metadata={"samples": [{"sample_id": "S1"}], "sample_count": 1},
            customer_id="test-customer",
        )

        assert result is True
        call_args = mock_table.put_item.call_args
        item = call_args.kwargs["Item"]
        assert item["bucket"] == "my-bucket-with-prefix"

    def test_register_workset_normalizes_bucket_with_path(self, state_db, mock_dynamodb):
        """Test register_workset extracts bucket from s3://bucket/path format."""
        mock_table = mock_dynamodb["table"]
        mock_table.put_item.return_value = {}

        result = state_db.register_workset(
            workset_id="test-ws-path",
            bucket="s3://my-bucket/some/path/component",
            prefix="worksets/test/",
            priority=WorksetPriority.NORMAL,
            metadata={"samples": [{"sample_id": "S1"}], "sample_count": 1},
            customer_id="test-customer",
        )

        assert result is True
        call_args = mock_table.put_item.call_args
        item = call_args.kwargs["Item"]
        assert item["bucket"] == "my-bucket"

    def test_register_workset_rejects_empty_bucket(self, state_db, mock_dynamodb):
        """Test register_workset raises ValueError for empty bucket."""
        with pytest.raises(ValueError, match="Invalid bucket name"):
            state_db.register_workset(
                workset_id="test-ws-empty",
                bucket="",
                prefix="worksets/test/",
                priority=WorksetPriority.NORMAL,
                metadata={"samples": [{"sample_id": "S1"}], "sample_count": 1},
                customer_id="test-customer",
            )

    def test_register_workset_rejects_s3_only_bucket(self, state_db, mock_dynamodb):
        """Test register_workset raises ValueError for 's3://' only bucket."""
        with pytest.raises(ValueError, match="Invalid bucket name"):
            state_db.register_workset(
                workset_id="test-ws-s3only",
                bucket="s3://",
                prefix="worksets/test/",
                priority=WorksetPriority.NORMAL,
                metadata={"samples": [{"sample_id": "S1"}], "sample_count": 1},
                customer_id="test-customer",
            )

    def test_register_workset_plain_bucket_unchanged(self, state_db, mock_dynamodb):
        """Test register_workset leaves plain bucket name unchanged."""
        mock_table = mock_dynamodb["table"]
        mock_table.put_item.return_value = {}

        result = state_db.register_workset(
            workset_id="test-ws-plain",
            bucket="plain-bucket-name",
            prefix="worksets/test/",
            priority=WorksetPriority.NORMAL,
            metadata={"samples": [{"sample_id": "S1"}], "sample_count": 1},
            customer_id="test-customer",
        )

        assert result is True
        call_args = mock_table.put_item.call_args
        item = call_args.kwargs["Item"]
        assert item["bucket"] == "plain-bucket-name"


class TestCostReportStorage:
    """Tests for Phase 5B: Snakemake cost report storage."""

    def test_update_cost_report_success(self, state_db, mock_dynamodb):
        """Test storing cost report data in DynamoDB."""
        mock_table = mock_dynamodb["table"]
        mock_table.update_item.return_value = {}

        result = state_db.update_cost_report(
            workset_id="test-ws-001",
            total_compute_cost_usd=12.3456,
            per_sample_costs={"HG002": 5.1234, "HG003": 7.2222},
            rule_count=42,
            sample_count=2,
        )

        assert result is True
        call_args = mock_table.update_item.call_args
        expr_values = call_args.kwargs["ExpressionAttributeValues"]
        assert expr_values[":cost"] == "12.3456"
        assert expr_values[":psc"] == {"HG002": "5.1234", "HG003": "7.2222"}
        assert expr_values[":rc"] == 42
        assert expr_values[":sc"] == 2

    def test_update_cost_report_minimal(self, state_db, mock_dynamodb):
        """Test storing cost report with only total cost."""
        mock_table = mock_dynamodb["table"]
        mock_table.update_item.return_value = {}

        result = state_db.update_cost_report(
            workset_id="test-ws-002",
            total_compute_cost_usd=0.5,
        )

        assert result is True
        call_args = mock_table.update_item.call_args
        expr_values = call_args.kwargs["ExpressionAttributeValues"]
        assert expr_values[":cost"] == "0.5"
        assert ":psc" not in expr_values
        assert ":rc" not in expr_values
        assert ":sc" not in expr_values

    def test_update_cost_report_handles_error(self, state_db, mock_dynamodb):
        """Test update_cost_report returns False on DynamoDB error."""
        mock_table = mock_dynamodb["table"]
        mock_table.update_item.side_effect = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "Test error"}},
            "UpdateItem",
        )

        result = state_db.update_cost_report(
            workset_id="test-ws-003",
            total_compute_cost_usd=1.0,
        )

        assert result is False

    def test_get_cost_report_success(self, state_db, mock_dynamodb):
        """Test retrieving cost report data from DynamoDB."""
        mock_table = mock_dynamodb["table"]
        mock_table.get_item.return_value = {
            "Item": {
                "total_compute_cost_usd": "12.3456",
                "per_sample_costs": {"HG002": "5.1234", "HG003": "7.2222"},
                "cost_report_parsed_at": "2026-01-24T10:00:00Z",
                "cost_report_rule_count": 42,
                "cost_report_sample_count": 2,
            }
        }

        result = state_db.get_cost_report("test-ws-001")

        assert result is not None
        assert result["total_compute_cost_usd"] == 12.3456
        assert result["per_sample_costs"] == {"HG002": 5.1234, "HG003": 7.2222}
        assert result["cost_report_parsed_at"] == "2026-01-24T10:00:00Z"
        assert result["cost_report_rule_count"] == 42
        assert result["cost_report_sample_count"] == 2

    def test_get_cost_report_not_found(self, state_db, mock_dynamodb):
        """Test get_cost_report returns None when workset not found."""
        mock_table = mock_dynamodb["table"]
        mock_table.get_item.return_value = {}

        result = state_db.get_cost_report("nonexistent-ws")

        assert result is None

    def test_get_cost_report_partial_data(self, state_db, mock_dynamodb):
        """Test get_cost_report handles partial data."""
        mock_table = mock_dynamodb["table"]
        mock_table.get_item.return_value = {
            "Item": {
                "total_compute_cost_usd": "5.0",
            }
        }

        result = state_db.get_cost_report("test-ws-partial")

        assert result is not None
        assert result["total_compute_cost_usd"] == 5.0
        assert "per_sample_costs" not in result


class TestStorageMetrics:
    """Tests for Phase 5C: FSx + S3 storage tracking."""

    def test_update_storage_metrics_success(self, state_db, mock_dynamodb):
        """Test storing storage metrics in DynamoDB."""
        mock_table = mock_dynamodb["table"]
        mock_table.update_item.return_value = {}

        result = state_db.update_storage_metrics(
            workset_id="test-ws-001",
            results_storage_bytes=1073741824,  # 1 GB
            fsx_storage_bytes=2147483648,  # 2 GB
        )

        assert result is True
        call_args = mock_table.update_item.call_args
        expr_values = call_args.kwargs["ExpressionAttributeValues"]
        assert expr_values[":rsb"] == 1073741824
        assert expr_values[":fsb"] == 2147483648

    def test_update_storage_metrics_without_fsx(self, state_db, mock_dynamodb):
        """Test storing storage metrics without FSx size."""
        mock_table = mock_dynamodb["table"]
        mock_table.update_item.return_value = {}

        result = state_db.update_storage_metrics(
            workset_id="test-ws-002",
            results_storage_bytes=536870912,  # 512 MB
        )

        assert result is True
        call_args = mock_table.update_item.call_args
        expr_values = call_args.kwargs["ExpressionAttributeValues"]
        assert expr_values[":rsb"] == 536870912
        assert ":fsb" not in expr_values

    def test_update_storage_metrics_handles_error(self, state_db, mock_dynamodb):
        """Test update_storage_metrics returns False on DynamoDB error."""
        mock_table = mock_dynamodb["table"]
        mock_table.update_item.side_effect = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "Test error"}},
            "UpdateItem",
        )

        result = state_db.update_storage_metrics(
            workset_id="test-ws-003",
            results_storage_bytes=1000,
        )

        assert result is False

    def test_get_storage_metrics_success(self, state_db, mock_dynamodb):
        """Test retrieving storage metrics from DynamoDB."""
        mock_table = mock_dynamodb["table"]
        mock_table.get_item.return_value = {
            "Item": {
                "results_storage_bytes": 1073741824,
                "fsx_storage_bytes": 2147483648,
                "storage_calculated_at": "2026-01-24T10:00:00Z",
            }
        }

        result = state_db.get_storage_metrics("test-ws-001")

        assert result is not None
        assert result["results_storage_bytes"] == 1073741824
        assert result["fsx_storage_bytes"] == 2147483648
        assert result["storage_calculated_at"] == "2026-01-24T10:00:00Z"

    def test_get_storage_metrics_not_found(self, state_db, mock_dynamodb):
        """Test get_storage_metrics returns None when workset not found."""
        mock_table = mock_dynamodb["table"]
        mock_table.get_item.return_value = {}

        result = state_db.get_storage_metrics("nonexistent-ws")

        assert result is None

    def test_get_storage_metrics_partial_data(self, state_db, mock_dynamodb):
        """Test get_storage_metrics handles partial data."""
        mock_table = mock_dynamodb["table"]
        mock_table.get_item.return_value = {
            "Item": {
                "results_storage_bytes": 500000,
            }
        }

        result = state_db.get_storage_metrics("test-ws-partial")

        assert result is not None
        assert result["results_storage_bytes"] == 500000
        assert "fsx_storage_bytes" not in result

