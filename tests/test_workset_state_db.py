"""Graph-native unit tests for WorksetStateDB."""

from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock

import pytest

from daylily_ursa.workset_state_db import (
    ErrorCategory,
    WorksetPriority,
    WorksetState,
    WorksetStateDB,
)


class _SessionCtx:
    def __init__(self, session: MagicMock):
        self._session = session

    def __enter__(self):
        return self._session

    def __exit__(self, exc_type, exc, tb):
        return False


class _Instance:
    def __init__(self, *, name: str = "ws", bstatus: str = "ready", json_addl: dict | None = None):
        self.name = name
        self.bstatus = bstatus
        self.json_addl = dict(json_addl or {})
        self.is_deleted = False


@pytest.fixture
def state_db() -> WorksetStateDB:
    db = WorksetStateDB.__new__(WorksetStateDB)
    db.lock_timeout_seconds = 300

    db.backend = MagicMock()
    db.cloudwatch = MagicMock()
    db._cloudwatch = None

    db._session = MagicMock()
    db.backend.session_scope.return_value = _SessionCtx(db._session)
    db.backend.ensure_templates.return_value = None
    db.backend.create_lineage.return_value = None
    db.backend.create_instance.return_value = _Instance(name="created")

    db._find_customer = MagicMock(return_value=None)
    db._write_state_event = MagicMock()
    db._write_lock_event = MagicMock()
    db._emit_metric = MagicMock()
    db._find_workset = MagicMock(return_value=None)
    return db


def test_register_workset_success(state_db: WorksetStateDB):
    ws = _Instance(name="ws-001")
    state_db.backend.create_instance.return_value = ws

    ok = state_db.register_workset(
        workset_id="ws-001",
        bucket="my-bucket",
        prefix="worksets/ws-001/",
        priority=WorksetPriority.NORMAL,
        metadata={"samples": [{"sample_id": "S1"}]},
        customer_id="cust-1",
    )

    assert ok is True
    state_db.backend.create_instance.assert_called_once()
    payload = state_db.backend.create_instance.call_args.kwargs["json_addl"]
    assert payload["workset_id"] == "ws-001"
    assert payload["state"] == "ready"


def test_register_workset_duplicate_returns_false(state_db: WorksetStateDB):
    state_db._find_workset.return_value = _Instance(name="existing")

    ok = state_db.register_workset(
        workset_id="ws-dup",
        bucket="my-bucket",
        prefix="worksets/ws-dup/",
        metadata={"samples": [{"sample_id": "S1"}]},
        customer_id="cust-1",
    )

    assert ok is False


def test_register_workset_rejects_invalid_customer(state_db: WorksetStateDB):
    with pytest.raises(ValueError):
        state_db.register_workset(
            workset_id="ws-bad",
            bucket="my-bucket",
            prefix="worksets/ws-bad/",
            metadata={"samples": [{"sample_id": "S1"}]},
            customer_id="",
        )


def test_register_workset_requires_samples(state_db: WorksetStateDB):
    with pytest.raises(ValueError):
        state_db.register_workset(
            workset_id="ws-nosamples",
            bucket="my-bucket",
            prefix="worksets/ws-nosamples/",
            metadata={},
            customer_id="cust-1",
        )


def test_acquire_lock_success(state_db: WorksetStateDB):
    ws = _Instance(name="ws-lock", bstatus="ready", json_addl={"state": "ready"})
    state_db._find_workset.return_value = ws

    ok = state_db.acquire_lock("ws-lock", owner_id="worker-1")

    assert ok is True
    assert ws.json_addl["lock_owner"] == "worker-1"
    assert "lock_expires_at" in ws.json_addl


def test_acquire_lock_denied_for_in_progress_without_force(state_db: WorksetStateDB):
    ws = _Instance(name="ws-lock", bstatus="in_progress", json_addl={"state": "in_progress"})
    state_db._find_workset.return_value = ws

    ok = state_db.acquire_lock("ws-lock", owner_id="worker-1", force=False)

    assert ok is False


def test_release_lock_owner_mismatch_returns_false(state_db: WorksetStateDB):
    ws = _Instance(name="ws-lock", json_addl={"lock_owner": "worker-a"})
    state_db._find_workset.return_value = ws

    ok = state_db.release_lock("ws-lock", owner_id="worker-b")

    assert ok is False


def test_release_lock_success(state_db: WorksetStateDB):
    ws = _Instance(
        name="ws-lock",
        json_addl={
            "lock_owner": "worker-a",
            "lock_acquired_at": "2026-01-01T00:00:00Z",
            "lock_expires_at": "2026-01-01T00:10:00Z",
            "lock": {"epoch": 1},
        },
    )
    state_db._find_workset.return_value = ws

    ok = state_db.release_lock("ws-lock", owner_id="worker-a")

    assert ok is True
    assert "lock_owner" not in ws.json_addl


def test_refresh_lock_success(state_db: WorksetStateDB):
    ws = _Instance(name="ws-lock", json_addl={"lock_owner": "worker-a", "lock": {"epoch": 2}})
    state_db._find_workset.return_value = ws

    ok = state_db.refresh_lock("ws-lock", owner_id="worker-a")

    assert ok is True
    assert ws.json_addl["lock"]["epoch"] == 3


def test_record_failure_sets_retrying(state_db: WorksetStateDB):
    ws = _Instance(name="ws-fail", json_addl={"retry_count": 0, "max_retries": 3, "state": "in_progress"})
    state_db.get_workset = MagicMock(return_value={"retry_count": 0, "max_retries": 3})
    state_db._find_workset.return_value = ws

    should_retry = state_db.record_failure("ws-fail", "boom", error_category=ErrorCategory.TRANSIENT)

    assert should_retry is True
    assert ws.json_addl["state"] == WorksetState.RETRYING.value


def test_record_failure_sets_failed_at_max_retries(state_db: WorksetStateDB):
    ws = _Instance(name="ws-fail", json_addl={"retry_count": 3, "max_retries": 3, "state": "in_progress"})
    state_db.get_workset = MagicMock(return_value={"retry_count": 3, "max_retries": 3})
    state_db._find_workset.return_value = ws

    should_retry = state_db.record_failure("ws-fail", "boom", error_category=ErrorCategory.TRANSIENT)

    assert should_retry is False
    assert ws.json_addl["state"] == WorksetState.FAILED.value


def test_get_retryable_worksets_filters_retry_after(state_db: WorksetStateDB):
    past = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    future = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    state_db.list_worksets_by_state = MagicMock(
        return_value=[
            {"workset_id": "ws-ready", "retry_after": past},
            {"workset_id": "ws-later", "retry_after": future},
            {"workset_id": "ws-now"},
        ]
    )

    out = state_db.get_retryable_worksets()

    assert {w["workset_id"] for w in out} == {"ws-ready", "ws-now"}


def test_reset_for_retry_calls_update_state(state_db: WorksetStateDB):
    state_db.update_state = MagicMock(return_value=None)
    assert state_db.reset_for_retry("ws-1") is True
    state_db.update_state.assert_called_once()


def test_get_ready_worksets_prioritized_orders_by_priority(state_db: WorksetStateDB):
    def _side_effect(state, priority=None, limit=100):
        if priority == WorksetPriority.URGENT:
            return [{"workset_id": "u1"}]
        if priority == WorksetPriority.NORMAL:
            return [{"workset_id": "n1"}]
        if priority == WorksetPriority.LOW:
            return [{"workset_id": "l1"}]
        return []

    state_db.list_worksets_by_state = MagicMock(side_effect=_side_effect)

    out = state_db.get_ready_worksets_prioritized(limit=10)

    assert [x["workset_id"] for x in out] == ["u1", "n1", "l1"]


def test_archive_delete_restore_flow(state_db: WorksetStateDB):
    ws = _Instance(name="ws-1", bstatus="ready", json_addl={"state": "ready"})
    state_db._find_workset.return_value = ws

    assert state_db.archive_workset("ws-1", archived_by="tester") is True
    assert ws.json_addl["state"] == WorksetState.ARCHIVED.value

    assert state_db.restore_workset("ws-1", restored_by="tester") is True
    assert ws.json_addl["state"] == WorksetState.READY.value

    assert state_db.delete_workset("ws-1", deleted_by="tester", hard_delete=False) is True
    assert ws.json_addl["state"] == WorksetState.DELETED.value
