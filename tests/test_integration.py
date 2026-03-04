"""Integration-style tests for graph-backed workset lifecycle behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from daylib.workset_diagnostics import classify_error
from daylib.workset_state_db import ErrorCategory, WorksetPriority, WorksetState, WorksetStateDB
from daylib.workset_validation import WorksetValidator


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

    db._find_customer = MagicMock(return_value=None)
    db._write_state_event = MagicMock()
    db._write_lock_event = MagicMock()
    db._emit_metric = MagicMock()
    db._find_workset = MagicMock(return_value=None)
    return db


class TestWorksetLifecycle:
    def test_register_to_complete_workflow(self, state_db: WorksetStateDB):
        ws = _Instance(
            name="integration-ws-001",
            bstatus="ready",
            json_addl={
                "workset_id": "integration-ws-001",
                "state": "ready",
                "priority": "normal",
                "bucket": "test-bucket",
                "prefix": "worksets/test/",
                "metadata": {"samples": [{"sample_id": "S0"}]},
            },
        )
        state_db.backend.create_instance.return_value = ws
        state_db._find_workset.side_effect = [None, ws, ws, ws]

        assert state_db.register_workset(
            workset_id="integration-ws-001",
            bucket="test-bucket",
            prefix="worksets/test/",
            priority=WorksetPriority.NORMAL,
            metadata={"samples": [{"sample_id": f"S{i}"} for i in range(5)], "sample_count": 5},
            customer_id="test-customer",
        )

        assert state_db.acquire_lock("integration-ws-001", "processor-1") is True

        state_db.update_state(
            "integration-ws-001",
            WorksetState.IN_PROGRESS,
            reason="Started processing",
            cluster_name="test-cluster",
        )
        state_db.update_state(
            "integration-ws-001",
            WorksetState.COMPLETE,
            reason="Processing finished",
            metrics={"duration_seconds": 300, "cost_usd": 5.0},
        )

        assert ws.bstatus == WorksetState.COMPLETE.value
        assert ws.json_addl["state"] == WorksetState.COMPLETE.value

    def test_register_to_error_to_retry_workflow(self, state_db: WorksetStateDB):
        ws = _Instance(name="error-ws-001", bstatus="in_progress", json_addl={"state": "in_progress", "retry_count": 0, "max_retries": 3})
        state_db.get_workset = MagicMock(return_value={"retry_count": 0, "max_retries": 3})
        state_db._find_workset.return_value = ws

        should_retry = state_db.record_failure(
            workset_id="error-ws-001",
            error_details="Connection timeout",
            error_category=ErrorCategory.TRANSIENT,
        )

        assert should_retry is True
        assert ws.json_addl["state"] == WorksetState.RETRYING.value

    def test_permanent_failure_workflow(self, state_db: WorksetStateDB):
        ws = _Instance(name="perm-error-ws-001", bstatus="retrying", json_addl={"state": "retrying", "retry_count": 3, "max_retries": 3})
        state_db.get_workset = MagicMock(return_value={"retry_count": 3, "max_retries": 3})
        state_db._find_workset.return_value = ws

        should_retry = state_db.record_failure(
            workset_id="perm-error-ws-001",
            error_details="Persistent error",
            error_category=ErrorCategory.TRANSIENT,
        )

        assert should_retry is False
        assert ws.json_addl["state"] == WorksetState.FAILED.value


class TestValidationToProcessing:
    def test_validation_before_registration(self, state_db: WorksetStateDB):
        with patch("daylib.workset_validation.boto3.Session"):
            WorksetValidator(region="us-west-2")

            config = {
                "workset_id": "validated-ws-001",
                "samples": [{"sample_id": "S1", "fastq_r1": "s3://bucket/S1_R1.fq.gz"}],
                "reference_genome": "hg38",
            }

            import jsonschema

            jsonschema.validate(config, WorksetValidator.WORK_YAML_SCHEMA)

            ws = _Instance(name="validated-ws-001", bstatus="ready", json_addl={"workset_id": "validated-ws-001", "state": "ready"})
            state_db.backend.create_instance.return_value = ws
            state_db._find_workset.side_effect = [None]

            result = state_db.register_workset(
                workset_id=config["workset_id"],
                bucket="test-bucket",
                prefix="worksets/validated-ws-001/",
                metadata={"samples": config["samples"]},
                customer_id="test-customer",
            )
            assert result is True


class TestDiagnosticsIntegration:
    def test_error_classification_and_recording(self, state_db: WorksetStateDB):
        error_text = "Out of memory: Cannot allocate 16GB"
        classification = classify_error(error_text)

        assert classification["error_code"] == "WS-RES-001"
        assert classification["category"] == "resource"
        assert classification["retryable"] is True

        ws = _Instance(name="diag-ws-001", bstatus="in_progress", json_addl={"state": "in_progress", "retry_count": 0, "max_retries": 3})
        state_db.get_workset = MagicMock(return_value={"retry_count": 0, "max_retries": 3})
        state_db._find_workset.return_value = ws

        error_category = ErrorCategory.TRANSIENT if classification["retryable"] else ErrorCategory.PERMANENT
        should_retry = state_db.record_failure(
            workset_id="diag-ws-001",
            error_details=error_text,
            error_category=error_category,
        )

        assert should_retry is True

    def test_non_retryable_error_classification(self, state_db: WorksetStateDB):
        error_text = "Invalid FASTQ format: truncated quality string"
        classification = classify_error(error_text)

        assert classification["error_code"] == "WS-DAT-001"
        assert classification["retryable"] is False

        ws = _Instance(name="data-error-ws-001", bstatus="in_progress", json_addl={"state": "in_progress", "retry_count": 0, "max_retries": 3})
        state_db.get_workset = MagicMock(return_value={"retry_count": 0, "max_retries": 3})
        state_db._find_workset.return_value = ws

        should_retry = state_db.record_failure(
            workset_id="data-error-ws-001",
            error_details=error_text,
            error_category=ErrorCategory.PERMANENT,
        )

        assert should_retry is False


class TestConcurrentProcessingIntegration:
    def test_state_db_concurrency_limits(self, state_db: WorksetStateDB):
        state_db.list_worksets_by_state = MagicMock(return_value=[{"workset_id": f"ws-{i}"} for i in range(5)])
        state_db.list_locked_worksets = MagicMock(return_value=[{"workset_id": f"l-{i}"} for i in range(5)])

        can_start = state_db.can_start_new_workset(max_concurrent=5)
        assert can_start is False

    def test_state_db_cluster_affinity(self, state_db: WorksetStateDB):
        ws = _Instance(name="test-ws-001", bstatus="ready", json_addl={"workset_id": "test-ws-001", "state": "ready"})
        state_db._find_workset.return_value = ws

        assert state_db.set_cluster_affinity("test-ws-001", "test-cluster") is True
        assert ws.json_addl["preferred_cluster"] == "test-cluster"
