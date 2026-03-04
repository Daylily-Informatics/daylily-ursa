"""TapDB graph-backed state management for Ursa worksets."""

from __future__ import annotations

import datetime as dt
import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy import and_

from daylib.config import normalize_bucket_name
from daylib.tapdb_graph import TapDBBackend, from_json_addl, utc_now_iso
from daylily_tapdb import generic_instance

LOGGER = logging.getLogger("daylily.workset_state_db")


class WorksetState(str, Enum):
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    ERROR = "error"
    IGNORED = "ignored"
    RETRYING = "retrying"
    FAILED = "failed"
    CANCELED = "canceled"
    ARCHIVED = "archived"
    DELETED = "deleted"


class WorksetProgressStep(str, Enum):
    STAGING = "staging"
    CLUSTER_PROVISIONING = "cluster_provisioning"
    CLUSTER_READY = "cluster_ready"
    PIPELINE_STARTING = "pipeline_starting"
    PIPELINE_RUNNING = "pipeline_running"
    PIPELINE_COMPLETE = "pipeline_complete"
    EXPORTING = "exporting"
    EXPORT_COMPLETE = "export_complete"
    CLEANUP_HEADNODE = "cleanup_headnode"
    CLEANUP_COMPLETE = "cleanup_complete"
    COLLECTING_METRICS = "collecting_metrics"
    METRICS_COMPLETE = "metrics_complete"
    FINALIZING = "finalizing"
    STAGING_FAILED = "staging_failed"
    CLUSTER_FAILED = "cluster_failed"
    PIPELINE_FAILED = "pipeline_failed"
    METRICS_FAILED = "metrics_failed"
    EXPORT_FAILED = "export_failed"
    CLEANUP_FAILED = "cleanup_failed"


class WorksetPriority(str, Enum):
    URGENT = "urgent"
    NORMAL = "normal"
    LOW = "low"


class WorksetType(str, Enum):
    CLINICAL = "clinical"
    RUO = "ruo"
    LSMC = "lsmc"


class ErrorCategory(str, Enum):
    TRANSIENT = "transient"
    RESOURCE = "resource"
    CONFIGURATION = "configuration"
    DATA = "data"
    PERMANENT = "permanent"


STATE_PRIORITY_ORDER = {
    WorksetState.ERROR: 0,
    WorksetState.RETRYING: 1,
    WorksetState.IN_PROGRESS: 2,
    WorksetState.READY: 3,
    WorksetState.COMPLETE: 4,
    WorksetState.FAILED: 5,
    WorksetState.IGNORED: 6,
}

EXECUTION_PRIORITY_ORDER = {
    WorksetPriority.URGENT: 0,
    WorksetPriority.NORMAL: 1,
    WorksetPriority.LOW: 2,
}

DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_BASE = 2
DEFAULT_RETRY_BACKOFF_MAX = 3600


class _NoopMetrics:
    def put_metric_data(self, **_: Any) -> None:
        return None


class WorksetStateDB:
    """TapDB-backed workset state manager with transactional lock semantics."""

    WORKSET_TEMPLATE = "workflow/workset/analysis/1.0/"
    CUSTOMER_TEMPLATE = "actor/customer/account/1.0/"
    STATE_EVENT_TEMPLATE = "action/workset/state-transition/1.0/"
    LOCK_EVENT_TEMPLATE = "action/workset/lock-event/1.0/"

    def __init__(
        self,
        *,
        lock_timeout_seconds: int = 3600,
    ):
        self.lock_timeout_seconds = lock_timeout_seconds
        self.backend = TapDBBackend(app_username="ursa")
        self.cloudwatch = _NoopMetrics()
        self._cloudwatch = None

    def bootstrap(self) -> None:
        with self.backend.session_scope(commit=True) as session:
            self.backend.ensure_templates(session)

    @staticmethod
    def _validate_customer_id(customer_id: Optional[str]) -> str:
        if customer_id is None:
            raise ValueError("customer_id is required and cannot be None")
        if not isinstance(customer_id, str):
            raise ValueError(f"customer_id must be a string, got {type(customer_id).__name__}")
        stripped = customer_id.strip()
        if not stripped:
            raise ValueError("customer_id cannot be empty")
        if stripped.lower() == "unknown":
            raise ValueError("customer_id cannot be 'Unknown'")
        return stripped

    @staticmethod
    def _validate_samples(metadata: Optional[Dict[str, Any]]) -> int:
        if not metadata:
            raise ValueError("Workset must have samples - no metadata provided")

        samples = metadata.get("samples", [])
        raw_sample_count = metadata.get("sample_count")
        tsv_content = metadata.get("stage_samples_tsv", "")

        if samples and len(samples) > 0:
            return len(samples)
        if raw_sample_count is not None:
            try:
                sample_count = int(raw_sample_count)
            except (TypeError, ValueError):
                sample_count = 0
            if sample_count > 0:
                return sample_count
        if tsv_content:
            lines = [line for line in tsv_content.strip().split("\n") if line.strip() and not line.startswith("#")]
            if lines and "\t" in lines[0]:
                return max(0, len(lines) - 1)

        raise ValueError("Workset must have at least one sample")

    def _workset_template_uuid(self, session) -> int:
        template = self.backend.templates.get_template(session, self.WORKSET_TEMPLATE)
        if template is None:
            self.backend.ensure_templates(session)
            template = self.backend.templates.get_template(session, self.WORKSET_TEMPLATE)
        if template is None:
            raise RuntimeError("Missing workset template")
        return int(template.uuid)

    def _event_template_uuid(self, session, template_code: str) -> int:
        template = self.backend.templates.get_template(session, template_code)
        if template is None:
            self.backend.ensure_templates(session)
            template = self.backend.templates.get_template(session, template_code)
        if template is None:
            raise RuntimeError(f"Missing event template: {template_code}")
        return int(template.uuid)

    def _find_workset(self, session, workset_id: str, *, for_update: bool = False) -> Optional[generic_instance]:
        query = session.query(generic_instance).filter(
            generic_instance.template_uuid == self._workset_template_uuid(session),
            generic_instance.is_deleted.is_(False),
            generic_instance.json_addl["workset_id"].as_string() == workset_id,
        )
        if for_update:
            query = query.with_for_update()
        return query.first()

    def _find_customer(self, session, customer_id: str) -> Optional[generic_instance]:
        template = self.backend.templates.get_template(session, self.CUSTOMER_TEMPLATE)
        if template is None:
            return None
        return (
            session.query(generic_instance)
            .filter(
                generic_instance.template_uuid == template.uuid,
                generic_instance.is_deleted.is_(False),
                generic_instance.json_addl["customer_id"].as_string() == customer_id,
            )
            .first()
        )

    def _to_dict(self, instance: generic_instance) -> Dict[str, Any]:
        result = from_json_addl(instance)
        if "state" not in result:
            result["state"] = instance.bstatus
        metadata = result.get("metadata", {})
        if isinstance(metadata, dict):
            if "sample_count" not in result and "sample_count" in metadata:
                result["sample_count"] = metadata["sample_count"]
            if "pipeline_type" not in result and "pipeline_type" in metadata:
                result["pipeline_type"] = metadata["pipeline_type"]
        return result

    @staticmethod
    def _coerce_state(state: WorksetState | str) -> WorksetState:
        if isinstance(state, WorksetState):
            return state
        return WorksetState(state)

    @staticmethod
    def _as_iso(value: Any) -> str:
        if isinstance(value, dt.datetime):
            return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")
        if isinstance(value, str):
            return value
        return utc_now_iso()

    def _with_history(self, workset: Dict[str, Any], entry: Dict[str, Any]) -> Dict[str, Any]:
        history = list(workset.get("state_history") or [])
        history.append(entry)
        workset["state_history"] = history
        return workset

    def _write_lock_event(
        self,
        session,
        *,
        workset: generic_instance,
        owner_id: str,
        action: str,
        expires_at: Optional[str] = None,
    ) -> None:
        event_payload: Dict[str, Any] = {
            "workset_id": (workset.json_addl or {}).get("workset_id", workset.name),
            "lock_action": action,
            "owner": owner_id,
            "timestamp": utc_now_iso(),
        }
        if expires_at:
            event_payload["expires_at"] = expires_at
        event = self.backend.create_instance(
            session,
            self.LOCK_EVENT_TEMPLATE,
            f"lock:{event_payload['workset_id']}:{event_payload['timestamp']}",
            json_addl=event_payload,
            bstatus="active",
        )
        self.backend.create_lineage(
            session,
            parent=workset,
            child=event,
            relationship_type="lock_event",
        )

    def _write_state_event(
        self,
        session,
        *,
        workset: generic_instance,
        new_state: WorksetState,
        reason: str,
        progress_step: Optional[WorksetProgressStep] = None,
        error_details: Optional[str] = None,
    ) -> None:
        event_payload: Dict[str, Any] = {
            "workset_id": (workset.json_addl or {}).get("workset_id", workset.name),
            "state": new_state.value,
            "timestamp": utc_now_iso(),
            "reason": reason,
        }
        if progress_step is not None:
            event_payload["progress_step"] = progress_step.value
        if error_details:
            event_payload["error_details"] = error_details

        event = self.backend.create_instance(
            session,
            self.STATE_EVENT_TEMPLATE,
            f"state:{event_payload['workset_id']}:{event_payload['timestamp']}",
            json_addl=event_payload,
            bstatus=new_state.value,
        )
        self.backend.create_lineage(
            session,
            parent=workset,
            child=event,
            relationship_type="state_transition",
        )

    def _emit_metric(self, metric_name: str, value: float) -> None:
        try:
            self.cloudwatch.put_metric_data(
                Namespace="Daylily/WorksetMonitor",
                MetricData=[
                    {
                        "MetricName": metric_name,
                        "Value": value,
                        "Unit": "Count",
                        "Timestamp": dt.datetime.now(dt.timezone.utc),
                    }
                ],
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("Metric emission skipped for %s: %s", metric_name, exc)

    def register_workset(
        self,
        workset_id: str,
        bucket: str,
        prefix: str,
        priority: WorksetPriority = WorksetPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
        customer_id: Optional[str] = None,
        skip_validation: bool = False,
        workset_type: Optional[WorksetType] = None,
        preferred_cluster: Optional[str] = None,
        cluster_region: Optional[str] = None,
    ) -> bool:
        if not skip_validation:
            customer_id = self._validate_customer_id(customer_id)
            self._validate_samples(metadata)

        if workset_type is None:
            workset_type = WorksetType.RUO

        normalized_bucket = normalize_bucket_name(bucket)
        if not normalized_bucket:
            raise ValueError(f"Invalid bucket name: {bucket}")

        now = utc_now_iso()
        payload: Dict[str, Any] = {
            "workset_id": workset_id,
            "state": WorksetState.READY.value,
            "priority": priority.value,
            "workset_type": workset_type.value,
            "bucket": normalized_bucket,
            "prefix": prefix,
            "created_at": now,
            "updated_at": now,
            "state_history": [
                {
                    "state": WorksetState.READY.value,
                    "timestamp": now,
                    "reason": "Initial registration",
                }
            ],
            "retry_count": 0,
            "max_retries": DEFAULT_MAX_RETRIES,
        }
        if customer_id:
            payload["customer_id"] = customer_id
        if preferred_cluster:
            payload["preferred_cluster"] = preferred_cluster
            payload["affinity_reason"] = "user_selected"
            if cluster_region:
                payload["cluster_region"] = cluster_region
        if metadata:
            payload["metadata"] = self._serialize_metadata(metadata)

        with self.backend.session_scope(commit=True) as session:
            self.backend.ensure_templates(session)
            if self._find_workset(session, workset_id) is not None:
                LOGGER.warning("Workset %s already exists", workset_id)
                return False

            ws = self.backend.create_instance(
                session,
                self.WORKSET_TEMPLATE,
                workset_id,
                json_addl=payload,
                bstatus=WorksetState.READY.value,
            )

            if customer_id:
                customer = self._find_customer(session, customer_id)
                if customer is not None:
                    self.backend.create_lineage(
                        session,
                        parent=customer,
                        child=ws,
                        relationship_type="owns",
                    )

            self._write_state_event(
                session,
                workset=ws,
                new_state=WorksetState.READY,
                reason="Initial registration",
            )

        self._emit_metric("WorksetRegistered", 1.0)
        LOGGER.info("Registered workset %s (priority=%s)", workset_id, priority.value)
        return True

    def acquire_lock(
        self,
        workset_id: str,
        owner_id: str,
        force: bool = False,
    ) -> bool:
        now = dt.datetime.now(dt.timezone.utc)
        now_iso = now.isoformat().replace("+00:00", "Z")
        expires_at = (now + dt.timedelta(seconds=self.lock_timeout_seconds)).isoformat().replace("+00:00", "Z")

        with self.backend.session_scope(commit=True) as session:
            ws = self._find_workset(session, workset_id, for_update=True)
            if ws is None:
                LOGGER.warning("Workset %s not found", workset_id)
                return False

            data = dict(ws.json_addl or {})
            current_state = str(data.get("state", ws.bstatus))
            lockable_states = {WorksetState.READY.value, WorksetState.RETRYING.value}
            if not force and current_state not in lockable_states:
                return False

            lock_owner = data.get("lock_owner")
            lock_expires_at = data.get("lock_expires_at")
            lock_is_stale = True
            if lock_owner:
                lock_is_stale = False
                if lock_expires_at:
                    try:
                        expires_time = dt.datetime.fromisoformat(str(lock_expires_at).rstrip("Z")).replace(tzinfo=dt.timezone.utc)
                        lock_is_stale = now >= expires_time
                    except ValueError:
                        lock_is_stale = True
                if lock_owner != owner_id and not (force or lock_is_stale):
                    return False

            data["lock_owner"] = owner_id
            data["lock_acquired_at"] = now_iso
            data["lock_expires_at"] = expires_at
            data["lock"] = {
                "owner": owner_id,
                "acquired_at": now_iso,
                "expires_at": expires_at,
                "epoch": int(data.get("lock", {}).get("epoch", 0)) + 1,
            }
            data["updated_at"] = now_iso
            ws.json_addl = data
            session.flush()

            self._write_lock_event(
                session,
                workset=ws,
                owner_id=owner_id,
                action="acquire",
                expires_at=expires_at,
            )

        self._emit_metric("LockAcquired", 1.0)
        return True

    def release_lock(self, workset_id: str, owner_id: str) -> bool:
        with self.backend.session_scope(commit=True) as session:
            ws = self._find_workset(session, workset_id, for_update=True)
            if ws is None:
                return False
            data = dict(ws.json_addl or {})
            if data.get("lock_owner") != owner_id:
                return False

            data.pop("lock_owner", None)
            data.pop("lock_acquired_at", None)
            data.pop("lock_expires_at", None)
            lock_payload = dict(data.get("lock") or {})
            lock_payload["released_at"] = utc_now_iso()
            data["lock"] = lock_payload
            data["updated_at"] = utc_now_iso()
            ws.json_addl = data
            session.flush()

            self._write_lock_event(
                session,
                workset=ws,
                owner_id=owner_id,
                action="release",
            )

        self._emit_metric("LockReleased", 1.0)
        return True

    def refresh_lock(self, workset_id: str, owner_id: str) -> bool:
        now = dt.datetime.now(dt.timezone.utc)
        now_iso = now.isoformat().replace("+00:00", "Z")
        expires_at = (now + dt.timedelta(seconds=self.lock_timeout_seconds)).isoformat().replace("+00:00", "Z")

        with self.backend.session_scope(commit=True) as session:
            ws = self._find_workset(session, workset_id, for_update=True)
            if ws is None:
                return False
            data = dict(ws.json_addl or {})
            if data.get("lock_owner") != owner_id:
                return False
            data["lock_acquired_at"] = now_iso
            data["lock_expires_at"] = expires_at
            lock_payload = dict(data.get("lock") or {})
            lock_payload.update({
                "owner": owner_id,
                "acquired_at": now_iso,
                "expires_at": expires_at,
                "epoch": int(lock_payload.get("epoch", 0)) + 1,
            })
            data["lock"] = lock_payload
            data["updated_at"] = now_iso
            ws.json_addl = data
            session.flush()
        return True

    def update_state(
        self,
        workset_id: str,
        new_state: WorksetState,
        reason: str,
        error_details: Optional[str] = None,
        cluster_name: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        progress_step: Optional[WorksetProgressStep] = None,
    ) -> None:
        now_iso = utc_now_iso()
        with self.backend.session_scope(commit=True) as session:
            ws = self._find_workset(session, workset_id, for_update=True)
            if ws is None:
                raise KeyError(f"workset not found: {workset_id}")

            data = dict(ws.json_addl or {})
            data["state"] = new_state.value
            data["updated_at"] = now_iso
            if error_details:
                data["error_details"] = error_details
            if cluster_name:
                data["cluster_name"] = cluster_name
            if metrics:
                data["metrics"] = self._serialize_metadata(metrics)
            if progress_step:
                data["progress_step"] = progress_step.value
            history_entry: Dict[str, Any] = {
                "state": new_state.value,
                "timestamp": now_iso,
                "reason": reason,
            }
            if progress_step:
                history_entry["progress_step"] = progress_step.value
            data = self._with_history(data, history_entry)

            ws.bstatus = new_state.value
            ws.json_addl = data
            session.flush()

            self._write_state_event(
                session,
                workset=ws,
                new_state=new_state,
                reason=reason,
                progress_step=progress_step,
                error_details=error_details,
            )

        self._emit_metric(f"WorksetState{new_state.value.title()}", 1.0)

    def update_progress_step(
        self,
        workset_id: str,
        progress_step: WorksetProgressStep,
        message: Optional[str] = None,
    ) -> None:
        with self.backend.session_scope(commit=True) as session:
            ws = self._find_workset(session, workset_id, for_update=True)
            if ws is None:
                return
            data = dict(ws.json_addl or {})
            data["progress_step"] = progress_step.value
            if message:
                data["progress_message"] = message
            data["updated_at"] = utc_now_iso()
            ws.json_addl = data
            session.flush()

    def update_progress(
        self,
        workset_id: str,
        current_step: Optional[str] = None,
        cluster_name: Optional[str] = None,
        started_at: Optional[str] = None,
        finished_at: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self.backend.session_scope(commit=True) as session:
            ws = self._find_workset(session, workset_id, for_update=True)
            if ws is None:
                return
            data = dict(ws.json_addl or {})
            data["updated_at"] = utc_now_iso()
            if current_step is not None:
                data["progress_step"] = current_step
            if cluster_name is not None:
                data["cluster_name"] = cluster_name
            if started_at is not None:
                data["started_at"] = started_at
            if finished_at is not None:
                data["finished_at"] = finished_at
            if metrics is not None:
                data["metrics"] = self._serialize_metadata(metrics)
            ws.json_addl = data
            session.flush()

    def update_metadata(
        self,
        workset_id: str,
        metadata_updates: Dict[str, Any],
    ) -> bool:
        with self.backend.session_scope(commit=True) as session:
            ws = self._find_workset(session, workset_id, for_update=True)
            if ws is None:
                return False
            data = dict(ws.json_addl or {})
            current_metadata = data.get("metadata")
            if not isinstance(current_metadata, dict):
                current_metadata = {}
            merged_metadata = {**current_metadata, **metadata_updates}
            data["metadata"] = self._serialize_metadata(merged_metadata)
            data["updated_at"] = utc_now_iso()
            ws.json_addl = data
            session.flush()
            return True

    def update_execution_environment(
        self,
        workset_id: str,
        cluster_name: Optional[str] = None,
        cluster_region: Optional[str] = None,
        headnode_ip: Optional[str] = None,
        headnode_analysis_path: Optional[str] = None,
        execution_s3_bucket: Optional[str] = None,
        execution_s3_prefix: Optional[str] = None,
        execution_started_at: Optional[str] = None,
        execution_ended_at: Optional[str] = None,
        results_s3_uri: Optional[str] = None,
    ) -> None:
        with self.backend.session_scope(commit=True) as session:
            ws = self._find_workset(session, workset_id, for_update=True)
            if ws is None:
                return
            data = dict(ws.json_addl or {})
            data["updated_at"] = utc_now_iso()
            if cluster_name is not None:
                data["execution_cluster_name"] = cluster_name
            if cluster_region is not None:
                data["execution_cluster_region"] = cluster_region
            if headnode_ip is not None:
                data["execution_headnode_ip"] = headnode_ip
            if headnode_analysis_path is not None:
                data["headnode_analysis_path"] = headnode_analysis_path
            if execution_s3_bucket is not None:
                data["execution_s3_bucket"] = execution_s3_bucket
            if execution_s3_prefix is not None:
                data["execution_s3_prefix"] = execution_s3_prefix
            if execution_started_at is not None:
                data["execution_started_at"] = execution_started_at
            if execution_ended_at is not None:
                data["execution_ended_at"] = execution_ended_at
            if results_s3_uri is not None:
                data["results_s3_uri"] = results_s3_uri
            ws.json_addl = data
            session.flush()

    def get_execution_environment(self, workset_id: str) -> Optional[Dict[str, Any]]:
        record = self.get_workset(workset_id)
        if record is None:
            return None
        return {
            "execution_cluster_name": record.get("execution_cluster_name"),
            "execution_cluster_region": record.get("execution_cluster_region"),
            "execution_headnode_ip": record.get("execution_headnode_ip"),
            "headnode_analysis_path": record.get("headnode_analysis_path"),
            "execution_s3_bucket": record.get("execution_s3_bucket"),
            "execution_s3_prefix": record.get("execution_s3_prefix"),
            "execution_started_at": record.get("execution_started_at"),
            "execution_ended_at": record.get("execution_ended_at"),
            "results_s3_uri": record.get("results_s3_uri"),
        }

    def set_execution_context(
        self,
        workset_id: str,
        execution_context: Dict[str, Any],
    ) -> bool:
        with self.backend.session_scope(commit=True) as session:
            ws = self._find_workset(session, workset_id, for_update=True)
            if ws is None:
                return False
            data = dict(ws.json_addl or {})
            data["updated_at"] = utc_now_iso()
            data["execution_context"] = self._serialize_metadata(execution_context)
            ws.json_addl = data
            session.flush()
            return True

    def get_execution_context(self, workset_id: str) -> Optional[Dict[str, Any]]:
        workset = self.get_workset(workset_id)
        if not workset:
            return None
        value = workset.get("execution_context")
        if isinstance(value, dict):
            return value
        return None

    def get_workset(self, workset_id: str) -> Optional[Dict[str, Any]]:
        with self.backend.session_scope() as session:
            ws = self._find_workset(session, workset_id)
            if ws is None:
                return None
            return self._deserialize_item(self._to_dict(ws))

    def update_performance_metrics(
        self,
        workset_id: str,
        performance_metrics: Dict[str, Any],
        is_final: bool = False,
    ) -> bool:
        with self.backend.session_scope(commit=True) as session:
            ws = self._find_workset(session, workset_id, for_update=True)
            if ws is None:
                return False
            data = dict(ws.json_addl or {})
            data["updated_at"] = utc_now_iso()
            data["performance_metrics"] = self._serialize_metadata(performance_metrics)
            data["performance_metrics_final"] = bool(is_final)
            ws.json_addl = data
            session.flush()
            return True

    def get_performance_metrics(self, workset_id: str) -> Optional[Dict[str, Any]]:
        workset = self.get_workset(workset_id)
        if workset is None:
            return None
        result: Dict[str, Any] = {
            "is_final": bool(workset.get("performance_metrics_final", False)),
        }
        if "performance_metrics" in workset:
            result["metrics"] = workset["performance_metrics"]
        return result

    def update_cost_report(
        self,
        workset_id: str,
        total_compute_cost_usd: float,
        per_sample_costs: Optional[Dict[str, float]] = None,
        rule_count: Optional[int] = None,
        sample_count: Optional[int] = None,
    ) -> bool:
        with self.backend.session_scope(commit=True) as session:
            ws = self._find_workset(session, workset_id, for_update=True)
            if ws is None:
                return False
            data = dict(ws.json_addl or {})
            now_iso = utc_now_iso()
            data["updated_at"] = now_iso
            data["total_compute_cost_usd"] = float(total_compute_cost_usd)
            data["cost_report_parsed_at"] = now_iso
            if per_sample_costs is not None:
                data["per_sample_costs"] = {k: float(v) for k, v in per_sample_costs.items()}
            if rule_count is not None:
                data["cost_report_rule_count"] = int(rule_count)
            if sample_count is not None:
                data["cost_report_sample_count"] = int(sample_count)
            ws.json_addl = data
            session.flush()
            return True

    def get_cost_report(self, workset_id: str) -> Optional[Dict[str, Any]]:
        workset = self.get_workset(workset_id)
        if workset is None:
            return None
        keys = {
            "total_compute_cost_usd",
            "per_sample_costs",
            "cost_report_parsed_at",
            "cost_report_rule_count",
            "cost_report_sample_count",
        }
        result = {k: workset[k] for k in keys if k in workset}
        return result or None

    def update_storage_metrics(
        self,
        workset_id: str,
        results_storage_bytes: int,
        fsx_storage_bytes: Optional[int] = None,
    ) -> bool:
        with self.backend.session_scope(commit=True) as session:
            ws = self._find_workset(session, workset_id, for_update=True)
            if ws is None:
                return False
            data = dict(ws.json_addl or {})
            now_iso = utc_now_iso()
            data["updated_at"] = now_iso
            data["results_storage_bytes"] = int(results_storage_bytes)
            data["storage_calculated_at"] = now_iso
            if fsx_storage_bytes is not None:
                data["fsx_storage_bytes"] = int(fsx_storage_bytes)
            ws.json_addl = data
            session.flush()
            return True

    def get_storage_metrics(self, workset_id: str) -> Optional[Dict[str, Any]]:
        workset = self.get_workset(workset_id)
        if workset is None:
            return None
        keys = {"results_storage_bytes", "fsx_storage_bytes", "storage_calculated_at"}
        result = {k: workset[k] for k in keys if k in workset}
        return result or None

    def _query_worksets(self, session, *, state: Optional[WorksetState] = None, limit: int = 100) -> List[generic_instance]:
        query = session.query(generic_instance).filter(
            generic_instance.template_uuid == self._workset_template_uuid(session),
            generic_instance.is_deleted.is_(False),
        )
        if state is not None:
            query = query.filter(generic_instance.bstatus == state.value)
        return query.order_by(generic_instance.created_dt.desc()).limit(limit).all()

    def list_worksets_by_state(
        self,
        state: WorksetState,
        priority: Optional[WorksetPriority] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        state = self._coerce_state(state)
        with self.backend.session_scope() as session:
            rows = self._query_worksets(session, state=state, limit=max(limit * 3, limit))
            out: List[Dict[str, Any]] = []
            for row in rows:
                payload = self._deserialize_item(self._to_dict(row))
                if priority is not None and payload.get("priority") != priority.value:
                    continue
                out.append(payload)
                if len(out) >= limit:
                    break
            return out

    def list_worksets_by_customer(
        self,
        customer_id: str,
        state: Optional[WorksetState] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        if not customer_id:
            return []
        with self.backend.session_scope() as session:
            query = session.query(generic_instance).filter(
                generic_instance.template_uuid == self._workset_template_uuid(session),
                generic_instance.is_deleted.is_(False),
                generic_instance.json_addl["customer_id"].as_string() == customer_id,
            )
            if state is not None:
                state = self._coerce_state(state)
                query = query.filter(generic_instance.bstatus == state.value)
            rows = query.order_by(generic_instance.created_dt.desc()).limit(limit).all()
            return [self._deserialize_item(self._to_dict(row)) for row in rows]

    def _list_all_worksets(self, limit: int = 1000) -> List[Dict[str, Any]]:
        with self.backend.session_scope() as session:
            rows = self._query_worksets(session, state=None, limit=limit)
            return [self._deserialize_item(self._to_dict(row)) for row in rows]

    def list_locked_worksets(self, limit: int = 100) -> List[Dict[str, Any]]:
        rows = self._list_all_worksets(limit=max(limit * 5, limit))
        locked = [row for row in rows if row.get("lock_owner")]
        return locked[:limit]

    def get_ready_worksets_prioritized(self, limit: int = 100) -> List[Dict[str, Any]]:
        worksets: List[Dict[str, Any]] = []
        for priority in [WorksetPriority.URGENT, WorksetPriority.NORMAL, WorksetPriority.LOW]:
            batch = self.list_worksets_by_state(
                WorksetState.READY,
                priority=priority,
                limit=limit - len(worksets),
            )
            worksets.extend(batch)
            if len(worksets) >= limit:
                break
        return worksets

    def get_actionable_worksets_prioritized(self, limit: int = 100) -> List[Dict[str, Any]]:
        worksets: List[Dict[str, Any]] = []
        in_progress = self.list_worksets_by_state(WorksetState.IN_PROGRESS, limit=limit)
        worksets.extend(in_progress)
        if len(worksets) < limit:
            for priority in [WorksetPriority.URGENT, WorksetPriority.NORMAL, WorksetPriority.LOW]:
                batch = self.list_worksets_by_state(
                    WorksetState.READY,
                    priority=priority,
                    limit=limit - len(worksets),
                )
                worksets.extend(batch)
                if len(worksets) >= limit:
                    break
        return worksets

    def get_queue_depth(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for ws_state in WorksetState:
            count = len(self.list_worksets_by_state(ws_state, limit=2000))
            counts[ws_state.value] = count
            self._emit_metric(f"QueueDepth{ws_state.value.title()}", float(count))
        return counts

    def _serialize_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        def convert(obj: Any) -> Any:
            if isinstance(obj, dt.datetime):
                return obj.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        return convert(data)

    def _deserialize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        def convert(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        result: Dict[str, Any] = convert(item)
        metadata = result.get("metadata", {})
        if isinstance(metadata, dict):
            if "sample_count" not in result and "sample_count" in metadata:
                result["sample_count"] = metadata["sample_count"]
            if "pipeline_type" not in result and "pipeline_type" in metadata:
                result["pipeline_type"] = metadata["pipeline_type"]
        return result

    def record_failure(
        self,
        workset_id: str,
        error_details: str,
        error_category: ErrorCategory = ErrorCategory.TRANSIENT,
        failed_step: Optional[str] = None,
    ) -> bool:
        workset = self.get_workset(workset_id)
        if not workset:
            return False

        retry_count = int(workset.get("retry_count", 0))
        max_retries = int(workset.get("max_retries", DEFAULT_MAX_RETRIES))

        should_retry = (
            retry_count < max_retries
            and error_category in [ErrorCategory.TRANSIENT, ErrorCategory.RESOURCE]
        )

        if should_retry:
            backoff_seconds = min(DEFAULT_RETRY_BACKOFF_BASE ** retry_count, DEFAULT_RETRY_BACKOFF_MAX)
            retry_after = (
                dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=backoff_seconds)
            ).isoformat().replace("+00:00", "Z")
            new_state = WorksetState.RETRYING
            reason = f"Retry {retry_count + 1}/{max_retries} after {error_category.value} error"
        else:
            retry_after = None
            new_state = WorksetState.FAILED
            reason = f"Permanent failure after {retry_count} retries: {error_category.value}"

        with self.backend.session_scope(commit=True) as session:
            ws = self._find_workset(session, workset_id, for_update=True)
            if ws is None:
                return False
            data = dict(ws.json_addl or {})
            data["state"] = new_state.value
            data["updated_at"] = utc_now_iso()
            data["retry_count"] = retry_count + 1
            data["error_details"] = error_details
            data["error_category"] = error_category.value
            data["failed_step"] = failed_step or "unknown"
            data["retry_after"] = retry_after
            data = self._with_history(
                data,
                {
                    "state": new_state.value,
                    "timestamp": utc_now_iso(),
                    "reason": reason,
                    "error_category": error_category.value,
                },
            )
            ws.bstatus = new_state.value
            ws.json_addl = data
            session.flush()
            self._write_state_event(
                session,
                workset=ws,
                new_state=new_state,
                reason=reason,
                error_details=error_details,
            )

        self._emit_metric("WorksetRetry" if should_retry else "WorksetPermanentFailure", 1.0)
        return should_retry

    def get_retryable_worksets(self) -> List[Dict[str, Any]]:
        worksets = self.list_worksets_by_state(WorksetState.RETRYING, limit=1000)
        now = utc_now_iso()
        retryable = []
        for workset in worksets:
            retry_after = workset.get("retry_after")
            if not retry_after or str(retry_after) <= now:
                retryable.append(workset)
        return retryable

    def reset_for_retry(self, workset_id: str) -> bool:
        try:
            self.update_state(workset_id, WorksetState.READY, reason="Reset for retry attempt")
            return True
        except Exception:
            return False

    def set_cluster_affinity(
        self,
        workset_id: str,
        cluster_name: str,
        affinity_reason: str = "manual",
    ) -> bool:
        with self.backend.session_scope(commit=True) as session:
            ws = self._find_workset(session, workset_id, for_update=True)
            if ws is None:
                return False
            data = dict(ws.json_addl or {})
            data["preferred_cluster"] = cluster_name
            data["affinity_reason"] = affinity_reason
            data["updated_at"] = utc_now_iso()
            ws.json_addl = data
            session.flush()
            return True

    def get_worksets_by_cluster(self, cluster_name: str) -> List[Dict[str, Any]]:
        rows = self._list_all_worksets(limit=5000)
        return [row for row in rows if row.get("cluster_name") == cluster_name]

    def get_concurrent_worksets_count(self) -> int:
        in_progress = len(self.list_worksets_by_state(WorksetState.IN_PROGRESS, limit=2000))
        locked = len(self.list_locked_worksets(limit=2000))
        return in_progress + locked

    def can_start_new_workset(self, max_concurrent: int) -> bool:
        return self.get_concurrent_worksets_count() < max_concurrent

    def get_next_workset_with_affinity(
        self,
        cluster_name: str,
        priority: Optional[WorksetPriority] = None,
    ) -> Optional[Dict[str, Any]]:
        ready_worksets = self.list_worksets_by_state(WorksetState.READY, limit=100)
        for workset in ready_worksets:
            if workset.get("preferred_cluster") == cluster_name:
                if priority is None or workset.get("priority") == priority.value:
                    return workset
        if ready_worksets:
            return ready_worksets[0]
        return None

    def archive_workset(
        self,
        workset_id: str,
        archived_by: str = "system",
        archive_reason: Optional[str] = None,
    ) -> bool:
        with self.backend.session_scope(commit=True) as session:
            ws = self._find_workset(session, workset_id, for_update=True)
            if ws is None:
                return False
            data = dict(ws.json_addl or {})
            original_state = data.get("state", ws.bstatus)
            data["state"] = WorksetState.ARCHIVED.value
            data["archived_at"] = utc_now_iso()
            data["archived_by"] = archived_by
            data["original_state"] = original_state
            if archive_reason:
                data["archive_reason"] = archive_reason
            ws.bstatus = WorksetState.ARCHIVED.value
            ws.json_addl = data
            session.flush()
            self._write_state_event(
                session,
                workset=ws,
                new_state=WorksetState.ARCHIVED,
                reason=archive_reason or "Archived",
            )
            return True

    def delete_workset(
        self,
        workset_id: str,
        deleted_by: str = "system",
        delete_reason: Optional[str] = None,
        hard_delete: bool = False,
    ) -> bool:
        with self.backend.session_scope(commit=True) as session:
            ws = self._find_workset(session, workset_id, for_update=True)
            if ws is None:
                return False
            data = dict(ws.json_addl or {})
            if hard_delete:
                ws.is_deleted = True
                ws.bstatus = WorksetState.DELETED.value
                data["state"] = WorksetState.DELETED.value
                data["deleted_at"] = utc_now_iso()
                data["deleted_by"] = deleted_by
                if delete_reason:
                    data["delete_reason"] = delete_reason
                ws.json_addl = data
            else:
                data["state"] = WorksetState.DELETED.value
                data["deleted_at"] = utc_now_iso()
                data["deleted_by"] = deleted_by
                if delete_reason:
                    data["delete_reason"] = delete_reason
                ws.bstatus = WorksetState.DELETED.value
                ws.json_addl = data
                self._write_state_event(
                    session,
                    workset=ws,
                    new_state=WorksetState.DELETED,
                    reason=delete_reason or "Deleted",
                )
            session.flush()
            return True

    def restore_workset(
        self,
        workset_id: str,
        restored_by: str = "system",
    ) -> bool:
        with self.backend.session_scope(commit=True) as session:
            ws = self._find_workset(session, workset_id, for_update=True)
            if ws is None:
                return False
            data = dict(ws.json_addl or {})
            if str(data.get("state", ws.bstatus)) != WorksetState.ARCHIVED.value:
                return False
            data["state"] = WorksetState.READY.value
            data["restored_at"] = utc_now_iso()
            data["restored_by"] = restored_by
            for key in ("archived_at", "archived_by", "archive_reason"):
                data.pop(key, None)
            ws.bstatus = WorksetState.READY.value
            ws.json_addl = data
            session.flush()
            self._write_state_event(
                session,
                workset=ws,
                new_state=WorksetState.READY,
                reason="Restored",
            )
            return True

    def list_archived_worksets(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self.list_worksets_by_state(WorksetState.ARCHIVED, limit=limit)
