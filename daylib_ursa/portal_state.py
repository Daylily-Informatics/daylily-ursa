"""SQLite-backed state for lightweight portal workflows."""

from __future__ import annotations

import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _quantile(values: List[float], fraction: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]

    values = sorted(values)
    index = (len(values) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    if lower == upper:
        return values[lower]
    weight = index - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _sample_count_from_manifest(manifest_tsv_content: Optional[str]) -> int:
    if not manifest_tsv_content:
        return 0
    lines = [line for line in manifest_tsv_content.splitlines() if line.strip()]
    return max(len(lines) - 1, 0)


class PortalState:
    """Persist pricing snapshots and lightweight workset records."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS pricing_runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trigger TEXT NOT NULL,
                    requested_by TEXT,
                    status TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    snapshot_captured_at TEXT
                );

                CREATE TABLE IF NOT EXISTS pricing_points (
                    point_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    captured_at TEXT NOT NULL,
                    region TEXT NOT NULL,
                    availability_zone TEXT NOT NULL,
                    partition TEXT NOT NULL,
                    instance_type TEXT NOT NULL,
                    vcpu_count INTEGER NOT NULL,
                    hourly_spot_price REAL NOT NULL,
                    vcpu_cost_per_hour REAL NOT NULL,
                    cluster_config_path TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES pricing_runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS worksets (
                    workset_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    workset_name TEXT NOT NULL,
                    pipeline_type TEXT NOT NULL,
                    reference_genome TEXT NOT NULL,
                    manifest_id TEXT,
                    manifest_tsv_content TEXT,
                    sample_count INTEGER NOT NULL,
                    priority TEXT NOT NULL,
                    workset_type TEXT NOT NULL,
                    notification_email TEXT,
                    enable_qc INTEGER NOT NULL,
                    archive_results INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    preferred_cluster TEXT,
                    cluster_name TEXT,
                    cluster_region TEXT,
                    target_region TEXT,
                    cluster_create_job_id TEXT,
                    message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def create_pricing_run(self, *, trigger: str, requested_by: Optional[str]) -> Dict[str, Any]:
        created_at = _now_iso()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO pricing_runs (trigger, requested_by, status, created_at)
                VALUES (?, ?, 'queued', ?)
                """,
                (trigger, requested_by, created_at),
            )
            run_id = int(cursor.lastrowid)
        return {
            "run_id": run_id,
            "trigger": trigger,
            "requested_by": requested_by,
            "status": "queued",
            "created_at": created_at,
        }

    def get_active_pricing_run(self) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM pricing_runs
                WHERE status IN ('queued', 'running')
                ORDER BY run_id DESC
                LIMIT 1
                """
            ).fetchone()
        return dict(row) if row else None

    def mark_pricing_run_running(self, run_id: int) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE pricing_runs
                SET status = 'running', started_at = ?
                WHERE run_id = ?
                """,
                (_now_iso(), run_id),
            )

    def mark_pricing_run_failed(self, run_id: int, error: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE pricing_runs
                SET status = 'failed', error = ?, completed_at = ?
                WHERE run_id = ?
                """,
                (error, _now_iso(), run_id),
            )

    def save_pricing_snapshot(self, run_id: int, snapshot: Dict[str, Any]) -> None:
        points = snapshot.get("points", [])
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE pricing_runs
                SET status = 'completed',
                    completed_at = ?,
                    snapshot_captured_at = ?,
                    error = NULL
                WHERE run_id = ?
                """,
                (_now_iso(), snapshot.get("captured_at"), run_id),
            )
            connection.executemany(
                """
                INSERT INTO pricing_points (
                    run_id,
                    captured_at,
                    region,
                    availability_zone,
                    partition,
                    instance_type,
                    vcpu_count,
                    hourly_spot_price,
                    vcpu_cost_per_hour,
                    cluster_config_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        point["captured_at"],
                        point["region"],
                        point["availability_zone"],
                        point["partition"],
                        point["instance_type"],
                        point["vcpu_count"],
                        point["hourly_spot_price"],
                        point["vcpu_cost_per_hour"],
                        snapshot["cluster_config_path"],
                    )
                    for point in points
                ],
            )

    def last_successful_pricing_capture(self) -> Optional[str]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT snapshot_captured_at
                FROM pricing_runs
                WHERE status = 'completed'
                ORDER BY snapshot_captured_at DESC
                LIMIT 1
                """
            ).fetchone()
        return str(row["snapshot_captured_at"]) if row and row["snapshot_captured_at"] else None

    def list_pricing_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM pricing_runs
                ORDER BY run_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_pricing_snapshot_payload(
        self,
        *,
        region: Optional[str] = None,
        partitions: Optional[Iterable[str]] = None,
        from_ts: Optional[str] = None,
        to_ts: Optional[str] = None,
    ) -> Dict[str, Any]:
        clauses = ["1=1"]
        params: List[Any] = []
        if region:
            clauses.append("region = ?")
            params.append(region)
        if from_ts:
            clauses.append("captured_at >= ?")
            params.append(from_ts)
        if to_ts:
            clauses.append("captured_at <= ?")
            params.append(to_ts)
        requested_partitions = [partition for partition in (partitions or []) if partition]
        if requested_partitions:
            placeholders = ", ".join("?" for _ in requested_partitions)
            clauses.append(f"partition IN ({placeholders})")
            params.extend(requested_partitions)

        query = f"""
            SELECT *
            FROM pricing_points
            WHERE {' AND '.join(clauses)}
            ORDER BY captured_at ASC, region ASC, partition ASC, availability_zone ASC, instance_type ASC
        """
        with self._connect() as connection:
            rows = [dict(row) for row in connection.execute(query, params).fetchall()]

        grouped: Dict[tuple[str, str], Dict[str, Any]] = {}
        latest_by_region_partition: Dict[tuple[str, str], Dict[str, Any]] = {}

        for row in rows:
            snapshot_key = (row["captured_at"], row["region"])
            snapshot = grouped.setdefault(
                snapshot_key,
                {
                    "captured_at": row["captured_at"],
                    "region": row["region"],
                    "partitions": [],
                    "_partition_map": {},
                },
            )
            partition_entry = snapshot["_partition_map"].setdefault(
                row["partition"],
                {
                    "partition": row["partition"],
                    "availability_zones": [],
                    "_az_map": {},
                },
            )
            az_entry = partition_entry["_az_map"].setdefault(
                row["availability_zone"],
                {
                    "availability_zone": row["availability_zone"],
                    "points": [],
                },
            )
            az_entry["points"].append(
                {
                    "instance_type": row["instance_type"],
                    "vcpu_count": row["vcpu_count"],
                    "hourly_spot_price": row["hourly_spot_price"],
                    "vcpu_cost_per_hour": row["vcpu_cost_per_hour"],
                }
            )

        snapshots: List[Dict[str, Any]] = []
        for snapshot_key in sorted(grouped):
            snapshot = grouped[snapshot_key]
            for partition_name, partition_entry in sorted(snapshot["_partition_map"].items()):
                zone_entries: List[Dict[str, Any]] = []
                for zone_name, zone_entry in sorted(partition_entry["_az_map"].items()):
                    costs = [point["vcpu_cost_per_hour"] for point in zone_entry["points"]]
                    zone_payload = {
                        "availability_zone": zone_name,
                        "points": zone_entry["points"],
                        "box": {
                            "min": round(min(costs), 8),
                            "q1": round(_quantile(costs, 0.25), 8),
                            "median": round(_quantile(costs, 0.5), 8),
                            "q3": round(_quantile(costs, 0.75), 8),
                            "max": round(max(costs), 8),
                        },
                    }
                    zone_entries.append(zone_payload)

                    latest_key = (snapshot["region"], partition_name)
                    current_latest = latest_by_region_partition.get(latest_key)
                    if (
                        current_latest is None
                        or snapshot["captured_at"] > current_latest["captured_at"]
                        or (
                            snapshot["captured_at"] == current_latest["captured_at"]
                            and zone_payload["box"]["median"] < current_latest["median_vcpu_cost_per_hour"]
                        )
                    ):
                        latest_by_region_partition[latest_key] = {
                            "captured_at": snapshot["captured_at"],
                            "region": snapshot["region"],
                            "partition": partition_name,
                            "availability_zone": zone_name,
                            "median_vcpu_cost_per_hour": zone_payload["box"]["median"],
                        }

                partition_entry["availability_zones"] = zone_entries
                partition_entry.pop("_az_map", None)
                snapshot["partitions"].append(partition_entry)
            snapshot.pop("_partition_map", None)
            snapshots.append(snapshot)

        latest_summary = [
            latest_by_region_partition[key]
            for key in sorted(latest_by_region_partition)
        ]
        return {
            "snapshots": snapshots,
            "latest_cheapest_az": latest_summary,
            "runs": self.list_pricing_runs(limit=5),
        }

    def create_workset(
        self,
        *,
        customer_id: str,
        payload: Dict[str, Any],
        state: str,
        cluster_name: Optional[str],
        cluster_region: Optional[str],
        target_region: Optional[str],
        cluster_create_job_id: Optional[str],
        message: Optional[str],
    ) -> Dict[str, Any]:
        created_at = _now_iso()
        workset_id = f"ws_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{secrets.token_hex(4)}"
        record = {
            "workset_id": workset_id,
            "customer_id": customer_id,
            "workset_name": str(payload.get("workset_name") or "").strip(),
            "pipeline_type": str(payload.get("pipeline_type") or "").strip(),
            "reference_genome": str(payload.get("reference_genome") or "").strip(),
            "manifest_id": payload.get("manifest_id"),
            "manifest_tsv_content": payload.get("manifest_tsv_content"),
            "sample_count": _sample_count_from_manifest(payload.get("manifest_tsv_content")),
            "priority": str(payload.get("priority") or "normal"),
            "workset_type": str(payload.get("workset_type") or "ruo"),
            "notification_email": payload.get("notification_email"),
            "enable_qc": 1 if payload.get("enable_qc", True) else 0,
            "archive_results": 1 if payload.get("archive_results", True) else 0,
            "state": state,
            "preferred_cluster": payload.get("preferred_cluster"),
            "cluster_name": cluster_name,
            "cluster_region": cluster_region,
            "target_region": target_region,
            "cluster_create_job_id": cluster_create_job_id,
            "message": message,
            "created_at": created_at,
            "updated_at": created_at,
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO worksets (
                    workset_id, customer_id, workset_name, pipeline_type, reference_genome,
                    manifest_id, manifest_tsv_content, sample_count, priority, workset_type,
                    notification_email, enable_qc, archive_results, state, preferred_cluster,
                    cluster_name, cluster_region, target_region, cluster_create_job_id, message,
                    created_at, updated_at
                ) VALUES (
                    :workset_id, :customer_id, :workset_name, :pipeline_type, :reference_genome,
                    :manifest_id, :manifest_tsv_content, :sample_count, :priority, :workset_type,
                    :notification_email, :enable_qc, :archive_results, :state, :preferred_cluster,
                    :cluster_name, :cluster_region, :target_region, :cluster_create_job_id, :message,
                    :created_at, :updated_at
                )
                """,
                record,
            )
        return record

    def update_workset_cluster_assignment(
        self,
        *,
        workset_id: str,
        cluster_name: str,
        cluster_region: str,
        state: str = "ready",
        message: Optional[str] = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE worksets
                SET cluster_name = ?,
                    cluster_region = ?,
                    state = ?,
                    message = ?,
                    updated_at = ?
                WHERE workset_id = ?
                """,
                (cluster_name, cluster_region, state, message, _now_iso(), workset_id),
            )

    def list_pending_worksets(self, *, target_region: Optional[str] = None) -> List[Dict[str, Any]]:
        clauses = ["state = 'pending_cluster_creation'"]
        params: List[Any] = []
        if target_region:
            clauses.append("target_region = ?")
            params.append(target_region)
        query = f"SELECT * FROM worksets WHERE {' AND '.join(clauses)} ORDER BY created_at ASC"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def list_worksets(self, customer_id: str, *, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        clauses = ["customer_id = ?"]
        params: List[Any] = [customer_id]
        if status:
            clauses.append("state = ?")
            params.append(status)
        params.append(limit)
        query = f"""
            SELECT *
            FROM worksets
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at DESC
            LIMIT ?
        """
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_workset(self, customer_id: str, workset_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM worksets
                WHERE customer_id = ? AND workset_id = ?
                LIMIT 1
                """,
                (customer_id, workset_id),
            ).fetchone()
        return dict(row) if row else None

    def get_dashboard_stats(self, customer_id: str) -> Dict[str, Any]:
        with self._connect() as connection:
            total_rows = connection.execute(
                """
                SELECT state, COUNT(*) AS count
                FROM worksets
                WHERE customer_id = ?
                GROUP BY state
                """,
                (customer_id,),
            ).fetchall()
        counts = {str(row["state"]): int(row["count"]) for row in total_rows}
        return {
            "in_progress_worksets": counts.get("running", 0),
            "ready_worksets": counts.get("ready", 0) + counts.get("pending_cluster_creation", 0),
            "completed_worksets": counts.get("complete", 0),
            "error_worksets": counts.get("error", 0),
            "cost_this_month": 0.0,
            "compute_cost_usd": 0.0,
            "registered_files": 0,
            "total_file_size_gb": 0.0,
            "storage_cost_usd": 0.0,
            "workset_storage_human": "0 B",
        }

    def get_activity_series(self, customer_id: str, *, days: int = 30) -> Dict[str, Any]:
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=max(days - 1, 0))
        labels: List[str] = []
        submitted: List[int] = []
        completed: List[int] = []
        failed: List[int] = []

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT substr(created_at, 1, 10) AS created_day,
                       COUNT(*) AS submitted_count,
                       SUM(CASE WHEN state = 'complete' THEN 1 ELSE 0 END) AS completed_count,
                       SUM(CASE WHEN state = 'error' THEN 1 ELSE 0 END) AS failed_count
                FROM worksets
                WHERE customer_id = ?
                GROUP BY created_day
                """,
                (customer_id,),
            ).fetchall()
        lookup = {str(row["created_day"]): dict(row) for row in rows}

        for offset in range(days):
            current_day = start_date + timedelta(days=offset)
            key = current_day.isoformat()
            row = lookup.get(key, {})
            labels.append(current_day.strftime("%b %d"))
            submitted.append(int(row.get("submitted_count", 0) or 0))
            completed.append(int(row.get("completed_count", 0) or 0))
            failed.append(int(row.get("failed_count", 0) or 0))

        return {
            "labels": labels,
            "datasets": {
                "submitted": submitted,
                "completed": completed,
                "failed": failed,
            },
        }
