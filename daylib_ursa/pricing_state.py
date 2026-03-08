"""SQLite-backed persistence for Ursa pricing snapshots."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


class PricingState:
    """Persist pricing runs and snapshot points for the admin API."""

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
                """
            )

    def create_pricing_run(self, *, trigger: str, requested_by: str | None) -> dict[str, Any]:
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

    def get_active_pricing_run(self) -> dict[str, Any] | None:
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

    def save_pricing_snapshot(self, run_id: int, snapshot: dict[str, Any]) -> None:
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

    def last_successful_pricing_capture(self) -> str | None:
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

    def list_pricing_runs(self, limit: int = 10) -> list[dict[str, Any]]:
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
        region: str | None = None,
        partitions: Iterable[str] | None = None,
        from_ts: str | None = None,
        to_ts: str | None = None,
    ) -> dict[str, Any]:
        clauses = ["1=1"]
        params: list[Any] = []
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
            SELECT
                captured_at,
                region,
                availability_zone,
                partition,
                instance_type,
                vcpu_count,
                hourly_spot_price,
                vcpu_cost_per_hour,
                cluster_config_path
            FROM pricing_points
            WHERE {' AND '.join(clauses)}
            ORDER BY captured_at DESC, region, partition, availability_zone, instance_type
        """
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()

        snapshots_by_capture: dict[str, dict[str, Any]] = {}
        for row in rows:
            captured_at = str(row["captured_at"])
            snapshot_entry = snapshots_by_capture.setdefault(
                captured_at,
                {
                    "captured_at": captured_at,
                    "region": str(row["region"]),
                    "cluster_config_path": str(row["cluster_config_path"]),
                    "partitions": {},
                },
            )
            partition_key = str(row["partition"])
            partition_entry = snapshot_entry["partitions"].setdefault(
                partition_key,
                {"partition": partition_key, "availability_zones": {}},
            )
            az_key = str(row["availability_zone"])
            az_entry = partition_entry["availability_zones"].setdefault(
                az_key,
                {"availability_zone": az_key, "points": []},
            )
            az_entry["points"].append(
                {
                    "instance_type": str(row["instance_type"]),
                    "vcpu_count": int(row["vcpu_count"]),
                    "hourly_spot_price": float(row["hourly_spot_price"]),
                    "vcpu_cost_per_hour": float(row["vcpu_cost_per_hour"]),
                }
            )

        snapshots: list[dict[str, Any]] = []
        for captured_at, snapshot in snapshots_by_capture.items():
            partitions_payload: list[dict[str, Any]] = []
            for partition_entry in snapshot["partitions"].values():
                az_payloads: list[dict[str, Any]] = []
                for zone_entry in partition_entry["availability_zones"].values():
                    costs = [point["vcpu_cost_per_hour"] for point in zone_entry["points"]]
                    az_payloads.append(
                        {
                            "availability_zone": zone_entry["availability_zone"],
                            "points": zone_entry["points"],
                            "box": {
                                "min": round(min(costs), 8),
                                "q1": round(_quantile(costs, 0.25), 8),
                                "median": round(_quantile(costs, 0.5), 8),
                                "q3": round(_quantile(costs, 0.75), 8),
                                "max": round(max(costs), 8),
                            },
                        }
                    )
                partitions_payload.append(
                    {
                        "partition": partition_entry["partition"],
                        "availability_zones": sorted(
                            az_payloads, key=lambda item: item["availability_zone"]
                        ),
                    }
                )
            snapshots.append(
                {
                    "captured_at": captured_at,
                    "region": snapshot["region"],
                    "cluster_config_path": snapshot["cluster_config_path"],
                    "partitions": sorted(partitions_payload, key=lambda item: item["partition"]),
                }
            )

        return {
            "snapshots": sorted(snapshots, key=lambda item: item["captured_at"], reverse=True),
            "runs": self.list_pricing_runs(limit=5),
        }
