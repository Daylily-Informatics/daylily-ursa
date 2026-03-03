"""Multi-region routing for TapDB-backed workset state operations."""

from __future__ import annotations

import datetime as dt
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from daylib.workset_state_db import WorksetPriority, WorksetState, WorksetStateDB

LOGGER = logging.getLogger("daylily.workset_multi_region")


class RegionStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class RegionHealth:
    region: str
    status: RegionStatus = RegionStatus.UNKNOWN
    latency_ms: float = 0.0
    last_check: Optional[dt.datetime] = None
    consecutive_failures: int = 0
    last_error: Optional[str] = None

    def is_available(self) -> bool:
        return self.status in (RegionStatus.HEALTHY, RegionStatus.DEGRADED)


@dataclass
class MultiRegionConfig:
    primary_region: str
    replica_regions: List[str] = field(default_factory=list)
    table_name: str = "daylily-worksets"
    enable_global_tables: bool = True
    health_check_interval_seconds: int = 30
    failover_threshold: int = 3
    latency_threshold_ms: float = 500.0
    profile: Optional[str] = None


class WorksetMultiRegionDB:
    """Region-aware wrapper around `WorksetStateDB` instances."""

    def __init__(self, config: MultiRegionConfig):
        self.config = config
        self.all_regions = [config.primary_region] + list(config.replica_regions)
        self._connections: Dict[str, WorksetStateDB] = {}
        self._region_health: Dict[str, RegionHealth] = {
            region: RegionHealth(region=region) for region in self.all_regions
        }
        self._active_region = config.primary_region
        self._health_check_thread: Optional[threading.Thread] = None
        self._stop_health_check = threading.Event()
        self._request_count: Dict[str, int] = {region: 0 for region in self.all_regions}
        self._error_count: Dict[str, int] = {region: 0 for region in self.all_regions}

        for region in self.all_regions:
            self._get_connection(region)

    def start_health_monitoring(self) -> None:
        if self._health_check_thread and self._health_check_thread.is_alive():
            return
        self._stop_health_check.clear()
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True,
            name="multi-region-health-monitor",
        )
        self._health_check_thread.start()

    def stop_health_monitoring(self) -> None:
        self._stop_health_check.set()
        if self._health_check_thread:
            self._health_check_thread.join(timeout=5.0)

    def _health_check_loop(self) -> None:
        while not self._stop_health_check.is_set():
            for region in self.all_regions:
                self._check_region_health(region)
            self._update_active_region()
            self._stop_health_check.wait(self.config.health_check_interval_seconds)

    def _check_region_health(self, region: str) -> None:
        health = self._region_health[region]
        conn = self._get_connection(region)
        if conn is None:
            health.status = RegionStatus.UNHEALTHY
            health.last_error = "No connection available"
            return

        start = time.monotonic()
        try:
            table = getattr(conn, "table", None)
            if table is not None and hasattr(table, "load"):
                table.load()
            else:
                conn.list_worksets_by_state(WorksetState.READY, limit=1)
            latency = (time.monotonic() - start) * 1000
            health.latency_ms = latency
            health.last_check = dt.datetime.now(dt.timezone.utc)
            health.consecutive_failures = 0
            health.last_error = None
            health.status = (
                RegionStatus.DEGRADED
                if latency > self.config.latency_threshold_ms
                else RegionStatus.HEALTHY
            )
        except Exception as exc:
            health.consecutive_failures += 1
            health.last_error = str(exc)
            health.last_check = dt.datetime.now(dt.timezone.utc)
            health.status = (
                RegionStatus.UNHEALTHY
                if health.consecutive_failures >= self.config.failover_threshold
                else RegionStatus.DEGRADED
            )

    def _update_active_region(self) -> None:
        primary_health = self._region_health.get(self.config.primary_region)
        if primary_health and primary_health.is_available():
            self._active_region = self.config.primary_region
            return

        for region in self.config.replica_regions:
            health = self._region_health.get(region)
            if health and health.is_available():
                self._active_region = region
                return

    def get_best_read_region(self) -> str:
        available = [
            (region, health)
            for region, health in self._region_health.items()
            if health.is_available()
        ]
        if not available:
            return self.config.primary_region
        available.sort(key=lambda item: item[1].latency_ms)
        return available[0][0]

    def _get_connection(self, region: str) -> Optional[WorksetStateDB]:
        conn = self._connections.get(region)
        if conn is not None:
            return conn
        try:
            conn = WorksetStateDB(
                table_name=self.config.table_name,
                region=region,
                profile=self.config.profile,
            )
            self._connections[region] = conn
            return conn
        except Exception as exc:
            self._region_health[region].status = RegionStatus.UNHEALTHY
            self._region_health[region].last_error = str(exc)
            return None

    def _with_retry(
        self,
        operation: Callable[[WorksetStateDB], Any],
        *,
        prefer_read_region: bool = False,
        max_retries: int = 2,
    ) -> Any:
        attempted_regions: List[str] = []
        for attempt in range(max_retries + 1):
            region = self.get_best_read_region() if prefer_read_region else self._active_region
            if region in attempted_regions and attempt < max_retries:
                for fallback in self.all_regions:
                    if fallback not in attempted_regions:
                        region = fallback
                        break

            conn = self._get_connection(region)
            attempted_regions.append(region)
            self._request_count[region] += 1

            if conn is None:
                self._error_count[region] += 1
                continue

            try:
                return operation(conn)
            except Exception as exc:
                self._error_count[region] += 1
                self._region_health[region].consecutive_failures += 1
                self._region_health[region].last_error = str(exc)
                self._region_health[region].status = RegionStatus.DEGRADED
                if attempt == max_retries:
                    raise

        raise RuntimeError("No region available for operation")

    def register_workset(
        self,
        workset_id: str,
        bucket: str,
        prefix: str,
        priority: WorksetPriority = WorksetPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return self._with_retry(
            lambda conn: conn.register_workset(
                workset_id=workset_id,
                bucket=bucket,
                prefix=prefix,
                priority=priority,
                metadata=metadata,
            ),
            prefer_read_region=False,
        )

    def get_workset(self, workset_id: str) -> Optional[Dict[str, Any]]:
        return self._with_retry(
            lambda conn: conn.get_workset(workset_id),
            prefer_read_region=True,
        )

    def update_state(
        self,
        workset_id: str,
        new_state: WorksetState,
        reason: str,
        error_details: Optional[str] = None,
    ) -> None:
        self._with_retry(
            lambda conn: conn.update_state(workset_id, new_state, reason, error_details),
            prefer_read_region=False,
        )

    def acquire_lock(
        self,
        workset_id: str,
        owner_id: str,
        force: bool = False,
    ) -> bool:
        return self._with_retry(
            lambda conn: conn.acquire_lock(workset_id, owner_id, force),
            prefer_read_region=False,
        )

    def release_lock(self, workset_id: str, owner_id: str) -> bool:
        return self._with_retry(
            lambda conn: conn.release_lock(workset_id, owner_id),
            prefer_read_region=False,
        )

    def list_worksets_by_state(
        self,
        state: WorksetState,
        priority: Optional[WorksetPriority] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        return self._with_retry(
            lambda conn: conn.list_worksets_by_state(state, priority, limit),
            prefer_read_region=True,
        )

    def get_ready_worksets_prioritized(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self._with_retry(
            lambda conn: conn.get_ready_worksets_prioritized(limit),
            prefer_read_region=True,
        )

    def get_queue_depth(self) -> Dict[str, int]:
        return self._with_retry(
            lambda conn: conn.get_queue_depth(),
            prefer_read_region=True,
        )

    def create_global_table(self) -> bool:
        LOGGER.info("TapDB global table creation is managed outside Ursa")
        return True

    def remove_replica(self, region: str) -> bool:
        if region == self.config.primary_region:
            LOGGER.error("Cannot remove primary region")
            return False
        if region not in self.config.replica_regions:
            LOGGER.warning("Region %s is not configured as replica", region)
            return False

        self.config.replica_regions = [r for r in self.config.replica_regions if r != region]
        self.all_regions = [self.config.primary_region] + list(self.config.replica_regions)
        self._connections.pop(region, None)
        self._region_health.pop(region, None)
        self._request_count.pop(region, None)
        self._error_count.pop(region, None)
        return True

    def get_region_status(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for region, health in self._region_health.items():
            out[region] = {
                "status": health.status.value,
                "latency_ms": health.latency_ms,
                "last_check": health.last_check.isoformat() if health.last_check else None,
                "consecutive_failures": health.consecutive_failures,
                "last_error": health.last_error,
                "requests": self._request_count.get(region, 0),
                "errors": self._error_count.get(region, 0),
                "is_primary": region == self.config.primary_region,
                "is_active": region == self._active_region,
            }
        return out

    def get_active_region(self) -> str:
        return self._active_region

    def force_failover(self, target_region: str) -> bool:
        if target_region not in self.all_regions:
            return False
        self._active_region = target_region
        return True

    def get_replication_status(self) -> Dict[str, Any]:
        return {
            "mode": "tapdb-multi-region-routing",
            "primary_region": self.config.primary_region,
            "active_region": self._active_region,
            "replica_regions": list(self.config.replica_regions),
            "health": self.get_region_status(),
        }
