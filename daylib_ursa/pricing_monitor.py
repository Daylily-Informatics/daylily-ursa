"""Run and persist daylily-ec pricing snapshots for the portal."""

from __future__ import annotations

import fcntl
import json
import logging
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from daylib_ursa.config import Settings
from daylib_ursa.ephemeral_cluster.runner import resolve_daylily_ec
from daylib_ursa.portal_state import PortalState

LOGGER = logging.getLogger("daylily.pricing_monitor")


class PricingMonitor:
    """Queue and persist pricing snapshots via the installed daylily-ec CLI."""

    def __init__(self, *, settings: Settings, store: PortalState):
        self.settings = settings
        self.store = store
        self._thread_lock = threading.Lock()
        self._scheduler_started = False
        self._scheduler_lock = threading.Lock()
        self._lock_path = Path.home() / ".ursa" / "pricing-monitor.lock"

    def start(self) -> None:
        with self._scheduler_lock:
            if self._scheduler_started:
                return
            thread = threading.Thread(target=self._scheduler_loop, daemon=True, name="ursa-pricing-scheduler")
            thread.start()
            self._scheduler_started = True

    def _scheduler_loop(self) -> None:
        while True:
            try:
                self.ensure_due_capture()
            except Exception:
                LOGGER.exception("Pricing scheduler iteration failed")
            time.sleep(60)

    def ensure_due_capture(self) -> None:
        last_capture = self.store.last_successful_pricing_capture()
        if last_capture:
            last_dt = datetime.fromisoformat(last_capture.replace("Z", "+00:00"))
            if datetime.now(timezone.utc) - last_dt < timedelta(hours=self.settings.ursa_cost_monitor_interval_hours):
                return
        self.queue_capture(trigger="scheduled", requested_by=None)

    def queue_capture(self, *, trigger: str, requested_by: Optional[str]) -> Dict[str, Any]:
        active = self.store.get_active_pricing_run()
        if active:
            return active
        run = self.store.create_pricing_run(trigger=trigger, requested_by=requested_by)
        thread = threading.Thread(
            target=self._run_capture_worker,
            args=(int(run["run_id"]),),
            daemon=True,
            name=f"ursa-pricing-run-{run['run_id']}",
        )
        thread.start()
        return run

    def _run_capture_worker(self, run_id: int) -> None:
        with self._thread_lock:
            self.store.mark_pricing_run_running(run_id)
            try:
                snapshot = self._capture_snapshot()
                self.store.save_pricing_snapshot(run_id, snapshot)
            except Exception as exc:
                LOGGER.exception("Pricing capture failed")
                self.store.mark_pricing_run_failed(run_id, str(exc))

    def _capture_snapshot(self) -> Dict[str, Any]:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            command = self._build_snapshot_command()
            LOGGER.info("Running pricing snapshot command: %s", command)
            result = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
                raise RuntimeError(f"daylily-ec pricing snapshot failed: {detail}")
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError as exc:
                raise RuntimeError("daylily-ec pricing snapshot did not return valid JSON") from exc

    def _build_snapshot_command(self) -> List[str]:
        command: List[str] = [str(resolve_daylily_ec()), "pricing", "snapshot"]
        for region in self.settings.get_cost_monitor_regions():
            command.extend(["--region", region])
        for partition in self.settings.get_cost_monitor_partitions():
            command.extend(["--partition", partition])
        if self.settings.ursa_cost_monitor_config_path:
            command.extend(["--config", self.settings.ursa_cost_monitor_config_path])
        if self.settings.aws_profile:
            command.extend(["--profile", self.settings.aws_profile])
        return command

    def get_snapshot_payload(
        self,
        *,
        region: Optional[str],
        partitions: Optional[List[str]],
        from_ts: Optional[str],
        to_ts: Optional[str],
    ) -> Dict[str, Any]:
        return self.store.get_pricing_snapshot_payload(
            region=region,
            partitions=partitions,
            from_ts=from_ts,
            to_ts=to_ts,
        )
