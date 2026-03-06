"""Concurrent workset processor with cluster affinity and resource management.

Enables parallel processing of multiple worksets across multiple clusters.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from daylily_ursa.workset_notifications import NotificationEvent, NotificationManager
from daylily_ursa.workset_scheduler import WorksetScheduler
from daylily_ursa.workset_state_db import ErrorCategory, WorksetState, WorksetStateDB
from daylily_ursa.workset_validation import WorksetValidator

LOGGER = logging.getLogger("daylily.workset_concurrent_processor")


def _send_notification(
    manager: Optional[NotificationManager],
    workset_euid: str,
    event_type: str,
    message: str,
    state: str = "unknown",
) -> None:
    """Helper to send notification via NotificationManager.notify()."""
    if manager is None:
        return
    event = NotificationEvent(
        workset_euid=workset_euid,
        event_type=event_type,
        state=state,
        message=message,
    )
    manager.notify(event)


@dataclass
class ProcessorConfig:
    """Configuration for concurrent processor."""
    max_concurrent_worksets: int = 10
    max_workers: int = 5
    poll_interval_seconds: int = 30
    enable_retry: bool = True
    enable_validation: bool = True
    enable_notifications: bool = True


class ConcurrentWorksetProcessor:
    """Process multiple worksets concurrently across clusters."""

    def __init__(
        self,
        state_db: WorksetStateDB,
        scheduler: WorksetScheduler,
        config: ProcessorConfig,
        validator: Optional[WorksetValidator] = None,
        notification_manager: Optional[NotificationManager] = None,
        workset_executor: Optional[Callable] = None,
    ):
        """Initialize concurrent processor.

        Args:
            state_db: Workset state database
            scheduler: Workset scheduler
            config: Processor configuration
            validator: Optional workset validator
            notification_manager: Optional notification manager
            workset_executor: Optional custom workset execution function
        """
        self.state_db = state_db
        self.scheduler = scheduler
        self.config = config
        self.validator = validator
        self.notification_manager = notification_manager
        self.workset_executor = workset_executor or self._default_workset_executor
        self.running = False

    def start(self) -> None:
        """Start the concurrent processor."""
        self.running = True
        LOGGER.info(
            "Starting concurrent processor (max_concurrent=%d, max_workers=%d)",
            self.config.max_concurrent_worksets,
            self.config.max_workers,
        )

        while self.running:
            try:
                self._process_cycle()
            except Exception as e:
                LOGGER.error("Error in processing cycle: %s", e, exc_info=True)

            time.sleep(self.config.poll_interval_seconds)

    def stop(self) -> None:
        """Stop the concurrent processor."""
        self.running = False
        LOGGER.info("Stopping concurrent processor")

    def _process_cycle(self) -> None:
        """Execute one processing cycle."""
        # Step 1: Handle retries
        if self.config.enable_retry:
            self._process_retries()

        # Step 2: Check if we can start new worksets
        current_count = self.state_db.get_concurrent_worksets_count()
        available_slots = self.config.max_concurrent_worksets - current_count

        if available_slots <= 0:
            LOGGER.debug("At max concurrent worksets (%d), waiting...", current_count)
            return

        LOGGER.info("Available slots: %d (current: %d)", available_slots, current_count)

        # Step 3: Get ready worksets
        ready_worksets = self.state_db.list_worksets_by_state(
            WorksetState.READY,
            limit=available_slots * 2,  # Get more than needed for filtering
        )

        if not ready_worksets:
            LOGGER.debug("No ready worksets found")
            return

        # Step 4: Schedule and process worksets
        worksets_to_process = []
        for workset in ready_worksets[:available_slots]:
            # Validate if enabled
            if self.config.enable_validation and self.validator:
                if not self._validate_workset(workset):
                    continue

            # Schedule workset
            decision = self.scheduler.schedule_workset(
                workset["euid"],
                workset.get("metadata", {}),
            )

            if decision.cluster_name or decision.should_create_cluster:
                worksets_to_process.append((workset, decision))

        # Step 5: Process worksets concurrently
        if worksets_to_process:
            self._process_worksets_parallel(worksets_to_process)

    def _process_retries(self) -> None:
        """Process worksets that are ready for retry."""
        retryable = self.state_db.get_retryable_worksets()

        for workset in retryable:
            euid = workset["euid"]
            LOGGER.info("Resetting workset %s for retry", euid)

            if self.state_db.reset_for_retry(euid):
                _send_notification(
                    self.notification_manager,
                    workset_euid=euid,
                    event_type="retry",
                    message=f"Workset {euid} reset for retry",
                )

    def _validate_workset(self, workset: Dict) -> bool:
        """Validate a workset before processing.

        Args:
            workset: Workset dict

        Returns:
            True if valid
        """
        euid = workset["euid"]
        bucket = workset["bucket"]
        prefix = workset["prefix"]

        if self.validator is None:
            LOGGER.warning("Validator not configured, skipping validation for %s", euid)
            return True

        try:
            result = self.validator.validate_workset(bucket, prefix)

            if not result.is_valid:
                error_msg = "; ".join(result.errors)
                LOGGER.error("Workset %s validation failed: %s", euid, error_msg)

                self.state_db.record_failure(
                    euid,
                    error_msg,
                    ErrorCategory.CONFIGURATION,
                )

                _send_notification(
                    self.notification_manager,
                    workset_euid=euid,
                    event_type="validation_failed",
                    message=f"Validation failed: {error_msg}",
                    state="error",
                )

                return False

            # Update workset with estimates
            if result.estimated_cost_usd:
                metadata = workset.get("metadata", {})
                metadata.update({
                    "estimated_cost_usd": result.estimated_cost_usd,
                    "estimated_duration_minutes": result.estimated_duration_minutes,
                    "estimated_vcpu_hours": result.estimated_vcpu_hours,
                })
                # Note: Would need to add update_metadata method to state_db

            if result.warnings:
                LOGGER.warning("Workset %s has warnings: %s", euid, result.warnings)

            return True

        except Exception as e:
            LOGGER.error("Validation error for %s: %s", euid, str(e))
            return False

    def _process_worksets_parallel(self, worksets_and_decisions: List) -> None:
        """Process multiple worksets in parallel.

        Args:
            worksets_and_decisions: List of (workset, decision) tuples
        """
        LOGGER.info("Processing %d worksets in parallel", len(worksets_and_decisions))

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {}

            for workset, decision in worksets_and_decisions:
                future = executor.submit(
                    self._process_single_workset,
                    workset,
                    decision,
                )
                futures[future] = workset["euid"]

            # Wait for completion and handle results
            for future in as_completed(futures):
                euid = futures[future]
                try:
                    success = future.result()
                    if success:
                        LOGGER.info("Workset %s completed successfully", euid)
                    else:
                        LOGGER.error("Workset %s failed", euid)
                except Exception as e:
                    LOGGER.error("Exception processing %s: %s", euid, e, exc_info=True)

    def _process_single_workset(self, workset: Dict, decision) -> bool:
        """Process a single workset.

        Args:
            workset: Workset dict
            decision: Scheduling decision

        Returns:
            True if successful
        """
        euid = workset["euid"]

        try:
            # Acquire lock
            owner_id = f"processor-{time.time()}"
            if not self.state_db.acquire_lock(euid, owner_id):
                LOGGER.warning("Failed to acquire lock for %s", euid)
                return False

            # Update to in-progress
            self.state_db.update_state(
                euid,
                WorksetState.IN_PROGRESS,
                reason=f"Processing on cluster {decision.cluster_name or 'new'}",
                cluster_name=decision.cluster_name,
            )

            _send_notification(
                self.notification_manager,
                workset_euid=euid,
                event_type="started",
                message=f"Processing started on {decision.cluster_name or 'new cluster'}",
                state="in_progress",
            )

            # Execute workset
            success = self.workset_executor(workset, decision)

            # Update final state
            if success:
                self.state_db.update_state(
                    euid,
                    WorksetState.COMPLETE,
                    reason="Processing completed successfully",
                )

                _send_notification(
                    self.notification_manager,
                    workset_euid=euid,
                    event_type="completed",
                    message="Processing completed successfully",
                    state="complete",
                )
            else:
                self.state_db.record_failure(
                    euid,
                    "Workset execution returned failure",
                    ErrorCategory.TRANSIENT,
                )

            # Release lock
            self.state_db.release_lock(euid, owner_id)

            return success

        except Exception as e:
            LOGGER.error("Error processing %s: %s", euid, e, exc_info=True)

            self.state_db.record_failure(
                euid,
                str(e),
                ErrorCategory.TRANSIENT,
            )

            _send_notification(
                self.notification_manager,
                workset_euid=euid,
                event_type="error",
                message=f"Processing error: {e}",
                state="error",
            )

            return False

    def _default_workset_executor(self, workset: Dict, decision) -> bool:
        """Default workset executor (placeholder).

        Args:
            workset: Workset dict
            decision: Scheduling decision

        Returns:
            True if successful
        """
        LOGGER.info(
            "Executing workset %s (cluster=%s)",
            workset["euid"],
            decision.cluster_name,
        )

        # This would call the actual workset processing logic
        # For now, just simulate success
        time.sleep(1)
        return True


