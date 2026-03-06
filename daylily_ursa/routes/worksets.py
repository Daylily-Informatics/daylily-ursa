"""Workset API routes.

Core workset CRUD operations, queue stats, and scheduling endpoints.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, status

from daylily_ursa.routes.dependencies import (
    QueueStats,
    SchedulingStats,
    WorksetCreate,
    WorksetResponse,
    WorksetStateUpdate,
)
from daylily_ursa.workset_state_db import WorksetPriority, WorksetState, WorksetStateDB
from daylily_ursa.workset_scheduler import WorksetScheduler

LOGGER = logging.getLogger("daylily.routes.worksets")


def create_worksets_router(
    state_db: WorksetStateDB,
    scheduler: Optional[WorksetScheduler] = None,
) -> APIRouter:
    """Create worksets router with required dependencies.

    Args:
        state_db: Workset state database
        scheduler: Optional workset scheduler for advanced scheduling

    Returns:
        Configured APIRouter with workset endpoints
    """
    router = APIRouter(tags=["worksets"])

    @router.get("/", tags=["health"], include_in_schema=False)
    async def root():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "daylily-workset-monitor",
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

    @router.post(
        "/api/v2/worksets", response_model=WorksetResponse, status_code=status.HTTP_201_CREATED
    )
    async def create_workset(workset: WorksetCreate):
        """Register a new workset."""
        try:
            euid = state_db.register_workset(
                name=workset.name,
                bucket=workset.bucket,
                prefix=workset.prefix,
                priority=workset.priority,
                workset_type=workset.workset_type,
                metadata=workset.metadata,
                customer_id=workset.customer_id,
                preferred_cluster=workset.preferred_cluster,
                cluster_region=workset.cluster_region,
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )

        if not euid:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Workset {workset.name} already exists",
            )

        # Retrieve the created workset by its new euid
        created = state_db.get_workset(euid)
        if not created:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve created workset",
            )

        return WorksetResponse(**created)

    @router.get("/api/v2/worksets/{euid}", response_model=WorksetResponse)
    async def get_workset(euid: str):
        """Get workset details by EUID."""
        workset = state_db.get_workset(euid)
        if not workset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workset {euid} not found",
            )

        return WorksetResponse(**workset)

    @router.get("/api/v2/worksets", response_model=List[WorksetResponse])
    async def list_worksets(
        state: Optional[WorksetState] = Query(None, description="Filter by state"),
        priority: Optional[WorksetPriority] = Query(None, description="Filter by priority"),
        limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    ):
        """List worksets with optional filters."""
        if state:
            worksets = state_db.list_worksets_by_state(state, priority=priority, limit=limit)
        else:
            # Get all states
            worksets = []
            for ws_state in WorksetState:
                batch = state_db.list_worksets_by_state(ws_state, priority=priority, limit=limit)
                worksets.extend(batch)
                if len(worksets) >= limit:
                    break
            worksets = worksets[:limit]

        return [WorksetResponse(**w) for w in worksets]

    @router.put("/api/v2/worksets/{euid}/state", response_model=WorksetResponse)
    async def update_workset_state(euid: str, update: WorksetStateUpdate):
        """Update workset state."""
        # Verify workset exists
        workset = state_db.get_workset(euid)
        if not workset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workset {euid} not found",
            )

        state_db.update_state(
            euid=euid,
            new_state=update.state,
            reason=update.reason,
            error_details=update.error_details,
            cluster_name=update.cluster_name,
            metrics=update.metrics if update.metrics else None,
        )

        # Return updated workset
        updated = state_db.get_workset(euid)
        if not updated:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve updated workset {euid}",
            )
        return WorksetResponse(**updated)

    @router.post("/api/v2/worksets/{euid}/lock")
    async def acquire_workset_lock(
        euid: str, owner_id: str = Query(..., description="Lock owner ID")
    ):
        """Acquire lock on a workset."""
        success = state_db.acquire_lock(euid, owner_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Failed to acquire lock on workset {euid}",
            )

        return {"status": "locked", "euid": euid, "owner_id": owner_id}

    @router.delete("/api/v2/worksets/{euid}/lock")
    async def release_workset_lock(
        euid: str, owner_id: str = Query(..., description="Lock owner ID")
    ):
        """Release lock on a workset."""
        success = state_db.release_lock(euid, owner_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Failed to release lock on workset {euid} (not owner)",
            )

        return {"status": "unlocked", "euid": euid}

    # ========== Monitoring Endpoints ==========

    @router.get("/api/v2/queue/stats", response_model=QueueStats, tags=["monitoring"])
    async def get_queue_stats():
        """Get queue statistics."""
        queue_depth = state_db.get_queue_depth()

        return QueueStats(
            queue_depth=queue_depth,
            total_worksets=sum(queue_depth.values()),
            ready_worksets=queue_depth.get(WorksetState.READY.value, 0),
            in_progress_worksets=queue_depth.get(WorksetState.IN_PROGRESS.value, 0),
            error_worksets=queue_depth.get(WorksetState.ERROR.value, 0),
        )

    @router.get("/api/v2/scheduler/stats", response_model=SchedulingStats, tags=["monitoring"])
    async def get_scheduler_stats():
        """Get scheduler statistics."""
        if not scheduler:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Scheduler not configured",
            )

        stats = scheduler.get_scheduling_stats()
        return SchedulingStats(**stats)

    # ========== Scheduling Endpoints ==========

    @router.get(
        "/api/v2/worksets/next", response_model=Optional[WorksetResponse], tags=["scheduling"]
    )
    async def get_next_workset():
        """Get the next workset to execute based on priority."""
        if not scheduler:
            # Fallback to simple priority-based selection
            worksets = state_db.get_ready_worksets_prioritized(limit=1)
            if not worksets:
                return None
            return WorksetResponse(**worksets[0])

        next_workset = scheduler.get_next_workset()
        if not next_workset:
            return None

        return WorksetResponse(**next_workset)

    return router
