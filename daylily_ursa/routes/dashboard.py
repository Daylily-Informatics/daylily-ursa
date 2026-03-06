"""Dashboard routes for Daylily API.

Contains routes tagged 'customer-dashboard' for customer dashboard data:
- Activity charts (submitted/completed/failed by day)
- Cost history charts
- Cost breakdown by category
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import DefaultDict, TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Query, status

from daylily_ursa.config import Settings
from daylily_ursa.workset_state_db import WorksetState, WorksetStateDB
from daylily_ursa.routes.dependencies import verify_workset_ownership

if TYPE_CHECKING:
    from daylily_ursa.workset_customer import CustomerManager

LOGGER = logging.getLogger("daylily.routes.dashboard")


class DashboardDependencies:
    """Container for dashboard route dependencies."""

    def __init__(
        self,
        state_db: WorksetStateDB,
        settings: Settings,
        customer_manager: "CustomerManager",
    ):
        self.state_db = state_db
        self.settings = settings
        self.customer_manager = customer_manager


def create_dashboard_router(deps: DashboardDependencies) -> APIRouter:
    """Create dashboard router with injected dependencies.

    Args:
        deps: DashboardDependencies container with all required dependencies

    Returns:
        Configured APIRouter with dashboard routes
    """
    router = APIRouter(tags=["customer-dashboard"])

    state_db = deps.state_db
    customer_manager = deps.customer_manager

    @router.get("/api/v2/customers/{customer_id}/dashboard/activity")
    async def get_dashboard_activity(
        customer_id: str,
        days: int = Query(30, ge=1, le=90, description="Number of days of activity data"),
    ):
        """Get workset activity data for charts (submitted/completed/failed by day).

        Returns daily counts of worksets by state for the specified time period.
        """
        if not customer_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Customer management not configured",
            )

        config = customer_manager.get_customer_config(customer_id)
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Customer {customer_id} not found",
            )

        # Get all worksets for this customer
        all_worksets = []
        for ws_state in WorksetState:
            batch = state_db.list_worksets_by_state(ws_state, limit=500)
            for ws in batch:
                if verify_workset_ownership(ws, customer_id):
                    all_worksets.append(ws)

        # Initialize date range
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=days - 1)

        # Initialize counters for each day
        daily_submitted: DefaultDict[str, int] = defaultdict(int)
        daily_completed: DefaultDict[str, int] = defaultdict(int)
        daily_failed: DefaultDict[str, int] = defaultdict(int)

        for ws in all_worksets:
            state_history = ws.get("state_history", [])
            for entry in state_history:
                ts = entry.get("timestamp")
                state = entry.get("state")
                if not ts or not state:
                    continue

                try:
                    entry_date = datetime.fromisoformat(ts.replace("Z", "+00:00")).date()
                except (ValueError, AttributeError):
                    continue

                if entry_date < start_date or entry_date > end_date:
                    continue

                date_str = entry_date.isoformat()
                if state == "ready":
                    daily_submitted[date_str] += 1
                elif state == "complete":
                    daily_completed[date_str] += 1
                elif state == "error":
                    daily_failed[date_str] += 1

        # Build response arrays
        labels = []
        submitted = []
        completed = []
        failed = []

        current = start_date
        while current <= end_date:
            date_str = current.isoformat()
            labels.append(current.strftime("%b %d"))
            submitted.append(daily_submitted.get(date_str, 0))
            completed.append(daily_completed.get(date_str, 0))
            failed.append(daily_failed.get(date_str, 0))
            current += timedelta(days=1)

        return {
            "labels": labels,
            "datasets": {
                "submitted": submitted,
                "completed": completed,
                "failed": failed,
            },
            "totals": {
                "submitted": sum(submitted),
                "completed": sum(completed),
                "failed": sum(failed),
            },
        }

    @router.get("/api/v2/customers/{customer_id}/dashboard/cost-history")
    async def get_dashboard_cost_history(
        customer_id: str,
        days: int = Query(30, ge=1, le=90, description="Number of days of cost data"),
    ):
        """Get daily cost data for charts.

        Returns daily costs aggregated from completed worksets.
        """
        if not customer_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Customer management not configured",
            )

        config = customer_manager.get_customer_config(customer_id)
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Customer {customer_id} not found",
            )

        # Get all worksets for this customer
        all_worksets = []
        for ws_state in WorksetState:
            batch = state_db.list_worksets_by_state(ws_state, limit=500)
            for ws in batch:
                if verify_workset_ownership(ws, customer_id):
                    all_worksets.append(ws)

        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=days - 1)

        # Track daily costs
        daily_costs: DefaultDict[str, float] = defaultdict(float)

        for ws in all_worksets:
            # Get cost from performance metrics or estimate
            cost = 0.0
            pm = ws.get("performance_metrics", {})
            if pm and isinstance(pm, dict):
                cost_summary = pm.get("cost_summary", {})
                if cost_summary and isinstance(cost_summary, dict):
                    cost = float(cost_summary.get("total_cost", 0))

            if cost == 0:
                # Fall back to estimated cost
                cost = float(ws.get("cost_usd", 0) or 0)
                if cost == 0:
                    metadata = ws.get("metadata", {})
                    if isinstance(metadata, dict):
                        cost = float(
                            metadata.get("cost_usd", 0)
                            or metadata.get("estimated_cost_usd", 0)
                            or 0
                        )

            if cost == 0:
                continue

            # Find completion date from state_history
            completion_date = None
            state_history = ws.get("state_history", [])
            for entry in reversed(state_history):
                if entry.get("state") == "complete":
                    ts = entry.get("timestamp")
                    if ts:
                        try:
                            completion_date = datetime.fromisoformat(
                                ts.replace("Z", "+00:00")
                            ).date()
                        except (ValueError, AttributeError):
                            pass
                    break

            if not completion_date:
                # Use updated_at as fallback
                updated = ws.get("updated_at")
                if updated:
                    try:
                        completion_date = datetime.fromisoformat(
                            updated.replace("Z", "+00:00")
                        ).date()
                    except (ValueError, AttributeError):
                        continue
                else:
                    continue

            if start_date <= completion_date <= end_date:
                daily_costs[completion_date.isoformat()] += cost

        # Build response arrays
        labels = []
        costs = []

        current = start_date
        while current <= end_date:
            date_str = current.isoformat()
            labels.append(current.strftime("%b %d"))
            costs.append(round(daily_costs.get(date_str, 0), 4))
            current += timedelta(days=1)

        return {
            "labels": labels,
            "costs": costs,
            "total": round(sum(costs), 2),
        }

    @router.get("/api/v2/customers/{customer_id}/dashboard/cost-breakdown")
    async def get_dashboard_cost_breakdown(
        customer_id: str,
    ):
        """Get cost breakdown by category.

        Returns compute, storage, and transfer costs (split into 3 transfer types).

        - Compute: From dedicated cost_report fields (Phase 5B) or estimates
        - Storage: From dedicated storage_metrics fields (Phase 5C) at $0.023/GB/month
        - Transfer: intra-region ($0), cross-region, and internet egress
        """
        if not state_db:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="State DB not configured",
            )

        if not customer_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Customer management not configured",
            )

        config = customer_manager.get_customer_config(customer_id)
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Customer {customer_id} not found",
            )

        # Shared billing logic (used by both portal Usage page + dashboard chart)
        from daylily_ursa.billing import calculate_customer_cost_breakdown

        breakdown = calculate_customer_cost_breakdown(
            state_db,
            customer_id,
            limit=500,
        )

        # Always return explicit transfer categories so the portal can present:
        # - intra-region transfer (typically $0/GB)
        # - cross-region transfer
        # - internet egress
        categories: list[str] = [
            "Compute",
            "Storage",
            "Transfer (Intra-region)",
            "Transfer (Cross-region)",
            "Internet egress",
        ]
        values: list[float] = [
            float(breakdown.get("compute_cost_usd", 0.0) or 0.0),
            float(breakdown.get("storage_cost_usd", 0.0) or 0.0),
            float(breakdown.get("transfer_intra_region_cost_usd", 0.0) or 0.0),
            float(breakdown.get("transfer_cross_region_cost_usd", 0.0) or 0.0),
            float(breakdown.get("transfer_internet_cost_usd", 0.0) or 0.0),
        ]
        if all(v == 0.0 for v in values):
            categories = ["Compute"]
            values = [0.0]

        return {
            "categories": categories,
            "values": values,
            "total": breakdown["total"],
            "has_actual_costs": breakdown["has_actual_costs"],
            "compute_cost_usd": breakdown["compute_cost_usd"],
            "storage_cost_usd": breakdown["storage_cost_usd"],
            "transfer_cost_usd": breakdown["transfer_cost_usd"],
            "transfer_intra_region_cost_usd": breakdown.get("transfer_intra_region_cost_usd", 0.0),
            "transfer_cross_region_cost_usd": breakdown.get("transfer_cross_region_cost_usd", 0.0),
            "transfer_internet_cost_usd": breakdown.get("transfer_internet_cost_usd", 0.0),
            "transfer_intra_region_gb": breakdown.get("transfer_intra_region_gb", 0.0),
            "transfer_cross_region_gb": breakdown.get("transfer_cross_region_gb", 0.0),
            "transfer_internet_gb": breakdown.get("transfer_internet_gb", 0.0),
            "storage_gb": breakdown["storage_gb"],
            "rates": breakdown.get("rates", {}),
        }

    return router
