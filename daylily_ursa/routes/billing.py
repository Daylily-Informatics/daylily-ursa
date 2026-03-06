"""Billing routes for Daylily API.

Contains routes tagged 'billing' for customer billing data:
- Billing summary
- Invoice generation
- Per-workset billing details
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Query, status

from daylily_ursa.config import Settings
from daylily_ursa.workset_state_db import WorksetStateDB
from daylily_ursa.routes.dependencies import verify_workset_ownership

if TYPE_CHECKING:
    from daylily_ursa.workset_customer import CustomerManager

LOGGER = logging.getLogger("daylily.routes.billing")


class BillingDependencies:
    """Container for billing route dependencies."""

    def __init__(
        self,
        state_db: WorksetStateDB,
        settings: Settings,
        customer_manager: "CustomerManager",
    ):
        self.state_db = state_db
        self.settings = settings
        self.customer_manager = customer_manager


def create_billing_router(deps: BillingDependencies) -> APIRouter:
    """Create billing router with injected dependencies.

    Args:
        deps: BillingDependencies container with all required dependencies

    Returns:
        Configured APIRouter with billing routes
    """
    router = APIRouter(tags=["billing"])

    state_db = deps.state_db
    customer_manager = deps.customer_manager

    from daylily_ursa.billing import BillingCalculator

    # Initialize billing calculator with default rates
    billing_calculator = BillingCalculator(state_db=state_db)

    @router.get("/api/v2/customers/{customer_id}/billing/summary")
    async def get_customer_billing_summary(
        customer_id: str,
        days: int = Query(30, ge=1, le=365, description="Number of days to include"),
    ):
        """Get billing summary for a customer.

        Returns aggregated costs from completed worksets including:
        - Compute costs (from Snakemake benchmark data)
        - Storage costs (per-GB S3 storage)
        - Transfer costs (data egress)
        - Platform fees (if configured)
        """
        config = customer_manager.get_customer_config(customer_id)
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Customer {customer_id} not found",
            )

        period_end = datetime.now(timezone.utc)
        period_start = period_end - timedelta(days=days)

        summary = billing_calculator.calculate_customer_billing(
            customer_id=customer_id,
            period_start=period_start,
            period_end=period_end,
        )

        return {
            "customer_id": customer_id,
            "period_start": summary.period_start,
            "period_end": summary.period_end,
            "total_worksets": summary.total_worksets,
            "billable_worksets": summary.billable_worksets,
            "total_samples": summary.total_samples,
            "total_storage_gb": summary.total_storage_gb,
            "costs": {
                "compute_usd": summary.total_compute_cost_usd,
                "storage_usd": summary.total_storage_cost_usd,
                "transfer_usd": summary.total_transfer_cost_usd,
                "platform_fee_usd": summary.total_platform_fee_usd,
                "grand_total_usd": summary.grand_total_usd,
            },
            "accuracy": {
                "has_actual_costs": summary.has_actual_costs,
                "estimated_worksets": summary.estimated_worksets,
            },
        }

    @router.get("/api/v2/customers/{customer_id}/billing/invoice")
    async def get_customer_invoice(
        customer_id: str,
        days: int = Query(30, ge=1, le=365, description="Number of days to include"),
    ):
        """Generate invoice data for a customer.

        Returns detailed invoice with line items for each workset.
        Suitable for rendering invoices or exporting to billing systems.
        """
        config = customer_manager.get_customer_config(customer_id)
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Customer {customer_id} not found",
            )

        period_end = datetime.now(timezone.utc)
        period_start = period_end - timedelta(days=days)

        invoice_data = billing_calculator.generate_invoice_data(
            customer_id=customer_id,
            period_start=period_start,
            period_end=period_end,
        )

        # Add customer info
        invoice_data["customer_name"] = config.customer_name
        invoice_data["customer_email"] = config.email
        invoice_data["billing_account_id"] = config.billing_account_id
        invoice_data["cost_center"] = config.cost_center

        return invoice_data


    @router.get("/api/v2/customers/{customer_id}/billing/workset/{workset_id}")
    async def get_workset_billing(
        customer_id: str,
        workset_id: str,
    ):
        """Get billing details for a specific workset.

        Returns detailed cost breakdown including actual vs estimated costs.
        """
        config = customer_manager.get_customer_config(customer_id)
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Customer {customer_id} not found",
            )

        workset = state_db.get_workset(workset_id)
        if not workset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workset {workset_id} not found",
            )

        if not verify_workset_ownership(workset, customer_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Workset does not belong to this customer",
            )

        item = billing_calculator.calculate_workset_billing(workset)

        return {
            "workset_id": item.workset_id,
            "customer_id": item.customer_id,
            "completed_at": item.completed_at,
            "samples": item.sample_count,
            "rules_executed": item.rule_count,
            "costs": {
                "compute_usd": billing_calculator._round_currency(item.compute_cost_usd, 2),
                "storage_gb": billing_calculator._round_currency(item.storage_gb, 2),
                "storage_usd": billing_calculator._round_currency(item.storage_cost_usd, 2),
                "transfer_usd": billing_calculator._round_currency(item.transfer_cost_usd, 2),
                "platform_fee_usd": billing_calculator._round_currency(item.platform_fee_usd, 2),
                "total_usd": billing_calculator._round_currency(item.total_cost_usd, 2),
            },
            "accuracy": {
                "has_actual_compute_cost": item.has_actual_compute_cost,
                "has_actual_storage": item.has_actual_storage,
            },
        }

    return router
