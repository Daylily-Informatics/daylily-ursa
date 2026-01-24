"""Billing calculation module for Daylily workset processing.

Provides per-sample and per-GB billing calculations based on actual
compute costs from Snakemake benchmark data and storage consumption.

Billing model:
- Compute: Actual cost from Snakemake benchmark data (spot instance pricing)
- Storage: Per-GB/month rate for S3 Standard storage
- Transfer: Per-GB rate for data egress (optional)
- Platform fee: Optional per-sample or percentage markup
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from daylib.workset_state_db import WorksetStateDB

LOGGER = logging.getLogger(__name__)


@dataclass
class BillingRates:
    """Configurable billing rates.
    
    All rates are in USD.
    """
    # Storage rates (per GB per month)
    s3_storage_per_gb_month: float = 0.023  # S3 Standard
    
    # Transfer rates (per GB)
    data_egress_per_gb: float = 0.09  # AWS data transfer out
    
    # Platform fees
    platform_fee_per_sample: float = 0.0  # Optional per-sample fee
    platform_fee_percentage: float = 0.0  # Optional percentage markup (0.10 = 10%)
    
    # Minimum charges
    minimum_compute_charge: float = 0.0
    minimum_storage_charge: float = 0.0


@dataclass
class WorksetBillingItem:
    """Billing details for a single workset."""
    workset_id: str
    customer_id: str
    completed_at: Optional[str] = None
    
    # Compute costs (from Snakemake benchmark)
    compute_cost_usd: float = 0.0
    sample_count: int = 0
    rule_count: int = 0
    
    # Storage costs
    storage_bytes: int = 0
    storage_gb: float = 0.0
    storage_cost_usd: float = 0.0
    
    # Transfer costs (estimated)
    transfer_gb: float = 0.0
    transfer_cost_usd: float = 0.0
    
    # Platform fees
    platform_fee_usd: float = 0.0
    
    # Totals
    total_cost_usd: float = 0.0
    
    # Source flags
    has_actual_compute_cost: bool = False
    has_actual_storage: bool = False


@dataclass
class CustomerBillingSummary:
    """Aggregated billing summary for a customer."""
    customer_id: str
    period_start: str
    period_end: str
    
    # Workset counts
    total_worksets: int = 0
    billable_worksets: int = 0
    
    # Sample counts
    total_samples: int = 0
    
    # Cost breakdown
    total_compute_cost_usd: float = 0.0
    total_storage_cost_usd: float = 0.0
    total_transfer_cost_usd: float = 0.0
    total_platform_fee_usd: float = 0.0
    
    # Storage totals
    total_storage_bytes: int = 0
    total_storage_gb: float = 0.0
    
    # Grand total
    grand_total_usd: float = 0.0
    
    # Line items
    workset_items: List[WorksetBillingItem] = field(default_factory=list)
    
    # Accuracy flags
    has_actual_costs: bool = False
    estimated_worksets: int = 0


class BillingCalculator:
    """Calculate billing for worksets and customers.
    
    Uses actual cost data from Snakemake benchmark reports (Phase 5B)
    and storage metrics (Phase 5C) when available, falling back to
    estimates when actual data is not present.
    """
    
    def __init__(
        self,
        state_db: "WorksetStateDB",
        rates: Optional[BillingRates] = None,
    ):
        """Initialize billing calculator.
        
        Args:
            state_db: WorksetStateDB instance for data access
            rates: Optional custom billing rates (defaults to standard rates)
        """
        self.state_db = state_db
        self.rates = rates or BillingRates()
    
    def _round_currency(self, value: float, places: int = 4) -> float:
        """Round currency value to specified decimal places."""
        d = Decimal(str(value))
        return float(d.quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP))
    
    def _bytes_to_gb(self, bytes_val: int) -> float:
        """Convert bytes to gigabytes."""
        return bytes_val / (1024 ** 3)

    def calculate_workset_billing(
        self,
        workset: Dict[str, Any],
    ) -> WorksetBillingItem:
        """Calculate billing for a single workset.

        Uses actual cost/storage data when available, estimates otherwise.

        Args:
            workset: Workset record from DynamoDB

        Returns:
            WorksetBillingItem with calculated costs
        """
        workset_id = workset.get("workset_id", "unknown")
        customer_id = workset.get("customer_id", "unknown")

        item = WorksetBillingItem(
            workset_id=workset_id,
            customer_id=customer_id,
            completed_at=workset.get("completed_at"),
        )

        # Get actual cost report data (Phase 5B)
        cost_report = self.state_db.get_cost_report(workset_id)
        if cost_report:
            item.compute_cost_usd = cost_report.get("total_compute_cost_usd", 0.0)
            item.sample_count = cost_report.get("cost_report_sample_count", 0)
            item.rule_count = cost_report.get("cost_report_rule_count", 0)
            item.has_actual_compute_cost = True
        else:
            # Fall back to estimates from metadata or performance_metrics
            pm = workset.get("performance_metrics", {})
            if pm and isinstance(pm, dict):
                cost_summary = pm.get("cost_summary", {})
                if cost_summary:
                    item.compute_cost_usd = float(cost_summary.get("total_cost", 0))
                    item.sample_count = int(cost_summary.get("sample_count", 0))
                    item.has_actual_compute_cost = True

            if item.compute_cost_usd == 0:
                # Try metadata estimates
                metadata = workset.get("metadata", {})
                if isinstance(metadata, dict):
                    item.compute_cost_usd = float(
                        metadata.get("cost_usd", 0) or
                        metadata.get("estimated_cost_usd", 0) or 0
                    )
                    item.sample_count = int(metadata.get("sample_count", 0))

        # Get actual storage metrics (Phase 5C)
        storage_metrics = self.state_db.get_storage_metrics(workset_id)
        if storage_metrics:
            item.storage_bytes = storage_metrics.get("results_storage_bytes", 0)
            item.has_actual_storage = True
        else:
            # Fall back to performance_metrics
            pm = workset.get("performance_metrics", {})
            if pm and isinstance(pm, dict):
                export_metrics = pm.get("post_export_metrics", {}) or pm.get("pre_export_metrics", {})
                if export_metrics:
                    item.storage_bytes = int(export_metrics.get("analysis_directory_size_bytes", 0) or 0)
                    item.has_actual_storage = True

        # Calculate storage cost
        item.storage_gb = self._bytes_to_gb(item.storage_bytes)
        item.storage_cost_usd = item.storage_gb * self.rates.s3_storage_per_gb_month

        # Calculate transfer cost (assume all storage is transferred once)
        item.transfer_gb = item.storage_gb
        item.transfer_cost_usd = item.transfer_gb * self.rates.data_egress_per_gb

        # Calculate platform fee
        if self.rates.platform_fee_per_sample > 0:
            item.platform_fee_usd += item.sample_count * self.rates.platform_fee_per_sample

        subtotal = item.compute_cost_usd + item.storage_cost_usd + item.transfer_cost_usd
        if self.rates.platform_fee_percentage > 0:
            item.platform_fee_usd += subtotal * self.rates.platform_fee_percentage

        # Apply minimum charges
        if item.compute_cost_usd > 0 and item.compute_cost_usd < self.rates.minimum_compute_charge:
            item.compute_cost_usd = self.rates.minimum_compute_charge
        if item.storage_cost_usd > 0 and item.storage_cost_usd < self.rates.minimum_storage_charge:
            item.storage_cost_usd = self.rates.minimum_storage_charge

        # Calculate total
        item.total_cost_usd = self._round_currency(
            item.compute_cost_usd +
            item.storage_cost_usd +
            item.transfer_cost_usd +
            item.platform_fee_usd
        )

        return item

    def calculate_customer_billing(
        self,
        customer_id: str,
        period_start: Optional[dt.datetime] = None,
        period_end: Optional[dt.datetime] = None,
        include_states: Optional[List[str]] = None,
    ) -> CustomerBillingSummary:
        """Calculate billing summary for a customer.

        Args:
            customer_id: Customer identifier
            period_start: Start of billing period (default: 30 days ago)
            period_end: End of billing period (default: now)
            include_states: Workset states to include (default: complete only)

        Returns:
            CustomerBillingSummary with aggregated costs
        """
        from daylib.workset_state_db import WorksetState

        # Default period: last 30 days
        if period_end is None:
            period_end = dt.datetime.now(dt.timezone.utc)
        if period_start is None:
            period_start = period_end - dt.timedelta(days=30)

        # Default to completed worksets only
        if include_states is None:
            include_states = ["complete"]

        summary = CustomerBillingSummary(
            customer_id=customer_id,
            period_start=period_start.isoformat().replace("+00:00", "Z"),
            period_end=period_end.isoformat().replace("+00:00", "Z"),
        )

        # Get worksets for customer
        worksets = self.state_db.list_worksets_by_customer(customer_id, limit=1000)

        for ws in worksets:
            ws_state = ws.get("state", "")
            if ws_state not in include_states:
                continue

            # Check if workset is within billing period
            completed_at = ws.get("completed_at") or ws.get("updated_at")
            if completed_at:
                try:
                    if isinstance(completed_at, str):
                        # Parse ISO format
                        completed_dt = dt.datetime.fromisoformat(
                            completed_at.replace("Z", "+00:00")
                        )
                    else:
                        completed_dt = completed_at

                    if completed_dt < period_start or completed_dt > period_end:
                        continue
                except (ValueError, TypeError):
                    pass  # Include if we can't parse the date

            summary.total_worksets += 1

            # Calculate billing for this workset
            item = self.calculate_workset_billing(ws)

            if item.total_cost_usd > 0:
                summary.billable_worksets += 1

            # Aggregate totals
            summary.total_samples += item.sample_count
            summary.total_compute_cost_usd += item.compute_cost_usd
            summary.total_storage_cost_usd += item.storage_cost_usd
            summary.total_transfer_cost_usd += item.transfer_cost_usd
            summary.total_platform_fee_usd += item.platform_fee_usd
            summary.total_storage_bytes += item.storage_bytes

            if item.has_actual_compute_cost:
                summary.has_actual_costs = True
            else:
                summary.estimated_worksets += 1

            summary.workset_items.append(item)

        # Calculate totals
        summary.total_storage_gb = self._bytes_to_gb(summary.total_storage_bytes)
        summary.grand_total_usd = self._round_currency(
            summary.total_compute_cost_usd +
            summary.total_storage_cost_usd +
            summary.total_transfer_cost_usd +
            summary.total_platform_fee_usd
        )

        # Round individual totals
        summary.total_compute_cost_usd = self._round_currency(summary.total_compute_cost_usd)
        summary.total_storage_cost_usd = self._round_currency(summary.total_storage_cost_usd)
        summary.total_transfer_cost_usd = self._round_currency(summary.total_transfer_cost_usd)
        summary.total_platform_fee_usd = self._round_currency(summary.total_platform_fee_usd)
        summary.total_storage_gb = self._round_currency(summary.total_storage_gb, 2)

        return summary

    def generate_invoice_data(
        self,
        customer_id: str,
        period_start: Optional[dt.datetime] = None,
        period_end: Optional[dt.datetime] = None,
    ) -> Dict[str, Any]:
        """Generate invoice-ready data for a customer.

        Args:
            customer_id: Customer identifier
            period_start: Start of billing period
            period_end: End of billing period

        Returns:
            Dict with invoice data suitable for rendering/export
        """
        summary = self.calculate_customer_billing(
            customer_id, period_start, period_end
        )

        # Build line items
        line_items = []
        for item in summary.workset_items:
            line_items.append({
                "workset_id": item.workset_id,
                "completed_at": item.completed_at,
                "samples": item.sample_count,
                "compute_usd": self._round_currency(item.compute_cost_usd, 2),
                "storage_gb": self._round_currency(item.storage_gb, 2),
                "storage_usd": self._round_currency(item.storage_cost_usd, 2),
                "transfer_usd": self._round_currency(item.transfer_cost_usd, 2),
                "platform_fee_usd": self._round_currency(item.platform_fee_usd, 2),
                "total_usd": self._round_currency(item.total_cost_usd, 2),
                "has_actual_cost": item.has_actual_compute_cost,
            })

        return {
            "customer_id": customer_id,
            "invoice_date": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "period_start": summary.period_start,
            "period_end": summary.period_end,
            "summary": {
                "total_worksets": summary.total_worksets,
                "billable_worksets": summary.billable_worksets,
                "total_samples": summary.total_samples,
                "total_storage_gb": summary.total_storage_gb,
                "compute_cost_usd": self._round_currency(summary.total_compute_cost_usd, 2),
                "storage_cost_usd": self._round_currency(summary.total_storage_cost_usd, 2),
                "transfer_cost_usd": self._round_currency(summary.total_transfer_cost_usd, 2),
                "platform_fee_usd": self._round_currency(summary.total_platform_fee_usd, 2),
                "grand_total_usd": self._round_currency(summary.grand_total_usd, 2),
            },
            "line_items": line_items,
            "rates": {
                "s3_storage_per_gb_month": self.rates.s3_storage_per_gb_month,
                "data_egress_per_gb": self.rates.data_egress_per_gb,
                "platform_fee_per_sample": self.rates.platform_fee_per_sample,
                "platform_fee_percentage": self.rates.platform_fee_percentage,
            },
            "accuracy": {
                "has_actual_costs": summary.has_actual_costs,
                "estimated_worksets": summary.estimated_worksets,
                "actual_worksets": summary.billable_worksets - summary.estimated_worksets,
            },
        }

