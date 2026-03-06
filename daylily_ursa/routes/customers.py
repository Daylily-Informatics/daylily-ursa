"""Customer management routes for Daylily API.

Contains routes for customer CRUD operations:
- POST /api/v2/customers
- GET /api/v2/customers/{customer_id}
- PATCH /api/v2/customers/{customer_id}
- GET /api/v2/customers
- GET /api/v2/customers/{customer_id}/usage
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Request, status

from daylily_ursa.routes.dependencies import (
    CustomerCreate,
    CustomerUpdate,
    CustomerResponse,
)

if TYPE_CHECKING:
    from daylily_ursa.workset_customer import CustomerManager

LOGGER = logging.getLogger("daylily.routes.customers")


class CustomerDependencies:
    """Container for customer route dependencies."""

    def __init__(
        self,
        customer_manager: "CustomerManager",
        get_current_user,
    ):
        self.customer_manager = customer_manager
        self.get_current_user = get_current_user


def create_customers_router(deps: CustomerDependencies) -> APIRouter:
    """Create customer router with injected dependencies.

    Args:
        deps: CustomerDependencies container with all required dependencies

    Returns:
        Configured APIRouter with customer routes
    """
    router = APIRouter(tags=["customers"])
    customer_manager = deps.customer_manager
    get_current_user = deps.get_current_user

    @router.post(
        "/api/v2/customers", response_model=CustomerResponse, status_code=status.HTTP_201_CREATED
    )
    async def create_customer(
        customer: CustomerCreate,
        current_user: Optional[Dict] = Depends(get_current_user),
    ):
        """Create a new customer with provisioned resources."""
        config = customer_manager.onboard_customer(
            customer_name=customer.customer_name,
            email=customer.email,
            max_concurrent_worksets=customer.max_concurrent_worksets,
            max_storage_gb=customer.max_storage_gb,
            billing_account_id=customer.billing_account_id,
            cost_center=customer.cost_center,
        )

        return CustomerResponse(
            customer_id=config.customer_id,
            customer_name=config.customer_name,
            email=config.email,
            s3_bucket=config.s3_bucket,
            max_concurrent_worksets=config.max_concurrent_worksets,
            max_storage_gb=config.max_storage_gb,
            billing_account_id=config.billing_account_id,
            cost_center=config.cost_center,
        )

    @router.get("/api/v2/customers/{customer_id}", response_model=CustomerResponse)
    async def get_customer(
        customer_id: str,
        current_user: Optional[Dict] = Depends(get_current_user),
    ):
        """Get customer details."""
        config = customer_manager.get_customer_config(customer_id)
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Customer {customer_id} not found",
            )

        return CustomerResponse(
            customer_id=config.customer_id,
            customer_name=config.customer_name,
            email=config.email,
            s3_bucket=config.s3_bucket,
            max_concurrent_worksets=config.max_concurrent_worksets,
            max_storage_gb=config.max_storage_gb,
            billing_account_id=config.billing_account_id,
            cost_center=config.cost_center,
        )

    @router.patch("/api/v2/customers/{customer_id}", response_model=CustomerResponse)
    async def update_customer(
        customer_id: str,
        update: CustomerUpdate,
        request: Request,
        current_user: Optional[Dict] = Depends(get_current_user),
    ):
        """Update customer details (partial update).

        Customers can only update their own account.
        Admins can update any account.
        """
        # Get session customer for authorization check
        session_customer_id = (
            request.session.get("customer_id") if hasattr(request, "session") else None
        )
        session_email = request.session.get("user_email") if hasattr(request, "session") else None

        # Check if user is admin
        is_admin = False
        if session_email:
            session_customer = customer_manager.get_customer_by_email(session_email)
            if session_customer and session_customer.is_admin:
                is_admin = True

        # Authorization: must be own account or admin
        if not is_admin and session_customer_id != customer_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only update your own account",
            )

        # Perform the update
        config = customer_manager.update_customer(
            customer_id=customer_id,
            customer_name=update.customer_name,
            email=update.email,
            billing_account_id=update.billing_account_id,
            cost_center=update.cost_center,
            max_concurrent_worksets=update.max_concurrent_worksets,
            max_storage_gb=update.max_storage_gb,
        )

        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Customer {customer_id} not found",
            )

        return CustomerResponse(
            customer_id=config.customer_id,
            customer_name=config.customer_name,
            email=config.email,
            s3_bucket=config.s3_bucket,
            max_concurrent_worksets=config.max_concurrent_worksets,
            max_storage_gb=config.max_storage_gb,
            billing_account_id=config.billing_account_id,
            cost_center=config.cost_center,
        )

    @router.get("/api/v2/customers", response_model=List[CustomerResponse])
    async def list_customers(
        current_user: Optional[Dict] = Depends(get_current_user),
    ):
        """List all customers."""
        configs = customer_manager.list_customers()

        return [
            CustomerResponse(
                customer_id=c.customer_id,
                customer_name=c.customer_name,
                email=c.email,
                s3_bucket=c.s3_bucket,
                max_concurrent_worksets=c.max_concurrent_worksets,
                max_storage_gb=c.max_storage_gb,
                billing_account_id=c.billing_account_id,
                cost_center=c.cost_center,
            )
            for c in configs
        ]

    @router.get("/api/v2/customers/{customer_id}/usage")
    async def get_customer_usage(
        customer_id: str,
        current_user: Optional[Dict] = Depends(get_current_user),
    ):
        """Get customer resource usage."""
        usage = customer_manager.get_customer_usage(customer_id)
        if not usage:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Customer {customer_id} not found",
            )

        return usage

    return router
