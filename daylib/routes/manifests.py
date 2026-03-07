"""Manifest management routes for Daylily API.

Contains routes for customer manifest operations:
- GET /api/customers/{customer_id}/manifests
- POST /api/customers/{customer_id}/manifests
- GET /api/customers/{customer_id}/manifests/{manifest_id}
- GET /api/customers/{customer_id}/manifests/{manifest_id}/download
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

from fastapi import APIRouter, Body, HTTPException, Query, status
from fastapi.responses import Response

if TYPE_CHECKING:
    from daylib.workset_customer import CustomerManager
    from daylib.manifest_registry import ManifestRegistry

LOGGER = logging.getLogger("daylily.routes.manifests")


class ManifestDependencies:
    """Container for manifest route dependencies."""

    def __init__(
        self,
        customer_manager: "CustomerManager",
        manifest_registry: Optional["ManifestRegistry"] = None,
    ):
        self.customer_manager = customer_manager
        self.manifest_registry = manifest_registry


def create_manifests_router(deps: ManifestDependencies) -> APIRouter:
    """Create manifest router with injected dependencies."""
    router = APIRouter(tags=["customer-manifests"])
    customer_manager = deps.customer_manager
    manifest_registry = deps.manifest_registry

    # Import ManifestTooLargeError for exception handling
    try:
        from daylib.manifest_registry import ManifestTooLargeError
    except ImportError:
        ManifestTooLargeError = None  # type: ignore[misc, assignment]

    def _check_manifest_registry() -> "ManifestRegistry":
        if manifest_registry is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Manifest storage not configured",
            )
        return manifest_registry

    @router.get("/api/customers/{customer_id}/manifests")
    async def list_customer_manifests(
        customer_id: str,
        limit: int = Query(200, ge=1, le=500),
    ):
        """List saved manifests for a customer (metadata only)."""
        if not customer_manager:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Customer management not configured")
        registry = _check_manifest_registry()

        config = customer_manager.get_customer_config(customer_id)
        if not config:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Customer {customer_id} not found")

        manifests = registry.list_customer_manifests(customer_id, limit=limit)
        return {"manifests": manifests}

    @router.post(
        "/api/customers/{customer_id}/manifests",
        status_code=status.HTTP_201_CREATED,
    )
    async def save_customer_manifest(
        customer_id: str,
        tsv_content: str = Body(..., embed=True),
        name: Optional[str] = Body(None, embed=True),
        description: Optional[str] = Body(None, embed=True),
    ):
        """Save a stage_samples.tsv manifest for later reuse."""
        registry = _check_manifest_registry()
        config = customer_manager.get_customer_config(customer_id)
        if not config:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Customer {customer_id} not found")

        try:
            saved = registry.save_manifest(
                customer_id=customer_id,
                tsv_content=tsv_content,
                name=name,
                description=description,
            )
        except Exception as e:
            if ManifestTooLargeError is not None and isinstance(e, ManifestTooLargeError):
                raise HTTPException(status_code=413, detail=str(e))
            if isinstance(e, ValueError):
                raise HTTPException(status_code=400, detail=str(e))
            raise

        manifest_ref = saved.manifest_euid or saved.manifest_id
        download_url = f"/api/customers/{customer_id}/manifests/{manifest_ref}/download"
        return {
            "manifest": saved.to_metadata_dict(),
            "download_url": download_url,
        }

    @router.get("/api/customers/{customer_id}/manifests/{manifest_id}")
    async def get_customer_manifest_metadata(customer_id: str, manifest_id: str):
        """Get saved manifest metadata (not content)."""
        registry = _check_manifest_registry()
        config = customer_manager.get_customer_config(customer_id)
        if not config:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Customer {customer_id} not found")

        m = registry.get_manifest(customer_id=customer_id, manifest_id=manifest_id)
        if not m:
            raise HTTPException(status_code=404, detail="Manifest not found")
        return {"manifest": m.to_metadata_dict()}

    @router.get("/api/customers/{customer_id}/manifests/{manifest_id}/download")
    async def download_customer_manifest(customer_id: str, manifest_id: str):
        """Download the saved stage_samples.tsv content."""
        registry = _check_manifest_registry()
        config = customer_manager.get_customer_config(customer_id)
        if not config:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Customer {customer_id} not found")

        tsv = registry.get_manifest_tsv(customer_id=customer_id, manifest_id=manifest_id)
        if tsv is None:
            raise HTTPException(status_code=404, detail="Manifest not found")

        filename = f"{manifest_id}.stage_samples.tsv"
        return Response(
            content=tsv,
            media_type="text/tab-separated-values",
            headers={"Content-Disposition": f"attachment; filename=\"{filename}\""},
        )

    return router

