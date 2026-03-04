"""Cluster management routes for Daylily API.

Contains routes for ParallelCluster operations:
- GET /api/clusters
- DELETE /api/clusters/{cluster_name}
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from daylib.config import Settings
from daylib.security import sanitize_for_log

LOGGER = logging.getLogger("daylily.routes.clusters")


class ClusterDependencies:
    """Container for cluster route dependencies."""

    def __init__(
        self,
        settings: Settings,
        get_current_user,
    ):
        self.settings = settings
        self.get_current_user = get_current_user


def create_clusters_router(deps: ClusterDependencies) -> APIRouter:
    """Create cluster management router with injected dependencies."""
    router = APIRouter(tags=["clusters"])
    settings = deps.settings
    get_current_user = deps.get_current_user

    @router.get("/api/clusters")
    async def list_clusters(
        request: Request,
        refresh: bool = Query(False, description="Force refresh cluster cache"),
        fetch_status: bool = Query(False, description="Fetch SSH-based status (budget/jobs) for running headnodes"),
    ):
        """List all ParallelCluster instances across configured regions.

        Returns cluster information including status, head node details,
        compute fleet status, and relevant tags.

        Uses caching to reduce API calls (5 minute TTL by default).
        Set refresh=true to force a cache refresh.
        Set fetch_status=true to also fetch budget and job queue info via SSH.
        """
        try:
            from daylib.cluster_service import get_cluster_service
            from daylib.ursa_config import get_ursa_config

            ursa_config = get_ursa_config()
            if ursa_config.is_configured:
                allowed_regions = ursa_config.get_allowed_regions()
            else:
                allowed_regions = settings.get_allowed_regions()

            if not allowed_regions:
                return {
                    "clusters": [], "regions": [],
                    "error": "No regions configured. Create ~/.ursa/ursa-config.yaml with region definitions.",
                }

            service = get_cluster_service(
                regions=allowed_regions,
                aws_profile=ursa_config.aws_profile or settings.aws_profile,
                cache_ttl_seconds=300,
            )

            all_clusters = service.get_all_clusters_with_status(
                force_refresh=refresh, fetch_ssh_status=fetch_status,
            )
            clusters_dicts = [c.to_dict() for c in all_clusters]

            return {
                "clusters": clusters_dicts, "regions": allowed_regions,
                "total_count": len(clusters_dicts), "cached": not refresh,
                "status_fetched": fetch_status,
            }
        except Exception as e:
            LOGGER.error(f"Failed to list clusters: {e}")
            return {"clusters": [], "regions": [], "error": str(e)}

    @router.delete("/api/clusters/{cluster_name}")
    async def delete_cluster(
        cluster_name: str,
        region: str = Query(..., description="AWS region where the cluster is located"),
        current_user: Optional[Dict] = Depends(get_current_user),
    ):
        """Delete a ParallelCluster instance (admin-only)."""
        from daylib.file_api import _enforce_admin_only

        _enforce_admin_only(current_user, operation="delete clusters")

        try:
            from daylib.cluster_service import get_cluster_service
            from daylib.ursa_config import get_ursa_config

            ursa_config = get_ursa_config()
            if ursa_config.is_configured:
                allowed_regions = ursa_config.get_allowed_regions()
            else:
                allowed_regions = settings.get_allowed_regions()

            if not allowed_regions:
                raise HTTPException(
                    status_code=400,
                    detail="No regions configured. Create ~/.ursa/ursa-config.yaml with region definitions.",
                )

            if region not in allowed_regions:
                raise HTTPException(status_code=400, detail=f"Region '{region}' is not configured for this deployment")

            service = get_cluster_service(
                regions=allowed_regions,
                aws_profile=ursa_config.aws_profile or settings.aws_profile,
                cache_ttl_seconds=300,
            )

            result = service.delete_cluster(cluster_name, region)
            if "error" in result:
                raise HTTPException(status_code=500, detail=f"Failed to delete cluster: {result['error']}")

            cmd = f"pcluster delete-cluster --region {region} -n {cluster_name}"
            aws_profile = ursa_config.aws_profile or settings.aws_profile

            return {
                "success": True, "cluster_name": cluster_name, "region": region,
                "pcluster_command": cmd, "aws_profile": aws_profile,
                "message": f"Cluster '{cluster_name}' deletion initiated",
            }
        except HTTPException:
            raise
        except Exception as e:
            LOGGER.error("Failed to delete cluster %s: %s", sanitize_for_log(cluster_name), e)
            raise HTTPException(status_code=500, detail=str(e))

    return router
