"""Cluster management routes for Daylily API.

Contains routes for ParallelCluster operations:
- GET /api/v2/clusters
- POST /api/v2/clusters/create
- GET /api/v2/clusters/create/jobs
- GET /api/v2/clusters/create/jobs/{job_id}
- GET /api/v2/clusters/create/jobs/{job_id}/logs
- GET /api/v2/clusters/create/options
- DELETE /api/v2/clusters/{cluster_name}
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

import boto3
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from daylily_ursa.config import Settings
from daylily_ursa.security import sanitize_for_log

LOGGER = logging.getLogger("daylily.routes.clusters")

# AWS ParallelCluster name pattern: alphanumeric start, up to 60 chars, alphanumeric/hyphen.
_CLUSTER_NAME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9-]{0,59}$")


def _find_region_for_region_az(region_az: str, *, allowed_regions: List[str]) -> Optional[str]:
    # region_az can be "us-west-2a" or "us-west-2-lax-1a". Choose the best
    # match from configured regions (longest prefix).
    candidates = [r for r in allowed_regions if region_az.startswith(r)]
    if not candidates:
        return None
    return max(candidates, key=len)


def _normalise_s3_bucket_location(location: Optional[str]) -> str:
    if not location:
        return "us-east-1"
    if location == "EU":
        return "eu-west-1"
    return location


class ClusterCreateRequest(BaseModel):
    """Request model for creating a ParallelCluster via daylily-ephemeral-cluster."""

    region_az: str = Field(..., description="Availability zone (e.g. us-west-2a)")
    cluster_name: str = Field(..., description="ParallelCluster name (max 60 chars, alnum + '-')")
    ssh_key_name: str = Field(..., description="EC2 keypair name to attach to headnode")
    s3_bucket_name: str = Field(
        ..., description="S3 bucket name used by daylily-ec for budget/reference config"
    )
    config_path: Optional[str] = Field(
        None, description="Optional existing daylily-ec config YAML path"
    )
    pass_on_warn: bool = Field(False, description="Proceed on warnings in daylily-ec validation")
    debug: bool = Field(False, description="Enable verbose daylily-ec debug output")


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

    def _get_allowed_regions_and_profile() -> tuple[List[str], Optional[str]]:
        from daylily_ursa.ursa_config import get_ursa_config

        ursa_config = get_ursa_config()
        if ursa_config.is_configured:
            allowed = ursa_config.get_allowed_regions()
            profile = ursa_config.aws_profile or settings.aws_profile
        else:
            allowed = settings.get_allowed_regions()
            profile = settings.aws_profile
        return (allowed or []), profile

    @router.get("/api/v2/clusters")
    async def list_clusters(
        request: Request,
        refresh: bool = Query(False, description="Force refresh cluster cache"),
        fetch_status: bool = Query(
            False, description="Fetch SSH-based status (budget/jobs) for running headnodes"
        ),
    ):
        """List all ParallelCluster instances across configured regions.

        Returns cluster information including status, head node details,
        compute fleet status, and relevant tags.

        Uses caching to reduce API calls (5 minute TTL by default).
        Set refresh=true to force a cache refresh.
        Set fetch_status=true to also fetch budget and job queue info via SSH.
        """
        try:
            from daylily_ursa.cluster_service import get_cluster_service
            from daylily_ursa.ursa_config import get_ursa_config

            ursa_config = get_ursa_config()
            if ursa_config.is_configured:
                allowed_regions = ursa_config.get_allowed_regions()
            else:
                allowed_regions = settings.get_allowed_regions()

            if not allowed_regions:
                return {
                    "clusters": [],
                    "regions": [],
                    "error": "No regions configured. Create ~/.ursa/ursa-config.yaml with region definitions.",
                }

            service = get_cluster_service(
                regions=allowed_regions,
                aws_profile=ursa_config.aws_profile or settings.aws_profile,
                cache_ttl_seconds=300,
            )

            all_clusters = service.get_all_clusters_with_status(
                force_refresh=refresh,
                fetch_ssh_status=fetch_status,
            )
            clusters_dicts = [c.to_dict() for c in all_clusters]

            return {
                "clusters": clusters_dicts,
                "regions": allowed_regions,
                "total_count": len(clusters_dicts),
                "cached": not refresh,
                "status_fetched": fetch_status,
            }
        except Exception as e:
            LOGGER.error(f"Failed to list clusters: {e}")
            return {"clusters": [], "regions": [], "error": str(e)}

    @router.post("/api/v2/clusters/create")
    async def create_cluster(
        create_req: ClusterCreateRequest,
        current_user: Optional[Dict] = Depends(get_current_user),
    ):
        """Create a ParallelCluster (admin-only) via daylily-ephemeral-cluster.

        This launches the external daylily-ec workflow as an Ursa-owned background job.
        """
        from daylily_ursa.file_api import _enforce_admin_only

        _enforce_admin_only(current_user, operation="create clusters")

        allowed_regions, aws_profile = _get_allowed_regions_and_profile()
        if not allowed_regions:
            raise HTTPException(
                status_code=400,
                detail="No regions configured. Create ~/.ursa/ursa-config.yaml with region definitions.",
            )

        region_az = str(create_req.region_az).strip()
        if not region_az:
            raise HTTPException(status_code=400, detail="region_az is required")

        region = _find_region_for_region_az(region_az, allowed_regions=allowed_regions)
        if not region:
            raise HTTPException(
                status_code=400,
                detail=f"Availability zone '{region_az}' does not match any configured region: {allowed_regions}",
            )

        cluster_name = str(create_req.cluster_name).strip()
        if not cluster_name or not _CLUSTER_NAME_PATTERN.match(cluster_name):
            raise HTTPException(
                status_code=400,
                detail="Invalid cluster_name. Must start with a letter and contain only letters, numbers, and '-' (max 60 chars).",
            )

        try:
            from daylily_ursa.ephemeral_cluster import runner as ec_runner

            contact_email = None
            if current_user:
                email = str(current_user.get("email") or "").strip()
                if email:
                    contact_email = email

            job = ec_runner.start_create_job(
                region_az=region_az,
                cluster_name=cluster_name,
                ssh_key_name=str(create_req.ssh_key_name).strip(),
                s3_bucket_name=str(create_req.s3_bucket_name).strip(),
                aws_profile=aws_profile,
                contact_email=contact_email,
                config_path_override=create_req.config_path,
                pass_on_warn=bool(create_req.pass_on_warn),
                debug=bool(create_req.debug),
            )
        except FileNotFoundError as e:
            # Missing script or config override path.
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            LOGGER.error("Failed to start cluster create job: %s", e)
            raise HTTPException(status_code=500, detail="Failed to start cluster create job")

        return {
            "job_id": job.job_id,
            "cluster_name": job.cluster_name,
            "region_az": job.region_az,
            "aws_profile": job.aws_profile,
            "log_path": str(job.log_path),
            "job_status_url": f"/api/v2/clusters/create/jobs/{job.job_id}",
            "job_logs_url": f"/api/v2/clusters/create/jobs/{job.job_id}/logs",
        }

    @router.get("/api/v2/clusters/create/jobs")
    async def list_cluster_create_jobs(
        limit: int = Query(20, ge=1, le=100),
        current_user: Optional[Dict] = Depends(get_current_user),
    ):
        """List recent cluster-create jobs (admin-only)."""
        from daylily_ursa.file_api import _enforce_admin_only

        _enforce_admin_only(current_user, operation="list cluster create jobs")
        from daylily_ursa.ephemeral_cluster import runner as ec_runner

        return {"jobs": ec_runner.list_cluster_create_jobs(limit=limit)}

    @router.get("/api/v2/clusters/create/jobs/{job_id}")
    async def get_cluster_create_job(
        job_id: str,
        current_user: Optional[Dict] = Depends(get_current_user),
    ):
        """Get a single cluster-create job record (admin-only)."""
        from daylily_ursa.file_api import _enforce_admin_only

        _enforce_admin_only(current_user, operation="read cluster create jobs")
        from daylily_ursa.ephemeral_cluster import runner as ec_runner

        try:
            return ec_runner.read_cluster_create_job(job_id)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @router.get("/api/v2/clusters/create/jobs/{job_id}/logs")
    async def get_cluster_create_job_logs(
        job_id: str,
        lines: int = Query(200, ge=1, le=2000),
        current_user: Optional[Dict] = Depends(get_current_user),
    ):
        """Tail the cluster-create job log (admin-only)."""
        from daylily_ursa.file_api import _enforce_admin_only

        _enforce_admin_only(current_user, operation="read cluster create job logs")
        from daylily_ursa.ephemeral_cluster import runner as ec_runner

        try:
            log_text = ec_runner.tail_job_log(job_id, lines=lines)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return {"job_id": job_id, "lines": lines, "log": log_text}

    @router.get("/api/v2/clusters/create/options")
    async def get_cluster_create_options(
        region: str = Query(..., description="AWS region for keypair lookup and bucket filtering"),
        current_user: Optional[Dict] = Depends(get_current_user),
    ):
        """Fetch create form options (admin-only).

        Returns keypairs in the selected region and S3 buckets in that region
        filtered to the daylily-ec naming convention (contains 'omics-analysis').
        """
        from daylily_ursa.file_api import _enforce_admin_only

        _enforce_admin_only(current_user, operation="fetch cluster create options")

        allowed_regions, aws_profile = _get_allowed_regions_and_profile()
        if not allowed_regions:
            raise HTTPException(
                status_code=400,
                detail="No regions configured. Create ~/.ursa/ursa-config.yaml with region definitions.",
            )
        if region not in allowed_regions:
            raise HTTPException(
                status_code=400, detail=f"Region '{region}' is not configured for this deployment"
            )

        sess = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()

        keypairs: List[str] = []
        try:
            ec2 = sess.client("ec2", region_name=region)
            resp = ec2.describe_key_pairs()
            for kp in resp.get("KeyPairs", []):
                name = kp.get("KeyName")
                if isinstance(name, str) and name:
                    keypairs.append(name)
            keypairs = sorted(set(keypairs))
        except Exception as e:  # pragma: no cover - best-effort options endpoint
            LOGGER.debug("Failed to list keypairs for %s: %s", region, e)

        buckets: List[str] = []
        try:
            s3 = sess.client("s3", region_name=region)
            resp = s3.list_buckets()
            for b in resp.get("Buckets", []):
                name = b.get("Name")
                if not isinstance(name, str) or not name:
                    continue
                if "omics-analysis" not in name:
                    continue
                try:
                    loc = s3.get_bucket_location(Bucket=name).get("LocationConstraint")
                    if _normalise_s3_bucket_location(loc) == region:
                        buckets.append(name)
                except Exception:
                    continue
            buckets = sorted(set(buckets))
        except Exception as e:  # pragma: no cover - best-effort options endpoint
            LOGGER.debug("Failed to list buckets for %s: %s", region, e)

        return {
            "aws_profile": aws_profile,
            "region": region,
            "keypairs": keypairs,
            "buckets": buckets,
        }

    @router.delete("/api/v2/clusters/{cluster_name}")
    async def delete_cluster(
        cluster_name: str,
        region: str = Query(..., description="AWS region where the cluster is located"),
        current_user: Optional[Dict] = Depends(get_current_user),
    ):
        """Delete a ParallelCluster instance (admin-only)."""
        from daylily_ursa.file_api import _enforce_admin_only

        _enforce_admin_only(current_user, operation="delete clusters")

        try:
            from daylily_ursa.cluster_service import get_cluster_service
            from daylily_ursa.ursa_config import get_ursa_config

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
                raise HTTPException(
                    status_code=400,
                    detail=f"Region '{region}' is not configured for this deployment",
                )

            service = get_cluster_service(
                regions=allowed_regions,
                aws_profile=ursa_config.aws_profile or settings.aws_profile,
                cache_ttl_seconds=300,
            )

            result = service.delete_cluster(cluster_name, region)
            if "error" in result:
                raise HTTPException(
                    status_code=500, detail=f"Failed to delete cluster: {result['error']}"
                )

            cmd = f"pcluster delete-cluster --region {region} -n {cluster_name}"
            aws_profile = ursa_config.aws_profile or settings.aws_profile

            return {
                "success": True,
                "cluster_name": cluster_name,
                "region": region,
                "pcluster_command": cmd,
                "aws_profile": aws_profile,
                "message": f"Cluster '{cluster_name}' deletion initiated",
            }
        except HTTPException:
            raise
        except Exception as e:
            LOGGER.error("Failed to delete cluster %s: %s", sanitize_for_log(cluster_name), e)
            raise HTTPException(status_code=500, detail=str(e))

    return router
