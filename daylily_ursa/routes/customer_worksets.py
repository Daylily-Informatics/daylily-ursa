"""Customer workset routes for Daylily API.

Contains routes tagged 'customer-worksets' for managing customer-scoped worksets:
- List, get, create, cancel, retry, archive, delete, restore worksets
- Workset logs and performance metrics
- Snakemake log download
"""

from __future__ import annotations

import logging
import re
import hashlib
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import boto3
import yaml  # type: ignore[import-untyped]

from fastapi import APIRouter, Body, HTTPException, Query, Request, status
from fastapi.responses import Response

from daylily_ursa.config import Settings
from daylily_ursa.security import sanitize_for_log
from daylily_ursa.workset_state_db import WorksetPriority, WorksetState, WorksetStateDB
from daylily_ursa.routes.dependencies import verify_workset_ownership

# Pipeline status monitoring
try:
    from daylily_ursa.pipeline_status import PipelineStatusFetcher, PipelineStatus
    PIPELINE_STATUS_AVAILABLE = True
except ImportError:
    PIPELINE_STATUS_AVAILABLE = False
    PipelineStatusFetcher = None  # type: ignore[misc, assignment]
    PipelineStatus = None  # type: ignore[misc, assignment]

# Optional integration layer import
try:
    from daylily_ursa.workset_integration import WorksetIntegration
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False
    WorksetIntegration = None  # type: ignore[misc, assignment]

# Manifest storage imports
try:
    from daylily_ursa.manifest_registry import ManifestRegistry, ManifestTooLargeError
    MANIFEST_STORAGE_AVAILABLE = True
except ImportError:
    MANIFEST_STORAGE_AVAILABLE = False
    ManifestRegistry = None  # type: ignore[misc, assignment]
    ManifestTooLargeError = None  # type: ignore[misc, assignment]

if TYPE_CHECKING:
    from daylily_ursa.workset_customer import CustomerManager

LOGGER = logging.getLogger("daylily.routes.customer_worksets")


class CustomerWorksetDependencies:
    """Container for customer workset route dependencies."""

    def __init__(
        self,
        state_db: WorksetStateDB,
        settings: Settings,
        customer_manager: "CustomerManager",
        integration: Optional[Any] = None,
        manifest_registry: Optional[Any] = None,
        get_current_user: Optional[Callable] = None,
    ):
        self.state_db = state_db
        self.settings = settings
        self.customer_manager = customer_manager
        self.integration = integration
        self.manifest_registry = manifest_registry
        self.get_current_user = get_current_user


def create_customer_worksets_router(deps: CustomerWorksetDependencies) -> APIRouter:
    """Create customer worksets router with injected dependencies.

    Args:
        deps: CustomerWorksetDependencies container with all required dependencies

    Returns:
        Configured APIRouter with customer workset routes
    """
    router = APIRouter(tags=["customer-worksets"])

    # Alias deps for brevity in route handlers
    state_db = deps.state_db
    customer_manager = deps.customer_manager
    settings = deps.settings
    integration = deps.integration
    manifest_registry = deps.manifest_registry
    get_current_user = deps.get_current_user

    @router.get("/api/v2/customers/{customer_id}/worksets")
    async def list_customer_worksets(
        customer_id: str,
        state: Optional[str] = None,
        limit: int = 100,
    ):
        """List worksets for a customer.

        Filters worksets by customer_id ownership (customer_id field or metadata.submitted_by).
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

        # Get all worksets and filter by customer_id ownership
        all_worksets: List[Dict[str, Any]] = []
        if state:
            try:
                ws_state = WorksetState(state)
                all_worksets = state_db.list_worksets_by_state(ws_state, limit=limit * 5)
            except ValueError:
                for ws_state in WorksetState:
                    batch = state_db.list_worksets_by_state(ws_state, limit=limit * 5)
                    all_worksets.extend(batch)
        else:
            for ws_state in WorksetState:
                batch = state_db.list_worksets_by_state(ws_state, limit=limit * 5)
                all_worksets.extend(batch)

        # SECURITY: Filter to only this customer's worksets (by customer_id, not bucket)
        customer_worksets = [
            w for w in all_worksets
            if verify_workset_ownership(w, customer_id)
        ]

        return {"worksets": customer_worksets[:limit]}

    @router.get("/api/v2/customers/{customer_id}/worksets/archived")
    async def list_archived_worksets(customer_id: str):
        """List all archived worksets for a customer."""
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

        # SECURITY: Filter archived worksets by customer_id ownership (not bucket)
        all_archived = state_db.list_archived_worksets(limit=500)
        customer_archived = [
            w for w in all_archived
            if verify_workset_ownership(w, customer_id)
        ]
        return customer_archived

    @router.get("/api/v2/customers/{customer_id}/worksets/{euid}")
    async def get_customer_workset(
        customer_id: str,
        euid: str,
    ):
        """Get a specific workset for a customer."""
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

        workset = state_db.get_workset(euid)
        if not workset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workset {euid} not found",
            )

        # Verify workset belongs to this customer (by customer_id, not bucket)
        if not verify_workset_ownership(workset, customer_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Workset does not belong to this customer",
            )

        return workset

    @router.post("/api/v2/customers/{customer_id}/worksets")
    async def create_customer_workset(
        request: Request,
        customer_id: str,
        workset_name: str = Body(..., embed=True),
        pipeline_type: str = Body(..., embed=True),
        reference_genome: str = Body(..., embed=True),
        s3_prefix: str = Body("", embed=True),
        priority: str = Body("normal", embed=True),
        workset_type: str = Body("ruo", embed=True),
        notification_email: Optional[str] = Body(None, embed=True),
        enable_qc: bool = Body(True, embed=True),
        archive_results: bool = Body(True, embed=True),
        s3_bucket: Optional[str] = Body(None, embed=True),
        samples: Optional[List[Dict[str, Any]]] = Body(None, embed=True),
        yaml_content: Optional[str] = Body(None, embed=True),
        manifest_id: Optional[str] = Body(None, embed=True),
        manifest_tsv_content: Optional[str] = Body(None, embed=True),
        preferred_cluster: Optional[str] = Body(None, embed=True),
    ):
        """Create a new workset for a customer from the portal form.

        This endpoint registers the workset in both TapDB (for UI state tracking)
        and writes S3 sentinel files (for processing engine discovery).

        Samples can be provided via:
        - samples: Direct list of sample dicts
        - yaml_content: YAML with samples array
        - manifest_id: ID of a saved manifest (retrieves TSV from ManifestRegistry)
        - manifest_tsv_content: Raw stage_samples.tsv content

        Bucket is determined from the selected cluster's tags:
        - The cluster's aws-parallelcluster-monitor-bucket tag specifies the S3 bucket
        - A cluster must be selected for workset creation
        """
        from daylily_ursa.ursa_config import get_ursa_config

        if not customer_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Customer management not configured",
            )

        # Validate customer_id - reject null, empty, or 'Unknown'
        if not customer_id or customer_id.strip() == "" or customer_id == "Unknown":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Valid customer ID is required. Please log in with a registered account.",
            )

        config = customer_manager.get_customer_config(customer_id)
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Customer {customer_id} not found",
            )

        # Generate a unique workset name from the user-provided name with date suffix
        import datetime as dt
        safe_name = workset_name.replace(" ", "-").lower()[:30]
        now_utc = dt.datetime.now(dt.timezone.utc)
        date_suffix = now_utc.strftime("%Y%m%d")
        time_hash = hashlib.sha256(f"{safe_name}-{now_utc.isoformat()}-{customer_id}".encode()).hexdigest()[:8]
        ws_name = f"{safe_name}-{time_hash}-{date_suffix}"

        # Determine bucket from cluster tags (aws-parallelcluster-monitor-bucket)
        # A cluster MUST be selected - bucket is discovered from its tags
        bucket = None
        cluster_region = None
        ursa_config = get_ursa_config()

        if not preferred_cluster:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A cluster must be selected for workset creation. "
                       "The S3 bucket is derived from the cluster's aws-parallelcluster-monitor-bucket tag.",
            )

        # Look up cluster to get region and bucket from tags
        try:
            from daylily_ursa.cluster_service import get_cluster_service, ClusterInfo
            service = get_cluster_service(
                regions=ursa_config.get_allowed_regions() or settings.get_allowed_regions(),
                aws_profile=ursa_config.aws_profile or settings.aws_profile,
            )
            # Get cluster info (includes tags with bucket)
            cluster_info = service.get_cluster_by_name(preferred_cluster, force_refresh=False)
            if not cluster_info:
                # Try refreshing cache in case cluster was just created
                cluster_info = service.get_cluster_by_name(preferred_cluster, force_refresh=True)

            if cluster_info:
                cluster_region = cluster_info.region
                bucket = cluster_info.get_monitor_bucket_name()
                if bucket:
                    LOGGER.info(
                        "Using bucket %s from cluster %s tag (region %s)",
                        sanitize_for_log(bucket), sanitize_for_log(preferred_cluster), cluster_region
                    )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Cluster '{preferred_cluster}' does not have a monitor bucket tag set. "
                               f"Clusters must have the '{ClusterInfo.MONITOR_BUCKET_TAG}' tag "
                               f"with the S3 bucket URI (e.g., s3://your-bucket).",
                    )
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Cluster '{preferred_cluster}' not found in configured regions. "
                           f"Scanned regions: {', '.join(ursa_config.get_allowed_regions() or settings.get_allowed_regions())}",
                )
        except HTTPException:
            raise
        except Exception as e:
            LOGGER.error("Failed to look up cluster %s: %s", sanitize_for_log(preferred_cluster), e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to query cluster metadata: {e}",
            )

        # Use provided prefix or generate one based on workset ID
        prefix = s3_prefix.strip() if s3_prefix else ""
        # Strip s3:// prefix from prefix if provided
        if prefix.startswith("s3://"):
            prefix = prefix[5:]
            # Extract bucket and prefix if full S3 URI was provided
            if "/" in prefix:
                parts = prefix.split("/", 1)
                # bucket = parts[0]  # Could use this if needed
                prefix = parts[1]
        if not prefix:
            prefix = f"worksets/{ws_name}/"
        if not prefix.endswith("/"):
            prefix += "/"

        # Process samples from various sources (priority: samples > manifest_id > manifest_tsv_content > yaml_content)
        workset_samples = samples or []
        manifest_tsv_for_s3 = None  # Raw TSV content to write to S3

        # Try manifest_id first (if no direct samples provided)
        if manifest_id and not workset_samples:
            if not manifest_registry:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Manifest storage not configured; cannot use manifest_id",
                )
            tsv = manifest_registry.get_manifest_tsv(customer_id=customer_id, manifest_id=manifest_id)
            if not tsv:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Manifest {manifest_id} not found for customer {customer_id}",
                )
            # Import parse function
            from daylily_ursa.manifest_registry import parse_tsv_to_samples
            workset_samples = parse_tsv_to_samples(tsv)
            manifest_tsv_for_s3 = tsv
            LOGGER.info("Loaded %d samples from saved manifest %s", len(workset_samples), manifest_id)

        # Try manifest_tsv_content next
        if manifest_tsv_content and not workset_samples:
            from daylily_ursa.manifest_registry import parse_tsv_to_samples
            workset_samples = parse_tsv_to_samples(manifest_tsv_content)
            manifest_tsv_for_s3 = manifest_tsv_content
            LOGGER.info("Parsed %d samples from provided TSV content", len(workset_samples))

        # Finally try YAML content
        if yaml_content and not workset_samples:
            try:
                yaml_data = yaml.safe_load(yaml_content)
                if yaml_data and isinstance(yaml_data.get("samples"), list):
                    workset_samples = yaml_data["samples"]
            except Exception as e:
                LOGGER.warning("Failed to parse YAML content: %s", str(e))

        # Normalize sample format and add default status
        normalized_samples = []
        for sample in workset_samples:
            if isinstance(sample, dict):
                normalized = {
                    "sample_id": sample.get("sample_id") or sample.get("id") or sample.get("name", "unknown"),
                    "r1_file": sample.get("r1_file") or sample.get("r1") or sample.get("fq1", ""),
                    "r2_file": sample.get("r2_file") or sample.get("r2") or sample.get("fq2", ""),
                    "status": sample.get("status", "pending"),
                }
                normalized_samples.append(normalized)

        # Issue 4: Validate that workset has at least one sample
        if len(normalized_samples) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Workset must contain at least one sample. Please upload files, specify an S3 path with samples, provide a saved manifest ID, or upload a manifest TSV.",
            )

        # Parse workset_type with fallback to RUO
        from daylily_ursa.workset_state_db import WorksetType
        try:
            ws_type = WorksetType(workset_type.lower())
        except ValueError:
            ws_type = WorksetType.RUO

        current_user_data = get_current_user(request) if get_current_user else None
        created_by_email = None
        if current_user_data and isinstance(current_user_data, dict):
            created_by_email = current_user_data.get("email")

        # Store additional metadata from the form
        metadata = {
            "workset_name": workset_name,
            "pipeline_type": pipeline_type,
            "reference_genome": reference_genome,
            "notification_email": notification_email,
            "enable_qc": enable_qc,
            "archive_results": archive_results,
            "submitted_by": customer_id,
            "priority": priority,
            "workset_type": ws_type.value,
            "samples": normalized_samples,
            "sample_count": len(normalized_samples),
            "data_bucket": config.s3_bucket,
            "data_buckets": [config.s3_bucket] if config.s3_bucket else [],
            "preferred_cluster": preferred_cluster,
            "cluster_region": cluster_region,
        }

        if created_by_email:
            metadata["created_by_email"] = created_by_email

        # If we have raw TSV content (from manifest), pass it for direct S3 write
        if manifest_tsv_for_s3:
            metadata["stage_samples_tsv"] = manifest_tsv_for_s3

        # Use integration layer for unified registration (TapDB + S3)
        # If no global integration exists but we have a bucket, create one ad-hoc
        effective_integration = integration
        if not effective_integration and bucket and INTEGRATION_AVAILABLE:
            LOGGER.info("Creating ad-hoc integration for bucket %s (region %s)", bucket, cluster_region)
            effective_integration = WorksetIntegration(
                state_db=state_db,
                bucket=bucket,
                region=cluster_region or settings.get_effective_region(),
                profile=ursa_config.aws_profile or settings.aws_profile,
            )

        if effective_integration:
            euid = effective_integration.register_workset(
                name=ws_name,
                bucket=bucket,
                prefix=prefix,
                priority=priority,
                workset_type=ws_type.value,
                metadata=metadata,
                customer_id=customer_id,
                preferred_cluster=preferred_cluster,
                cluster_region=cluster_region,
                write_s3=True,
                write_tapdb=True,
            )
        else:
            # Fallback to TapDB-only registration (no S3 files)
            LOGGER.warning("No integration layer available - S3 files will NOT be created")
            try:
                ws_priority = WorksetPriority(priority)
            except ValueError:
                ws_priority = WorksetPriority.NORMAL

            try:
                euid = state_db.register_workset(
                    name=ws_name,
                    bucket=bucket,
                    prefix=prefix,
                    priority=ws_priority,
                    workset_type=ws_type,
                    metadata=metadata,
                    customer_id=customer_id,
                    preferred_cluster=preferred_cluster,
                    cluster_region=cluster_region,
                )
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e),
                )

        if not euid:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Workset {ws_name} already exists",
            )

        created = state_db.get_workset(euid)
        return created

    @router.post("/api/v2/customers/{customer_id}/worksets/{euid}/cancel")
    async def cancel_customer_workset(
        customer_id: str,
        euid: str,
    ):
        """Cancel a customer's workset."""
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

        workset = state_db.get_workset(euid)
        if not workset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workset {euid} not found",
            )

        # Verify workset belongs to this customer (by customer_id, not bucket)
        if not verify_workset_ownership(workset, customer_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Workset does not belong to this customer",
            )

        state_db.update_state(euid=euid, new_state=WorksetState.CANCELED, reason="Canceled by user")
        updated = state_db.get_workset(euid)
        return updated

    @router.post("/api/v2/customers/{customer_id}/worksets/{euid}/retry")
    async def retry_customer_workset(
        customer_id: str,
        euid: str,
    ):
        """Retry a failed customer workset by creating a new cloned workset.

        Creates a new workset with a new datetime suffix, cloning all
        configuration from the original. The original workset is left
        unchanged for audit trail purposes.

        Returns:
            The newly created retry workset (not the original).
        """
        import datetime as dt

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

        original = state_db.get_workset(euid)
        if not original:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workset {euid} not found",
            )

        # Verify workset belongs to this customer (by customer_id, not bucket)
        if not verify_workset_ownership(original, customer_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Workset does not belong to this customer",
            )

        # Extract base name from original workset name (remove hash8-YYYYMMDD suffix)
        # Format: {safe-name}-{hash8}-{YYYYMMDD}
        original_name = original.get("name", "")
        match = re.match(r"^(.+)-[a-f0-9]{8}-\d{8}$", original_name)
        if match:
            base_name = match.group(1)
        else:
            # Fallback: use original workset name from metadata
            original_meta = original.get("metadata", {})
            base_name = original_meta.get("workset_name", original_name)[:30]
            base_name = base_name.replace(" ", "-").lower()

        # Generate new workset name with new datetime suffix
        now_utc = dt.datetime.now(dt.timezone.utc)
        date_suffix = now_utc.strftime("%Y%m%d")
        time_hash = hashlib.sha256(f"{base_name}-{now_utc.isoformat()}-{customer_id}".encode()).hexdigest()[:8]
        new_ws_name = f"{base_name}-{time_hash}-{date_suffix}"

        # Clone metadata from original workset
        original_meta = original.get("metadata", {})
        new_metadata = {
            "workset_name": original_meta.get("workset_name", base_name),
            "archive_results": original_meta.get("archive_results", True),
            "submitted_by": customer_id,
            "priority": original.get("priority", "normal"),
            "workset_type": original.get("workset_type", "ruo"),
            "samples": original_meta.get("samples", []),
            "sample_count": original_meta.get("sample_count", 0),
            "data_bucket": original_meta.get("data_bucket", config.s3_bucket),
            "data_buckets": original_meta.get("data_buckets", []),
            "preferred_cluster": original.get("preferred_cluster"),
            "cluster_region": original.get("cluster_region"),
            # Retry tracking metadata
            "retried_from_euid": euid,
            "retry_reason": "User requested retry",
        }

        # Preserve optional fields if present
        for key in ["created_by_email", "pipeline_type"]:
            if key in original_meta:
                new_metadata[key] = original_meta[key]

        # Determine bucket and prefix for new workset
        bucket = original.get("bucket") or config.s3_bucket
        prefix = f"worksets/{new_ws_name}/"
        priority = original.get("priority", "normal")
        workset_type_val = original.get("workset_type", "ruo")
        preferred_cluster = original.get("preferred_cluster")
        cluster_region = original.get("cluster_region")

        # Register the new workset using integration layer if available
        from daylily_ursa.ursa_config import get_ursa_config
        from daylily_ursa.workset_state_db import WorksetType

        ursa_config = get_ursa_config()
        effective_integration = integration
        if not effective_integration and bucket and INTEGRATION_AVAILABLE:
            effective_integration = WorksetIntegration(
                state_db=state_db,
                bucket=bucket,
                region=cluster_region or settings.get_effective_region(),
                profile=ursa_config.aws_profile or settings.aws_profile,
            )

        if effective_integration:
            new_euid = effective_integration.register_workset(
                name=new_ws_name,
                bucket=bucket,
                prefix=prefix,
                priority=priority,
                workset_type=workset_type_val,
                metadata=new_metadata,
                customer_id=customer_id,
                preferred_cluster=preferred_cluster,
                cluster_region=cluster_region,
                write_s3=True,
                write_tapdb=True,
            )
        else:
            # Fallback to TapDB-only registration
            try:
                ws_priority = WorksetPriority(priority)
            except ValueError:
                ws_priority = WorksetPriority.NORMAL
            try:
                ws_type = WorksetType(workset_type_val)
            except ValueError:
                ws_type = WorksetType.RUO

            new_euid = state_db.register_workset(
                name=new_ws_name,
                bucket=bucket,
                prefix=prefix,
                priority=ws_priority,
                workset_type=ws_type,
                metadata=new_metadata,
                customer_id=customer_id,
                preferred_cluster=preferred_cluster,
                cluster_region=cluster_region,
            )

        if not new_euid:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create retry workset",
            )

        # Update original workset metadata to track retry
        try:
            state_db.update_metadata(
                euid,
                {"retried_as_euid": new_euid},
            )
        except Exception as e:
            LOGGER.warning("Failed to update original workset with retry link: %s", e)

        # Return the new workset
        new_workset = state_db.get_workset(new_euid)
        return new_workset

    @router.post("/api/v2/customers/{customer_id}/worksets/{euid}/archive")
    async def archive_customer_workset(
        request: Request,
        customer_id: str,
        euid: str,
        reason: Optional[str] = Body(None, embed=True),
    ):
        """Archive a customer's workset.

        Moves workset to archived state. Archived worksets can be restored.
        Admins can archive any workset.
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

        workset = state_db.get_workset(euid)
        if not workset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workset {euid} not found",
            )

        # Check if user is admin (can archive any workset) or owns the workset (by customer_id)
        is_admin = getattr(request, "session", {}).get("is_admin", False)
        if not is_admin and not verify_workset_ownership(workset, customer_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Workset does not belong to this customer",
            )

        # Archive the workset
        success = state_db.archive_workset(
            euid, archived_by=customer_id, archive_reason=reason
        )
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to archive workset",
            )

        # Optionally move S3 files to archive prefix
        bucket = workset.get("bucket")
        prefix = workset.get("prefix", "").rstrip("/")
        if bucket and prefix and integration:
            try:
                archive_prefix = f"archived/{prefix.split('/')[-1]}/"
                s3 = boto3.client("s3")
                # Copy files to archive location
                paginator = s3.get_paginator("list_objects_v2")
                for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                    for obj in page.get("Contents", []):
                        old_key = obj["Key"]
                        new_key = old_key.replace(prefix, archive_prefix.rstrip("/"), 1)
                        s3.copy_object(
                            Bucket=bucket,
                            CopySource={"Bucket": bucket, "Key": old_key},
                            Key=new_key,
                        )
                        s3.delete_object(Bucket=bucket, Key=old_key)
                LOGGER.info("Moved workset %s files to archive: %s", euid, archive_prefix)
            except Exception as e:
                LOGGER.warning("Failed to move workset files to archive: %s", str(e))

        return state_db.get_workset(euid)

    @router.post("/api/v2/customers/{customer_id}/worksets/{euid}/delete")
    async def delete_customer_workset(
        request: Request,
        customer_id: str,
        euid: str,
        hard_delete: bool = Body(False, embed=True),
        reason: Optional[str] = Body(None, embed=True),
    ):
        """Delete a customer's workset.

        Args:
            hard_delete: If True, permanently removes all S3 data and TapDB record.
                        If False (default), marks as deleted but preserves data.

        Admins can delete any workset.
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

        workset = state_db.get_workset(euid)
        if not workset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workset {euid} not found",
            )

        # Check if user is admin (can delete any workset) or owns the workset (by customer_id)
        is_admin = getattr(request, "session", {}).get("is_admin", False)
        if not is_admin and not verify_workset_ownership(workset, customer_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Workset does not belong to this customer",
            )

        # If hard delete, remove S3 files first
        if hard_delete:
            bucket = workset.get("bucket")
            prefix = workset.get("prefix", "").rstrip("/") + "/"
            if bucket and prefix:
                try:
                    s3 = boto3.client("s3")
                    paginator = s3.get_paginator("list_objects_v2")
                    objects_to_delete = []
                    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                        for obj in page.get("Contents", []):
                            objects_to_delete.append({"Key": obj["Key"]})

                    if objects_to_delete:
                        # Delete in batches of 1000 (S3 limit)
                        for i in range(0, len(objects_to_delete), 1000):
                            batch = objects_to_delete[i:i + 1000]
                            s3.delete_objects(
                                Bucket=bucket,
                                Delete={"Objects": batch},
                            )
                        LOGGER.info(
                            "Deleted %d S3 objects for workset %s",
                            len(objects_to_delete),
                            euid,
                        )
                except Exception as e:
                    LOGGER.error("Failed to delete S3 objects for workset %s: %s", euid, str(e))
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Failed to delete S3 data: {str(e)}",
                    )

        # Update TapDB state
        success = state_db.delete_workset(
            euid,
            deleted_by=customer_id,
            delete_reason=reason,
            hard_delete=hard_delete,
        )
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete workset from database",
            )

        if hard_delete:
            return {"status": "deleted", "euid": euid, "hard_delete": True}
        return state_db.get_workset(euid)

    @router.post("/api/v2/customers/{customer_id}/worksets/{euid}/restore")
    async def restore_customer_workset(
        customer_id: str,
        euid: str,
    ):
        """Restore an archived workset back to ready state."""
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

        workset = state_db.get_workset(euid)
        if not workset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workset {euid} not found",
            )

        # Verify workset belongs to this customer (by customer_id, not bucket)
        if not verify_workset_ownership(workset, customer_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Workset does not belong to this customer",
            )

        if workset.get("state") != "archived":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only archived worksets can be restored",
            )

        success = state_db.restore_workset(euid, restored_by=customer_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to restore workset",
            )

        return state_db.get_workset(euid)

    @router.get("/api/v2/customers/{customer_id}/worksets/{euid}/logs")
    async def get_customer_workset_logs(
        customer_id: str,
        euid: str,
    ):
        """Get logs for a customer's workset including live pipeline status.

        Returns:
            - euid: The workset EUID
            - state_history: List of state transitions from TapDB
            - pipeline_status: Live status from headnode (null if unavailable)
              - is_running: Whether the tmux session is active
              - steps_completed: Number of Snakemake steps completed
              - steps_total: Total number of Snakemake steps
              - percent_complete: Completion percentage
              - current_rule: Currently executing Snakemake rule
              - duration_seconds: Pipeline runtime in seconds
              - storage_bytes: Size of analysis directory
              - recent_log_lines: Last 50 lines from Snakemake log
              - log_files: List of available Snakemake log files
              - errors: Error lines found in logs
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

        workset = state_db.get_workset(euid)
        if not workset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workset {euid} not found",
            )

        # Verify workset belongs to this customer (by customer_id, not bucket)
        if not verify_workset_ownership(workset, customer_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Workset does not belong to this customer",
            )

        # Get state history from TapDB
        history = workset.get("state_history", [])

        # Attempt to fetch live pipeline status from headnode
        pipeline_status = None
        if PIPELINE_STATUS_AVAILABLE and PipelineStatusFetcher is not None:
            workset_name = workset.get("name") or workset.get("workset_name")
            # Use cached headnode IP from TapDB (stored by monitor when workset started)
            headnode_ip = workset.get("execution_headnode_ip")

            if headnode_ip and workset_name:
                try:
                    # Get workset region for region-specific SSH key
                    workset_region = (
                        workset.get("execution_cluster_region")
                        or workset.get("cluster_region")
                        or settings.get_effective_region()
                    )

                    # Try to get region-specific SSH key from ursa_config
                    ssh_key = settings.pipeline_ssh_identity_file
                    try:
                        from daylily_ursa.ursa_config import get_ursa_config
                        ursa_cfg = get_ursa_config()
                        region_key = ursa_cfg.get_ssh_key_for_region(workset_region)
                        if region_key:
                            ssh_key = region_key
                            LOGGER.debug("Using region-specific SSH key for %s: %s", workset_region, ssh_key)
                    except Exception as e:
                        LOGGER.debug("Could not load ursa_config for region SSH key: %s", e)

                    # Create fetcher with region-aware SSH key
                    fetcher = PipelineStatusFetcher(
                        ssh_user=settings.pipeline_ssh_user,
                        ssh_identity_file=ssh_key,
                        timeout=settings.pipeline_ssh_timeout,
                        clone_dest_root=settings.pipeline_clone_dest_root,
                        repo_dir_name=settings.pipeline_repo_dir_name,
                    )

                    # Use headnode IP from TapDB (no pcluster call needed)
                    # Derive tmux session name (matches monitor convention)
                    tmux_session = f"daylily-{workset_name}"

                    # Fetch status
                    status_obj = fetcher.fetch_status(
                        headnode_ip=headnode_ip,
                        workset_name=workset_name,
                        tmux_session_name=tmux_session,
                    )
                    pipeline_status = status_obj.to_dict()
                except Exception as e:
                    LOGGER.warning(
                        "Failed to fetch pipeline status for %s: %s",
                        euid,
                        str(e),
                    )
                    # Graceful fallback - return null pipeline_status
                    pipeline_status = None

        return {
            "euid": euid,
            "state_history": history,
            "pipeline_status": pipeline_status,
        }

    @router.get("/api/v2/customers/{customer_id}/worksets/{euid}/performance-metrics")
    async def get_customer_workset_performance_metrics(
        customer_id: str,
        euid: str,
        force_refresh: bool = Query(False, description="Force refresh from headnode even if cached"),
    ):
        """Get performance metrics for a customer's workset.

        Metrics are pulled from the cluster headnode:
        - alignment_stats: Per-sample alignment quality metrics from alignstats_combo_mqc.tsv
        - benchmark_data: Per-rule performance metrics from rules_benchmark_data_singleton.tsv
        - cost_summary: Computed per-sample and total costs

        Caching behavior:
        - While workset is running/pending: fetches fresh data from headnode
        - Once complete/error: fetches once, then returns cached data
        - Use force_refresh=true to bypass cache
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

        workset = state_db.get_workset(euid)
        if not workset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workset {euid} not found",
            )

        # Verify workset belongs to this customer
        if not verify_workset_ownership(workset, customer_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Workset does not belong to this customer",
            )

        workset_state = workset.get("state", "")
        is_terminal_state = workset_state in ("complete", "error", "archived", "deleted")

        # Check for cached metrics first
        if not force_refresh:
            cached = state_db.get_performance_metrics(euid)
            if cached and cached.get("is_final"):
                cached_metrics = cached.get("metrics", {})
                if isinstance(cached_metrics, dict):
                    post_export_metrics = cached_metrics.get("post_export_metrics")
                    if isinstance(post_export_metrics, dict):
                        post_export_metrics.setdefault(
                            "data_transfer_intra_region_bytes", 0
                        )
                        post_export_metrics.setdefault(
                            "data_transfer_cross_region_bytes", 0
                        )
                        post_export_metrics.setdefault(
                            "data_transfer_internet_bytes", 0
                        )
                # Have final cached metrics - return them
                return {
                    "euid": euid,
                    "cached": True,
                    "is_final": True,
                    **(cached_metrics if isinstance(cached_metrics, dict) else {}),
                }

        # Try to fetch metrics - first from headnode, then fall back to S3
        metrics_data = None
        metrics_source = None

        if PIPELINE_STATUS_AVAILABLE and PipelineStatusFetcher is not None:
            workset_name = workset.get("name") or workset.get("workset_name") or euid
            results_s3_uri = workset.get("results_s3_uri")
            # Use cached headnode IP from TapDB (stored by monitor when workset started)
            headnode_ip = workset.get("execution_headnode_ip")

            # Get workset region for region-specific SSH key
            workset_region = (
                workset.get("execution_cluster_region")
                or workset.get("cluster_region")
                or settings.get_effective_region()
            )

            # Try to get region-specific SSH key from ursa_config
            ssh_key = settings.pipeline_ssh_identity_file
            try:
                from daylily_ursa.ursa_config import get_ursa_config
                ursa_cfg = get_ursa_config()
                region_key = ursa_cfg.get_ssh_key_for_region(workset_region)
                if region_key:
                    ssh_key = region_key
            except Exception:
                pass

            fetcher = PipelineStatusFetcher(
                ssh_user=settings.pipeline_ssh_user,
                ssh_identity_file=ssh_key,
                timeout=settings.pipeline_ssh_timeout,
                clone_dest_root=settings.pipeline_clone_dest_root,
                repo_dir_name=settings.pipeline_repo_dir_name,
            )

            # Try headnode first (for running worksets) - use IP from TapDB
            if headnode_ip and workset_name and not is_terminal_state:
                try:
                    metrics_data = fetcher.fetch_performance_metrics(
                        headnode_ip, workset_name
                    )
                    if metrics_data and any(metrics_data.values()):
                        metrics_source = "headnode"
                except Exception as e:
                    LOGGER.warning(
                        "Failed to fetch performance metrics from headnode for %s: %s",
                        euid,
                        str(e),
                    )

            # Fall back to S3 if headnode didn't work or workset is complete
            if not metrics_data or not any(metrics_data.values()):
                if results_s3_uri:
                    LOGGER.debug(
                        "Attempting S3 fallback for metrics: %s", results_s3_uri
                    )
                    try:
                        metrics_data = fetcher.fetch_performance_metrics_from_s3(
                            results_s3_uri,
                            region=settings.get_effective_region(),
                        )
                        if metrics_data and any(metrics_data.values()):
                            metrics_source = "s3"
                            LOGGER.info(
                                "Fetched performance metrics from S3 for %s",
                                euid,
                            )
                    except Exception as e:
                        LOGGER.warning(
                            "Failed to fetch performance metrics from S3 for %s: %s",
                            euid,
                            str(e),
                        )

            # Calculate S3 directory size for completed worksets with force_refresh
            if force_refresh and results_s3_uri and is_terminal_state:
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(results_s3_uri)
                    s3_bucket = parsed.netloc
                    s3_prefix = parsed.path.lstrip("/")
                    if s3_prefix and not s3_prefix.endswith("/"):
                        s3_prefix = s3_prefix + "/"

                    session_kwargs = {"region_name": settings.get_effective_region()}
                    if settings.aws_profile:
                        session_kwargs["profile_name"] = settings.aws_profile
                    session = boto3.Session(**session_kwargs)
                    s3_client = session.client("s3")
                    paginator = s3_client.get_paginator("list_objects_v2")

                    total_size = 0
                    for page in paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix):
                        for obj in page.get("Contents", []):
                            total_size += obj.get("Size", 0)

                    if total_size > 0:
                        # Coarse transfer metering
                        bucket_region = None
                        try:
                            loc = s3_client.get_bucket_location(Bucket=s3_bucket)
                            bucket_region = loc.get("LocationConstraint") or "us-east-1"
                        except Exception:
                            bucket_region = None

                        intra_bytes = 0
                        cross_bytes = 0
                        internet_bytes = 0
                        cluster_region_str = str(workset_region) if workset_region else None
                        if cluster_region_str and bucket_region:
                            if cluster_region_str == bucket_region:
                                intra_bytes = int(total_size)
                            else:
                                cross_bytes = int(total_size)
                        else:
                            internet_bytes = int(total_size)

                        # Format human-readable size
                        def format_bytes(size_bytes: int) -> str:
                            size = float(size_bytes)
                            for unit in ["B", "KB", "MB", "GB", "TB"]:
                                if abs(size) < 1024.0:
                                    return f"{size:.1f}{unit}"
                                size /= 1024.0
                            return f"{size:.1f}PB"

                        post_export_metrics = {
                            "analysis_directory_size_bytes": total_size,
                            "analysis_directory_size_human": format_bytes(total_size),
                            "data_transfer_intra_region_bytes": intra_bytes,
                            "data_transfer_cross_region_bytes": cross_bytes,
                            "data_transfer_internet_bytes": internet_bytes,
                        }
                        if metrics_data is None:
                            metrics_data = {}
                        metrics_data["post_export_metrics"] = post_export_metrics
                        LOGGER.info(
                            "Calculated S3 directory size for %s: %s (%d bytes)",
                            euid,
                            post_export_metrics["analysis_directory_size_human"],
                            total_size,
                        )
                except Exception as e:
                    LOGGER.warning(
                        "Failed to calculate S3 directory size for %s: %s",
                        euid,
                        str(e),
                    )

        # Cache the results if we got any
        if metrics_data and any(metrics_data.values()):
            post_export_metrics = None
            if isinstance(metrics_data, dict):
                post_export_metrics = metrics_data.get("post_export_metrics")
            if isinstance(post_export_metrics, dict):
                post_export_metrics.setdefault("data_transfer_intra_region_bytes", 0)
                post_export_metrics.setdefault("data_transfer_cross_region_bytes", 0)
                post_export_metrics.setdefault("data_transfer_internet_bytes", 0)

            # Cache with is_final=True if workset is in terminal state
            state_db.update_performance_metrics(
                euid, metrics_data, is_final=is_terminal_state
            )

        return {
            "euid": euid,
            "cached": False,
            "is_final": is_terminal_state,
            "source": metrics_source,
            **(metrics_data or {"alignment_stats": None, "benchmark_data": None, "cost_summary": None, "duration_info": None, "post_export_metrics": None}),
        }

    @router.get(
        "/api/v2/customers/{customer_id}/worksets/{euid}/snakemake-log/{log_filename}",
    )
    async def download_snakemake_log(
        customer_id: str,
        euid: str,
        log_filename: str,
    ):
        """Download a specific Snakemake log file from the headnode.

        Args:
            customer_id: Customer identifier
            euid: Workset EUID
            log_filename: Name of the Snakemake log file (e.g., 2026-01-18T135855.907380.snakemake.log)

        Returns:
            Plain text content of the log file
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

        workset = state_db.get_workset(euid)
        if not workset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workset {euid} not found",
            )

        # Verify workset belongs to this customer
        if not verify_workset_ownership(workset, customer_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Workset does not belong to this customer",
            )

        # Validate log filename (prevent path traversal)
        if "/" in log_filename or "\\" in log_filename or ".." in log_filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid log filename",
            )
        if not log_filename.endswith(".snakemake.log"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid log filename format",
            )

        if not PIPELINE_STATUS_AVAILABLE or PipelineStatusFetcher is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Pipeline status module not available",
            )

        workset_name = workset.get("name") or workset.get("workset_name")
        # Use cached headnode IP from TapDB (stored by monitor when workset started)
        headnode_ip = workset.get("execution_headnode_ip")

        if not headnode_ip or not workset_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Workset missing execution_headnode_ip or name",
            )

        # Get workset region for region-specific SSH key
        workset_region = (
            workset.get("execution_cluster_region")
            or workset.get("cluster_region")
            or settings.get_effective_region()
        )

        # Try to get region-specific SSH key from ursa_config
        ssh_key = settings.pipeline_ssh_identity_file
        try:
            from daylily_ursa.ursa_config import get_ursa_config
            ursa_cfg = get_ursa_config()
            region_key = ursa_cfg.get_ssh_key_for_region(workset_region)
            if region_key:
                ssh_key = region_key
        except Exception:
            pass

        try:
            fetcher = PipelineStatusFetcher(
                ssh_user=settings.pipeline_ssh_user,
                ssh_identity_file=ssh_key,
                timeout=settings.pipeline_ssh_timeout,
                clone_dest_root=settings.pipeline_clone_dest_root,
                repo_dir_name=settings.pipeline_repo_dir_name,
            )

            content = fetcher.get_full_log_content(headnode_ip, workset_name, log_filename)
            if content is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Log file {log_filename} not found",
                )

            return Response(
                content=content,
                media_type="text/plain",
                headers={
                    "Content-Disposition": f'attachment; filename="{log_filename}"',
                },
            )

        except HTTPException:
            raise
        except Exception as e:
            LOGGER.error("Failed to download Snakemake log: %s", str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to fetch log file: {str(e)}",
            )

    return router
