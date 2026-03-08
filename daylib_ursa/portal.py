"""Lightweight portal routes for pricing, clusters, and workset submission."""

from __future__ import annotations

import logging
import os
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from daylib_ursa.cluster_service import ClusterInfo, get_cluster_service
from daylib_ursa.config import Settings
from daylib_ursa.ephemeral_cluster import (
    list_cluster_create_jobs,
    start_create_job,
    tail_job_log,
)
from daylib_ursa.portal_state import PortalState
from daylib_ursa.pricing_monitor import PricingMonitor

LOGGER = logging.getLogger("daylily.portal")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TEMPLATES = Jinja2Templates(directory=str(_REPO_ROOT / "templates"))


def _portal_customer(customer_id: str) -> Dict[str, Any]:
    return {
        "customer_id": customer_id,
        "customer_name": customer_id.replace("-", " ").title(),
        "email": "",
        "s3_bucket": "",
        "max_concurrent_worksets": 10,
        "billing_account_id": "",
        "cost_center": "",
    }


def _request_is_admin(request: Request, settings: Settings) -> bool:
    header_value = (request.headers.get("X-Ursa-Admin") or request.query_params.get("admin") or "").strip().lower()
    if header_value in {"1", "true", "yes", "admin"}:
        return True
    if header_value in {"0", "false", "no"}:
        return False
    return not settings.enable_auth


def _customer_id(request: Request, settings: Settings) -> str:
    return (
        request.query_params.get("customer_id")
        or request.headers.get("X-Ursa-Customer-Id")
        or settings.ursa_portal_default_customer_id
    )


def _template_context(
    request: Request,
    settings: Settings,
    *,
    customer_id: Optional[str] = None,
    is_admin: Optional[bool] = None,
    **extra: Any,
) -> Dict[str, Any]:
    resolved_customer_id = customer_id or _customer_id(request, settings)
    resolved_is_admin = _request_is_admin(request, settings) if is_admin is None else is_admin
    return {
        "request": request,
        "auth_enabled": settings.enable_auth,
        "user_authenticated": settings.enable_auth,
        "user_email": "",
        "is_admin": resolved_is_admin,
        "customer_id": resolved_customer_id,
        "customer": _portal_customer(resolved_customer_id),
        "current_year": datetime.now().year,
        "cache_bust": os.environ.get("SOURCE_DATE_EPOCH", "1"),
        "api_base": "",
        "csrf_token": "",
        **extra,
    }


def _cluster_payload(cluster: ClusterInfo, *, include_sensitive: bool) -> Dict[str, Any]:
    return cluster.to_dict(include_sensitive=include_sensitive)


def _sorted_running_clusters_by_region(clusters: List[ClusterInfo]) -> Dict[str, ClusterInfo]:
    running: Dict[str, ClusterInfo] = {}
    for cluster in sorted(clusters, key=lambda item: (item.region, item.cluster_name)):
        if cluster.cluster_status == "CREATE_COMPLETE" and cluster.region not in running:
            running[cluster.region] = cluster
    return running


def _find_inflight_create_job(region: str) -> Optional[Dict[str, Any]]:
    for job in list_cluster_create_jobs(limit=100):
        status = str(job.get("status") or "")
        region_az = str(job.get("region_az") or "")
        if region_az.startswith(region) and status in {"queued", "running"}:
            return job
    return None


def _generate_cluster_name(region: str) -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    compact_region = region.replace("-", "")
    return f"daylily-{compact_region}-{stamp}-{secrets.token_hex(2)}"


def _validate_admin(request: Request, settings: Settings) -> None:
    if not _request_is_admin(request, settings):
        raise HTTPException(status_code=403, detail="Admin privileges are required for this action")


def _build_boto_session(settings: Settings, region: Optional[str] = None) -> Any:
    kwargs: Dict[str, Any] = {}
    if settings.aws_profile:
        kwargs["profile_name"] = settings.aws_profile
    if region:
        kwargs["region_name"] = region
    return boto3.Session(**kwargs)


def _load_create_options(settings: Settings, region: str) -> Dict[str, List[str]]:
    session = _build_boto_session(settings, region=region)
    ec2 = session.client("ec2", region_name=region)
    s3 = session.client("s3")

    keypairs = sorted(
        str(item.get("KeyName") or "").strip()
        for item in ec2.describe_key_pairs().get("KeyPairs", [])
        if str(item.get("KeyName") or "").strip()
    )
    buckets = sorted(
        str(item.get("Name") or "").strip()
        for item in s3.list_buckets().get("Buckets", [])
        if str(item.get("Name") or "").strip()
    )
    return {"keypairs": keypairs, "buckets": buckets}


def _reconcile_pending_worksets(state: PortalState, clusters: List[ClusterInfo]) -> None:
    running_by_region = _sorted_running_clusters_by_region(clusters)
    for workset in state.list_pending_worksets():
        target_region = str(workset.get("target_region") or "")
        running_cluster = running_by_region.get(target_region)
        if not running_cluster:
            continue
        state.update_workset_cluster_assignment(
            workset_id=str(workset["workset_id"]),
            cluster_name=running_cluster.cluster_name,
            cluster_region=running_cluster.region,
            state="ready",
            message="Cluster became available automatically",
        )


def mount_portal(app: FastAPI, settings: Settings) -> None:
    """Attach the lightweight portal routes to the existing FastAPI app."""
    state_dir = Path.home() / ".ursa"
    portal_state = PortalState(state_dir / "portal.sqlite3")
    pricing_monitor = PricingMonitor(settings=settings, store=portal_state)

    app.state.portal_state = portal_state
    app.state.pricing_monitor = pricing_monitor

    if not any(getattr(route, "path", None) == "/static" for route in app.routes):
        app.mount("/static", StaticFiles(directory=str(_REPO_ROOT / "static")), name="static")

    @app.on_event("startup")
    async def _start_portal_background_services() -> None:
        pricing_monitor.start()

    router = APIRouter()

    @router.get("/portal", response_class=HTMLResponse)
    async def dashboard(request: Request) -> HTMLResponse:
        customer_id = _customer_id(request, settings)
        stats = portal_state.get_dashboard_stats(customer_id)
        worksets = portal_state.list_worksets(customer_id, limit=5)
        return _TEMPLATES.TemplateResponse(
            "dashboard.html",
            _template_context(
                request,
                settings,
                customer_id=customer_id,
                stats=stats,
                worksets=worksets,
            ),
        )

    @router.get("/portal/clusters", response_class=HTMLResponse)
    async def clusters_page(
        request: Request,
        action: Optional[str] = Query(default=None),
        region: Optional[str] = Query(default=None),
    ) -> HTMLResponse:
        cluster_service = get_cluster_service(
            regions=settings.get_allowed_regions() or settings.get_cost_monitor_regions(),
            aws_profile=settings.aws_profile,
        )
        clusters = cluster_service.get_all_clusters(force_refresh=False)
        _reconcile_pending_worksets(portal_state, clusters)
        return _TEMPLATES.TemplateResponse(
            "clusters.html",
            _template_context(
                request,
                settings,
                clusters=[_cluster_payload(cluster, include_sensitive=_request_is_admin(request, settings)) for cluster in clusters],
                regions=settings.get_allowed_regions() or settings.get_cost_monitor_regions(),
                create_mode=action == "create",
                prefill_region=region or "",
            ),
        )

    @router.get("/portal/worksets/new", response_class=HTMLResponse)
    async def new_workset_page(request: Request) -> HTMLResponse:
        regions = settings.get_allowed_regions() or settings.get_cost_monitor_regions()
        return _TEMPLATES.TemplateResponse("worksets/new.html", _template_context(request, settings, allowed_regions=regions))

    @router.get("/portal/worksets", response_class=HTMLResponse)
    async def worksets_page(
        request: Request,
        status: Optional[str] = Query(default=None),
    ) -> HTMLResponse:
        customer_id = _customer_id(request, settings)
        cluster_service = get_cluster_service(
            regions=settings.get_allowed_regions() or settings.get_cost_monitor_regions(),
            aws_profile=settings.aws_profile,
        )
        _reconcile_pending_worksets(portal_state, cluster_service.get_all_clusters(force_refresh=False))
        worksets = portal_state.list_worksets(customer_id, status=status)
        return _TEMPLATES.TemplateResponse(
            "worksets/minimal_list.html",
            _template_context(
                request,
                settings,
                customer_id=customer_id,
                worksets=worksets,
                selected_status=status or "",
            ),
        )

    @router.get("/portal/worksets/{workset_id}", response_class=HTMLResponse)
    async def workset_detail_page(request: Request, workset_id: str) -> HTMLResponse:
        customer_id = _customer_id(request, settings)
        cluster_service = get_cluster_service(
            regions=settings.get_allowed_regions() or settings.get_cost_monitor_regions(),
            aws_profile=settings.aws_profile,
        )
        _reconcile_pending_worksets(portal_state, cluster_service.get_all_clusters(force_refresh=False))
        workset = portal_state.get_workset(customer_id, workset_id)
        if workset is None:
            raise HTTPException(status_code=404, detail="Workset not found")
        return _TEMPLATES.TemplateResponse(
            "worksets/minimal_detail.html",
            _template_context(
                request,
                settings,
                customer_id=customer_id,
                workset=workset,
            ),
        )

    @router.get("/api/clusters")
    async def list_clusters(request: Request, refresh: bool = False) -> Dict[str, Any]:
        cluster_service = get_cluster_service(
            regions=settings.get_allowed_regions() or settings.get_cost_monitor_regions(),
            aws_profile=settings.aws_profile,
        )
        clusters = cluster_service.get_all_clusters(force_refresh=refresh)
        _reconcile_pending_worksets(portal_state, clusters)
        include_sensitive = _request_is_admin(request, settings)
        return {
            "clusters": [_cluster_payload(cluster, include_sensitive=include_sensitive) for cluster in clusters]
        }

    @router.delete("/api/clusters/{cluster_name}")
    async def delete_cluster(request: Request, cluster_name: str, region: str) -> Dict[str, Any]:
        _validate_admin(request, settings)
        cluster_service = get_cluster_service(
            regions=settings.get_allowed_regions() or settings.get_cost_monitor_regions(),
            aws_profile=settings.aws_profile,
        )
        result = cluster_service.delete_cluster(cluster_name, region)
        return {"success": True, "result": result}

    @router.get("/api/clusters/create/options")
    async def create_cluster_options(request: Request, region: str) -> Dict[str, Any]:
        _validate_admin(request, settings)
        try:
            return _load_create_options(settings, region)
        except Exception as exc:
            LOGGER.exception("Failed to load cluster create options")
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @router.post("/api/clusters/create")
    async def create_cluster(request: Request) -> Dict[str, Any]:
        _validate_admin(request, settings)
        payload = await request.json()
        region_az = str(payload.get("region_az") or "").strip()
        cluster_name = str(payload.get("cluster_name") or "").strip()
        ssh_key_name = str(payload.get("ssh_key_name") or "").strip()
        s3_bucket_name = str(payload.get("s3_bucket_name") or "").strip()
        if not region_az or not cluster_name or not ssh_key_name or not s3_bucket_name:
            raise HTTPException(status_code=400, detail="region_az, cluster_name, ssh_key_name, and s3_bucket_name are required")

        job = start_create_job(
            region_az=region_az,
            cluster_name=cluster_name,
            ssh_key_name=ssh_key_name,
            s3_bucket_name=s3_bucket_name,
            aws_profile=settings.aws_profile,
            contact_email=None,
            config_path_override=payload.get("config_path"),
            pass_on_warn=bool(payload.get("pass_on_warn")),
            debug=bool(payload.get("debug")),
        )
        return {
            "job_id": job.job_id,
            "cluster_name": job.cluster_name,
            "region_az": job.region_az,
            "status": job.status,
        }

    @router.get("/api/clusters/create/jobs")
    async def cluster_create_jobs(limit: int = 20) -> Dict[str, Any]:
        return {"jobs": list_cluster_create_jobs(limit=limit)}

    @router.get("/api/clusters/create/jobs/{job_id}/logs")
    async def cluster_create_logs(job_id: str, lines: int = 200) -> Dict[str, Any]:
        return {"job_id": job_id, "log": tail_job_log(job_id, lines=lines)}

    @router.get("/api/pricing-snapshots")
    async def pricing_snapshots(
        region: Optional[str] = Query(default=None),
        partitions: Optional[str] = Query(default=None),
        from_ts: Optional[str] = Query(default=None, alias="from"),
        to_ts: Optional[str] = Query(default=None, alias="to"),
    ) -> Dict[str, Any]:
        requested_partitions = [part.strip() for part in partitions.split(",")] if partitions else None
        return pricing_monitor.get_snapshot_payload(
            region=region,
            partitions=requested_partitions,
            from_ts=from_ts,
            to_ts=to_ts,
        )

    @router.post("/api/pricing-snapshots/run")
    async def run_pricing_snapshot(request: Request) -> Dict[str, Any]:
        _validate_admin(request, settings)
        queued = pricing_monitor.queue_capture(trigger="manual", requested_by="admin")
        return queued

    @router.get("/api/customers/{customer_id}/dashboard/stats")
    async def dashboard_stats(customer_id: str) -> Dict[str, Any]:
        return portal_state.get_dashboard_stats(customer_id)

    @router.get("/api/customers/{customer_id}/dashboard/activity")
    async def dashboard_activity(customer_id: str, days: int = 30) -> Dict[str, Any]:
        return portal_state.get_activity_series(customer_id, days=days)

    @router.get("/api/customers/{customer_id}/manifests")
    async def list_manifests(customer_id: str) -> Dict[str, Any]:
        return {"manifests": []}

    @router.get("/api/customers/{customer_id}/worksets")
    async def list_customer_worksets(customer_id: str) -> Dict[str, Any]:
        cluster_service = get_cluster_service(
            regions=settings.get_allowed_regions() or settings.get_cost_monitor_regions(),
            aws_profile=settings.aws_profile,
        )
        _reconcile_pending_worksets(portal_state, cluster_service.get_all_clusters(force_refresh=False))
        return {"worksets": portal_state.list_worksets(customer_id)}

    @router.get("/api/customers/{customer_id}/worksets/{workset_id}")
    async def get_customer_workset(customer_id: str, workset_id: str) -> Dict[str, Any]:
        cluster_service = get_cluster_service(
            regions=settings.get_allowed_regions() or settings.get_cost_monitor_regions(),
            aws_profile=settings.aws_profile,
        )
        _reconcile_pending_worksets(portal_state, cluster_service.get_all_clusters(force_refresh=False))
        workset = portal_state.get_workset(customer_id, workset_id)
        if workset is None:
            raise HTTPException(status_code=404, detail="Workset not found")
        return workset

    @router.get("/api/customers/{customer_id}/worksets/{workset_id}/logs")
    async def get_customer_workset_logs(customer_id: str, workset_id: str) -> Dict[str, Any]:
        if portal_state.get_workset(customer_id, workset_id) is None:
            raise HTTPException(status_code=404, detail="Workset not found")
        return {"content": ""}

    @router.post("/api/customers/{customer_id}/worksets")
    async def create_customer_workset(customer_id: str, request: Request) -> Dict[str, Any]:
        payload = await request.json()
        cluster_service = get_cluster_service(
            regions=settings.get_allowed_regions() or settings.get_cost_monitor_regions(),
            aws_profile=settings.aws_profile,
        )
        clusters = cluster_service.get_all_clusters(force_refresh=False)
        _reconcile_pending_worksets(portal_state, clusters)
        running_by_region = _sorted_running_clusters_by_region(clusters)

        preferred_cluster = str(payload.get("preferred_cluster") or "").strip()
        target_region = str(payload.get("target_region") or "").strip()
        is_admin = _request_is_admin(request, settings)
        cluster_name: Optional[str] = None
        cluster_region: Optional[str] = None
        cluster_create_job_id: Optional[str] = None
        state = "ready"
        message: Optional[str] = "Queued for execution"

        if preferred_cluster:
            selected_cluster = next((cluster for cluster in clusters if cluster.cluster_name == preferred_cluster), None)
            if selected_cluster is None or selected_cluster.cluster_status != "CREATE_COMPLETE":
                raise HTTPException(status_code=409, detail="Selected cluster is not currently available")
            cluster_name = selected_cluster.cluster_name
            cluster_region = selected_cluster.region
        else:
            if target_region and target_region in running_by_region:
                selected_cluster = running_by_region[target_region]
                cluster_name = selected_cluster.cluster_name
                cluster_region = selected_cluster.region
            else:
                if not is_admin:
                    region_label = target_region or "the selected region"
                    raise HTTPException(
                        status_code=409,
                        detail=f"No running cluster exists in {region_label}; contact an admin to create one.",
                    )
                bootstrap = payload.get("cluster_bootstrap") or {}
                target_region = target_region or str(bootstrap.get("region") or "").strip()
                if not target_region:
                    raise HTTPException(status_code=400, detail="target_region is required when no cluster is selected")
                existing_job = _find_inflight_create_job(target_region)
                if existing_job:
                    cluster_create_job_id = str(existing_job.get("job_id") or "")
                else:
                    ssh_key_name = str(bootstrap.get("ssh_key_name") or "").strip()
                    s3_bucket_name = str(bootstrap.get("s3_bucket_name") or "").strip()
                    az_suffix = str(bootstrap.get("az_suffix") or "a").strip() or "a"
                    if not ssh_key_name or not s3_bucket_name:
                        raise HTTPException(
                            status_code=400,
                            detail="ssh_key_name and s3_bucket_name are required to bootstrap a cluster",
                        )
                    job = start_create_job(
                        region_az=f"{target_region}{az_suffix}",
                        cluster_name=str(bootstrap.get("cluster_name") or _generate_cluster_name(target_region)),
                        ssh_key_name=ssh_key_name,
                        s3_bucket_name=s3_bucket_name,
                        aws_profile=settings.aws_profile,
                        contact_email=None,
                        config_path_override=bootstrap.get("config_path"),
                        pass_on_warn=bool(bootstrap.get("pass_on_warn")),
                        debug=bool(bootstrap.get("debug")),
                    )
                    cluster_create_job_id = job.job_id
                state = "pending_cluster_creation"
                cluster_region = target_region
                message = f"Waiting for a cluster in {target_region}"

        workset = portal_state.create_workset(
            customer_id=customer_id,
            payload=payload,
            state=state,
            cluster_name=cluster_name,
            cluster_region=cluster_region,
            target_region=target_region or cluster_region,
            cluster_create_job_id=cluster_create_job_id,
            message=message,
        )
        return {
            "workset_id": workset["workset_id"],
            "state": workset["state"],
            "cluster_name": workset["cluster_name"],
            "cluster_region": workset["cluster_region"],
            "cluster_create_job_id": workset["cluster_create_job_id"],
            "message": workset["message"],
        }

    app.include_router(router)
