"""Portal routes for clusters, worksets, files, manifests, and usage."""

from __future__ import annotations

import csv
import io
import logging
import os
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import boto3
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from daylib_ursa.cluster_service import ClusterInfo, get_cluster_service
from daylib_ursa.config import Settings
from daylib_ursa.ephemeral_cluster import (
    list_cluster_create_jobs,
    start_create_job,
    tail_job_log,
)
from daylib_ursa.portal_auth import (
    PORTAL_SESSION_COOKIE_NAME,
    decode_portal_session,
)
from daylib_ursa.portal_graph_state import GraphPortalState
from daylib_ursa.pricing_monitor import PricingMonitor
from daylib_ursa.s3_utils import RegionAwareS3Client, normalize_bucket_name

LOGGER = logging.getLogger("daylily.portal")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TEMPLATES = Jinja2Templates(directory=str(_REPO_ROOT / "templates"))


def _session_identity(request: Request, settings: Settings) -> Dict[str, Any]:
    raw = request.cookies.get(PORTAL_SESSION_COOKIE_NAME)
    if not raw:
        return {}
    session = decode_portal_session(settings.session_secret_key, raw) or {}
    return session if isinstance(session, dict) else {}


def _portal_customer(
    customer_id: str,
    *,
    settings: Settings,
    identity: Optional[Dict[str, Any]] = None,
    portal_state: GraphPortalState | None = None,
) -> Dict[str, Any]:
    customer: Dict[str, Any] = {
        "customer_id": customer_id,
        "customer_name": customer_id.replace("-", " ").title(),
        "email": "",
        "s3_bucket": "",
        "max_concurrent_worksets": 10,
        "max_storage_gb": 500,
        "billing_account_id": "",
        "cost_center": "",
    }
    if identity:
        customer["customer_name"] = str(identity.get("customer_name") or customer["customer_name"])
        customer["email"] = str(identity.get("user_email") or "")
        customer["s3_bucket"] = str(identity.get("s3_bucket") or "")
        customer["billing_account_id"] = str(identity.get("billing_account_id") or "")
        customer["cost_center"] = str(identity.get("cost_center") or "")
        try:
            customer["max_concurrent_worksets"] = int(
                identity.get("max_concurrent_worksets") or customer["max_concurrent_worksets"]
            )
        except (TypeError, ValueError):
            pass

    if not customer["s3_bucket"] and portal_state is not None:
        try:
            buckets = portal_state.list_buckets(customer_id=customer_id)
            primary = next((item for item in buckets if item.get("bucket_type") == "primary"), None)
            selected = primary or (buckets[0] if buckets else None)
            if selected:
                customer["s3_bucket"] = str(selected.get("bucket_name") or "")
        except Exception:
            LOGGER.exception("Failed to resolve customer buckets for %s", customer_id)
    return customer

def _request_is_admin(request: Request, settings: Settings) -> bool:
    header_value = (
        request.headers.get("X-Ursa-Admin") or request.query_params.get("admin") or ""
    ).strip().lower()
    if header_value in {"1", "true", "yes", "admin"}:
        return True
    if header_value in {"0", "false", "no"}:
        return False
    if settings.enable_auth:
        return bool(_session_identity(request, settings).get("is_admin"))
    return True


def _customer_id(request: Request, settings: Settings) -> str:
    session = _session_identity(request, settings)
    return (
        request.query_params.get("customer_id")
        or request.headers.get("X-Ursa-Customer-Id")
        or str(session.get("customer_id") or "").strip()
        or settings.ursa_portal_default_customer_id
    )


def _template_context(
    request: Request,
    settings: Settings,
    portal_state: GraphPortalState,
    *,
    customer_id: Optional[str] = None,
    is_admin: Optional[bool] = None,
    **extra: Any,
) -> Dict[str, Any]:
    identity = _session_identity(request, settings)
    resolved_customer_id = customer_id or _customer_id(request, settings)
    resolved_is_admin = _request_is_admin(request, settings) if is_admin is None else is_admin
    auth_enabled = bool(settings.enable_auth)
    user_authenticated = (not auth_enabled) or bool(identity.get("logged_in"))
    user_email = str(identity.get("user_email") or "")
    customer = _portal_customer(
        resolved_customer_id,
        settings=settings,
        identity=identity,
        portal_state=portal_state,
    )
    return {
        "request": request,
        "auth_enabled": auth_enabled,
        "user_authenticated": user_authenticated,
        "user_email": user_email,
        "is_admin": resolved_is_admin,
        "customer_id": resolved_customer_id,
        "customer": customer,
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


def _hosted_ui_logout_url(
    *,
    domain: str,
    client_id: str,
    logout_uri: str,
) -> str:
    normalized_domain = domain.strip()
    if not normalized_domain.startswith(("http://", "https://")):
        normalized_domain = f"https://{normalized_domain}"
    query = urlencode({"client_id": client_id, "logout_uri": logout_uri})
    return f"{normalized_domain.rstrip('/')}/logout?{query}"


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


def _reconcile_pending_worksets(state: GraphPortalState, clusters: List[ClusterInfo]) -> None:
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


def _manifest_template_tsv() -> str:
    columns = GraphPortalState._manifest_columns()
    return "\t".join(columns) + "\n"


def mount_portal(app: FastAPI, settings: Settings) -> None:
    """Attach portal routes to the existing FastAPI app."""
    portal_state = GraphPortalState(
        region=settings.get_effective_region(),
        profile=settings.aws_profile,
    )
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
                portal_state,
                customer_id=customer_id,
                stats=stats,
                worksets=worksets,
            ),
        )

    @router.get("/portal/manifest-generator", response_class=HTMLResponse)
    async def manifest_generator_page(request: Request) -> HTMLResponse:
        return _TEMPLATES.TemplateResponse(
            "manifest_generator.html",
            _template_context(request, settings, portal_state, customer_id=_customer_id(request, settings)),
        )

    @router.get("/portal/files", response_class=HTMLResponse)
    async def files_page(
        request: Request,
        search: Optional[str] = Query(default=None),
        subject_id: Optional[str] = Query(default=None),
        biosample_id: Optional[str] = Query(default=None),
        file_format: Optional[str] = Query(default=None),
    ) -> HTMLResponse:
        customer_id = _customer_id(request, settings)
        files = portal_state.list_files(
            customer_id=customer_id,
            limit=500,
            search=search,
            filters={
                "subject_id": subject_id or "",
                "biosample_id": biosample_id or "",
                "file_format": file_format or "",
            },
        )
        filesets = portal_state.list_filesets(customer_id=customer_id)
        buckets = portal_state.list_buckets(customer_id=customer_id)
        stats = {
            "total_files": len(files),
            "total_filesets": len(filesets),
            "linked_buckets": len(buckets),
            "unique_subjects": len({str(item.get("subject_id") or "") for item in files if item.get("subject_id")}),
        }
        return _TEMPLATES.TemplateResponse(
            "files/index.html",
            _template_context(
                request,
                settings,
                portal_state,
                customer_id=customer_id,
                files=files,
                stats=stats,
            ),
        )

    @router.get("/portal/files/register", response_class=HTMLResponse)
    async def file_register_page(request: Request) -> HTMLResponse:
        customer_id = _customer_id(request, settings)
        buckets = portal_state.list_buckets(customer_id=customer_id)
        return _TEMPLATES.TemplateResponse(
            "files/register.html",
            _template_context(
                request,
                settings,
                portal_state,
                customer_id=customer_id,
                buckets=buckets,
            ),
        )

    @router.post("/portal/files/register")
    async def portal_register_discovered(request: Request) -> Dict[str, Any]:
        payload = await request.json()
        customer_id = _customer_id(request, settings)
        bucket_id = str(payload.get("bucket_id") or "").strip()
        prefix = str(payload.get("prefix") or "")
        selected_keys = [str(item).strip() for item in (payload.get("selected_keys") or []) if str(item).strip()]
        if not bucket_id or not selected_keys:
            raise HTTPException(status_code=400, detail="bucket_id and selected_keys are required")

        discovered = portal_state.discover_bucket_files(
            customer_id=customer_id,
            bucket_id=bucket_id,
            prefix=prefix,
            max_files=int(payload.get("max_files") or 10000),
            file_formats=None,
        )
        discovered_by_key = {str(item.get("key") or ""): item for item in discovered.get("files") or []}

        biosample_id = str(payload.get("biosample_id") or "").strip()
        subject_id = str(payload.get("subject_id") or "").strip()
        platform = str(payload.get("sequencing_platform") or "NOVASEQX")

        registered_count = 0
        errors: list[str] = []
        bucket = portal_state.get_bucket(customer_id=customer_id, bucket_id=bucket_id)
        bucket_name = str((bucket or {}).get("bucket_name") or "")
        for key in selected_keys:
            item = discovered_by_key.get(key)
            if item is None:
                errors.append(f"Not found during discovery: {key}")
                continue
            s3_uri = f"s3://{bucket_name}/{key}"
            try:
                portal_state.register_file(
                    customer_id=customer_id,
                    bucket_id=bucket_id,
                    payload={
                        "file_metadata": {
                            "s3_uri": s3_uri,
                            "file_size_bytes": int(item.get("file_size_bytes") or 0),
                            "file_format": item.get("detected_format") or "",
                        },
                        "biosample_metadata": {
                            "biosample_id": biosample_id,
                            "subject_id": subject_id,
                            "sample_type": None,
                        },
                        "sequencing_metadata": {
                            "platform": platform,
                            "vendor": "ILMN",
                        },
                    },
                )
                registered_count += 1
            except Exception as exc:
                errors.append(f"{key}: {exc}")

        return {
            "registered_count": registered_count,
            "selected_count": len(selected_keys),
            "errors": errors,
        }

    @router.get("/portal/files/upload", response_class=HTMLResponse)
    async def file_upload_page(request: Request) -> HTMLResponse:
        customer_id = _customer_id(request, settings)
        buckets = portal_state.list_buckets(customer_id=customer_id)
        return _TEMPLATES.TemplateResponse(
            "files/upload.html",
            _template_context(
                request,
                settings,
                portal_state,
                customer_id=customer_id,
                buckets=buckets,
            ),
        )

    @router.post("/portal/files/upload")
    async def upload_file_portal(
        request: Request,
        bucket_id: str = Form(...),
        prefix: str = Form(default=""),
        auto_register: str = Form(default="false"),
        file: UploadFile = File(...),
    ) -> Dict[str, Any]:
        customer_id = _customer_id(request, settings)
        data = await file.read()
        result = portal_state.upload_file_bytes(
            customer_id=customer_id,
            bucket_id=bucket_id,
            filename=file.filename or "upload.bin",
            content=data,
            prefix=prefix,
            auto_register=str(auto_register).strip().lower() in {"1", "true", "yes", "on"},
        )
        return result

    @router.get("/portal/files/buckets", response_class=HTMLResponse)
    async def buckets_page(request: Request) -> HTMLResponse:
        customer_id = _customer_id(request, settings)
        buckets = portal_state.list_buckets(customer_id=customer_id)
        return _TEMPLATES.TemplateResponse(
            "files/buckets.html",
            _template_context(
                request,
                settings,
                portal_state,
                customer_id=customer_id,
                buckets=buckets,
            ),
        )

    @router.get("/portal/files/browse/{bucket_id}", response_class=HTMLResponse)
    async def bucket_browse_page(
        request: Request,
        bucket_id: str,
        prefix: Optional[str] = Query(default=""),
    ) -> HTMLResponse:
        customer_id = _customer_id(request, settings)
        bucket = portal_state.get_bucket(customer_id=customer_id, bucket_id=bucket_id)
        if bucket is None:
            raise HTTPException(status_code=404, detail="Bucket not found")
        listing = portal_state.browse_bucket(
            customer_id=customer_id,
            bucket_id=bucket_id,
            prefix=prefix or "",
        )
        return _TEMPLATES.TemplateResponse(
            "files/browse.html",
            _template_context(
                request,
                settings,
                portal_state,
                customer_id=customer_id,
                bucket=bucket,
                current_prefix=listing.get("current_prefix") or "",
                items=listing.get("items") or [],
                breadcrumbs=listing.get("breadcrumbs") or [],
            ),
        )

    @router.get("/portal/files/browser", response_class=HTMLResponse)
    async def bucket_browser_alias(
        request: Request,
        bucket_id: str = Query(...),
        prefix: Optional[str] = Query(default=""),
    ) -> HTMLResponse:
        return await bucket_browse_page(request, bucket_id=bucket_id, prefix=prefix)

    @router.get("/portal/files/filesets", response_class=HTMLResponse)
    async def filesets_page(request: Request) -> HTMLResponse:
        customer_id = _customer_id(request, settings)
        filesets = portal_state.list_filesets(customer_id=customer_id)
        return _TEMPLATES.TemplateResponse(
            "files/filesets.html",
            _template_context(
                request,
                settings,
                portal_state,
                customer_id=customer_id,
                filesets=filesets,
            ),
        )

    @router.get("/portal/files/filesets/{fileset_id}", response_class=HTMLResponse)
    async def fileset_detail_page(request: Request, fileset_id: str) -> HTMLResponse:
        customer_id = _customer_id(request, settings)
        fileset = portal_state.get_fileset(customer_id=customer_id, fileset_id=fileset_id)
        if fileset is None:
            raise HTTPException(status_code=404, detail="File set not found")
        return _TEMPLATES.TemplateResponse(
            "files/fileset_detail.html",
            _template_context(
                request,
                settings,
                portal_state,
                customer_id=customer_id,
                fileset=fileset,
                files=fileset.get("files") or [],
            ),
        )

    @router.get("/portal/files/{file_id}", response_class=HTMLResponse)
    async def file_detail_page(request: Request, file_id: str) -> HTMLResponse:
        customer_id = _customer_id(request, settings)
        file_payload = portal_state.get_file(customer_id=customer_id, file_id=file_id)
        if file_payload is None:
            raise HTTPException(status_code=404, detail="File not found")

        filesets = portal_state.list_filesets(customer_id=customer_id)
        membership = [
            {
                "fileset_id": item.get("fileset_id"),
                "name": item.get("name"),
                "file_count": item.get("file_count"),
            }
            for item in filesets
            if any(str(child.get("file_id")) == str(file_id) for child in (item.get("files") or []))
        ]

        return _TEMPLATES.TemplateResponse(
            "files/detail.html",
            _template_context(
                request,
                settings,
                portal_state,
                customer_id=customer_id,
                file=file_payload,
                workset_history=[],
                file_sets=membership,
            ),
        )

    @router.get("/portal/files/{file_id}/edit", response_class=HTMLResponse)
    async def file_edit_page(request: Request, file_id: str) -> HTMLResponse:
        customer_id = _customer_id(request, settings)
        file_payload = portal_state.get_file(customer_id=customer_id, file_id=file_id)
        if file_payload is None:
            raise HTTPException(status_code=404, detail="File not found")
        return _TEMPLATES.TemplateResponse(
            "files/edit_file.html",
            _template_context(
                request,
                settings,
                portal_state,
                customer_id=customer_id,
                file=file_payload,
            ),
        )

    @router.get("/portal/usage", response_class=HTMLResponse)
    async def usage_page(request: Request) -> HTMLResponse:
        customer_id = _customer_id(request, settings)
        usage = portal_state.get_usage_summary(customer_id)
        usage_details = portal_state.get_usage_details(customer_id)
        return _TEMPLATES.TemplateResponse(
            "usage.html",
            _template_context(
                request,
                settings,
                portal_state,
                customer_id=customer_id,
                usage=usage,
                usage_details=usage_details,
                workset_storage=[],
            ),
        )

    @router.get("/portal/usage/export")
    async def usage_export(request: Request) -> PlainTextResponse:
        customer_id = _customer_id(request, settings)
        usage_details = portal_state.get_usage_details(customer_id)
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=["date", "type", "workset_id", "workset_label", "quantity", "unit", "cost", "is_actual"],
        )
        writer.writeheader()
        for row in usage_details:
            writer.writerow(row)
        body = output.getvalue()
        return PlainTextResponse(
            body,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=usage-report.csv"},
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
                portal_state,
                clusters=[
                    _cluster_payload(cluster, include_sensitive=_request_is_admin(request, settings))
                    for cluster in clusters
                ],
                regions=settings.get_allowed_regions() or settings.get_cost_monitor_regions(),
                create_mode=action == "create",
                prefill_region=region or "",
            ),
        )

    @router.get("/portal/worksets/new", response_class=HTMLResponse)
    async def new_workset_page(request: Request) -> HTMLResponse:
        regions = settings.get_allowed_regions() or settings.get_cost_monitor_regions()
        return _TEMPLATES.TemplateResponse(
            "worksets/new.html",
            _template_context(request, settings, portal_state, allowed_regions=regions),
        )

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
                portal_state,
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
                portal_state,
                customer_id=customer_id,
                workset=workset,
            ),
        )

    @router.get("/portal/logout")
    async def logout(request: Request) -> RedirectResponse:
        if settings.enable_auth and settings.cognito_domain and settings.cognito_app_client_id:
            logout_target = f"{request.url.scheme}://{request.url.netloc}/"
            response = RedirectResponse(
                url=_hosted_ui_logout_url(
                    domain=str(settings.cognito_domain),
                    client_id=str(settings.cognito_app_client_id),
                    logout_uri=logout_target,
                ),
                status_code=307,
            )
        else:
            response = RedirectResponse(url="/portal/login", status_code=307)
        response.delete_cookie(PORTAL_SESSION_COOKIE_NAME, path="/")
        return response

    @router.get("/portal/account", response_class=HTMLResponse)
    async def account_page(request: Request) -> HTMLResponse:
        customer_id = _customer_id(request, settings)
        identity = _session_identity(request, settings)
        customer = _portal_customer(
            customer_id,
            settings=settings,
            identity=identity,
            portal_state=portal_state,
        )
        env_vars = dict(os.environ)
        session_info = {
            "session_user_email": identity.get("user_email"),
            "session_customer_id": identity.get("customer_id"),
            "session_logged_in": bool(identity.get("logged_in")),
        }
        return _TEMPLATES.TemplateResponse(
            "account.html",
            _template_context(
                request,
                settings,
                portal_state,
                customer_id=customer_id,
                customer=customer,
                env_vars=env_vars,
                session_info=session_info,
                db_customer_id=customer_id,
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
            "clusters": [
                _cluster_payload(cluster, include_sensitive=include_sensitive) for cluster in clusters
            ]
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
            raise HTTPException(
                status_code=400,
                detail="region_az, cluster_name, ssh_key_name, and s3_bucket_name are required",
            )

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

    @router.get("/api/customers/{customer_id}/dashboard/cost-history")
    async def dashboard_cost_history(customer_id: str, days: int = 30) -> Dict[str, Any]:
        return portal_state.get_cost_history(customer_id, days=days)

    @router.get("/api/customers/{customer_id}/dashboard/cost-breakdown")
    async def dashboard_cost_breakdown(customer_id: str) -> Dict[str, Any]:
        return portal_state.get_cost_breakdown(customer_id)

    @router.get("/api/customers/{customer_id}/usage")
    async def usage_summary(customer_id: str) -> Dict[str, Any]:
        return portal_state.get_usage_summary(customer_id)

    @router.get("/api/customers/{customer_id}/usage/details")
    async def usage_details(customer_id: str) -> Dict[str, Any]:
        return {"usage_details": portal_state.get_usage_details(customer_id)}

    @router.get("/api/customers/{customer_id}/manifests")
    async def list_manifests(customer_id: str, limit: int = 200) -> Dict[str, Any]:
        return {"manifests": portal_state.list_manifests(customer_id=customer_id, limit=limit)}

    @router.post("/api/customers/{customer_id}/manifests")
    async def create_manifest(customer_id: str, request: Request) -> Dict[str, Any]:
        payload = await request.json()
        tsv_content = str(payload.get("tsv_content") or "")
        if not tsv_content.strip():
            raise HTTPException(status_code=400, detail="tsv_content is required")
        manifest = portal_state.create_manifest(
            customer_id=customer_id,
            tsv_content=tsv_content,
            name=str(payload.get("name") or ""),
            description=payload.get("description"),
        )
        return {"manifest": manifest}

    @router.get("/api/customers/{customer_id}/manifests/{manifest_id}")
    async def get_manifest(customer_id: str, manifest_id: str) -> Dict[str, Any]:
        manifest = portal_state.get_manifest(customer_id=customer_id, manifest_id=manifest_id)
        if manifest is None:
            raise HTTPException(status_code=404, detail="Manifest not found")
        return manifest

    @router.get("/api/customers/{customer_id}/manifests/{manifest_id}/download")
    async def download_manifest(customer_id: str, manifest_id: str) -> PlainTextResponse:
        manifest = portal_state.get_manifest(customer_id=customer_id, manifest_id=manifest_id)
        if manifest is None:
            raise HTTPException(status_code=404, detail="Manifest not found")
        tsv_content = str(manifest.get("tsv_content") or "")
        file_name = str(manifest.get("name") or manifest_id)
        return PlainTextResponse(
            tsv_content,
            media_type="text/tab-separated-values",
            headers={"Content-Disposition": f"attachment; filename={file_name}.tsv"},
        )

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
            selected_cluster = next(
                (cluster for cluster in clusters if cluster.cluster_name == preferred_cluster),
                None,
            )
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
                    raise HTTPException(
                        status_code=400,
                        detail="target_region is required when no cluster is selected",
                    )
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
                        cluster_name=str(
                            bootstrap.get("cluster_name") or _generate_cluster_name(target_region)
                        ),
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

    @router.get("/api/files")
    async def api_list_files(
        request: Request,
        limit: int = 200,
        search: Optional[str] = Query(default=None),
        file_format: Optional[str] = Query(default=None),
        subject_id: Optional[str] = Query(default=None),
        biosample_id: Optional[str] = Query(default=None),
    ) -> List[Dict[str, Any]]:
        customer_id = _customer_id(request, settings)
        return portal_state.list_files(
            customer_id=customer_id,
            limit=limit,
            search=search,
            filters={
                "file_format": file_format or "",
                "subject_id": subject_id or "",
                "biosample_id": biosample_id or "",
            },
        )

    @router.post("/api/files/search")
    async def api_search_files(
        request: Request,
        customer_id: Optional[str] = Query(default=None),
        search: Optional[str] = Query(default=None),
        biosample_id: Optional[str] = Query(default=None),
        subject_id: Optional[str] = Query(default=None),
        file_format: Optional[str] = Query(default=None),
        platform: Optional[str] = Query(default=None),
    ) -> Dict[str, Any]:
        body = await request.json()
        resolved_customer = customer_id or _customer_id(request, settings)
        files = portal_state.list_files(
            customer_id=resolved_customer,
            limit=500,
            search=search,
            filters={
                "biosample_id": biosample_id or body.get("biosample_id") or "",
                "subject_id": subject_id or body.get("subject_id") or "",
                "file_format": file_format or "",
                "platform": platform or "",
            },
        )
        return {"files": files, "file_count": len(files)}

    @router.post("/api/files/register")
    async def api_register_file(request: Request, customer_id: str = Query(...)) -> Dict[str, Any]:
        payload = await request.json()
        file_payload = portal_state.register_file(customer_id=customer_id, payload=payload)
        return file_payload

    @router.post("/api/files/bulk-import")
    async def api_bulk_import(request: Request, customer_id: str = Query(...)) -> Dict[str, Any]:
        payload = await request.json()
        files = payload.get("files") or []
        fileset_name = str(payload.get("fileset_name") or "").strip()
        fileset_description = payload.get("fileset_description")

        imported: list[str] = []
        errors: list[str] = []
        for entry in files:
            try:
                registered = portal_state.register_file(customer_id=customer_id, payload=dict(entry or {}))
                imported.append(str(registered.get("file_id") or ""))
            except Exception as exc:
                errors.append(str(exc))

        fileset_id: str | None = None
        if fileset_name and imported:
            created = portal_state.create_fileset(
                customer_id=customer_id,
                name=fileset_name,
                description=fileset_description,
                tags=[],
                file_ids=imported,
            )
            fileset_id = str(created.get("fileset_id") or "")

        return {
            "imported_count": len(imported),
            "failed_count": len(errors),
            "errors": errors,
            "fileset_id": fileset_id,
        }

    @router.get("/api/files/manifest/template")
    async def manifest_template_download() -> PlainTextResponse:
        return PlainTextResponse(
            _manifest_template_tsv(),
            media_type="text/tab-separated-values",
            headers={"Content-Disposition": "attachment; filename=manifest_template.tsv"},
        )

    @router.get("/api/files/buckets/list")
    async def api_list_buckets(request: Request, customer_id: str = Query(...)) -> Dict[str, Any]:
        return {"buckets": portal_state.list_buckets(customer_id=customer_id)}

    @router.post("/api/files/buckets/validate")
    async def api_validate_bucket(bucket_name: str = Query(...)) -> Dict[str, Any]:
        return portal_state.validate_bucket_name(bucket_name)

    @router.post("/api/files/buckets/link")
    async def api_link_bucket(request: Request, customer_id: str = Query(...)) -> Dict[str, Any]:
        payload = await request.json()
        return portal_state.link_bucket(
            customer_id=customer_id,
            bucket_name=str(payload.get("bucket_name") or "").strip(),
            bucket_type=str(payload.get("bucket_type") or "primary"),
            display_name=payload.get("display_name"),
            description=payload.get("description"),
            prefix_restriction=payload.get("prefix_restriction"),
            read_only=bool(payload.get("read_only", False)),
            validate=bool(payload.get("validate", True)),
        )

    @router.get("/api/files/buckets/{bucket_id}")
    async def api_get_bucket(request: Request, bucket_id: str) -> Dict[str, Any]:
        customer_id = _customer_id(request, settings)
        bucket = portal_state.get_bucket(customer_id=customer_id, bucket_id=bucket_id)
        if bucket is None:
            raise HTTPException(status_code=404, detail="Bucket not found")
        return bucket

    @router.patch("/api/files/buckets/{bucket_id}")
    async def api_update_bucket(request: Request, bucket_id: str) -> Dict[str, Any]:
        customer_id = _customer_id(request, settings)
        payload = await request.json()
        bucket = portal_state.update_bucket(customer_id=customer_id, bucket_id=bucket_id, updates=payload)
        if bucket is None:
            raise HTTPException(status_code=404, detail="Bucket not found")
        return bucket

    @router.post("/api/files/buckets/{bucket_id}/unlink")
    async def api_unlink_bucket(request: Request, bucket_id: str) -> Dict[str, Any]:
        customer_id = _customer_id(request, settings)
        ok = portal_state.unlink_bucket(customer_id=customer_id, bucket_id=bucket_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Bucket not found")
        return {"success": True}

    @router.post("/api/files/buckets/{bucket_id}/revalidate")
    async def api_revalidate_bucket(request: Request, bucket_id: str) -> Dict[str, Any]:
        customer_id = _customer_id(request, settings)
        result = portal_state.revalidate_bucket(customer_id=customer_id, bucket_id=bucket_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Bucket not found")
        return result

    @router.get("/api/files/buckets/{bucket_id}/browse")
    async def api_browse_bucket(
        request: Request,
        bucket_id: str,
        customer_id: Optional[str] = Query(default=None),
        prefix: Optional[str] = Query(default=""),
    ) -> Dict[str, Any]:
        resolved_customer = customer_id or _customer_id(request, settings)
        return portal_state.browse_bucket(
            customer_id=resolved_customer,
            bucket_id=bucket_id,
            prefix=prefix,
        )

    @router.post("/api/files/buckets/{bucket_id}/discover")
    async def api_discover_bucket(
        request: Request,
        bucket_id: str,
        customer_id: str = Query(...),
        prefix: str = Query(default=""),
        max_files: int = Query(default=1000),
        file_formats: Optional[str] = Query(default=None),
    ) -> Dict[str, Any]:
        formats = [part.strip() for part in str(file_formats or "").split(",") if part.strip()]
        return portal_state.discover_bucket_files(
            customer_id=customer_id,
            bucket_id=bucket_id,
            prefix=prefix,
            max_files=max_files,
            file_formats=formats,
        )

    @router.post("/api/files/buckets/{bucket_id}/folders")
    async def api_create_folder(
        request: Request,
        bucket_id: str,
        customer_id: str = Query(...),
        prefix: str = Query(default=""),
    ) -> Dict[str, Any]:
        payload = await request.json()
        folder_name = str(payload.get("folder_name") or "").strip()
        portal_state.create_bucket_folder(
            customer_id=customer_id,
            bucket_id=bucket_id,
            prefix=prefix,
            folder_name=folder_name,
        )
        return {"success": True}

    @router.delete("/api/files/buckets/{bucket_id}/files")
    async def api_delete_bucket_file(
        request: Request,
        bucket_id: str,
        customer_id: str = Query(...),
        file_key: str = Query(...),
    ) -> Dict[str, Any]:
        portal_state.delete_bucket_file(customer_id=customer_id, bucket_id=bucket_id, file_key=file_key)
        return {"success": True}

    @router.get("/api/files/filesets")
    async def api_list_filesets(request: Request, customer_id: Optional[str] = Query(default=None)) -> List[Dict[str, Any]]:
        resolved_customer = customer_id or _customer_id(request, settings)
        return portal_state.list_filesets(customer_id=resolved_customer)

    @router.post("/api/files/filesets")
    async def api_create_fileset(request: Request, customer_id: str = Query(...)) -> Dict[str, Any]:
        payload = await request.json()
        return portal_state.create_fileset(
            customer_id=customer_id,
            name=str(payload.get("name") or "").strip(),
            description=payload.get("description"),
            tags=list(payload.get("tags") or []),
            file_ids=[str(item) for item in (payload.get("file_ids") or []) if str(item).strip()],
        )

    @router.patch("/api/files/filesets/{fileset_id}")
    async def api_update_fileset(request: Request, fileset_id: str) -> Dict[str, Any]:
        customer_id = _customer_id(request, settings)
        payload = await request.json()
        fileset = portal_state.update_fileset(
            customer_id=customer_id,
            fileset_id=fileset_id,
            name=payload.get("name"),
            description=payload.get("description"),
            tags=payload.get("tags"),
        )
        if fileset is None:
            raise HTTPException(status_code=404, detail="File set not found")
        return fileset

    @router.delete("/api/files/filesets/{fileset_id}")
    async def api_delete_fileset(request: Request, fileset_id: str) -> Dict[str, Any]:
        customer_id = _customer_id(request, settings)
        ok = portal_state.delete_fileset(customer_id=customer_id, fileset_id=fileset_id)
        if not ok:
            raise HTTPException(status_code=404, detail="File set not found")
        return {"success": True}

    @router.post("/api/files/filesets/{fileset_id}/add-files")
    async def api_add_files_to_fileset(request: Request, fileset_id: str) -> Dict[str, Any]:
        customer_id = _customer_id(request, settings)
        payload = await request.json()
        file_ids = [str(item) for item in (payload or []) if str(item).strip()]
        fileset = portal_state.add_files_to_fileset(
            customer_id=customer_id,
            fileset_id=fileset_id,
            file_ids=file_ids,
        )
        if fileset is None:
            raise HTTPException(status_code=404, detail="File set not found")
        return fileset

    @router.post("/api/files/filesets/{fileset_id}/remove-files")
    async def api_remove_files_from_fileset(request: Request, fileset_id: str) -> Dict[str, Any]:
        customer_id = _customer_id(request, settings)
        payload = await request.json()
        file_ids = [str(item) for item in (payload or []) if str(item).strip()]
        fileset = portal_state.remove_files_from_fileset(
            customer_id=customer_id,
            fileset_id=fileset_id,
            file_ids=file_ids,
        )
        if fileset is None:
            raise HTTPException(status_code=404, detail="File set not found")
        return fileset

    @router.post("/api/files/filesets/{fileset_id}/clone")
    async def api_clone_fileset(request: Request, fileset_id: str) -> Dict[str, Any]:
        customer_id = _customer_id(request, settings)
        payload = await request.json()
        new_name = str(payload.get("new_name") or "").strip()
        if not new_name:
            raise HTTPException(status_code=400, detail="new_name is required")
        fileset = portal_state.clone_fileset(
            customer_id=customer_id,
            fileset_id=fileset_id,
            new_name=new_name,
        )
        if fileset is None:
            raise HTTPException(status_code=404, detail="File set not found")
        return fileset

    @router.post("/api/files/filesets/{fileset_id}/manifest")
    async def api_fileset_manifest(request: Request, fileset_id: str) -> PlainTextResponse:
        customer_id = _customer_id(request, settings)
        try:
            tsv = portal_state.render_fileset_manifest_tsv(customer_id=customer_id, fileset_id=fileset_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return PlainTextResponse(
            tsv,
            media_type="text/tab-separated-values",
            headers={"Content-Disposition": "attachment; filename=stage_samples.tsv"},
        )

    @router.get("/api/files/{file_id}/download")
    async def api_file_download_url(
        request: Request,
        file_id: str,
        customer_id: Optional[str] = Query(default=None),
        expires_in: int = Query(default=3600),
    ) -> Dict[str, Any]:
        resolved_customer = customer_id or _customer_id(request, settings)
        url = portal_state.generate_file_download_url(
            customer_id=resolved_customer,
            file_id=file_id,
            expires_in=expires_in,
        )
        if not url:
            raise HTTPException(status_code=404, detail="File not found")
        return {"url": url}

    @router.post("/api/files/{file_id}/tags")
    async def api_add_tag(
        request: Request,
        file_id: str,
        customer_id: Optional[str] = Query(default=None),
    ) -> Dict[str, Any]:
        payload = await request.json()
        tag = str(payload.get("tag") or "").strip()
        if not tag:
            raise HTTPException(status_code=400, detail="tag is required")
        resolved_customer = customer_id or _customer_id(request, settings)
        file_payload = portal_state.add_tag(customer_id=resolved_customer, file_id=file_id, tag=tag)
        if file_payload is None:
            raise HTTPException(status_code=404, detail="File not found")
        return file_payload

    @router.delete("/api/files/{file_id}/tags/{tag}")
    async def api_remove_tag(
        request: Request,
        file_id: str,
        tag: str,
        customer_id: Optional[str] = Query(default=None),
    ) -> Dict[str, Any]:
        resolved_customer = customer_id or _customer_id(request, settings)
        file_payload = portal_state.remove_tag(customer_id=resolved_customer, file_id=file_id, tag=tag)
        if file_payload is None:
            raise HTTPException(status_code=404, detail="File not found")
        return file_payload

    @router.patch("/api/v1/files/{file_id}")
    async def api_patch_file_v1(request: Request, file_id: str) -> Dict[str, Any]:
        customer_id = _customer_id(request, settings)
        payload = await request.json()
        file_payload = portal_state.update_file(customer_id=customer_id, file_id=file_id, payload=payload)
        if file_payload is None:
            raise HTTPException(status_code=404, detail="File not found")
        return file_payload

    @router.get("/api/s3/discover-samples")
    async def api_discover_samples(request: Request, customer_id: Optional[str] = Query(default=None)) -> Dict[str, Any]:
        resolved_customer = customer_id or _customer_id(request, settings)
        return portal_state.discover_samples(customer_id=resolved_customer)

    @router.get("/api/s3/bucket-region/{bucket_name}")
    async def api_bucket_region(bucket_name: str) -> Dict[str, Any]:
        region_client = RegionAwareS3Client(
            default_region=settings.get_effective_region(),
            profile=settings.aws_profile,
        )
        normalized = normalize_bucket_name(bucket_name)
        if not normalized:
            raise HTTPException(status_code=400, detail="Invalid bucket name")
        return {"bucket_name": normalized, "region": region_client.get_bucket_region(normalized)}

    app.include_router(router)
