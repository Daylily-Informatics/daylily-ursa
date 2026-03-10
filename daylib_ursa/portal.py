"""Portal routes for clusters, worksets, manifests, and usage."""

from __future__ import annotations

import csv
import io
import logging
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import boto3
from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
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
    PORTAL_SESSION_MAX_AGE_SECONDS,
    decode_portal_session,
    encode_portal_session,
)
from daylib_ursa.portal_graph_state import GraphPortalState
from daylib_ursa.pricing_monitor import PricingMonitor
from daylib_ursa.s3_utils import RegionAwareS3Client, normalize_bucket_name
from daylib_ursa.user_preferences import (
    DEFAULT_DISPLAY_TIMEZONE,
    list_display_timezone_options,
    normalize_display_timezone,
    set_display_timezone_for_email,
)

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
    display_timezone = normalize_display_timezone(
        identity.get("display_timezone"),
        default=DEFAULT_DISPLAY_TIMEZONE,
    )
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
        "current_year": datetime.now(timezone.utc).year,
        "display_timezone": display_timezone,
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
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
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


def _region_from_region_az(region_az: str) -> str:
    value = str(region_az or "").strip()
    if len(value) >= 2 and value[-1].isalpha() and value[-2].isdigit():
        return value[:-1]
    return value


def _validate_cluster_create_identity(settings: Settings, *, region_az: str) -> Dict[str, str]:
    region = _region_from_region_az(region_az)
    session = _build_boto_session(settings, region=region or None)
    sts = session.client("sts", region_name=region or None)
    identity = sts.get_caller_identity()
    arn = str(identity.get("Arn") or "").strip()
    if not arn:
        profile_label = settings.aws_profile or "default"
        raise RuntimeError(
            f"AWS STS caller identity is missing Arn for profile '{profile_label}'. "
            "Check profile credentials before creating a cluster."
        )
    return {
        "account_id": str(identity.get("Account") or "").strip(),
        "arn": arn,
    }


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


def _monitor_state_dirs() -> list[Path]:
    return [Path.home() / ".ursa", Path.home() / ".config" / "ursa"]


def _read_monitor_process() -> tuple[Optional[int], Optional[Path]]:
    for state_dir in _monitor_state_dirs():
        pid_file = state_dir / "monitor.pid"
        if not pid_file.exists():
            continue
        try:
            pid = int(pid_file.read_text().strip())
        except ValueError:
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        except PermissionError:
            return pid, pid_file
        except OSError:
            continue
        return pid, pid_file
    return None, None


def _list_monitor_logs(*, limit: int = 20) -> list[Path]:
    discovered: dict[str, Path] = {}
    for state_dir in _monitor_state_dirs():
        log_dir = state_dir / "logs"
        if not log_dir.exists():
            continue
        for path in log_dir.glob("monitor_*.log"):
            discovered[str(path)] = path

    ordered = sorted(
        discovered.values(),
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )
    return ordered[: max(limit, 1)]


def _tail_monitor_log_lines(path: Path, *, lines: int = 100) -> list[str]:
    try:
        all_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    return all_lines[-max(lines, 1) :]


def _format_uptime(seconds: int) -> str:
    remaining = max(int(seconds), 0)
    minutes_total, _ = divmod(remaining, 60)
    hours_total, minutes = divmod(minutes_total, 60)
    days, hours = divmod(hours_total, 24)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _monitor_stats_for_customer(*, portal_state: GraphPortalState, customer_id: str) -> Dict[str, int]:
    worksets = portal_state.list_worksets(customer_id, limit=10_000)
    stats = {"ready": 0, "in_progress": 0, "complete": 0, "error": 0, "total_processed": 0}
    for workset in worksets:
        state = str(workset.get("state") or "").strip().lower()
        if state in {"running", "submitted", "in_progress", "processing"}:
            stats["in_progress"] += 1
        elif state in {"ready", "queued", "pending", "pending_cluster_creation"}:
            stats["ready"] += 1
        elif state in {"complete", "completed", "returned"}:
            stats["complete"] += 1
        elif state in {"error", "failed"}:
            stats["error"] += 1
    stats["total_processed"] = stats["complete"] + stats["error"]
    return stats


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
            "session_display_timezone": normalize_display_timezone(
                identity.get("display_timezone"),
                default=DEFAULT_DISPLAY_TIMEZONE,
            ),
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
                display_timezone_options=list_display_timezone_options(),
            ),
        )

    @router.post("/api/account/preferences")
    async def account_preferences_update(request: Request) -> JSONResponse:
        identity = _session_identity(request, settings)
        if settings.enable_auth and not bool(identity.get("logged_in")):
            raise HTTPException(status_code=401, detail="Authentication required")

        payload = await request.json()
        requested_timezone = payload.get("display_timezone")
        display_timezone = normalize_display_timezone(
            requested_timezone,
            default=DEFAULT_DISPLAY_TIMEZONE,
        )
        user_email = str(identity.get("user_email") or "").strip()
        if user_email:
            display_timezone = set_display_timezone_for_email(user_email, display_timezone)

        identity["display_timezone"] = display_timezone
        response = JSONResponse(
            {
                "status": "success",
                "preferences": {"display_timezone": display_timezone},
            }
        )
        response.set_cookie(
            key=PORTAL_SESSION_COOKIE_NAME,
            value=encode_portal_session(settings.session_secret_key, identity),
            httponly=True,
            secure=request.url.scheme == "https",
            samesite="lax",
            max_age=PORTAL_SESSION_MAX_AGE_SECONDS,
            path="/",
        )
        return response

    @router.get("/portal/admin/users", response_class=HTMLResponse)
    async def admin_users_page(
        request: Request,
        q: Optional[str] = Query(default=None),
        error: Optional[str] = Query(default=None),
        success: Optional[str] = Query(default=None),
    ) -> HTMLResponse:
        _validate_admin(request, settings)
        customer_id = _customer_id(request, settings)
        identity = _session_identity(request, settings)
        customer = _portal_customer(
            customer_id,
            settings=settings,
            identity=identity,
            portal_state=portal_state,
        )
        users = [
            {
                "email": str(customer.get("email") or identity.get("user_email") or ""),
                "customer_name": str(customer.get("customer_name") or customer_id),
                "customer_id": customer_id,
                "is_admin": _request_is_admin(request, settings),
            }
        ]
        query = str(q or "").strip().lower()
        if query:
            users = [
                item
                for item in users
                if query in str(item.get("email") or "").lower()
                or query in str(item.get("customer_name") or "").lower()
                or query in str(item.get("customer_id") or "").lower()
            ]
        return _TEMPLATES.TemplateResponse(
            "admin/users.html",
            _template_context(
                request,
                settings,
                portal_state,
                customer_id=customer_id,
                customers=users,
                user_query=q or "",
                error=error or "",
                success=success or "",
            ),
        )

    @router.get("/portal/monitor", response_class=HTMLResponse)
    async def monitor_dashboard(request: Request) -> HTMLResponse:
        _validate_admin(request, settings)
        customer_id = _customer_id(request, settings)
        monitor_pid, pid_path = _read_monitor_process()
        monitor_running = monitor_pid is not None
        start_time_str = ""
        uptime_str = "--"
        if pid_path is not None and pid_path.exists():
            start_dt = datetime.fromtimestamp(pid_path.stat().st_mtime, tz=timezone.utc)
            start_time_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
            uptime_str = _format_uptime(
                int((datetime.now(timezone.utc) - start_dt).total_seconds())
            )

        log_paths = _list_monitor_logs(limit=20)
        latest_log = log_paths[0] if log_paths else None
        log_lines = _tail_monitor_log_lines(latest_log, lines=100) if latest_log else []
        stats = _monitor_stats_for_customer(portal_state=portal_state, customer_id=customer_id)

        monitor_config = _REPO_ROOT / "config" / "workset-monitor-config.yaml"
        config_display = {
            "config_path": str(monitor_config),
            "prefix": settings.s3_prefix,
            "poll_interval_seconds": 60,
            "max_concurrent_worksets": int(
                _portal_customer(
                    customer_id,
                    settings=settings,
                    identity=_session_identity(request, settings),
                    portal_state=portal_state,
                ).get("max_concurrent_worksets")
                or 10
            ),
            "reuse_cluster_name": "",
            "archive_prefix": "archive/",
        }

        return _TEMPLATES.TemplateResponse(
            "monitor/dashboard.html",
            _template_context(
                request,
                settings,
                portal_state,
                customer_id=customer_id,
                monitor_running=monitor_running,
                monitor_pid=monitor_pid,
                uptime_str=uptime_str,
                start_time_str=start_time_str,
                stats=stats,
                config_display=config_display,
                log_files=[path.name for path in log_paths],
                latest_log_path=str(latest_log) if latest_log is not None else "",
                log_lines=log_lines,
                log_error="",
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

    @router.get("/api/monitor/status")
    async def monitor_status(request: Request) -> Dict[str, Any]:
        _validate_admin(request, settings)
        customer_id = _customer_id(request, settings)
        monitor_pid, _ = _read_monitor_process()
        return {
            "monitor_running": monitor_pid is not None,
            "monitor_pid": monitor_pid,
            "stats": _monitor_stats_for_customer(portal_state=portal_state, customer_id=customer_id),
        }

    @router.get("/api/monitor/logs")
    async def monitor_logs(request: Request, lines: int = 100) -> Dict[str, Any]:
        _validate_admin(request, settings)
        latest = next(iter(_list_monitor_logs(limit=1)), None)
        return {
            "path": str(latest) if latest is not None else "",
            "lines": _tail_monitor_log_lines(latest, lines=max(lines, 1)) if latest is not None else [],
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

        try:
            _validate_cluster_create_identity(settings, region_az=region_az)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"AWS identity preflight failed: {exc}") from exc

        try:
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
        except (RuntimeError, FileNotFoundError) as exc:
            raise HTTPException(status_code=502, detail=f"Cluster create preflight failed: {exc}") from exc
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

    @router.post("/api/customers/{customer_id}/manifests", status_code=201)
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
        manifest_id = str(manifest.get("manifest_id") or "")
        return {
            "manifest": manifest,
            "download_url": f"/api/customers/{customer_id}/manifests/{manifest_id}/download",
        }

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
                    region_az = f"{target_region}{az_suffix}"
                    try:
                        _validate_cluster_create_identity(settings, region_az=region_az)
                    except Exception as exc:
                        raise HTTPException(
                            status_code=502,
                            detail=f"AWS identity preflight failed: {exc}",
                        ) from exc
                    try:
                        job = start_create_job(
                            region_az=region_az,
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
                    except (RuntimeError, FileNotFoundError) as exc:
                        raise HTTPException(
                            status_code=502,
                            detail=f"Cluster create preflight failed: {exc}",
                        ) from exc
                    cluster_create_job_id = job.job_id
                state = "pending_cluster_creation"
                cluster_region = target_region
                message = f"Waiting for a cluster in {target_region}"

        try:
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
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "workset_id": workset["workset_id"],
            "state": workset["state"],
            "cluster_name": workset["cluster_name"],
            "cluster_region": workset["cluster_region"],
            "cluster_create_job_id": workset["cluster_create_job_id"],
            "message": workset["message"],
        }

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
