from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

from daylily_auth_cognito import complete_cognito_callback, start_cognito_login
from daylily_auth_cognito.browser.session import CognitoWebAuthError
from fastapi import FastAPI, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from daylib_ursa import __version__
from daylib_ursa.anomalies import open_anomaly_repository
from daylib_ursa.auth import (
    AuthError,
    CurrentUser,
    USER_TOKEN_TEMPLATE,
    build_web_session_config,
    clear_session_user,
    get_current_user,
    persist_session_user,
    session_principal_from_current_user,
)
from daylib_ursa.cluster_jobs import region_from_region_az
from daylib_ursa.config import _require_bare_cognito_domain
from daylib_ursa.analysis_commands import command_catalog_payload
from daylib_ursa.observability import (
    build_api_health_payload,
    build_auth_health_payload,
    build_db_health_payload,
    build_endpoint_health_payload,
    build_health_payload,
    build_obs_services_payload,
)
from daylib_ursa.ursa_config import (
    _stable_deployment_color_hex,
    _stable_region_color_hex,
    get_ursa_config,
)

LOGGER = logging.getLogger(__name__)
CLUSTER_CREATE_REGION_SUGGESTIONS = [
    "us-west-2",
    "us-east-1",
    "us-east-2",
    "ap-south-1",
    "eu-central-1",
]
_SENSITIVE_CONFIG_TOKENS = (
    "secret",
    "token",
    "password",
    "passwd",
    "key",
    "credential",
    "private",
    "signing",
    "session",
    "cookie",
    "authorization",
    "client_secret",
    "api_key",
    "access_key",
    "secret_key",
)


def _run_git_command(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return str(completed.stdout or "").strip()


def _resolve_git_metadata(repo_root: Path) -> dict[str, str]:
    metadata = {"branch": "unavailable", "tag": "unreleased", "commit": "unavailable"}
    try:
        branch = _run_git_command(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
        if branch:
            metadata["branch"] = "detached" if branch == "HEAD" else branch
        metadata["commit"] = (
            _run_git_command(repo_root, "rev-parse", "--short", "HEAD") or "unavailable"
        )
        try:
            tag = _run_git_command(repo_root, "describe", "--tags", "--exact-match")
        except subprocess.CalledProcessError:
            tag = ""
        if tag:
            metadata["tag"] = tag
    except (OSError, subprocess.CalledProcessError):
        return metadata
    return metadata


def _is_sensitive_config_path(path: str) -> bool:
    lowered = str(path or "").lower()
    return any(token in lowered for token in _SENSITIVE_CONFIG_TOKENS)


def _format_config_value(value: Any) -> str:
    if value is None:
        return "<unset>"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, indent=2, sort_keys=True, default=str)
    return str(value)


def _flatten_effective_config(
    value: Any,
    *,
    prefix: str = "",
    rows: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    rows = [] if rows is None else rows
    if prefix and _is_sensitive_config_path(prefix):
        rows.append({"path": prefix, "value": "<redacted>" if value else "<unset>"})
        return rows
    if isinstance(value, dict):
        if prefix:
            rows.append({"path": prefix, "value": _format_config_value(value)})
        for key in sorted(value.keys(), key=str):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_effective_config(value[key], prefix=child_prefix, rows=rows)
        return rows
    if isinstance(value, list):
        if prefix:
            rows.append({"path": prefix, "value": _format_config_value(value) if value else "[]"})
        for index, item in enumerate(value):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            _flatten_effective_config(item, prefix=child_prefix, rows=rows)
        return rows
    if prefix:
        rows.append({"path": prefix, "value": _format_config_value(value)})
    return rows


def _safe_settings_snapshot(settings: Any, *, config_path: Path | None = None) -> dict[str, Any]:
    effective = {}
    if hasattr(settings, "model_dump"):
        effective = dict(settings.model_dump(mode="json", exclude_none=False))
    else:
        effective = {
            key: getattr(settings, key)
            for key in dir(settings)
            if not key.startswith("_") and not callable(getattr(settings, key))
        }
    region_value = (
        getattr(settings, "day_aws_region", "")
        or getattr(settings, "get_effective_region", lambda: "")()
        or ""
    )
    effective["build_version"] = __version__
    effective["config_path"] = str(config_path or "")
    effective["environment"] = str(getattr(settings, "daylily_env", "") or "")
    effective["deployment"] = dict(getattr(settings, "deployment", {}) or {})
    effective["region"] = str(region_value)
    return effective


def _build_environment_chrome(settings: Any) -> dict[str, Any]:
    deployment_name = str(getattr(settings, "deployment_name", "") or "").strip()
    region_name = str(
        getattr(settings, "day_aws_region", "")
        or getattr(settings, "get_effective_region", lambda: "")()
        or ""
    ).strip()
    return {
        "show": bool(getattr(settings, "ui_show_environment_chrome", True)),
        "deployment": {
            "name": deployment_name,
            "color": _stable_deployment_color_hex(deployment_name) if deployment_name else "",
        },
        "region": {
            "name": region_name,
            "color": _stable_region_color_hex(region_name) if region_name else "",
        },
    }


def mount_gui(app: FastAPI) -> None:
    gui_root = Path(__file__).resolve().parent / "gui"
    repo_root = Path(__file__).resolve().parents[1]
    templates = Jinja2Templates(directory=str(gui_root / "templates"))
    static_root = gui_root / "static"
    if static_root.is_dir():
        app.mount("/ui/static", StaticFiles(directory=str(static_root)), name="ui-static")

    app.state.ursa_config = getattr(app.state, "ursa_config", None) or get_ursa_config()
    git_meta = _resolve_git_metadata(repo_root)

    def _config_source():
        cfg = getattr(app.state, "ursa_config", None)
        if cfg is None:
            cfg = get_ursa_config()
            app.state.ursa_config = cfg
        return cfg

    def _config_path() -> Path | None:
        cfg = _config_source()
        path = getattr(cfg, "config_path", None)
        return path if isinstance(path, Path) else None

    def _environment_chrome_context() -> dict[str, object]:
        settings = getattr(app.state, "settings", None)
        deployment_name = str(getattr(settings, "deployment_name", "") or "").strip()
        region_name = str(
            getattr(settings, "day_aws_region", "")
            or getattr(settings, "get_effective_region", lambda: "")()
            or ""
        ).strip()
        return {
            "show": bool(getattr(settings, "ui_show_environment_chrome", True)),
            "deployment": {
                "name": deployment_name,
                "color": _stable_deployment_color_hex(deployment_name) if deployment_name else "",
            },
            "region": {
                "name": region_name,
                "color": _stable_region_color_hex(region_name) if region_name else "",
            },
        }

    def _next_path(raw_value: str | None) -> str:
        value = str(raw_value or "").strip()
        return value if value.startswith("/") else "/"

    def _cognito_login_path(next_path: str) -> str:
        return f"/auth/login?next={_next_path(next_path)}"

    def _cognito_settings() -> dict[str, str]:
        settings = getattr(app.state, "settings", None)
        values = {
            "domain": str(getattr(settings, "cognito_domain", "") or "").strip(),
            "client_id": str(getattr(settings, "cognito_app_client_id", "") or "").strip(),
            "client_secret": str(getattr(settings, "cognito_app_client_secret", "") or "").strip(),
            "callback_url": str(getattr(settings, "cognito_callback_url", "") or "").strip(),
            "logout_url": str(getattr(settings, "cognito_logout_url", "") or "").strip(),
        }
        missing = [
            key for key in ("domain", "client_id", "callback_url", "logout_url") if not values[key]
        ]
        if missing:
            raise HTTPException(
                status_code=503,
                detail=f"Cognito authentication is not configured: missing {', '.join(missing)}",
            )
        try:
            values["domain"] = _require_bare_cognito_domain(
                values["domain"],
                field_name="cognito_domain",
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Cognito authentication is not configured: {exc}",
            ) from exc
        return values

    def _web_session_config():
        settings = getattr(app.state, "settings", None)
        server_instance_id = str(getattr(app.state, "server_instance_id", "") or "").strip()
        if settings is None or not server_instance_id:
            raise HTTPException(status_code=503, detail="Authentication provider is not configured")
        return build_web_session_config(settings, server_instance_id)

    def _build_cognito_logout_url(*, state: str | None = None) -> str:
        cognito = _cognito_settings()
        query = {
            "client_id": cognito["client_id"],
            "redirect_uri": cognito["callback_url"].rstrip("/"),
            "response_type": "code",
        }
        if state:
            query["state"] = state
        return f"https://{cognito['domain']}/logout?{urlencode(query)}"

    def _auth_error_message(reason: str | None) -> str | None:
        messages = {
            "auth_error": "An authentication error prevented sign-in from completing.",
            "session_expired": "Your session ended before the requested page loaded.",
            "not_authorized": "This account is not provisioned for Ursa access.",
            "invalid_state": "The sign-in state was invalid or expired. Start sign-in again.",
            "missing_code": "The sign-in response was incomplete. Start sign-in again.",
            "token_exchange_failed": "The sign-in exchange failed. Start sign-in again.",
            "cognito_sign_in_misconfigured": (
                "Ursa Cognito sign-in is misconfigured. The shared app client callback/logout "
                "URLs or redirect URI do not match this Ursa deployment."
            ),
            "cognito_logout_misconfigured": (
                "Ursa cleared your local session, but the shared Cognito logout contract is "
                "misconfigured. Update the shared app client redirect URLs for this Ursa deployment."
            ),
        }
        clean_reason = str(reason or "").strip()
        return messages.get(clean_reason) or None

    def _require_allowed_cognito_email(email: str) -> None:
        settings = getattr(app.state, "settings", None)
        if settings is None:
            raise AuthError("Authentication provider is not configured")
        valid, message = settings.validate_email_domain(email)
        if not valid:
            raise AuthError(f"not authorized: {message}")

    def _auth_error_redirect(reason: str) -> RedirectResponse:
        return RedirectResponse(
            url=f"/auth/error?reason={quote(reason)}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    def _login_redirect_response(request: Request) -> RedirectResponse:
        next_path = quote(str(request.url.path or "/"), safe="/?=&")
        reason = str(getattr(request.state, "cognito_auth_reason", "") or "").strip()
        suffix = f"&reason={quote(reason)}" if reason else ""
        return RedirectResponse(
            url=f"/login?next={next_path}{suffix}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    async def _resolve_cognito_session_principal(token_payload: dict[str, Any], request: Request):
        try:
            id_token = str(token_payload.get("id_token") or "").strip()
            access_token = str(token_payload.get("access_token") or "").strip()
            if not id_token and not access_token:
                raise AuthError("Cognito token response missing access_token or id_token")
            auth_provider = getattr(app.state, "auth_provider", None)
            if auth_provider is None:
                raise AuthError("Authentication provider is not configured")
            actor = auth_provider.resolve_access_token(
                id_token or access_token,
                paired_access_token=access_token or None,
            )
            _require_allowed_cognito_email(actor.email)
            return session_principal_from_current_user(actor)
        except AuthError as exc:
            LOGGER.warning(
                "Ursa Cognito principal resolution failed during callback: %s",
                exc,
                exc_info=exc,
            )
            reason = "not_authorized" if "not authorized" in str(exc).lower() else "auth_error"
            raise CognitoWebAuthError(
                reason,
                str(exc),
                status_code=status.HTTP_401_UNAUTHORIZED,
                redirect_to_error=True,
            ) from exc

    def _session_actor(request: Request) -> CurrentUser | None:
        try:
            return get_current_user(request)
        except HTTPException:
            return None

    def _resource_store():
        resources = getattr(app.state, "resource_store", None)
        if resources is None:
            raise HTTPException(status_code=503, detail="Resource store is not configured")
        return resources

    def _token_service():
        service = getattr(app.state, "token_service", None)
        if service is None:
            raise HTTPException(status_code=503, detail="Token service is not configured")
        return service

    def _list_all_tokens_for_admin(actor: CurrentUser) -> list[Any]:
        service = _token_service()
        try:
            return service.list_tokens(actor=actor, owner_user_id="*")
        except AuthError:
            if not actor.is_admin or not hasattr(service, "backend"):
                return []
            with service.backend.session_scope(commit=False) as session:
                tokens = service.backend.list_instances_by_template(
                    session,
                    template_code=USER_TOKEN_TEMPLATE,
                    limit=500,
                )
                return [service._token_record(session, token) for token in tokens]

    def _cluster_service():
        service = getattr(app.state, "cluster_service", None)
        if service is None:
            raise HTTPException(status_code=503, detail="Cluster service is not configured")
        return service

    def _observability_store():
        store = getattr(app.state, "observability", None)
        if store is None:
            raise HTTPException(status_code=503, detail="Observability store is not configured")
        return store

    def _anomaly_repository():
        resources = _resource_store()
        token_service = _token_service()
        backend = getattr(resources, "backend", None) or getattr(token_service, "backend", None)
        if backend is None:
            raise HTTPException(status_code=503, detail="Anomaly repository is not configured")
        return open_anomaly_repository(
            resource_store=resources,
            settings=app.state.settings,
            backend=backend,
        )

    def _render_page(
        request: Request,
        *,
        template_name: str,
        page_title: str,
        active_page: str,
        secondary_page: str | None = None,
        admin_only: bool = False,
        context: dict[str, Any] | None = None,
    ) -> HTMLResponse | RedirectResponse:
        actor = _session_actor(request)
        if actor is None:
            return _login_redirect_response(request)
        if admin_only and not actor.is_admin:
            raise HTTPException(status_code=403, detail="Admin privileges are required")

        def _json_default(value: Any):
            if hasattr(value, "__dict__"):
                return value.__dict__
            return str(value)

        template_context = {
            "request": request,
            "actor": actor,
            "page_title": page_title,
            "active_page": active_page,
            "secondary_page": secondary_page,
            "page_data_json": json.dumps(context or {}, default=_json_default),
            "environment_chrome": _environment_chrome_context(),
            "git_meta": git_meta,
            "app_version": __version__,
        }
        template_context.update(context or {})
        return templates.TemplateResponse(request, template_name, template_context)

    def _admin_config_context() -> dict[str, Any]:
        settings = getattr(app.state, "settings", None)
        if settings is None:
            raise HTTPException(status_code=503, detail="Settings are not configured")
        config_snapshot = _safe_settings_snapshot(settings, config_path=_config_path())
        rows = _flatten_effective_config(config_snapshot)
        rows.sort(key=lambda item: item["path"])
        return {
            "config_path": str(_config_path() or ""),
            "effective_config_rows": rows,
            "ui_show_environment_chrome": bool(
                getattr(settings, "ui_show_environment_chrome", True)
            ),
        }

    def _json_text(value: Any) -> str:
        def _json_default(inner: Any):
            if hasattr(inner, "__dict__"):
                return inner.__dict__
            return str(inner)

        return json.dumps(value, indent=2, default=_json_default)

    def _list_worksets(actor: CurrentUser) -> list[Any]:
        return _resource_store().list_worksets(tenant_id=actor.tenant_id)

    def _list_manifests(actor: CurrentUser) -> list[Any]:
        return _resource_store().list_manifests(tenant_id=actor.tenant_id)

    def _list_analysis_jobs(actor: CurrentUser) -> list[Any]:
        return _resource_store().list_analysis_jobs(
            tenant_id=None if actor.is_admin else actor.tenant_id
        )

    def _list_analyses(actor: CurrentUser) -> list[Any]:
        return app.state.store.list_analyses(
            tenant_id=None if actor.is_admin else actor.tenant_id,
        )

    def _list_buckets(actor: CurrentUser) -> list[Any]:
        return _resource_store().list_linked_buckets(tenant_id=actor.tenant_id)

    def _allowed_regions() -> list[str]:
        service = getattr(app.state, "cluster_service", None)
        runtime_regions = [
            str(region or "").strip()
            for region in list(getattr(service, "regions", []) or [])
            if str(region or "").strip()
        ]
        if runtime_regions:
            return runtime_regions
        settings = getattr(app.state, "settings", None)
        if settings is None or not hasattr(settings, "get_allowed_regions"):
            return []
        return list(settings.get_allowed_regions())

    def _cluster_create_regions() -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        for region in [*CLUSTER_CREATE_REGION_SUGGESTIONS, *_allowed_regions()]:
            normalized = str(region or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            values.append(normalized)
        return values

    def _active_cluster_create_jobs(jobs: list[Any]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for job in jobs:
            state = str(getattr(job, "state", "") or "").strip().upper()
            if state not in {"QUEUED", "RUNNING"}:
                continue
            cluster_name = str(getattr(job, "cluster_name", "") or "").strip()
            region = str(getattr(job, "region", "") or "").strip()
            region_az = str(getattr(job, "region_az", "") or "").strip()
            if not region:
                region = region_from_region_az(region_az)
            if not cluster_name or not region:
                continue
            items.append(
                {
                    "job_euid": str(getattr(job, "job_euid", "") or "").strip(),
                    "cluster_name": cluster_name,
                    "region": region,
                    "region_az": region_az,
                    "state": state,
                    "created_at": str(getattr(job, "created_at", "") or "").strip(),
                }
            )
        return items

    def _cluster_region_sections(
        clusters: list[dict[str, Any]],
        jobs: list[Any],
    ) -> list[dict[str, Any]]:
        scanned_regions = _allowed_regions()
        live_by_region: dict[str, list[dict[str, Any]]] = {}
        live_cluster_keys: set[tuple[str, str]] = set()
        for cluster in clusters:
            region = str(cluster.get("region") or "").strip()
            cluster_name = str(cluster.get("cluster_name") or "").strip()
            if not region:
                continue
            live_by_region.setdefault(region, []).append(cluster)
            if cluster_name:
                live_cluster_keys.add((region, cluster_name))

        pending_by_region: dict[str, list[dict[str, Any]]] = {}
        for job in _active_cluster_create_jobs(jobs):
            if (job["region"], job["cluster_name"]) in live_cluster_keys:
                continue
            pending_by_region.setdefault(job["region"], []).append(job)

        ordered_regions: list[str] = []
        seen_regions: set[str] = set()
        for region in [*scanned_regions, *live_by_region.keys(), *pending_by_region.keys()]:
            normalized = str(region or "").strip()
            if not normalized or normalized in seen_regions:
                continue
            seen_regions.add(normalized)
            ordered_regions.append(normalized)

        sections: list[dict[str, Any]] = []
        for region in ordered_regions:
            live_clusters = sorted(
                list(live_by_region.get(region) or []),
                key=lambda item: str(item.get("cluster_name") or ""),
            )
            pending_jobs = sorted(
                list(pending_by_region.get(region) or []),
                key=lambda item: str(item.get("created_at") or ""),
                reverse=True,
            )
            sections.append(
                {
                    "region": region,
                    "clusters": live_clusters,
                    "pending_jobs": pending_jobs,
                    "live_count": len(live_clusters),
                    "pending_count": len(pending_jobs),
                }
            )
        return sections

    def _aws_profile_label() -> str:
        settings = getattr(app.state, "settings", None)
        value = str(getattr(settings, "aws_profile", "") or "").strip()
        return value or "default"

    def _analysis_command_catalog_context() -> dict[str, Any]:
        try:
            return command_catalog_payload()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _analysis_command_summary(metadata: dict[str, Any]) -> dict[str, Any]:
        command = dict(metadata.get("analysis_command") or {})
        profile = dict(command.get("profile") or {})
        pipeline_type = (
            str(profile.get("display_name") or "")
            or str(command.get("command_id") or "")
            or str(metadata.get("pipeline_type") or "")
            or "germline"
        )
        reference_genome = str(profile.get("genome") or "") or str(
            metadata.get("reference_genome") or ""
        )
        return {
            "pipeline_type": pipeline_type,
            "reference_genome": reference_genome,
            "execution_profile": "daylily-ec",
        }

    def _workset_view_model(workset: Any) -> dict[str, Any]:
        metadata = dict(getattr(workset, "metadata", {}) or {})
        command_summary = _analysis_command_summary(metadata)
        manifests = list(getattr(workset, "manifests", []) or [])
        analysis_euids = list(getattr(workset, "analysis_euids", []) or [])
        sample_count = int(metadata.get("sample_count") or 0)
        if sample_count <= 0:
            sample_count = sum(
                len(getattr(manifest, "artifact_euids", []) or []) for manifest in manifests
            )
        if sample_count <= 0:
            sample_count = len(getattr(workset, "artifact_set_euids", []) or [])
        return {
            "workset_id": getattr(workset, "workset_euid", ""),
            "workset_name": getattr(workset, "name", ""),
            "name": getattr(workset, "name", ""),
            "workset_euid": getattr(workset, "workset_euid", ""),
            "state": getattr(workset, "state", "ACTIVE"),
            "workset_type": str(metadata.get("workset_type") or "ruo"),
            "pipeline_type": command_summary["pipeline_type"],
            "reference_genome": command_summary["reference_genome"],
            "execution_profile": command_summary["execution_profile"],
            "customer_id": str(getattr(workset, "tenant_id", "") or ""),
            "s3_status": str(metadata.get("s3_status") or "unknown"),
            "execution_cluster_name": str(
                metadata.get("preferred_cluster") or metadata.get("cluster_name") or ""
            ),
            "execution_cluster_region": str(metadata.get("cluster_region") or ""),
            "progress": int(metadata.get("progress") or 0),
            "progress_step": str(
                metadata.get("progress_step") or metadata.get("current_step") or ""
            ),
            "started_at": str(
                metadata.get("started_at") or metadata.get("execution_started_at") or ""
            ),
            "updated_at": getattr(workset, "updated_at", ""),
            "created_at": getattr(workset, "created_at", ""),
            "compute_cost": float(metadata.get("compute_cost") or 0.0),
            "storage_bytes": int(metadata.get("storage_bytes") or 0),
            "storage_human": str(metadata.get("storage_human") or "—"),
            "storage_available": bool(
                metadata.get("storage_available") or metadata.get("storage_bytes")
            ),
            "sample_count": sample_count,
            "manifests": manifests,
            "analysis_euids": analysis_euids,
            "artifact_set_euids": list(getattr(workset, "artifact_set_euids", []) or []),
            "metadata": metadata,
        }

    def _filter_worksets(worksets: list[Any], request: Request) -> dict[str, Any]:
        items = [_workset_view_model(item) for item in worksets]
        filter_status = str(request.query_params.get("status") or "").strip().lower()
        filter_type = str(request.query_params.get("type") or "").strip().lower()
        filter_search = str(request.query_params.get("search") or "").strip().lower()
        filter_sort = str(request.query_params.get("sort") or "created_desc").strip().lower()
        filtered = items
        if filter_status:
            filtered = [
                item
                for item in filtered
                if str(item.get("state") or "").strip().lower() == filter_status
            ]
        if filter_type:
            filtered = [
                item
                for item in filtered
                if str(item.get("workset_type") or "").strip().lower() == filter_type
            ]
        if filter_search:
            filtered = [
                item
                for item in filtered
                if filter_search in str(item.get("workset_name") or "").lower()
                or filter_search in str(item.get("workset_id") or "").lower()
            ]
        if filter_sort == "created_asc":
            filtered.sort(key=lambda item: str(item.get("created_at") or ""))
        elif filter_sort == "status":
            filtered.sort(
                key=lambda item: (str(item.get("state") or ""), str(item.get("created_at") or "")),
                reverse=True,
            )
        else:
            filtered.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return {
            "worksets": filtered,
            "total_count": len(items),
            "filter_status": filter_status,
            "filter_type": filter_type,
            "filter_search": request.query_params.get("search") or "",
            "filter_sort": filter_sort,
            "current_page": 1,
            "total_pages": 1,
        }

    def _format_file_size(size_bytes: int | None) -> str:
        size = int(size_bytes or 0)
        if size < 1024:
            return f"{size} B"
        units = ["KB", "MB", "GB", "TB"]
        scaled = float(size)
        for unit in units:
            scaled /= 1024.0
            if scaled < 1024.0 or unit == units[-1]:
                precision = 0 if scaled >= 100 else 1
                return f"{scaled:.{precision}f} {unit}"
        return f"{size} B"

    def _detect_file_format(filename: str) -> str | None:
        lower = str(filename or "").lower()
        suffix_map = (
            (".fastq.gz", "fastq"),
            (".fq.gz", "fastq"),
            (".fastq", "fastq"),
            (".fq", "fastq"),
            (".bam", "bam"),
            (".cram", "cram"),
            (".vcf.gz", "vcf"),
            (".vcf", "vcf"),
            (".tsv", "tsv"),
            (".csv", "csv"),
            (".txt", "txt"),
        )
        for suffix, label in suffix_map:
            if lower.endswith(suffix):
                return label
        return None

    def _bucket_browse_context(
        actor: CurrentUser, bucket_id: str, prefix: str = ""
    ) -> dict[str, Any]:
        bucket = _resource_store().get_linked_bucket(bucket_id)
        if bucket is None or str(bucket.state or "").upper() == "DELETED":
            raise HTTPException(status_code=404, detail="Bucket not found")
        if not actor.is_admin and bucket.tenant_id != actor.tenant_id:
            raise HTTPException(status_code=403, detail="Bucket is outside the caller tenant")
        normalized_prefix = str(prefix or "").lstrip("/")
        restricted_prefix = str(getattr(bucket, "prefix_restriction", "") or "").strip().lstrip("/")
        if restricted_prefix:
            restricted_prefix = restricted_prefix.rstrip("/") + "/"
        if (
            restricted_prefix
            and normalized_prefix
            and not normalized_prefix.startswith(restricted_prefix)
        ):
            raise HTTPException(
                status_code=403, detail="Prefix is outside the linked bucket restriction"
            )
        current_prefix = normalized_prefix or restricted_prefix
        response = app.state.s3_client.list_objects_v2(
            Bucket=bucket.bucket_name,
            Prefix=current_prefix or "",
            Delimiter="/",
            MaxKeys=500,
        )
        items: list[dict[str, Any]] = []
        for common_prefix in response.get("CommonPrefixes", []):
            folder_path = str(common_prefix.get("Prefix") or "")
            folder_name = folder_path.rstrip("/").split("/")[-1]
            items.append(
                {
                    "name": folder_name,
                    "is_folder": True,
                    "key": folder_path,
                    "size_bytes": None,
                    "size_human": "--",
                    "last_modified": None,
                    "file_format": None,
                    "is_registered": False,
                }
            )
        for obj in response.get("Contents", []):
            key = str(obj.get("Key") or "")
            if not key or key == current_prefix:
                continue
            name = key.split("/")[-1]
            if not name:
                continue
            size_bytes = int(obj.get("Size") or 0)
            last_modified = obj.get("LastModified")
            items.append(
                {
                    "name": name,
                    "is_folder": False,
                    "key": key,
                    "size_bytes": size_bytes,
                    "size_human": _format_file_size(size_bytes),
                    "last_modified": last_modified.isoformat()
                    if hasattr(last_modified, "isoformat")
                    else None,
                    "file_format": _detect_file_format(name),
                    "is_registered": False,
                }
            )
        breadcrumbs = [{"name": "/", "prefix": restricted_prefix or ""}]
        if current_prefix:
            root_prefix = restricted_prefix or ""
            suffix = (
                current_prefix[len(root_prefix) :]
                if root_prefix and current_prefix.startswith(root_prefix)
                else current_prefix
            )
            running_prefix = root_prefix
            for part in [segment for segment in suffix.rstrip("/").split("/") if segment]:
                running_prefix = f"{running_prefix}{part}/"
                breadcrumbs.append({"name": part, "prefix": running_prefix})
        if not current_prefix:
            parent_prefix = None
        else:
            parent_parts = current_prefix.rstrip("/").split("/")[:-1]
            parent_prefix = (
                "/".join(parent_parts) + "/" if parent_parts else (restricted_prefix or "")
            )
            if (
                restricted_prefix
                and parent_prefix
                and not parent_prefix.startswith(restricted_prefix)
            ):
                parent_prefix = restricted_prefix
        return {
            "bucket": bucket,
            "items": items,
            "breadcrumbs": breadcrumbs,
            "current_prefix": current_prefix or "",
            "parent_prefix": parent_prefix,
        }

    def _dashboard_context(actor: CurrentUser) -> dict[str, Any]:
        worksets = _list_worksets(actor)
        manifests = _list_manifests(actor)
        analyses = _list_analyses(actor)
        buckets = _list_buckets(actor)
        tokens = _token_service().list_tokens(actor=actor)
        stats = {
            "worksets": len(worksets),
            "manifests": len(manifests),
            "analyses": len(analyses),
            "tokens": len(tokens),
            "buckets": len(buckets),
            "active_worksets": len(
                [
                    item
                    for item in worksets
                    if str(item.state).upper() not in {"COMPLETE", "COMPLETED", "ERROR"}
                ]
            ),
            "completed_worksets": len(
                [item for item in worksets if str(item.state).upper() in {"COMPLETE", "COMPLETED"}]
            ),
            "errored_worksets": len(
                [item for item in worksets if str(item.state).upper() == "ERROR"]
            ),
        }
        if actor.is_admin:
            cluster_items = _cluster_service().get_all_clusters_with_status(
                force_refresh=False, fetch_ssh_status=False
            )
            cluster_jobs = _resource_store().list_cluster_jobs(tenant_id=None)
            stats["clusters"] = len(cluster_items)
            stats["cluster_jobs"] = len(cluster_jobs)
        return {
            "stats": stats,
            "worksets": worksets[:8],
            "recent_manifests": manifests[:5],
            "recent_analyses": analyses[:5],
        }

    def _usage_context(actor: CurrentUser) -> dict[str, Any]:
        worksets = _list_worksets(actor)
        manifests = _list_manifests(actor)
        analyses = _list_analyses(actor)
        buckets = _list_buckets(actor)
        total_manifest_refs = sum(len(item.artifact_euids) for item in manifests)
        return {
            "usage": {
                "worksets_total": len(worksets),
                "active_worksets": len(
                    [
                        item
                        for item in worksets
                        if str(item.state).upper() not in {"COMPLETE", "COMPLETED", "ERROR"}
                    ]
                ),
                "completed_worksets": len(
                    [
                        item
                        for item in worksets
                        if str(item.state).upper() in {"COMPLETE", "COMPLETED"}
                    ]
                ),
                "manifests_total": len(manifests),
                "analysis_total": len(analyses),
                "linked_buckets": len(buckets),
                "artifact_references": total_manifest_refs,
                "estimated_compute_cost_usd": 0.0,
                "estimated_storage_cost_usd": 0.0,
                "estimated_transfer_cost_usd": 0.0,
            },
            "worksets": worksets[:20],
            "manifests": manifests[:20],
            "analyses": analyses[:20],
        }

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request, next: str = "/", reason: str = ""):
        actor = _session_actor(request)
        if actor is not None:
            return RedirectResponse(url=_next_path(next), status_code=status.HTTP_303_SEE_OTHER)
        return templates.TemplateResponse(
            request,
            "login.html",
            {
                "request": request,
                "next_path": _next_path(next),
                "cognito_login_url": _cognito_login_path(next),
                "error": _auth_error_message(reason),
                "environment_chrome": _environment_chrome_context(),
                "git_meta": git_meta,
                "app_version": __version__,
            },
        )

    @app.get("/auth/login", include_in_schema=False)
    async def auth_login(request: Request, next: str = "/"):
        try:
            return start_cognito_login(request, _web_session_config(), _next_path(next))
        except (HTTPException, ValueError) as exc:
            LOGGER.error("Ursa Cognito sign-in is misconfigured: %s", exc)
            return _auth_error_redirect("cognito_sign_in_misconfigured")

    @app.get("/auth/callback", include_in_schema=False)
    async def auth_callback(request: Request, code: str = "", state: str = ""):
        try:
            return await complete_cognito_callback(
                request,
                _web_session_config(),
                code.strip() or None,
                state.strip() or None,
                _resolve_cognito_session_principal,
            )
        except CognitoWebAuthError as exc:
            LOGGER.warning("Ursa Cognito callback failed: %s", exc)
            request.state.cognito_auth_reason = exc.reason
            return RedirectResponse(
                url=f"/auth/error?reason={quote(exc.reason)}",
                status_code=status.HTTP_303_SEE_OTHER,
            )

    @app.post("/login", response_class=HTMLResponse)
    async def login_submit(
        request: Request,
        access_token: str = Form(...),
        next_path: str = Form("/"),
    ):
        token = str(access_token or "").strip()
        if not token:
            return templates.TemplateResponse(
                request,
                "login.html",
                {
                    "request": request,
                    "next_path": _next_path(next_path),
                    "cognito_login_url": _cognito_login_path(next_path),
                    "error": "Authentication token is required",
                    "environment_chrome": _environment_chrome_context(),
                    "git_meta": git_meta,
                    "app_version": __version__,
                },
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        try:
            auth_provider = getattr(app.state, "auth_provider", None)
            if auth_provider is None:
                raise AuthError("Authentication provider is not configured")
            actor = auth_provider.resolve_access_token(token)
            _require_allowed_cognito_email(actor.email)
        except AuthError as exc:
            return templates.TemplateResponse(
                request,
                "login.html",
                {
                    "request": request,
                    "next_path": _next_path(next_path),
                    "cognito_login_url": _cognito_login_path(next_path),
                    "error": str(exc),
                    "environment_chrome": _environment_chrome_context(),
                    "git_meta": git_meta,
                    "app_version": __version__,
                },
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        persist_session_user(request, actor)
        return RedirectResponse(url=_next_path(next_path), status_code=status.HTTP_303_SEE_OTHER)

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon_redirect() -> RedirectResponse:
        return RedirectResponse(
            url="/ui/static/favicon.svg", status_code=status.HTTP_307_TEMPORARY_REDIRECT
        )

    @app.get("/auth/error", include_in_schema=False)
    async def auth_error(request: Request, reason: str = "auth_error"):
        message = (
            _auth_error_message(reason)
            or _auth_error_message("auth_error")
            or "Authentication failed."
        )
        return templates.TemplateResponse(
            request,
            "login.html",
            {
                "request": request,
                "next_path": "/",
                "cognito_login_url": _cognito_login_path("/"),
                "error": message,
                "auth_badge": "Access Review",
                "auth_title": "This account could not complete sign-in.",
                "auth_description": "Ursa access is provisioned per deployment and user role.",
                "auth_card_title": "Sign-in was blocked",
                "auth_card_copy": message,
                "auth_primary_href": "/auth/login",
                "auth_primary_label": "Return to Sign In",
                "environment_chrome": _environment_chrome_context(),
                "git_meta": git_meta,
                "app_version": __version__,
            },
            status_code=status.HTTP_403_FORBIDDEN,
        )

    async def _logout_response(request: Request):
        logout_reason: str | None = None
        try:
            logout_url = _build_cognito_logout_url()
        except (HTTPException, ValueError) as exc:
            LOGGER.error("Ursa Cognito logout is misconfigured: %s", exc)
            logout_reason = "cognito_logout_misconfigured"
            logout_url = ""
        clear_session_user(request)
        return (
            _auth_error_redirect(logout_reason)
            if logout_reason
            else RedirectResponse(
                url=logout_url,
                status_code=status.HTTP_303_SEE_OTHER,
            )
        )

    @app.get("/auth/logout", include_in_schema=False)
    async def auth_logout_get(request: Request):
        return await _logout_response(request)

    @app.post("/auth/logout", include_in_schema=False)
    async def auth_logout_post(request: Request):
        return await _logout_response(request)

    @app.get("/logout")
    async def logout(request: Request):
        return await _logout_response(request)

    @app.get("/", response_class=HTMLResponse)
    async def dashboard_page(request: Request):
        actor = _session_actor(request)
        if actor is None:
            return _login_redirect_response(request)
        return _render_page(
            request,
            template_name="dashboard.html",
            page_title="Dashboard",
            active_page="dashboard",
            context=_dashboard_context(actor),
        )

    @app.get("/usage", response_class=HTMLResponse)
    async def usage_page(request: Request):
        actor = _session_actor(request)
        if actor is None:
            return _login_redirect_response(request)
        return _render_page(
            request,
            template_name="usage.html",
            page_title="Usage Summary",
            active_page="usage",
            context=_usage_context(actor),
        )

    @app.get("/worksets", response_class=HTMLResponse)
    async def worksets_page(request: Request):
        actor = _session_actor(request)
        if actor is None:
            return _login_redirect_response(request)
        return _render_page(
            request,
            template_name="worksets/list.html",
            page_title="Worksets",
            active_page="worksets",
            context=_filter_worksets(_list_worksets(actor), request),
        )

    @app.get("/worksets/new", response_class=HTMLResponse)
    async def worksets_new_page(request: Request):
        actor = _session_actor(request)
        if actor is None:
            return _login_redirect_response(request)
        manifests = _list_manifests(actor)
        clusters = (
            [
                item.to_dict(include_sensitive=False)
                for item in _cluster_service().get_all_clusters_with_status(
                    force_refresh=False, fetch_ssh_status=False
                )
            ]
            if actor.is_admin
            else []
        )
        return _render_page(
            request,
            template_name="worksets/new.html",
            page_title="Create Workset",
            active_page="worksets",
            context={
                "worksets": _list_worksets(actor),
                "manifests": manifests,
                "allowed_regions": _allowed_regions(),
                "clusters": clusters,
                "is_admin": actor.is_admin,
                "analysis_command_catalog": _analysis_command_catalog_context(),
            },
        )

    @app.get("/worksets/{workset_euid}", response_class=HTMLResponse)
    async def workset_detail_page(request: Request, workset_euid: str):
        actor = _session_actor(request)
        if actor is None:
            return RedirectResponse(
                url=f"/login?next={request.url.path}", status_code=status.HTTP_303_SEE_OTHER
            )
        workset = _resource_store().get_workset(workset_euid)
        if workset is None:
            raise HTTPException(status_code=404, detail="Workset not found")
        if not actor.is_admin and workset.tenant_id != actor.tenant_id:
            raise HTTPException(status_code=403, detail="Workset is outside the caller tenant")
        analyses = [
            item
            for item in _list_analyses(actor)
            if str(getattr(item, "workset_euid", "") or "") == workset_euid
        ]
        return _render_page(
            request,
            template_name="worksets/detail.html",
            page_title=f"Workset {workset.name}",
            active_page="worksets",
            context={
                "workset": workset,
                "analyses": analyses,
                "workset_payload_json": _json_text(workset),
            },
        )

    @app.get("/manifests", response_class=HTMLResponse)
    async def manifests_page(request: Request):
        actor = _session_actor(request)
        if actor is None:
            return RedirectResponse(
                url=f"/login?next={request.url.path}", status_code=status.HTTP_303_SEE_OTHER
            )
        return _render_page(
            request,
            template_name="manifests/index.html",
            page_title="Workset Manifest Generator",
            active_page="manifests",
            context={
                "worksets": _list_worksets(actor),
                "manifests": _list_manifests(actor),
                "buckets": _list_buckets(actor),
            },
        )

    @app.get("/manifests/{manifest_euid}", response_class=HTMLResponse)
    async def manifest_detail_page(request: Request, manifest_euid: str):
        actor = _session_actor(request)
        if actor is None:
            return RedirectResponse(
                url=f"/login?next={request.url.path}", status_code=status.HTTP_303_SEE_OTHER
            )
        manifest = _resource_store().get_manifest(manifest_euid)
        if manifest is None:
            raise HTTPException(status_code=404, detail="Manifest not found")
        if not actor.is_admin and manifest.tenant_id != actor.tenant_id:
            raise HTTPException(status_code=403, detail="Manifest is outside the caller tenant")
        return _render_page(
            request,
            template_name="manifests/detail.html",
            page_title=f"Manifest {manifest.name}",
            active_page="manifests",
            context={"manifest": manifest, "manifest_payload_json": _json_text(manifest)},
        )

    @app.get("/analysis-jobs", response_class=HTMLResponse)
    async def analysis_jobs_page(request: Request):
        actor = _session_actor(request)
        if actor is None:
            return RedirectResponse(
                url=f"/login?next={request.url.path}", status_code=status.HTTP_303_SEE_OTHER
            )
        clusters = (
            [
                item.to_dict(include_sensitive=False)
                for item in _cluster_service().get_all_clusters_with_status(
                    force_refresh=False,
                    fetch_ssh_status=False,
                )
            ]
            if actor.is_admin
            else []
        )
        return _render_page(
            request,
            template_name="analysis_jobs.html",
            page_title="Analysis Launches",
            active_page="analysis_jobs",
            context={
                "worksets": _list_worksets(actor),
                "manifests": _list_manifests(actor),
                "analysis_jobs": _list_analysis_jobs(actor),
                "analysis_command_catalog": _analysis_command_catalog_context(),
                "clusters": clusters,
                "allowed_regions": _allowed_regions(),
                "is_admin": actor.is_admin,
            },
        )

    @app.get("/buckets", response_class=HTMLResponse)
    async def buckets_page(request: Request):
        actor = _session_actor(request)
        if actor is None:
            return RedirectResponse(
                url=f"/login?next={request.url.path}", status_code=status.HTTP_303_SEE_OTHER
            )
        return _render_page(
            request,
            template_name="buckets.html",
            page_title="Linked Buckets",
            active_page="buckets",
            context={
                "buckets": _list_buckets(actor),
                "admin_bucket_profile": _aws_profile_label(),
            },
        )

    @app.get("/buckets/{bucket_id}", response_class=HTMLResponse)
    async def bucket_browse_page(request: Request, bucket_id: str, prefix: str = ""):
        actor = _session_actor(request)
        if actor is None:
            return RedirectResponse(
                url=f"/login?next={request.url.path}", status_code=status.HTTP_303_SEE_OTHER
            )
        return _render_page(
            request,
            template_name="buckets_browse.html",
            page_title="Browse Bucket",
            active_page="buckets",
            context=_bucket_browse_context(actor, bucket_id, prefix),
        )

    @app.get("/analyses", response_class=HTMLResponse)
    async def analyses_page(request: Request):
        actor = _session_actor(request)
        if actor is None:
            return RedirectResponse(
                url=f"/login?next={request.url.path}", status_code=status.HTTP_303_SEE_OTHER
            )
        return _render_page(
            request,
            template_name="analyses/list.html",
            page_title="Analyses",
            active_page="tools",
            secondary_page="analyses",
            context={"analyses": _list_analyses(actor)},
        )

    @app.get("/analyses/{analysis_euid}", response_class=HTMLResponse)
    async def analysis_detail_page(request: Request, analysis_euid: str):
        actor = _session_actor(request)
        if actor is None:
            return RedirectResponse(
                url=f"/login?next={request.url.path}", status_code=status.HTTP_303_SEE_OTHER
            )
        analysis = app.state.store.get_analysis(analysis_euid)
        if analysis is None:
            raise HTTPException(status_code=404, detail="Analysis not found")
        if not actor.is_admin and analysis.tenant_id != actor.tenant_id:
            raise HTTPException(status_code=403, detail="Analysis is outside the caller tenant")
        return _render_page(
            request,
            template_name="analyses/detail.html",
            page_title=f"Analysis {analysis.analysis_euid}",
            active_page="tools",
            secondary_page="analyses",
            context={"analysis": analysis, "analysis_payload_json": _json_text(analysis)},
        )

    @app.get("/artifacts", response_class=HTMLResponse)
    async def artifacts_page(request: Request):
        return _render_page(
            request,
            template_name="artifacts.html",
            page_title="Artifact Tools",
            active_page="tools",
            secondary_page="artifacts",
        )

    @app.get("/tokens", response_class=HTMLResponse)
    async def tokens_page(request: Request):
        actor = _session_actor(request)
        if actor is None:
            return RedirectResponse(
                url=f"/login?next={request.url.path}", status_code=status.HTTP_303_SEE_OTHER
            )
        return _render_page(
            request,
            template_name="tokens/list.html",
            page_title="User Tokens",
            active_page="tools",
            secondary_page="tokens",
            context={"tokens": _token_service().list_tokens(actor=actor)},
        )

    @app.get("/tokens/{token_euid}", response_class=HTMLResponse)
    async def token_detail_page(request: Request, token_euid: str):
        actor = _session_actor(request)
        if actor is None:
            return RedirectResponse(
                url=f"/login?next={request.url.path}", status_code=status.HTTP_303_SEE_OTHER
            )
        service = _token_service()
        tokens = service.list_tokens(actor=actor)
        token = next((item for item in tokens if item.token_euid == token_euid), None)
        if token is None:
            raise HTTPException(status_code=404, detail="User token not found")
        usage = service.list_usage(actor=actor, token_euid=token_euid)
        return _render_page(
            request,
            template_name="tokens/detail.html",
            page_title=f"User Token {token.token_name}",
            active_page="tools",
            secondary_page="tokens",
            context={"token": token, "usage": usage},
        )

    @app.get("/clusters", response_class=HTMLResponse)
    async def clusters_page(request: Request):
        actor = _session_actor(request)
        if actor is None:
            return RedirectResponse(
                url=f"/login?next={request.url.path}", status_code=status.HTTP_303_SEE_OTHER
            )
        clusters = _cluster_service().get_all_clusters_with_status(
            force_refresh=False, fetch_ssh_status=False
        )
        cluster_payload = [item.to_dict(include_sensitive=False) for item in clusters]
        jobs = _resource_store().list_cluster_jobs(
            tenant_id=None if actor.is_admin else actor.tenant_id
        )
        scanned_regions = _allowed_regions()
        active_create_jobs = _active_cluster_create_jobs(jobs)
        return _render_page(
            request,
            template_name="clusters.html",
            page_title="Clusters",
            active_page="clusters",
            admin_only=True,
            context={
                "clusters": cluster_payload,
                "cluster_regions": _cluster_region_sections(cluster_payload, jobs),
                "jobs": jobs,
                "regions": scanned_regions,
                "scan_regions_csv": ",".join(scanned_regions),
                "create_regions": _cluster_create_regions(),
                "is_admin": actor.is_admin,
                "create_mode": False,
                "active_create_jobs_count": len(active_create_jobs),
                "aws_profile_label": _aws_profile_label(),
                "prefill_region": (
                    _cluster_create_regions()[0] if _cluster_create_regions() else ""
                ),
            },
        )

    @app.get("/clusters/{cluster_name}", response_class=HTMLResponse)
    async def cluster_detail_page(request: Request, cluster_name: str, region: str | None = None):
        actor = _session_actor(request)
        if actor is None:
            return RedirectResponse(
                url=f"/login?next={request.url.path}", status_code=status.HTTP_303_SEE_OTHER
            )
        service = _cluster_service()
        resolved_region = str(region or service.get_region_for_cluster(cluster_name) or "").strip()
        if not resolved_region:
            cluster = service.get_cluster_by_name(cluster_name, force_refresh=False)
            if cluster is None:
                raise HTTPException(status_code=404, detail="Cluster not found")
            resolved_region = cluster.region
        cluster = service.describe_cluster(cluster_name, resolved_region)
        jobs = [
            item
            for item in _resource_store().list_cluster_jobs(tenant_id=None)
            if item.cluster_name == cluster_name
        ]
        return _render_page(
            request,
            template_name="cluster_detail.html",
            page_title=f"Cluster {cluster_name}",
            active_page="clusters",
            admin_only=True,
            context={
                "cluster": cluster.to_dict(include_sensitive=False),
                "jobs": jobs,
                "cluster_payload_json": _json_text(cluster.to_dict(include_sensitive=False)),
            },
        )

    @app.get("/clusters/jobs/{job_euid}", response_class=HTMLResponse)
    async def cluster_job_detail_page(request: Request, job_euid: str):
        actor = _session_actor(request)
        if actor is None:
            return RedirectResponse(
                url=f"/login?next={request.url.path}", status_code=status.HTTP_303_SEE_OTHER
            )
        job = _resource_store().get_cluster_job(job_euid)
        if job is None:
            raise HTTPException(status_code=404, detail="Cluster job not found")
        return _render_page(
            request,
            template_name="cluster_job_detail.html",
            page_title=f"Cluster Job {job.job_euid}",
            active_page="clusters",
            admin_only=True,
            context={"job": job, "job_payload_json": _json_text(job)},
        )

    @app.get("/admin", response_class=HTMLResponse, include_in_schema=False)
    async def admin_home(request: Request):
        actor = _session_actor(request)
        if actor is None:
            return RedirectResponse(
                url=f"/login?next={request.url.path}", status_code=status.HTTP_303_SEE_OTHER
            )
        if not actor.is_admin:
            raise HTTPException(status_code=403, detail="Admin privileges are required")
        return RedirectResponse(url="/admin/tokens", status_code=status.HTTP_303_SEE_OTHER)

    @app.get("/admin/tokens", response_class=HTMLResponse)
    async def admin_tokens_page(request: Request):
        actor = _session_actor(request)
        if actor is None:
            return RedirectResponse(
                url=f"/login?next={request.url.path}", status_code=status.HTTP_303_SEE_OTHER
            )
        return _render_page(
            request,
            template_name="admin_tokens.html",
            page_title="Admin Tokens",
            active_page="admin",
            secondary_page="admin_tokens",
            admin_only=True,
            context={"tokens": _list_all_tokens_for_admin(actor)},
        )

    @app.get("/admin/clients", response_class=HTMLResponse)
    async def admin_clients_page(request: Request):
        actor = _session_actor(request)
        if actor is None:
            return RedirectResponse(
                url=f"/login?next={request.url.path}", status_code=status.HTTP_303_SEE_OTHER
            )
        return _render_page(
            request,
            template_name="admin_clients.html",
            page_title="Client Registrations",
            active_page="admin",
            secondary_page="admin_clients",
            admin_only=True,
            context={"clients": _resource_store().list_client_registrations()},
        )

    @app.get("/admin/clients/{client_registration_euid}", response_class=HTMLResponse)
    async def admin_client_detail_page(request: Request, client_registration_euid: str):
        actor = _session_actor(request)
        if actor is None:
            return RedirectResponse(
                url=f"/login?next={request.url.path}", status_code=status.HTTP_303_SEE_OTHER
            )
        resources = _resource_store()
        client = resources.get_client_registration(client_registration_euid)
        if client is None:
            raise HTTPException(status_code=404, detail="Client registration not found")
        tokens = [
            item
            for item in _list_all_tokens_for_admin(actor)
            if item.client_registration_euid == client_registration_euid
        ]
        return _render_page(
            request,
            template_name="admin_client_detail.html",
            page_title=f"Client {client.client_name}",
            active_page="admin",
            secondary_page="admin_clients",
            admin_only=True,
            context={
                "client_registration": client,
                "tokens": tokens,
                "client_payload_json": _json_text(client),
            },
        )

    @app.get("/admin/observability", response_class=HTMLResponse)
    async def admin_observability_page(request: Request):
        actor = _session_actor(request)
        if actor is None:
            return RedirectResponse(
                url=f"/login?next={request.url.path}", status_code=status.HTTP_303_SEE_OTHER
            )
        store = _observability_store()
        projection, service_catalog = store.obs_services_snapshot()
        health_snapshot = store.health_snapshot()
        api_projection, families = store.api_health()
        endpoint_projection, endpoint_page = store.endpoint_health(offset=0, limit=25)
        db_projection, db_rollup = store.db_health()
        auth_projection, auth_rollup = store.auth_health()
        context = {
            "health_payload": build_health_payload(
                request,
                settings=app.state.settings,
                app_version=__version__,
                projection=store.projection(
                    observed_at=(
                        health_snapshot.get("checks", {}).get("database", {}).get("observed_at")
                        or health_snapshot.get("checks", {}).get("auth", {}).get("observed_at")
                    )
                ),
                health_snapshot=health_snapshot,
            ),
            "obs_services_payload": build_obs_services_payload(
                request,
                settings=app.state.settings,
                app_version=__version__,
                projection=projection,
                snapshot=service_catalog,
            ),
            "api_health_payload": build_api_health_payload(
                request,
                settings=app.state.settings,
                app_version=__version__,
                projection=api_projection,
                families=families,
            ),
            "endpoint_health_payload": build_endpoint_health_payload(
                request,
                settings=app.state.settings,
                app_version=__version__,
                projection=endpoint_projection,
                total=int(endpoint_page["total"]),
                offset=int(endpoint_page["offset"]),
                limit=int(endpoint_page["limit"]),
                items=list(endpoint_page["items"]),
            ),
            "db_health_payload": build_db_health_payload(
                request,
                settings=app.state.settings,
                app_version=__version__,
                projection=db_projection,
                db_health=db_rollup,
            ),
            "auth_health_payload": build_auth_health_payload(
                request,
                settings=app.state.settings,
                app_version=__version__,
                projection=auth_projection,
                auth_rollup=auth_rollup,
            ),
        }
        return _render_page(
            request,
            template_name="observability.html",
            page_title="Observability",
            active_page="admin",
            secondary_page="admin_observability",
            admin_only=True,
            context=context,
        )

    @app.get("/admin/config", response_class=HTMLResponse)
    async def admin_config_page(request: Request):
        actor = _session_actor(request)
        if actor is None:
            return RedirectResponse(
                url=f"/login?next={request.url.path}", status_code=status.HTTP_303_SEE_OTHER
            )
        if not actor.is_admin:
            raise HTTPException(status_code=403, detail="Admin privileges are required")
        return _render_page(
            request,
            template_name="admin_config.html",
            page_title="Configuration",
            active_page="admin",
            secondary_page="admin_config",
            admin_only=True,
            context=_admin_config_context(),
        )

    @app.get("/admin/anomalies", response_class=HTMLResponse)
    async def admin_anomalies_page(request: Request):
        repository = _anomaly_repository()
        anomalies = repository.list()
        return _render_page(
            request,
            template_name="admin_anomalies.html",
            page_title="Anomalies",
            active_page="admin",
            secondary_page="admin_anomalies",
            admin_only=True,
            context={
                "anomalies": anomalies,
                "anomaly": None,
            },
        )

    @app.get("/admin/anomalies/{anomaly_id}", response_class=HTMLResponse)
    async def admin_anomaly_detail_page(anomaly_id: str, request: Request):
        repository = _anomaly_repository()
        anomaly = repository.get(anomaly_id)
        if anomaly is None:
            raise HTTPException(status_code=404, detail="Anomaly not found")
        return _render_page(
            request,
            template_name="admin_anomalies.html",
            page_title="Anomalies",
            active_page="admin",
            secondary_page="admin_anomalies",
            admin_only=True,
            context={
                "anomalies": repository.list(limit=25),
                "anomaly": anomaly,
            },
        )
