from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from daylily_cognito import build_authorization_url, exchange_authorization_code
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
    clear_session_user,
    get_current_user,
    persist_session_user,
)
from daylib_ursa.observability import (
    build_api_health_payload,
    build_auth_health_payload,
    build_db_health_payload,
    build_endpoint_health_payload,
    build_health_payload,
    build_obs_services_payload,
)


def mount_gui(app: FastAPI) -> None:
    gui_root = Path(__file__).resolve().parent / "gui"
    templates = Jinja2Templates(directory=str(gui_root / "templates"))
    static_root = gui_root / "static"
    if static_root.is_dir():
        app.mount("/ui/static", StaticFiles(directory=str(static_root)), name="ui-static")

    def _deployment_context() -> dict[str, object]:
        settings = getattr(app.state, "settings", None)
        return {
            "name": str(getattr(settings, "deployment_name", "") or ""),
            "color": str(getattr(settings, "deployment_color", "#0f766e") or "#0f766e"),
            "is_production": bool(getattr(settings, "deployment_is_production", False)),
        }

    def _next_path(raw_value: str | None) -> str:
        value = str(raw_value or "").strip()
        return value if value.startswith("/") else "/"

    def _cognito_login_path(next_path: str) -> str:
        return f"/auth/login?next={_next_path(next_path)}"

    def _oauth_state() -> str:
        return secrets.token_urlsafe(24)

    def _cognito_settings() -> dict[str, str]:
        settings = getattr(app.state, "settings", None)
        values = {
            "domain": str(getattr(settings, "cognito_domain", "") or "").strip().rstrip("/"),
            "client_id": str(getattr(settings, "cognito_app_client_id", "") or "").strip(),
            "client_secret": str(getattr(settings, "cognito_app_client_secret", "") or "").strip(),
            "callback_url": str(getattr(settings, "cognito_callback_url", "") or "").strip(),
            "logout_url": str(getattr(settings, "cognito_logout_url", "") or "").strip(),
        }
        if values["domain"].startswith("https://"):
            values["domain"] = values["domain"][len("https://") :]
        missing = [key for key in ("domain", "client_id", "callback_url", "logout_url") if not values[key]]
        if missing:
            raise HTTPException(
                status_code=503,
                detail=f"Cognito authentication is not configured: missing {', '.join(missing)}",
            )
        return values

    def _build_cognito_login_url(*, state: str) -> str:
        cognito = _cognito_settings()
        return build_authorization_url(
            domain=cognito["domain"],
            client_id=cognito["client_id"],
            redirect_uri=cognito["callback_url"],
            state=state,
        )

    def _build_cognito_logout_url(*, state: str | None = None) -> str:
        cognito = _cognito_settings()
        query = {
            "client_id": cognito["client_id"],
            "redirect_uri": cognito["logout_url"],
            "response_type": "code",
        }
        if state:
            query["state"] = state
        return f"https://{cognito['domain']}/logout?{urlencode(query)}"

    def _exchange_auth_code(code: str) -> dict[str, Any]:
        cognito = _cognito_settings()
        try:
            return exchange_authorization_code(
                domain=cognito["domain"],
                client_id=cognito["client_id"],
                code=code,
                redirect_uri=cognito["callback_url"],
                client_secret=cognito["client_secret"] or None,
            )
        except RuntimeError as exc:
            raise AuthError(str(exc)) from exc

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
            return RedirectResponse(
                url=f"/login?next={request.url.path}", status_code=status.HTTP_303_SEE_OTHER
            )
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
            "deployment": _deployment_context(),
        }
        template_context.update(context or {})
        return templates.TemplateResponse(request, template_name, template_context)

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

    def _list_analyses(actor: CurrentUser) -> list[Any]:
        return app.state.store.list_analyses(
            tenant_id=None if actor.is_admin else actor.tenant_id,
        )

    def _list_buckets(actor: CurrentUser) -> list[Any]:
        return _resource_store().list_linked_buckets(tenant_id=actor.tenant_id)

    def _allowed_regions() -> list[str]:
        settings = getattr(app.state, "settings", None)
        if settings is None or not hasattr(settings, "get_allowed_regions"):
            return []
        return list(settings.get_allowed_regions())

    def _workset_view_model(workset: Any) -> dict[str, Any]:
        metadata = dict(getattr(workset, "metadata", {}) or {})
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
            "pipeline_type": str(metadata.get("pipeline_type") or "germline"),
            "reference_genome": str(metadata.get("reference_genome") or ""),
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
    async def login_page(request: Request, next: str = "/"):
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
                "error": None,
                "deployment": _deployment_context(),
            },
        )

    @app.get("/auth/login", include_in_schema=False)
    async def auth_login(request: Request, next: str = "/"):
        state = _oauth_state()
        request.session["ursa_oauth_state"] = state
        request.session["ursa_post_auth_redirect"] = _next_path(next)
        return RedirectResponse(
            url=_build_cognito_login_url(state=state),
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @app.get("/auth/callback", include_in_schema=False)
    async def auth_callback(request: Request, code: str = "", state: str = ""):
        expected_state = str(request.session.get("ursa_oauth_state") or "").strip()
        if not code.strip():
            raise HTTPException(status_code=400, detail="Missing authorization code")
        if not state.strip() or state.strip() != expected_state:
            raise HTTPException(status_code=400, detail="Invalid oauth state")
        try:
            token_payload = _exchange_auth_code(code.strip())
            # Hosted UI code exchange should prefer the access token for session auth.
            # Falling back to id_token can trigger at_hash verification paths that expect
            # the paired access token to be available.
            token = str(token_payload.get("access_token") or token_payload.get("id_token") or "").strip()
            if not token:
                raise AuthError("Cognito token response missing access_token or id_token")
            auth_provider = getattr(app.state, "auth_provider", None)
            if auth_provider is None:
                raise AuthError("Authentication provider is not configured")
            actor = auth_provider.resolve_access_token(token)
        except AuthError as exc:
            request.session.pop("ursa_oauth_state", None)
            request.session.pop("ursa_post_auth_redirect", None)
            return templates.TemplateResponse(
                request,
                "login.html",
                {
                    "request": request,
                    "next_path": _next_path(request.query_params.get("next") or "/"),
                    "cognito_login_url": _cognito_login_path(request.query_params.get("next") or "/"),
                    "error": str(exc),
                    "deployment": _deployment_context(),
                },
                status_code=status.HTTP_401_UNAUTHORIZED,
            )

        persist_session_user(request, actor)
        redirect_to = _next_path(request.session.pop("ursa_post_auth_redirect", "/"))
        request.session.pop("ursa_oauth_state", None)
        return RedirectResponse(url=redirect_to, status_code=status.HTTP_303_SEE_OTHER)

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
                    "deployment": _deployment_context(),
                },
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        try:
            auth_provider = getattr(app.state, "auth_provider", None)
            if auth_provider is None:
                raise AuthError("Authentication provider is not configured")
            actor = auth_provider.resolve_access_token(token)
        except AuthError as exc:
            return templates.TemplateResponse(
                request,
                "login.html",
                {
                    "request": request,
                    "next_path": _next_path(next_path),
                    "cognito_login_url": _cognito_login_path(next_path),
                    "error": str(exc),
                    "deployment": _deployment_context(),
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

    @app.get("/logout")
    async def logout(request: Request):
        clear_session_user(request)
        try:
            state = _oauth_state()
            request.session["ursa_oauth_state"] = state
            return RedirectResponse(
                url=_build_cognito_logout_url(state=state),
                status_code=status.HTTP_303_SEE_OTHER,
            )
        except HTTPException:
            return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)

    @app.get("/", response_class=HTMLResponse)
    async def dashboard_page(request: Request):
        actor = _session_actor(request)
        if actor is None:
            return RedirectResponse(
                url=f"/login?next={request.url.path}", status_code=status.HTTP_303_SEE_OTHER
            )
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
            return RedirectResponse(
                url=f"/login?next={request.url.path}", status_code=status.HTTP_303_SEE_OTHER
            )
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
            return RedirectResponse(
                url=f"/login?next={request.url.path}", status_code=status.HTTP_303_SEE_OTHER
            )
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
            return RedirectResponse(
                url=f"/login?next={request.url.path}", status_code=status.HTTP_303_SEE_OTHER
            )
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
            context={"buckets": _list_buckets(actor)},
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
        jobs = _resource_store().list_cluster_jobs(
            tenant_id=None if actor.is_admin else actor.tenant_id
        )
        return _render_page(
            request,
            template_name="clusters.html",
            page_title="Clusters",
            active_page="clusters",
            admin_only=True,
            context={
                "clusters": [item.to_dict(include_sensitive=False) for item in clusters],
                "jobs": jobs,
                "regions": _allowed_regions(),
                "is_admin": actor.is_admin,
                "create_mode": False,
                "prefill_region": (_allowed_regions()[0] if _allowed_regions() else ""),
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
            active_page="tools",
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
            active_page="tools",
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
            active_page="tools",
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
            active_page="tools",
            secondary_page="admin_observability",
            admin_only=True,
            context=context,
        )

    @app.get("/admin/anomalies", response_class=HTMLResponse)
    async def admin_anomalies_page(request: Request):
        repository = _anomaly_repository()
        anomalies = repository.list()
        return _render_page(
            request,
            template_name="admin_anomalies.html",
            page_title="Anomalies",
            active_page="tools",
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
            active_page="tools",
            secondary_page="admin_anomalies",
            admin_only=True,
            context={
                "anomalies": repository.list(limit=25),
                "anomaly": anomaly,
            },
        )
