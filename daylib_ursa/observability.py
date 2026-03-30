from __future__ import annotations

import hashlib
import os
import socket
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from math import ceil
from threading import RLock
from typing import Any

from fastapi import Request
from sqlalchemy import event

from daylib_ursa import __version__
from daylib_ursa.auth import CurrentUser
from daylib_ursa.config import Settings
from daylib_ursa.integrations.tapdb_runtime import (
    TapDBRuntimeError,
    run_schema_drift_check as run_tapdb_schema_drift_check,
)

CONTRACT_VERSION = "v3"
SERVICE_NAME = "ursa"
_SCHEMA_DRIFT_CACHE: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _instance_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def _build_sha() -> str:
    return (
        os.environ.get("URSA_BUILD_SHA")
        or os.environ.get("BUILD_SHA")
        or os.environ.get("GIT_SHA")
        or ""
    )


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, ceil(len(ordered) * quantile) - 1))
    return round(float(ordered[index]), 3)


def _fingerprint(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _normalize_sql(statement: str) -> tuple[str, str]:
    text = " ".join(str(statement or "").split())
    statement_kind = text.split(" ", 1)[0].upper() if text else "UNKNOWN"
    digest = _fingerprint(text or statement_kind)
    return digest, statement_kind


def _tool_version() -> str:
    try:
        return version("daylily-tapdb")
    except PackageNotFoundError:
        return ""


def _default_schema_drift_payload(environment: str = "") -> dict[str, Any]:
    return {
        "status": "not_run",
        "checked_at": None,
        "environment": environment,
        "tool_version": _tool_version(),
        "summary": "Schema drift check has not been run.",
        "report": {},
        "strict": False,
    }


@dataclass
class ProjectionMetadata:
    state: str = "ready"
    stale: bool = False
    observed_at: str | None = None
    last_synced_at: str | None = None
    detail: str | None = None

    def model_dump(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "stale": self.stale,
            "observed_at": self.observed_at,
            "last_synced_at": self.last_synced_at,
            "detail": self.detail,
        }


@dataclass
class EndpointRollup:
    method: str
    route_template: str
    request_count: int = 0
    error_count: int = 0
    status_class_counts: Counter[str] = field(default_factory=Counter)
    durations_ms: deque[float] = field(default_factory=lambda: deque(maxlen=512))
    fingerprints: set[str] = field(default_factory=set)
    observed_at: str = field(default_factory=_utcnow)

    def record(self, *, status_code: int, duration_ms: float, fingerprint: str) -> None:
        self.request_count += 1
        if status_code >= 500:
            self.error_count += 1
        self.status_class_counts[f"{status_code // 100}xx"] += 1
        self.durations_ms.append(float(duration_ms))
        if fingerprint:
            self.fingerprints.add(fingerprint)
        self.observed_at = _utcnow()

    def to_dict(self) -> dict[str, Any]:
        durations = list(self.durations_ms)
        return {
            "method": self.method,
            "route_template": self.route_template,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "status_class_counts": dict(self.status_class_counts),
            "p50_ms": _percentile(durations, 0.50),
            "p95_ms": _percentile(durations, 0.95),
            "p99_ms": _percentile(durations, 0.99),
            "fingerprint_count": len(self.fingerprints),
            "observed_at": self.observed_at,
        }


@dataclass
class FamilyRollup:
    family: str
    request_count: int = 0
    error_count: int = 0
    durations_ms: deque[float] = field(default_factory=lambda: deque(maxlen=512))
    observed_at: str = field(default_factory=_utcnow)

    def record(self, *, status_code: int, duration_ms: float) -> None:
        self.request_count += 1
        if status_code >= 500:
            self.error_count += 1
        self.durations_ms.append(float(duration_ms))
        self.observed_at = _utcnow()

    def to_dict(self) -> dict[str, Any]:
        durations = list(self.durations_ms)
        return {
            "family": self.family,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "p50_ms": _percentile(durations, 0.50),
            "p95_ms": _percentile(durations, 0.95),
            "p99_ms": _percentile(durations, 0.99),
            "observed_at": self.observed_at,
        }


@dataclass
class DbQueryRollup:
    fingerprint: str
    statement_kind: str
    label: str
    request_count: int = 0
    error_count: int = 0
    durations_ms: deque[float] = field(default_factory=lambda: deque(maxlen=256))
    observed_at: str = field(default_factory=_utcnow)

    def record(self, *, duration_ms: float, success: bool) -> None:
        self.request_count += 1
        if not success:
            self.error_count += 1
        self.durations_ms.append(float(duration_ms))
        self.observed_at = _utcnow()

    def to_dict(self) -> dict[str, Any]:
        durations = list(self.durations_ms)
        return {
            "fingerprint": self.fingerprint,
            "statement_kind": self.statement_kind,
            "label": self.label,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "p50_ms": _percentile(durations, 0.50),
            "p95_ms": _percentile(durations, 0.95),
            "p99_ms": _percentile(durations, 0.99),
            "observed_at": self.observed_at,
        }


class UrsaObservabilityStore:
    def __init__(self, *, settings: Settings, app_version: str) -> None:
        self.settings = settings
        self.app_version = app_version
        self._lock = RLock()
        self._started_at = _utcnow()
        self._dependency_observed_at = self._started_at
        self._endpoint_rollups: dict[tuple[str, str], EndpointRollup] = {}
        self._family_rollups: dict[str, FamilyRollup] = {}
        self._db_rollups: dict[str, DbQueryRollup] = {}
        self._db_probes: deque[dict[str, Any]] = deque(maxlen=25)
        self._auth_recent: deque[dict[str, Any]] = deque(maxlen=25)
        self._auth_status_counts: Counter[str] = Counter()
        self._observed_dependencies: set[str] = set()
        self._schema_drift = _default_schema_drift_payload(
            str(self.settings.tapdb_env or self.settings.daylily_env or "")
        )
        self._refresh_schema_drift_status()
        self._obs_services_snapshot = self._build_obs_services_snapshot()

    def _build_obs_services_snapshot(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "endpoints": [
                {"path": "/healthz", "auth": "none", "kind": "liveness"},
                {"path": "/readyz", "auth": "none", "kind": "readiness"},
                {"path": "/health", "auth": "operator_or_service_token", "kind": "summary"},
                {"path": "/obs_services", "auth": "operator_or_service_token", "kind": "discovery"},
                {"path": "/api_health", "auth": "operator_or_service_token", "kind": "api_rollup"},
                {"path": "/endpoint_health", "auth": "operator_or_service_token", "kind": "endpoint_rollup"},
                {"path": "/db_health", "auth": "operator_or_service_token", "kind": "database"},
                {"path": "/api/anomalies", "auth": "operator_or_service_token", "kind": "anomaly_list"},
                {
                    "path": "/api/anomalies/{anomaly_id}",
                    "auth": "operator_or_service_token",
                    "kind": "anomaly_detail",
                },
                {"path": "/my_health", "auth": "authenticated_self", "kind": "self"},
                {"path": "/auth_health", "auth": "operator_or_service_token", "kind": "auth"},
            ],
            "extensions": [
                "ursa.admin_observability_ui",
                "ursa.anomalies_v1",
                "ursa.topology_v1",
            ],
            "dependencies": self._dependencies_snapshot(),
            "observed_at": self._dependency_observed_at,
        }

    def _configured_dependencies(self) -> list[str]:
        configured: list[str] = []
        if str(self.settings.atlas_base_url or "").strip():
            configured.append("atlas")
        if str(self.settings.bloom_base_url or "").strip():
            configured.append("bloom")
        if self.settings.dewey_enabled and str(self.settings.dewey_base_url or "").strip():
            configured.append("dewey")
        return configured

    def _dependencies_snapshot(self) -> dict[str, Any]:
        return {
            "configured_services": self._configured_dependencies(),
            "observed_services": sorted(self._observed_dependencies),
        }

    def _session_summary(self, observed_at: str | None = None) -> dict[str, Any]:
        return {
            "supported": False,
            "active_session_count": None,
            "recent_user_count": None,
            "observed_at": observed_at or self._started_at,
        }

    def _refresh_schema_drift_status(self) -> None:
        environment = str(self.settings.tapdb_env or self.settings.daylily_env or "").strip()
        cache_key = (
            str(self.settings.database_target or "").strip(),
            str(self.settings.tapdb_client_id or "").strip(),
            str(self.settings.tapdb_database_name or "").strip(),
            environment,
            str(self.settings.aws_profile or "").strip(),
        )
        cached = _SCHEMA_DRIFT_CACHE.get(cache_key)
        if cached is not None:
            self._schema_drift = dict(cached)
            return
        try:
            result = run_tapdb_schema_drift_check(
                target=self.settings.database_target,
                client_id=self.settings.tapdb_client_id,
                profile=self.settings.aws_profile or "",
                region=self.settings.day_aws_region or self.settings.daylily_primary_region or "us-west-2",
                namespace=self.settings.tapdb_database_name,
                tapdb_env=environment or None,
            )
        except TapDBRuntimeError as exc:
            result = {
                **_default_schema_drift_payload(environment),
                "status": "check_failed",
                "summary": f"Unable to execute tapdb drift-check: {exc}",
            }
        except Exception as exc:
            result = {
                **_default_schema_drift_payload(environment),
                "status": "check_failed",
                "summary": f"Unable to execute tapdb drift-check: {exc}",
            }
        self._schema_drift = dict(result)
        _SCHEMA_DRIFT_CACHE[cache_key] = dict(result)

    def projection(self, *, observed_at: str | None = None, detail: str | None = None) -> ProjectionMetadata:
        seen_at = observed_at or self._started_at
        return ProjectionMetadata(
            state="ready",
            stale=False,
            observed_at=seen_at,
            last_synced_at=seen_at,
            detail=detail,
        )

    def record_http_request(
        self,
        *,
        method: str,
        route_template: str,
        status_code: int,
        duration_ms: float,
    ) -> None:
        family = self._classify_family(route_template)
        key = (method.upper(), route_template)
        fingerprint = f"{method.upper()}:{route_template}:{status_code // 100}xx"
        with self._lock:
            endpoint_rollup = self._endpoint_rollups.setdefault(
                key,
                EndpointRollup(method=method.upper(), route_template=route_template),
            )
            endpoint_rollup.record(
                status_code=status_code,
                duration_ms=duration_ms,
                fingerprint=fingerprint,
            )
            family_rollup = self._family_rollups.setdefault(family, FamilyRollup(family=family))
            family_rollup.record(status_code=status_code, duration_ms=duration_ms)

    def record_db_query(self, *, statement: str, duration_ms: float, success: bool) -> None:
        fingerprint, statement_kind = _normalize_sql(statement)
        with self._lock:
            rollup = self._db_rollups.setdefault(
                fingerprint,
                DbQueryRollup(
                    fingerprint=fingerprint,
                    statement_kind=statement_kind,
                    label=statement_kind,
                ),
            )
            rollup.record(duration_ms=duration_ms, success=success)

    def record_db_probe(self, *, status: str, latency_ms: float, detail: str) -> None:
        with self._lock:
            self._db_probes.appendleft(
                {
                    "status": status,
                    "latency_ms": round(float(latency_ms), 3),
                    "detail": str(detail or ""),
                    "fingerprint": _fingerprint(detail),
                    "observed_at": _utcnow(),
                }
            )

    def record_observed_dependency(self, service_id: str) -> None:
        candidate = str(service_id or "").strip().lower()
        if not candidate:
            return
        with self._lock:
            self._observed_dependencies.add(candidate)
            self._dependency_observed_at = _utcnow()
            self._obs_services_snapshot = self._build_obs_services_snapshot()

    def record_auth_event(
        self,
        *,
        status: str,
        mode: str,
        detail: str,
        service_principal: bool,
    ) -> None:
        event = {
            "status": status,
            "mode": mode,
            "detail": str(detail or ""),
            "service_principal": service_principal,
            "fingerprint": _fingerprint(detail),
            "observed_at": _utcnow(),
        }
        with self._lock:
            self._auth_recent.appendleft(event)
            self._auth_status_counts[status] += 1

    def health_snapshot(self) -> dict[str, Any]:
        latest_db = self.latest_db_probe()
        latest_auth = self._auth_recent[0] if self._auth_recent else None
        database_status = str((latest_db or {}).get("status") or "unknown")
        overall_status = "ok" if database_status in {"ok", "unknown"} else "degraded"
        return {
            "status": overall_status,
            "checks": {
                "process": {"status": "ok", "observed_at": _utcnow()},
                "database": latest_db
                or {
                    "status": "unknown",
                    "latency_ms": None,
                    "detail": None,
                    "observed_at": None,
                },
                "auth": {
                    "status": str((latest_auth or {}).get("status") or "unknown"),
                    "mode": str((latest_auth or {}).get("mode") or ""),
                    "cognito_configured": bool(
                        self.settings.cognito_domain
                        and self.settings.cognito_user_pool_id
                        and self.settings.cognito_app_client_id
                    ),
                    "observed_at": (latest_auth or {}).get("observed_at"),
                },
            },
        }

    def obs_services_snapshot(self) -> tuple[ProjectionMetadata, dict[str, Any]]:
        with self._lock:
            snapshot = dict(self._build_obs_services_snapshot())
        observed_at = str(snapshot.get("observed_at") or self._started_at)
        return self.projection(observed_at=observed_at), snapshot

    def api_health(self) -> tuple[ProjectionMetadata, list[dict[str, Any]]]:
        with self._lock:
            families = [rollup.to_dict() for rollup in self._family_rollups.values()]
        families.sort(key=lambda item: (-int(item["request_count"]), item["family"]))
        observed_at = families[0]["observed_at"] if families else self._started_at
        return self.projection(observed_at=observed_at), families

    def endpoint_health(self, *, offset: int, limit: int) -> tuple[ProjectionMetadata, dict[str, Any]]:
        with self._lock:
            items = [rollup.to_dict() for rollup in self._endpoint_rollups.values()]
        items.sort(
            key=lambda item: (-int(item["request_count"]), item["route_template"], item["method"])
        )
        total = len(items)
        sliced = items[offset : offset + limit]
        observed_at = (
            sliced[0]["observed_at"] if sliced else (items[0]["observed_at"] if items else self._started_at)
        )
        return self.projection(observed_at=observed_at), {
            "total": total,
            "offset": offset,
            "limit": limit,
            "items": sliced,
        }

    def latest_db_probe(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._db_probes[0]) if self._db_probes else None

    def db_health(self) -> tuple[ProjectionMetadata, dict[str, Any]]:
        with self._lock:
            latest = dict(self._db_probes[0]) if self._db_probes else None
            recent_queries = [rollup.to_dict() for rollup in self._db_rollups.values()]
            schema_drift = dict(self._schema_drift)
        recent_queries.sort(key=lambda item: (-float(item["p95_ms"]), -int(item["request_count"]), item["label"]))
        hottest = sorted(recent_queries, key=lambda item: (-int(item["request_count"]), item["label"]))[:10]
        slowest = sorted(recent_queries, key=lambda item: (-float(item["p95_ms"]), item["label"]))[:10]
        observed_at = (
            (latest or {}).get("observed_at")
            or (recent_queries[0]["observed_at"] if recent_queries else self._started_at)
        )
        payload = {
            "status": str((latest or {}).get("status") or "unknown"),
            "latest": latest,
            "recent": recent_queries[:25],
            "slowest": slowest,
            "hottest": hottest,
            "schema_drift": schema_drift,
            "observed_at": observed_at,
        }
        return self.projection(observed_at=observed_at), payload

    def auth_health(self) -> tuple[ProjectionMetadata, dict[str, Any]]:
        with self._lock:
            recent = list(self._auth_recent)
            status_counts = dict(self._auth_status_counts)
        latest = recent[0] if recent else None
        observed_at = str((latest or {}).get("observed_at") or self._started_at)
        return self.projection(observed_at=observed_at), {
            "status": str((latest or {}).get("status") or "unknown"),
            "mode": str((latest or {}).get("mode") or "unknown"),
            "cognito_configured": bool(
                self.settings.cognito_domain
                and self.settings.cognito_user_pool_id
                and self.settings.cognito_app_client_id
            ),
            "cognito_domain": str(self.settings.cognito_domain or ""),
            "user_pool_id": str(self.settings.cognito_user_pool_id or ""),
            "app_client_id_present": bool(self.settings.cognito_app_client_id),
            "recent": recent,
            "status_counts": status_counts,
            "sessions": self._session_summary(observed_at),
            "observed_at": observed_at,
        }

    def _classify_family(self, route_template: str) -> str:
        path = route_template or "/"
        if path.startswith("/api/v1/"):
            parts = [part for part in path.split("/") if part]
            return parts[2] if len(parts) > 2 else "api"
        if path.startswith("/auth"):
            return "auth"
        if path.startswith("/admin"):
            return "admin"
        if path in {
            "/health",
            "/healthz",
            "/readyz",
            "/obs_services",
            "/api_health",
            "/endpoint_health",
            "/db_health",
            "/my_health",
            "/auth_health",
        }:
            return "observability"
        return "web"


def base_frame(request: Request, *, status: str, settings: Settings, app_version: str) -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "service": SERVICE_NAME,
        "environment": settings.deployment_name or settings.daylily_env or "development",
        "instance_id": _instance_id(),
        "observed_at": _utcnow(),
        "status": status,
        "request_id": getattr(request.state, "request_id", ""),
        "correlation_id": getattr(request.state, "correlation_id", ""),
        "build": {
            "version": app_version or __version__,
            "sha": _build_sha(),
        },
    }


def _status_for_projection(projection: ProjectionMetadata, ready_status: str) -> str:
    return ready_status if projection.state == "ready" else "unknown"


def _with_projection(payload: dict[str, Any], projection: ProjectionMetadata) -> dict[str, Any]:
    payload["projection"] = projection.model_dump()
    return payload


def build_health_payload(
    request: Request,
    *,
    settings: Settings,
    app_version: str,
    projection: ProjectionMetadata,
    health_snapshot: dict[str, Any],
) -> dict[str, Any]:
    payload = base_frame(
        request,
        status=_status_for_projection(projection, str(health_snapshot.get("status") or "unknown")),
        settings=settings,
        app_version=app_version,
    )
    payload["checks"] = dict(health_snapshot.get("checks") or {})
    return _with_projection(payload, projection)


def build_obs_services_payload(
    request: Request,
    *,
    settings: Settings,
    app_version: str,
    projection: ProjectionMetadata,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    payload = base_frame(
        request,
        status=_status_for_projection(projection, str(snapshot.get("status") or "ok")),
        settings=settings,
        app_version=app_version,
    )
    payload["endpoints"] = list(snapshot.get("endpoints") or [])
    payload["extensions"] = list(snapshot.get("extensions") or [])
    payload["dependencies"] = dict(snapshot.get("dependencies") or {})
    return _with_projection(payload, projection)


def build_api_health_payload(
    request: Request,
    *,
    settings: Settings,
    app_version: str,
    projection: ProjectionMetadata,
    families: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = base_frame(
        request,
        status=_status_for_projection(projection, "ok"),
        settings=settings,
        app_version=app_version,
    )
    payload["families"] = families
    return _with_projection(payload, projection)


def build_endpoint_health_payload(
    request: Request,
    *,
    settings: Settings,
    app_version: str,
    projection: ProjectionMetadata,
    total: int,
    offset: int,
    limit: int,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = base_frame(
        request,
        status=_status_for_projection(projection, "ok"),
        settings=settings,
        app_version=app_version,
    )
    payload["page"] = {"total": total, "offset": offset, "limit": limit}
    payload["items"] = items
    return _with_projection(payload, projection)


def build_db_health_payload(
    request: Request,
    *,
    settings: Settings,
    app_version: str,
    projection: ProjectionMetadata,
    db_health: dict[str, Any],
) -> dict[str, Any]:
    payload = base_frame(
        request,
        status=_status_for_projection(projection, str(db_health.get("status") or "unknown")),
        settings=settings,
        app_version=app_version,
    )
    payload["database"] = db_health
    return _with_projection(payload, projection)


def build_auth_health_payload(
    request: Request,
    *,
    settings: Settings,
    app_version: str,
    projection: ProjectionMetadata,
    auth_rollup: dict[str, Any],
) -> dict[str, Any]:
    payload = base_frame(
        request,
        status=_status_for_projection(projection, str(auth_rollup.get("status") or "unknown")),
        settings=settings,
        app_version=app_version,
    )
    payload["auth"] = {
        "mode": str(auth_rollup.get("mode") or ""),
        "cognito_configured": bool(auth_rollup.get("cognito_configured", False)),
        "cognito_domain": str(auth_rollup.get("cognito_domain") or ""),
        "user_pool_id": str(auth_rollup.get("user_pool_id") or ""),
        "app_client_id_present": bool(auth_rollup.get("app_client_id_present", False)),
        "recent": list(auth_rollup.get("recent") or []),
        "status_counts": dict(auth_rollup.get("status_counts") or {}),
        "sessions": dict(auth_rollup.get("sessions") or {}),
    }
    return _with_projection(payload, projection)


def build_my_health_payload(
    request: Request,
    *,
    settings: Settings,
    app_version: str,
    user: CurrentUser,
) -> dict[str, Any]:
    payload = base_frame(request, status="ok", settings=settings, app_version=app_version)
    payload["principal"] = {
        "subject": str(user.user_id),
        "email": user.email,
        "name": user.name,
        "roles": user.roles,
        "auth_mode": user.auth_source,
        "expires_at": None,
        "service_principal": user.auth_source == "service_token",
    }
    return payload


def install_sqlalchemy_observability(store: UrsaObservabilityStore, engine: Any) -> Any:
    if engine is None or getattr(engine, "_ursa_observability_installed", False):
        return lambda: None

    start_key = "_ursa_observability_start"

    def before_cursor_execute(conn, _cursor, statement, _parameters, _context, _executemany):
        stack = conn.info.setdefault(start_key, [])
        stack.append((time.monotonic(), statement))

    def after_cursor_execute(conn, _cursor, statement, _parameters, _context, _executemany):
        stack = conn.info.get(start_key, [])
        start_time, started_statement = stack.pop() if stack else (time.monotonic(), statement)
        store.record_db_query(
            statement=str(started_statement or statement),
            duration_ms=(time.monotonic() - start_time) * 1000,
            success=True,
        )

    def handle_error(exception_context):
        conn = exception_context.connection
        stack = conn.info.get(start_key, []) if conn is not None else []
        start_time, started_statement = stack.pop() if stack else (
            time.monotonic(),
            exception_context.statement,
        )
        store.record_db_query(
            statement=str(started_statement or exception_context.statement or "unknown"),
            duration_ms=(time.monotonic() - start_time) * 1000,
            success=False,
        )

    event.listen(engine, "before_cursor_execute", before_cursor_execute)
    event.listen(engine, "after_cursor_execute", after_cursor_execute)
    event.listen(engine, "handle_error", handle_error)
    setattr(engine, "_ursa_observability_installed", True)

    def cleanup() -> None:
        if not getattr(engine, "_ursa_observability_installed", False):
            return
        event.remove(engine, "before_cursor_execute", before_cursor_execute)
        event.remove(engine, "after_cursor_execute", after_cursor_execute)
        event.remove(engine, "handle_error", handle_error)
        setattr(engine, "_ursa_observability_installed", False)

    return cleanup
