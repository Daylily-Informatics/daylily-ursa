from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from daylib_ursa.config import Settings
from daylib_ursa.resource_store import ResourceStore
from daylib_ursa.tapdb_graph import from_json_addl, utc_now_iso
from daylib_ursa.tapdb_templates import seed_ursa_templates

ANOMALY_TEMPLATE = "ops/anomaly/local-record/1.0/"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _fingerprint(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _environment_name(settings: Settings) -> str:
    return (
        str(settings.deployment_name or "").strip()
        or str(settings.tapdb_env or "").strip()
        or str(settings.database_target or "").strip()
        or "unknown"
    )


def redact_context(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in {"authorization", "cookie", "set-cookie", "token", "secret", "password"}:
                redacted[str(key)] = "[redacted]"
            elif lowered == "sql":
                redacted[str(key)] = "[redacted-sql]"
            else:
                redacted[str(key)] = redact_context(item)
        return redacted
    if isinstance(value, list):
        return [redact_context(item) for item in value]
    if isinstance(value, str):
        return value
    return value


@dataclass(frozen=True)
class AnomalyRecord:
    id: str
    service: str
    environment: str
    category: str
    severity: str
    fingerprint: str
    summary: str
    first_seen_at: str
    last_seen_at: str
    occurrence_count: int
    redacted_context: dict[str, Any]
    source_view_url: str


class AnomalyRepository:
    def __init__(self, *, backend: Any, resource_store: ResourceStore | None, settings: Settings) -> None:
        self.resource_store = resource_store
        self.backend = backend
        self.settings = settings

    def record(
        self,
        *,
        category: str,
        severity: str,
        fingerprint: str,
        summary: str,
        redacted_context: dict[str, Any] | None = None,
    ) -> AnomalyRecord:
        normalized_context = dict(redact_context(redacted_context or {}))
        environment = _environment_name(self.settings)
        now = utc_now_iso()
        with self.backend.session_scope(commit=True) as session:
            self._ensure_templates(session)
            existing = self._find_existing(
                session,
                category=category,
                severity=severity,
                fingerprint=fingerprint,
                environment=environment,
            )
            if existing is None:
                instance = self.backend.create_instance(
                    session,
                    template_code=ANOMALY_TEMPLATE,
                    name=summary[:120] or "Ursa anomaly",
                    json_addl={
                        "service": "ursa",
                        "environment": environment,
                        "category": category,
                        "severity": severity,
                        "fingerprint": fingerprint,
                        "summary": summary,
                        "first_seen_at": now,
                        "last_seen_at": now,
                        "occurrence_count": 1,
                        "redacted_context": normalized_context,
                    },
                    bstatus="active",
                )
            else:
                payload = dict(getattr(existing, "json_addl", {}) or {})
                payload.update(
                    {
                        "service": "ursa",
                        "environment": environment,
                        "category": category,
                        "severity": severity,
                        "fingerprint": fingerprint,
                        "summary": summary,
                        "first_seen_at": str(payload.get("first_seen_at") or now),
                        "last_seen_at": now,
                        "occurrence_count": int(payload.get("occurrence_count") or 0) + 1,
                        "redacted_context": normalized_context,
                    }
                )
                existing.json_addl = payload
                if hasattr(existing, "modified_dt"):
                    existing.modified_dt = _utcnow()
                if hasattr(session, "flush"):
                    session.flush()
                instance = existing
        return self._to_record(instance)

    def list(self, *, limit: int = 100) -> list[AnomalyRecord]:
        with self.backend.session_scope(commit=False) as session:
            rows = [
                self._to_record(item)
                for item in self.backend.list_instances_by_template(
                    session,
                    template_code=ANOMALY_TEMPLATE,
                    limit=limit,
                )
            ]
        rows.sort(key=lambda item: item.last_seen_at, reverse=True)
        return rows

    def get(self, anomaly_id: str) -> AnomalyRecord | None:
        with self.backend.session_scope(commit=False) as session:
            instance = self.backend.find_instance_by_euid(
                session,
                template_code=ANOMALY_TEMPLATE,
                value=anomaly_id,
            )
            if instance is None:
                return None
            return self._to_record(instance)

    def record_db_probe_failure(self, *, detail: str, latency_ms: float) -> AnomalyRecord:
        return self.record(
            category="database",
            severity="error",
            fingerprint=_fingerprint(detail or "database-probe-failure"),
            summary="Ursa database probe failed",
            redacted_context={
                "detail": str(detail or ""),
                "latency_ms": round(float(latency_ms), 3),
            },
        )

    def _ensure_templates(self, session: Any) -> None:
        ensure_templates = getattr(self.backend, "ensure_templates", None)
        if callable(ensure_templates):
            if callable(seed_ursa_templates):
                seed_ursa_templates(session)
            ensure_templates(session)

    def _find_existing(
        self,
        session: Any,
        *,
        category: str,
        severity: str,
        fingerprint: str,
        environment: str,
    ) -> Any | None:
        matches = self.backend.list_instances_by_property(
            session,
            template_code=ANOMALY_TEMPLATE,
            key="fingerprint",
            value=fingerprint,
            limit=100,
        )
        for instance in matches:
            payload = from_json_addl(instance)
            if (
                str(payload.get("category") or "") == category
                and str(payload.get("severity") or "") == severity
                and str(payload.get("environment") or "") == environment
            ):
                return instance
        return None

    def _to_record(self, instance: Any) -> AnomalyRecord:
        payload = from_json_addl(instance)
        return AnomalyRecord(
            id=str(getattr(instance, "euid", "") or ""),
            service=str(payload.get("service") or "ursa"),
            environment=str(payload.get("environment") or _environment_name(self.settings)),
            category=str(payload.get("category") or "unknown"),
            severity=str(payload.get("severity") or "unknown"),
            fingerprint=str(payload.get("fingerprint") or ""),
            summary=str(payload.get("summary") or getattr(instance, "name", "") or ""),
            first_seen_at=str(payload.get("first_seen_at") or utc_now_iso()),
            last_seen_at=str(payload.get("last_seen_at") or payload.get("first_seen_at") or utc_now_iso()),
            occurrence_count=int(payload.get("occurrence_count") or 0),
            redacted_context=dict(payload.get("redacted_context") or {}),
            source_view_url=f"/admin/anomalies/{getattr(instance, 'euid', '')}",
        )


def open_anomaly_repository(
    *,
    resource_store: ResourceStore | None,
    settings: Settings,
    backend: Any | None = None,
) -> AnomalyRepository:
    resolved_backend = backend or getattr(resource_store, "backend", None)
    if resolved_backend is None:
        raise RuntimeError("Anomaly backend is not configured")
    return AnomalyRepository(
        backend=resolved_backend,
        resource_store=resource_store,
        settings=settings,
    )
