"""Ursa ↔ TapDB adapter built on the TapDB 5.x runtime surface."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import String, cast, or_

from daylily_tapdb import generic_instance, generic_instance_lineage, utc_now_iso

from daylib_ursa import __version__
from daylib_ursa.integrations.tapdb_runtime import (
    DEFAULT_TAPDB_CLIENT_ID,
    DEFAULT_TAPDB_DATABASE_NAME,
    TapdbClientBundle,
    ensure_tapdb_version,
    get_tapdb_bundle,
)

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TemplateSpec:
    template_code: str


URSA_TEMPLATE_DEFINITIONS: list[TemplateSpec] = [
    TemplateSpec("workflow/analysis/run-linked/1.0/"),
    TemplateSpec("data/artifact/analysis-output/1.0/"),
    TemplateSpec("action/analysis/review-event/1.0/"),
    TemplateSpec("action/analysis/atlas-return/1.0/"),
    TemplateSpec("integration/reference/sequenced-assignment-context/1.0/"),
    TemplateSpec("workflow/workset/gui-ready/1.0/"),
    TemplateSpec("data/manifest/dewey-bound/1.0/"),
    TemplateSpec("action/artifact/dewey-import/1.0/"),
    TemplateSpec("integration/auth/user-token/1.0/"),
    TemplateSpec("integration/auth/user-token-revision/1.0/"),
    TemplateSpec("integration/auth/user-token-usage/1.0/"),
    TemplateSpec("integration/auth/client-registration/1.0/"),
    TemplateSpec("integration/storage/linked-bucket/1.0/"),
    TemplateSpec("workflow/cluster/ephemeral-job/1.0/"),
    TemplateSpec("action/cluster/ephemeral-job-revision/1.0/"),
    TemplateSpec("action/cluster/ephemeral-job-event/1.0/"),
    TemplateSpec("ops/anomaly/local-record/1.0/"),
]

TEMPLATE_DEFINITIONS = URSA_TEMPLATE_DEFINITIONS


def from_json_addl(instance) -> dict[str, Any]:
    raw = dict(getattr(instance, "json_addl", {}) or {})
    properties = raw.get("properties")
    if isinstance(properties, dict):
        merged = dict(raw)
        merged.update(properties)
        return merged
    return raw


def to_action_history_entry(*args, **kwargs) -> dict[str, Any]:
    return {
        "args": list(args),
        "kwargs": dict(kwargs),
    }


def _normalize_properties(payload: dict[str, Any] | None) -> dict[str, Any]:
    flattened = dict(payload or {})
    properties = flattened.pop("properties", None)
    if isinstance(properties, dict):
        merged = dict(properties)
        merged.update(flattened)
        flattened = merged
    return flattened


def _coerce_tenant_uuid(value: Any) -> uuid.UUID | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return uuid.UUID(raw)
    except ValueError:
        return None


class TapDBBackend:
    """Ursa persistence facade over TapDBConnection, TemplateManager, and InstanceFactory."""

    def __init__(
        self,
        *,
        app_username: str | None = None,
        client_id: str | None = None,
        namespace: str | None = None,
        tapdb_env: str | None = None,
        bundle: TapdbClientBundle | None = None,
    ) -> None:
        ensure_tapdb_version()
        if app_username is None or client_id is None or namespace is None or tapdb_env is None:
            try:
                from daylib_ursa.config import get_settings

                settings = get_settings()
            except Exception:
                settings = None
        else:
            settings = None

        resolved_client_id = (
            client_id
            or str(getattr(settings, "tapdb_client_id", "") or "").strip()
            or DEFAULT_TAPDB_CLIENT_ID
        )
        resolved_namespace = (
            namespace
            or str(getattr(settings, "tapdb_database_name", "") or "").strip()
            or DEFAULT_TAPDB_DATABASE_NAME
        )
        resolved_tapdb_env = (
            tapdb_env or str(getattr(settings, "tapdb_env", "") or "").strip() or None
        )
        resolved_app_username = (
            app_username
            or str(getattr(settings, "tapdb_client_id", "") or "").strip()
            or resolved_client_id
        )
        self.bundle = bundle or get_tapdb_bundle(
            client_id=resolved_client_id,
            namespace=resolved_namespace,
            tapdb_env=resolved_tapdb_env,
            app_username=resolved_app_username,
        )
        self._conn = self.bundle.connection
        self._tm = self.bundle.template_manager
        self._factory = self.bundle.instance_factory

    def session_scope(self, commit: bool = False):
        return self._conn.session_scope(commit=commit)

    def ensure_templates(self, session) -> None:
        missing = []
        for spec in URSA_TEMPLATE_DEFINITIONS:
            if self._tm.get_template(session, spec.template_code) is None:
                missing.append(spec.template_code)
        if missing:
            raise RuntimeError(
                "Missing Ursa templates. Seed the Ursa TapDB JSON pack before "
                f"running the service: {', '.join(missing)}"
            )

    def create_instance(
        self,
        session,
        template_code: str,
        name: str,
        *,
        json_addl: dict[str, Any] | None = None,
        bstatus: str | None = None,
        tenant_id: uuid.UUID | str | None = None,
        create_children: bool = False,
    ) -> Any:
        properties = _normalize_properties(json_addl)
        tenant_uuid = _coerce_tenant_uuid(tenant_id or properties.get("tenant_id"))
        instance = self._factory.create_instance(
            session,
            template_code=template_code,
            name=name,
            properties=properties,
            create_children=create_children,
            tenant_id=tenant_uuid,
        )
        if bstatus is not None:
            instance.bstatus = str(bstatus)
            session.flush()
        return instance

    def find_instance_by_euid(
        self,
        session,
        template_code: str,
        value: str,
        *,
        for_update: bool = False,
    ) -> Any | None:
        tmpl = self._tm.get_template(session, template_code)
        if tmpl is None:
            return None
        query = session.query(generic_instance).filter(
            generic_instance.template_uid == tmpl.uid,
            generic_instance.euid == value,
            generic_instance.is_deleted.is_(False),
        )
        if for_update:
            query = query.with_for_update()
        return query.first()

    def _property_filter(self, key: str, value: str):
        filters = [
            generic_instance.json_addl[key].astext == value,
            generic_instance.json_addl["properties"][key].astext == value,
        ]
        if key == "tenant_id":
            tenant_uuid = _coerce_tenant_uuid(value)
            if tenant_uuid is not None:
                filters.append(cast(generic_instance.tenant_id, String) == str(tenant_uuid))
        return or_(*filters)

    def find_instance_by_external_id(
        self,
        session,
        template_code: str,
        key: str,
        value: str,
    ) -> Any | None:
        tmpl = self._tm.get_template(session, template_code)
        if tmpl is None:
            return None
        return (
            session.query(generic_instance)
            .filter(
                generic_instance.template_uid == tmpl.uid,
                self._property_filter(key, value),
                generic_instance.is_deleted.is_(False),
            )
            .first()
        )

    def list_instances_by_template(
        self,
        session,
        template_code: str,
        *,
        limit: int = 200,
    ) -> list[Any]:
        tmpl = self._tm.get_template(session, template_code)
        if tmpl is None:
            return []
        return (
            session.query(generic_instance)
            .filter(
                generic_instance.template_uid == tmpl.uid,
                generic_instance.is_deleted.is_(False),
            )
            .order_by(generic_instance.created_dt.desc())
            .limit(limit)
            .all()
        )

    def list_instances_by_property(
        self,
        session,
        template_code: str,
        key: str,
        value: str,
        *,
        limit: int = 200,
    ) -> list[Any]:
        tmpl = self._tm.get_template(session, template_code)
        if tmpl is None:
            return []
        return (
            session.query(generic_instance)
            .filter(
                generic_instance.template_uid == tmpl.uid,
                self._property_filter(key, value),
                generic_instance.is_deleted.is_(False),
            )
            .order_by(generic_instance.created_dt.desc())
            .limit(limit)
            .all()
        )

    def create_lineage(
        self,
        session,
        *,
        parent,
        child,
        relationship_type: str = "generic",
    ) -> Any:
        lineage = generic_instance_lineage(
            name=f"{parent.euid}->{child.euid}",
            polymorphic_discriminator="generic_instance_lineage",
            category="generic",
            type="lineage",
            subtype="instance_lineage",
            version=__version__,
            bstatus="active",
            parent_instance_uid=parent.uid,
            child_instance_uid=child.uid,
            relationship_type=relationship_type,
            parent_type=parent.polymorphic_discriminator,
            child_type=child.polymorphic_discriminator,
        )
        session.add(lineage)
        session.flush()
        return lineage

    def list_children(
        self,
        session,
        *,
        parent,
        relationship_type: str,
    ) -> list[Any]:
        child_uids = [
            row.child_instance_uid
            for row in session.query(generic_instance_lineage)
            .filter(
                generic_instance_lineage.parent_instance_uid == parent.uid,
                generic_instance_lineage.relationship_type == relationship_type,
                generic_instance_lineage.is_deleted.is_(False),
            )
            .all()
        ]
        if not child_uids:
            return []
        return (
            session.query(generic_instance)
            .filter(
                generic_instance.uid.in_(child_uids),
                generic_instance.is_deleted.is_(False),
            )
            .order_by(generic_instance.created_dt.asc())
            .all()
        )

    def list_parents(
        self,
        session,
        *,
        child,
        relationship_type: str,
    ) -> list[Any]:
        parent_uids = [
            row.parent_instance_uid
            for row in session.query(generic_instance_lineage)
            .filter(
                generic_instance_lineage.child_instance_uid == child.uid,
                generic_instance_lineage.relationship_type == relationship_type,
                generic_instance_lineage.is_deleted.is_(False),
            )
            .all()
        ]
        if not parent_uids:
            return []
        return (
            session.query(generic_instance)
            .filter(
                generic_instance.uid.in_(parent_uids),
                generic_instance.is_deleted.is_(False),
            )
            .order_by(generic_instance.created_dt.asc())
            .all()
        )

    def update_instance_json(
        self,
        session,
        instance,
        updates: dict[str, Any],
    ) -> None:
        raw = dict(instance.json_addl or {})
        props = raw.get("properties")
        if not isinstance(props, dict):
            props = {}
            raw["properties"] = props
        normalized = _normalize_properties(updates)
        for key, value in normalized.items():
            props[key] = value
        if "tenant_id" in normalized:
            instance.tenant_id = _coerce_tenant_uuid(normalized.get("tenant_id"))
        instance.json_addl = raw
        session.flush()


__all__ = [
    "TEMPLATE_DEFINITIONS",
    "TapDBBackend",
    "TemplateSpec",
    "URSA_TEMPLATE_DEFINITIONS",
    "from_json_addl",
    "to_action_history_entry",
    "utc_now_iso",
]
