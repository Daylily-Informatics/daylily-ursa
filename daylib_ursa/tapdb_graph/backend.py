"""Ursa ↔ TapDB adapter.

Provides the ``TapDBBackend`` class that Ursa stores (auth, analysis,
resource) use for all persistence operations.  Prior to TapDB 3.x a
helper surface (``UrsaTapdbRepository``, ``URSA_TEMPLATE_DEFINITIONS``,
etc.) was exported from ``daylily_tapdb`` itself.  TapDB 3.x removed
those Ursa-specific symbols, so this module now implements them locally
on top of the generic 3.x primitives (``TAPDBConnection``,
``TemplateManager``, ``generic_instance``, ``generic_instance_lineage``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from typing import Any

_tapdb_import_error: Exception | None = None
try:
    from daylily_tapdb import (
        TAPDBConnection,
        TemplateManager,
        generic_instance,
        generic_instance_lineage,
        utc_now_iso,
    )
    from daylily_tapdb.cli.db_config import get_db_config_for_env
except ImportError as exc:  # pragma: no cover - compatibility path for reduced test envs
    _tapdb_import_error = exc

    def utc_now_iso() -> str:  # type: ignore[misc]
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version gate
# ---------------------------------------------------------------------------

_MINIMUM_TAPDB_VERSION = (2, 0, 1)
_MAXIMUM_TAPDB_MAJOR = 4  # bumped: TapDB 3.x is now supported


def _parse_version_parts(raw_version: str) -> tuple[int, int, int]:
    cleaned = raw_version.strip()
    parts = cleaned.split(".")
    numeric: list[int] = []
    for part in parts[:3]:
        digits = []
        for char in part:
            if char.isdigit():
                digits.append(char)
            else:
                break
        numeric.append(int("".join(digits) or "0"))
    while len(numeric) < 3:
        numeric.append(0)
    return tuple(numeric[:3])


def _validate_tapdb_version() -> None:
    if _tapdb_import_error is not None:
        return
    version = importlib_metadata.version("daylily-tapdb")
    parsed = _parse_version_parts(version)
    if parsed < _MINIMUM_TAPDB_VERSION or parsed[0] >= _MAXIMUM_TAPDB_MAJOR:
        raise RuntimeError(
            f"Ursa requires daylily-tapdb>=2.0.1,<4. "
            f"Installed version is {version}."
        )


_validate_tapdb_version()

# ---------------------------------------------------------------------------
# Template catalogue
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemplateSpec:
    """Lightweight descriptor for an Ursa-managed TapDB template."""

    template_code: str


URSA_TEMPLATE_DEFINITIONS: list[TemplateSpec] = [
    # analysis
    TemplateSpec("workflow/analysis/run-linked/1.0/"),
    TemplateSpec("data/artifact/analysis-output/1.0/"),
    TemplateSpec("action/analysis/review-event/1.0/"),
    TemplateSpec("action/analysis/atlas-return/1.0/"),
    TemplateSpec("integration/reference/sequenced-assignment-context/1.0/"),
    # worksets & manifests
    TemplateSpec("workflow/workset/gui-ready/1.0/"),
    TemplateSpec("data/manifest/dewey-bound/1.0/"),
    TemplateSpec("action/artifact/dewey-import/1.0/"),
    # auth
    TemplateSpec("integration/auth/user-token/1.0/"),
    TemplateSpec("integration/auth/user-token-revision/1.0/"),
    TemplateSpec("integration/auth/user-token-usage/1.0/"),
    TemplateSpec("integration/auth/client-registration/1.0/"),
    # storage
    TemplateSpec("integration/storage/linked-bucket/1.0/"),
    # cluster jobs
    TemplateSpec("workflow/cluster/ephemeral-job/1.0/"),
    TemplateSpec("action/cluster/ephemeral-job-revision/1.0/"),
    TemplateSpec("action/cluster/ephemeral-job-event/1.0/"),
]

TEMPLATE_DEFINITIONS = URSA_TEMPLATE_DEFINITIONS

# ---------------------------------------------------------------------------
# Utility helpers (were previously re-exported from daylily_tapdb)
# ---------------------------------------------------------------------------


def from_json_addl(instance) -> dict[str, Any]:
    """Extract the ``json_addl`` dict from a TapDB instance."""
    return dict(getattr(instance, "json_addl", {}) or {})


def to_action_history_entry(*args, **kwargs) -> dict[str, Any]:
    """Build a simple action-history record."""
    return {
        "args": list(args),
        "kwargs": dict(kwargs),
    }


# ---------------------------------------------------------------------------
# UrsaTapdbRepository — local implementation over TapDB 3.x primitives
# ---------------------------------------------------------------------------


class UrsaTapdbRepository:
    """Persistence facade consumed by Ursa stores.

    Wraps ``TAPDBConnection`` + ``TemplateManager`` and exposes a
    high-level API that mirrors the surface Ursa expects:

    * ``session_scope``
    * ``create_instance``
    * ``find_instance_by_euid`` / ``find_instance_by_external_id``
    * ``list_instances_by_template`` / ``list_instances_by_property``
    * ``list_children`` / ``list_parents``
    * ``create_lineage``
    * ``update_instance_json``
    * ``ensure_templates``
    """

    def __init__(self, *, app_username: str = "ursa") -> None:
        if _tapdb_import_error is not None:
            raise RuntimeError(
                "TapDB backend is unavailable in this environment"
            ) from _tapdb_import_error

        # Use the canonical TapDB config loader to resolve the correct
        # host, port, database name, and credentials for the active env.
        import os

        env = os.environ.get("TAPDB_ENV", "dev")
        try:
            cfg = get_db_config_for_env(env)
            self._conn = TAPDBConnection(
                app_username=app_username,
                db_hostname=f"{cfg['host']}:{cfg['port']}",
                db_user=cfg.get("user"),
                db_pass=cfg.get("password", ""),
                db_name=cfg["database"],
            )
        except Exception as exc:
            _log.warning(
                "get_db_config_for_env(%r) failed (%s); falling back to "
                "TAPDBConnection defaults",
                env,
                exc,
            )
            self._conn = TAPDBConnection(app_username=app_username)

        self._tm = TemplateManager()

    # -- session management -------------------------------------------------

    def session_scope(self, commit: bool = False):
        """Delegate to ``TAPDBConnection.session_scope``."""
        return self._conn.session_scope(commit=commit)

    # -- template bootstrap -------------------------------------------------

    def ensure_templates(self, session) -> None:
        """Log a warning for any Ursa templates missing from the DB."""
        missing = []
        for spec in URSA_TEMPLATE_DEFINITIONS:
            if self._tm.get_template(session, spec.template_code) is None:
                missing.append(spec.template_code)
        if missing:
            _log.warning(
                "Missing Ursa templates (seed with 'tapdb db data seed'): %s",
                missing,
            )

    # -- instance CRUD ------------------------------------------------------

    def create_instance(
        self,
        session,
        template_code: str,
        name: str,
        *,
        json_addl: dict[str, Any] | None = None,
    ) -> Any:
        """Create and flush a new ``generic_instance``."""
        tmpl = self._tm.get_template(session, template_code)
        if tmpl is None:
            raise ValueError(f"Template not found: {template_code}")
        inst = generic_instance(
            name=name,
            polymorphic_discriminator=(
                tmpl.instance_polymorphic_identity
                or tmpl.polymorphic_discriminator.replace("_template", "_instance")
            ),
            category=tmpl.category,
            type=tmpl.type,
            subtype=tmpl.subtype,
            version=tmpl.version,
            template_uid=tmpl.uid,
            json_addl=json_addl or {},
            bstatus=tmpl.json_addl.get("default_status", "created"),
            is_singleton=bool(tmpl.is_singleton),
        )
        session.add(inst)
        session.flush()
        return inst

    def find_instance_by_euid(
        self,
        session,
        template_code: str,
        value: str,
        *,
        for_update: bool = False,
    ) -> Any | None:
        """Find an instance by its EUID within a template scope."""
        tmpl = self._tm.get_template(session, template_code)
        if tmpl is None:
            return None
        q = session.query(generic_instance).filter(
            generic_instance.template_uid == tmpl.uid,
            generic_instance.euid == value,
            generic_instance.is_deleted.is_(False),
        )
        if for_update:
            q = q.with_for_update()
        return q.first()

    def find_instance_by_external_id(
        self,
        session,
        template_code: str,
        key: str,
        value: str,
    ) -> Any | None:
        """Find an instance where ``json_addl[key] == value``."""
        tmpl = self._tm.get_template(session, template_code)
        if tmpl is None:
            return None
        return (
            session.query(generic_instance)
            .filter(
                generic_instance.template_uid == tmpl.uid,
                generic_instance.json_addl[key].astext == value,
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
        """Return instances for a given template code."""
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
        """Return instances where ``json_addl[key] == value``."""
        tmpl = self._tm.get_template(session, template_code)
        if tmpl is None:
            return []
        return (
            session.query(generic_instance)
            .filter(
                generic_instance.template_uid == tmpl.uid,
                generic_instance.json_addl[key].astext == value,
                generic_instance.is_deleted.is_(False),
            )
            .order_by(generic_instance.created_dt.desc())
            .limit(limit)
            .all()
        )

    # -- lineage ------------------------------------------------------------

    def create_lineage(
        self,
        session,
        *,
        parent,
        child,
        relationship_type: str = "generic",
    ) -> Any:
        """Create a directed edge between two instances."""
        lineage = generic_instance_lineage(
            name=f"{parent.euid}->{child.euid}",
            polymorphic_discriminator="generic_instance_lineage",
            category="generic",
            type="lineage",
            subtype="instance_lineage",
            version="1.0.0",
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
        """Return child instances linked to *parent* via *relationship_type*."""
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
        """Return parent instances linked to *child* via *relationship_type*."""
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

    # -- json_addl mutation -------------------------------------------------

    def update_instance_json(
        self,
        session,
        instance,
        updates: dict[str, Any],
    ) -> None:
        """Merge *updates* into ``instance.json_addl`` and flush."""
        current = dict(instance.json_addl or {})
        current.update(updates)
        # Re-assign to ensure ORM dirty-tracking detects the change.
        instance.json_addl = current
        session.flush()


class TapDBBackend(UrsaTapdbRepository):
    """Thin Ursa adapter — inherits from the local repository shim."""


__all__ = [
    "TapDBBackend",
    "TEMPLATE_DEFINITIONS",
    "URSA_TEMPLATE_DEFINITIONS",
    "TemplateSpec",
    "from_json_addl",
    "to_action_history_entry",
    "utc_now_iso",
]
