from __future__ import annotations

import os
import datetime as dt
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

# Canonical EUID type alias — use this everywhere outside TapDB internals
Euid = str

from sqlalchemy import and_, text
from sqlalchemy.orm import Session

from daylily_tapdb import (
    TAPDBConnection,
    TemplateManager,
    InstanceFactory,
    generic_template,
    generic_instance,
    generic_instance_lineage,
)
from daylily_tapdb.cli.context import resolve_context
from daylily_tapdb.cli.db_config import get_db_config_for_env
from daylily_tapdb.sequences import ensure_instance_prefix_sequence


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def expected_ursa_database_name(env: str) -> str:
    return f"daylily-ursa-{str(env or '').strip().lower()}"


def _parse_template_code(template_code: str) -> tuple[str, str, str, str]:
    parts = template_code.strip("/").split("/")
    if len(parts) != 4:
        raise ValueError(f"Invalid template code: {template_code}")
    return parts[0], parts[1], parts[2], parts[3]


@dataclass(frozen=True)
class TemplateDefinition:
    template_code: str
    template_discriminator: str
    instance_discriminator: str
    instance_prefix: str
    name: str


TEMPLATE_DEFINITIONS: tuple[TemplateDefinition, ...] = (
    TemplateDefinition("actor/customer/account/1.0/", "actor_template", "actor_instance", "CT", "Customer"),
    TemplateDefinition("actor/user/account/1.0/", "actor_template", "actor_instance", "AG", "User"),
    TemplateDefinition("workflow/workset/analysis/1.0/", "workflow_template", "workflow_instance", "WS", "Workset"),
    TemplateDefinition("action/workset/state-transition/1.0/", "action_template", "action_instance", "ST", "State Transition"),
    TemplateDefinition("action/workset/lock-event/1.0/", "action_template", "action_instance", "CK", "Lock Event"),
    TemplateDefinition("file/object/registered/1.0/", "file_template", "file_instance", "FF", "Registered File"),
    TemplateDefinition("container/fileset/group/1.0/", "container_template", "container_instance", "FT", "Fileset"),
    TemplateDefinition("content/manifest/stage-samples/1.0/", "content_template", "content_instance", "MF", "Manifest"),
    TemplateDefinition("subject/person/participant/1.0/", "subject_template", "subject_instance", "SJ", "Subject"),
    TemplateDefinition("content/biospecimen/entity/1.0/", "content_template", "content_instance", "BP", "Biospecimen"),
    TemplateDefinition("content/biosample/entity/1.0/", "content_template", "content_instance", "BS", "Biosample"),
    TemplateDefinition("content/library/entity/1.0/", "content_template", "content_instance", "BR", "Library"),
    TemplateDefinition("data/storage/s3-bucket-link/1.0/", "data_template", "data_instance", "BK", "Linked Bucket"),
)


class TapDBBackend:
    """Shared TapDB backend wiring for Ursa graph persistence."""

    def __init__(self, app_username: str = "ursa"):
        # Ursa uses TapDB in strict namespace mode to avoid cross-app collisions.
        os.environ.setdefault("TAPDB_STRICT_NAMESPACE", "1")

        env = (os.environ.get("TAPDB_ENV") or "").strip()
        if not env:
            raise RuntimeError(
                "TAPDB_ENV is required (dev|test|prod).\n"
                "Example:\n"
                "  export TAPDB_ENV=dev\n"
            )

        try:
            # Ensures TAPDB_CLIENT_ID + TAPDB_DATABASE_NAME are set.
            resolve_context(require_keys=True)
            cfg = get_db_config_for_env(env)
        except Exception as exc:
            raise RuntimeError(
                "TapDB is not configured for Ursa.\n\n"
                "Required environment variables:\n"
                "  export TAPDB_STRICT_NAMESPACE=1\n"
                "  export TAPDB_CLIENT_ID=local\n"
                "  export TAPDB_DATABASE_NAME=ursa\n"
                "  export TAPDB_ENV=dev\n\n"
                "Then bootstrap TapDB (preferred):\n"
                "  tapdb config init --client-id local --database-name ursa --env dev\n"
                "  tapdb bootstrap local\n"
            ) from exc

        expected_db_name = expected_ursa_database_name(env)
        configured_db_name = str(cfg.get("database") or "").strip()
        if configured_db_name != expected_db_name:
            raise RuntimeError(
                "Invalid TapDB database name for Ursa.\n\n"
                f"TAPDB_ENV={env!r} requires database {expected_db_name!r}, "
                f"but config resolved to {configured_db_name!r}.\n\n"
                "Update your TapDB config so environments.<env>.database matches "
                "daylily-ursa-<env> for each Ursa environment."
            )

        db_hostname = f"{cfg['host']}:{cfg['port']}"
        engine_type = (cfg.get("engine_type") or "local").strip().lower()
        region = (cfg.get("region") or os.environ.get("AWS_REGION") or "us-west-2").strip()
        iam_auth_raw = str(cfg.get("iam_auth") or "").strip().lower()
        iam_auth = iam_auth_raw in {"1", "true", "yes", "on"}
        secret_arn = cfg.get("secret_arn") or cfg.get("master_secret_arn")
        self.connection = TAPDBConnection(
            db_hostname=db_hostname,
            db_user=cfg["user"],
            db_pass=cfg["password"],
            db_name=cfg["database"],
            app_username=app_username,
            engine_type=engine_type if engine_type != "local" else None,
            region=region,
            iam_auth=iam_auth,
            secret_arn=secret_arn,
        )
        # Best-effort: attach TapDB's query metrics instrumentation so Ursa can render
        # the same DB metrics page as the TapDB admin GUI.
        try:
            from admin.db_metrics import maybe_install_engine_metrics

            maybe_install_engine_metrics(self.connection.engine, env_name=env)
        except Exception:
            # Metrics are optional; never break Ursa startup for instrumentation issues.
            pass
        self.templates = TemplateManager()
        self.factory = InstanceFactory(self.templates)

    @contextmanager
    def session_scope(self, commit: bool = False) -> Generator[Session, None, None]:
        db_username_var = None
        token = None
        try:
            from admin.db_metrics import db_username_var as _db_username_var

            db_username_var = _db_username_var
            token = db_username_var.set(str(getattr(self.connection, "app_username", "")) or "unknown")
        except Exception:
            pass
        try:
            with self.connection.session_scope(commit=commit) as session:
                yield session
        finally:
            if db_username_var is not None and token is not None:
                try:
                    db_username_var.reset(token)
                except Exception:
                    pass

    @staticmethod
    def _normalize_prefix(prefix: str) -> str:
        return prefix.strip().upper()

    def _required_instance_prefixes(self, session: Session) -> list[str]:
        prefixes = {
            self._normalize_prefix(spec.instance_prefix)
            for spec in TEMPLATE_DEFINITIONS
            if spec.instance_prefix and spec.instance_prefix.strip()
        }

        template_rows = (
            session.query(generic_template.instance_prefix)
            .filter(
                generic_template.is_deleted.is_(False),
                generic_template.instance_prefix.isnot(None),
            )
            .all()
        )
        for (prefix,) in template_rows:
            value = str(prefix or "").strip()
            if value:
                prefixes.add(self._normalize_prefix(value))
        return sorted(prefixes)

    def _required_instance_sequence_names(self, session: Session) -> list[str]:
        return [f"{prefix.lower()}_instance_seq" for prefix in self._required_instance_prefixes(session)]

    def list_required_instance_sequences(self, session: Session) -> list[str]:
        return self._required_instance_sequence_names(session)

    def ensure_instance_sequences(self, session: Session) -> None:
        for prefix in sorted(set(self._required_instance_prefixes(session))):
            ensure_instance_prefix_sequence(session, prefix)

    def get_missing_instance_sequences(self, session: Session) -> list[str]:
        required = set(self._required_instance_sequence_names(session))
        if not required:
            return []
        rows = session.execute(
            text(
                "SELECT sequence_name "
                "FROM information_schema.sequences "
                "WHERE sequence_schema = 'public'"
            )
        ).fetchall()
        existing = {str(row[0]) for row in rows}
        return sorted(seq for seq in required if seq not in existing)

    def ensure_templates(self, session: Session) -> None:
        for spec in TEMPLATE_DEFINITIONS:
            self._ensure_template(session, spec)
        self.ensure_instance_sequences(session)

    def _ensure_template(self, session: Session, spec: TemplateDefinition) -> generic_template:
        template = self.templates.get_template(session, spec.template_code)
        if template is not None:
            if template.instance_prefix != spec.instance_prefix:
                template.instance_prefix = spec.instance_prefix
                session.flush()
            if template.instance_polymorphic_identity != spec.instance_discriminator:
                template.instance_polymorphic_identity = spec.instance_discriminator
                session.flush()
            return template

        category, type_, subtype, version = _parse_template_code(spec.template_code)
        template = generic_template(
            name=spec.name,
            polymorphic_discriminator=spec.template_discriminator,
            category=category,
            type=type_,
            subtype=subtype,
            version=version,
            bstatus="active",
            json_addl={},
            is_singleton=False,
            instance_prefix=spec.instance_prefix,
            instance_polymorphic_identity=spec.instance_discriminator,
            json_addl_schema=None,
        )
        session.add(template)
        session.flush()
        self.templates.clear_cache()
        return template

    def create_instance(
        self,
        session: Session,
        template_code: str,
        name: str,
        *,
        json_addl: Optional[Dict[str, Any]] = None,
        bstatus: str = "active",
        singleton: bool = False,
    ) -> generic_instance:
        template = self.templates.get_template(session, template_code)
        if template is None:
            self.ensure_templates(session)
            template = self.templates.get_template(session, template_code)
        if template is None:
            raise RuntimeError(f"Missing template: {template_code}")
        prefix = str(template.instance_prefix or "").strip()
        if prefix:
            ensure_instance_prefix_sequence(session, self._normalize_prefix(prefix))

        instance = self.factory.create_instance(
            session=session,
            template_code=template_code,
            name=name,
            properties={},
            create_children=False,
        )
        payload = dict(instance.json_addl or {})
        if json_addl:
            payload.update(json_addl)
        instance.json_addl = payload
        instance.bstatus = bstatus
        instance.is_singleton = singleton
        session.flush()
        return instance

    def update_instance_json(self, session: Session, instance: generic_instance, updates: Dict[str, Any]) -> None:
        payload = dict(instance.json_addl or {})
        payload.update(updates)
        instance.json_addl = payload
        session.flush()

    def find_instance_by_euid(
        self,
        session: Session,
        euid: str,
    ) -> Optional[generic_instance]:
        """Look up a non-deleted instance by its TapDB EUID."""
        return (
            session.query(generic_instance)
            .filter(
                generic_instance.euid == euid,
                generic_instance.is_deleted.is_(False),
            )
            .first()
        )

    def find_instance_by_external_id(
        self,
        session: Session,
        *,
        template_code: str,
        key: str,
        value: str,
    ) -> Optional[generic_instance]:
        template = self.templates.get_template(session, template_code)
        if template is None:
            return None
        return (
            session.query(generic_instance)
            .filter(
                generic_instance.template_uuid == template.uuid,
                generic_instance.is_deleted.is_(False),
                generic_instance.json_addl[key].as_string() == value,
            )
            .first()
        )

    def list_instances_by_template(
        self,
        session: Session,
        *,
        template_code: str,
        limit: int = 100,
    ) -> list[generic_instance]:
        template = self.templates.get_template(session, template_code)
        if template is None:
            return []
        return (
            session.query(generic_instance)
            .filter(
                generic_instance.template_uuid == template.uuid,
                generic_instance.is_deleted.is_(False),
            )
            .order_by(generic_instance.created_dt.desc())
            .limit(limit)
            .all()
        )

    def create_lineage(
        self,
        session: Session,
        *,
        parent: generic_instance,
        child: generic_instance,
        relationship_type: str,
        name: Optional[str] = None,
    ) -> generic_instance_lineage:
        existing = (
            session.query(generic_instance_lineage)
            .filter(
                generic_instance_lineage.parent_instance_uuid == parent.uuid,
                generic_instance_lineage.child_instance_uuid == child.uuid,
                generic_instance_lineage.relationship_type == relationship_type,
                generic_instance_lineage.is_deleted.is_(False),
            )
            .first()
        )
        if existing is not None:
            return existing

        lineage = generic_instance_lineage(
            name=name or f"{parent.euid}->{child.euid}:{relationship_type}",
            polymorphic_discriminator="generic_instance_lineage",
            category="generic",
            type="lineage",
            subtype="instance_lineage",
            version="1.0.0",
            bstatus="active",
            json_addl={},
            is_singleton=False,
            parent_type=parent.polymorphic_discriminator,
            child_type=child.polymorphic_discriminator,
            relationship_type=relationship_type,
            parent_instance_uuid=parent.uuid,
            child_instance_uuid=child.uuid,
        )
        session.add(lineage)
        session.flush()
        return lineage

    def list_children(
        self,
        session: Session,
        *,
        parent: generic_instance,
        relationship_type: Optional[str] = None,
    ) -> list[generic_instance]:
        query = (
            session.query(generic_instance)
            .join(
                generic_instance_lineage,
                generic_instance_lineage.child_instance_uuid == generic_instance.uuid,
            )
            .filter(
                generic_instance_lineage.parent_instance_uuid == parent.uuid,
                generic_instance_lineage.is_deleted.is_(False),
                generic_instance.is_deleted.is_(False),
            )
        )
        if relationship_type:
            query = query.filter(generic_instance_lineage.relationship_type == relationship_type)
        return query.all()

    def list_parents(
        self,
        session: Session,
        *,
        child: generic_instance,
        relationship_type: Optional[str] = None,
    ) -> list[generic_instance]:
        query = (
            session.query(generic_instance)
            .join(
                generic_instance_lineage,
                generic_instance_lineage.parent_instance_uuid == generic_instance.uuid,
            )
            .filter(
                generic_instance_lineage.child_instance_uuid == child.uuid,
                generic_instance_lineage.is_deleted.is_(False),
                generic_instance.is_deleted.is_(False),
            )
        )
        if relationship_type:
            query = query.filter(generic_instance_lineage.relationship_type == relationship_type)
        return query.all()

    def get_customer_owned(
        self,
        session: Session,
        *,
        customer: generic_instance,
        template_code: str,
        relationship_type: str = "owns",
        limit: int = 200,
    ) -> list[generic_instance]:
        template = self.templates.get_template(session, template_code)
        if template is None:
            return []
        return (
            session.query(generic_instance)
            .join(
                generic_instance_lineage,
                and_(
                    generic_instance_lineage.child_instance_uuid == generic_instance.uuid,
                    generic_instance_lineage.parent_instance_uuid == customer.uuid,
                    generic_instance_lineage.relationship_type == relationship_type,
                    generic_instance_lineage.is_deleted.is_(False),
                ),
            )
            .filter(
                generic_instance.template_uuid == template.uuid,
                generic_instance.is_deleted.is_(False),
            )
            .order_by(generic_instance.created_dt.desc())
            .limit(limit)
            .all()
        )


def require_euid(euid: Optional[str]) -> str:
    """Validate that an EUID is present and non-empty.  Raises ValueError otherwise."""
    if not euid:
        raise ValueError("EUID is required but was empty or None")
    return euid


def from_json_addl(instance: generic_instance) -> Dict[str, Any]:
    payload = dict(instance.json_addl or {})
    # ALWAYS use the instance-level euid — never trust json_addl["euid"]
    payload["euid"] = instance.euid
    payload.setdefault("name", instance.name)
    payload.setdefault("created_at", instance.created_dt.isoformat().replace("+00:00", "Z") if instance.created_dt else utc_now_iso())
    payload.setdefault("updated_at", instance.modified_dt.isoformat().replace("+00:00", "Z") if instance.modified_dt else payload["created_at"])
    payload.setdefault("state", instance.bstatus)
    return payload


def to_action_history_entry(instance: generic_instance) -> Dict[str, Any]:
    payload = dict(instance.json_addl or {})
    payload.setdefault("timestamp", instance.created_dt.isoformat().replace("+00:00", "Z") if instance.created_dt else utc_now_iso())
    payload.setdefault("event_euid", instance.euid)
    payload.setdefault("event_name", instance.name)
    return payload
