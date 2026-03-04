from __future__ import annotations

import os
import datetime as dt
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, Iterable, Optional

from sqlalchemy import and_
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


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


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
        self.templates = TemplateManager()
        self.factory = InstanceFactory(self.templates)

    @contextmanager
    def session_scope(self, commit: bool = False) -> Generator[Session, None, None]:
        with self.connection.session_scope(commit=commit) as session:
            yield session

    def ensure_templates(self, session: Session) -> None:
        for spec in TEMPLATE_DEFINITIONS:
            self._ensure_template(session, spec)

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


def from_json_addl(instance: generic_instance) -> Dict[str, Any]:
    payload = dict(instance.json_addl or {})
    payload.setdefault("euid", instance.euid)
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
