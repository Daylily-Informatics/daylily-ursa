from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from daylib_ursa.auth import ActorContext, AuthError, UserTokenService
from daylib_ursa.config import Settings
from daylib_ursa.workset_api import create_app


@dataclass
class _Instance:
    uid: int
    euid: str
    name: str
    json_addl: dict
    bstatus: str
    template_code: str
    created_dt: datetime
    modified_dt: datetime
    polymorphic_discriminator: str = "generic_instance"


class MemoryBackend:
    def __init__(self) -> None:
        self._uid = 0
        self.instances: list[_Instance] = []
        self.lineages: list[tuple[_Instance, _Instance, str]] = []

    @contextmanager
    def session_scope(self, commit: bool = False):
        _ = commit
        yield object()

    def create_instance(self, session, template_code: str, name: str, *, json_addl, bstatus, singleton=False):
        _ = (session, singleton)
        self._uid += 1
        prefix = {
            "integration/auth/user-token/1.0/": "UT",
            "integration/auth/user-token-revision/1.0/": "UR",
            "integration/auth/user-token-usage/1.0/": "UG",
        }.get(template_code, "GI")
        now = datetime.now(timezone.utc)
        instance = _Instance(
            uid=self._uid,
            euid=f"{prefix}-{self._uid}",
            name=name,
            json_addl=dict(json_addl),
            bstatus=bstatus,
            template_code=template_code,
            created_dt=now,
            modified_dt=now,
        )
        self.instances.append(instance)
        return instance

    def create_lineage(self, session, *, parent, child, relationship_type, name=None):
        _ = (session, name)
        self.lineages.append((parent, child, relationship_type))

    def list_children(self, session, *, parent, relationship_type=None):
        _ = session
        return [
            child
            for source, child, rel in self.lineages
            if source is parent and (relationship_type is None or rel == relationship_type)
        ]

    def list_parents(self, session, *, child, relationship_type=None):
        _ = session
        return [
            parent
            for parent, target, rel in self.lineages
            if target is child and (relationship_type is None or rel == relationship_type)
        ]

    def find_instance_by_euid(self, session, *, template_code: str, value: str, for_update: bool = False):
        _ = (session, for_update)
        for instance in self.instances:
            if instance.template_code == template_code and instance.euid == value:
                return instance
        return None

    def find_instance_by_external_id(self, session, *, template_code: str, key: str, value: str):
        _ = session
        for instance in self.instances:
            if instance.template_code != template_code:
                continue
            if str(instance.json_addl.get(key) or "") == value:
                return instance
        return None

    def list_instances_by_property(self, session, *, template_code: str, key: str, value: str, limit: int = 200):
        _ = session
        rows = [
            instance
            for instance in self.instances
            if instance.template_code == template_code and str(instance.json_addl.get(key) or "") == value
        ]
        return list(reversed(rows))[:limit]

    def list_instances_by_template(self, session, *, template_code: str, limit: int = 100):
        _ = session
        rows = [instance for instance in self.instances if instance.template_code == template_code]
        return list(reversed(rows))[:limit]


class DummyIdentityClient:
    def resolve_access_token(self, access_token: str) -> ActorContext:
        assert access_token == "atlas-token"
        return ActorContext(
            user_id="user-1",
            atlas_tenant_id="TEN-1",
            roles=("admin",),
            auth_source="atlas_bearer",
        )

    def resolve_user(self, user_id: str) -> ActorContext:
        return ActorContext(
            user_id=user_id,
            atlas_tenant_id="TEN-1",
            roles=("admin",),
            auth_source="ursa_token",
        )


class DummyResourceStore:
    def list_worksets(self, *, atlas_tenant_id: str, limit: int = 100):
        _ = (atlas_tenant_id, limit)
        return []


class DummyAnalysisStore:
    def list_analyses(self, *, atlas_tenant_id=None, workset_euid=None, limit=200):  # pragma: no cover - not used
        _ = (atlas_tenant_id, workset_euid, limit)
        return []


def _settings() -> Settings:
    return Settings(
        cors_origins="*",
        ursa_internal_api_key="ursa-test-key",
        bloom_base_url="https://bloom.example",
        atlas_base_url="https://atlas.example",
        ursa_internal_output_bucket="ursa-internal",
        ursa_tapdb_mount_enabled=False,
    )


def test_user_token_service_create_validate_revoke_and_usage_flow() -> None:
    backend = MemoryBackend()
    service = UserTokenService(backend=backend, identity_client=DummyIdentityClient())
    actor = DummyIdentityClient().resolve_access_token("atlas-token")

    record, plaintext = service.create_token(
        actor=actor,
        owner_user_id=actor.user_id,
        token_name="local dev",
        scope="internal_rw",
        note="first token",
    )
    assert plaintext.startswith("urs_")
    assert service.list_tokens(actor=actor)[0].token_euid == record.token_euid

    validated = service.validate_token(plaintext)
    assert validated.actor.user_id == "user-1"
    assert validated.token.scope == "internal_rw"

    service.log_usage(
        token_euid=record.token_euid,
        actor_user_id=actor.user_id,
        endpoint="/api/v1/worksets",
        http_method="GET",
        response_status=200,
        ip_address="127.0.0.1",
        user_agent="pytest",
        request_metadata={"request_id": "abc"},
    )
    usage = service.list_usage(actor=actor, token_euid=record.token_euid)
    assert usage[0].endpoint == "/api/v1/worksets"

    revoked = service.revoke_token(actor=actor, token_euid=record.token_euid, note="cleanup")
    assert revoked.status == "REVOKED"
    with pytest.raises(AuthError, match="revoked"):
        service.validate_token(plaintext)


def test_user_routes_reject_shared_api_key_and_accept_ursa_bearer_tokens() -> None:
    backend = MemoryBackend()
    identity = DummyIdentityClient()
    service = UserTokenService(backend=backend, identity_client=identity)
    actor = identity.resolve_access_token("atlas-token")
    token_record, plaintext = service.create_token(
        actor=actor,
        owner_user_id=actor.user_id,
        token_name="gui token",
        scope="internal_rw",
    )

    app = create_app(
        DummyAnalysisStore(),
        bloom_client=object(),
        identity_client=identity,
        resource_store=DummyResourceStore(),
        token_service=service,
        settings=_settings(),
    )

    with TestClient(app) as client:
        rejected = client.get("/api/v1/worksets", headers={"X-API-Key": "ursa-test-key"})
        accepted = client.get("/api/v1/worksets", headers={"Authorization": f"Bearer {plaintext}"})

    assert rejected.status_code == 401
    assert "not allowed" in rejected.json()["detail"]
    assert accepted.status_code == 200
    usage = service.list_usage(actor=actor, token_euid=token_record.token_euid)
    assert usage[0].response_status == 200
