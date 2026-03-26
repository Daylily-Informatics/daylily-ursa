from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import httpx

from daylib_ursa.resource_store import ResourceStore
from daylib_ursa.tapdb_graph import TapDBBackend, from_json_addl, utc_now_iso

USER_TOKEN_TEMPLATE = "integration/auth/user-token/1.0/"
USER_TOKEN_REVISION_TEMPLATE = "integration/auth/user-token-revision/1.0/"
USER_TOKEN_USAGE_TEMPLATE = "integration/auth/user-token-usage/1.0/"

USER_TOKEN_PREFIX = "urs_"
TOKEN_STATUS_ACTIVE = "ACTIVE"
TOKEN_STATUS_REVOKED = "REVOKED"


class AuthError(RuntimeError):
    """Raised when Ursa auth or token operations fail."""


@dataclass(frozen=True)
class ActorContext:
    user_id: str
    atlas_tenant_id: str
    roles: tuple[str, ...]
    email: str | None = None
    display_name: str | None = None
    organization: str | None = None
    site: str | None = None
    auth_source: Literal["atlas_bearer", "ursa_token"] = "atlas_bearer"
    token_euid: str | None = None
    token_scope: str | None = None
    client_registration_euid: str | None = None

    @property
    def is_admin(self) -> bool:
        admin_markers = {"admin", "atlas_admin", "ursa_admin", "tenant_admin", "org_admin"}
        return any(role in admin_markers or role.endswith(":admin") for role in self.roles)


@dataclass(frozen=True)
class UserTokenRecord:
    token_euid: str
    owner_user_id: str
    token_name: str
    token_prefix: str
    scope: str
    status: str
    expires_at: str
    created_at: str
    updated_at: str
    created_by: str | None
    last_used_at: str | None
    revoked_at: str | None
    note: str | None
    client_registration_euid: str | None


@dataclass(frozen=True)
class UserTokenUsageRecord:
    usage_euid: str
    token_euid: str
    actor_user_id: str
    endpoint: str
    http_method: str
    response_status: int
    ip_address: str | None
    user_agent: str | None
    request_metadata: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class TokenValidationResult:
    actor: ActorContext
    token: UserTokenRecord


@dataclass(frozen=True)
class AtlasUserDirectoryEntry:
    user_id: str
    atlas_tenant_id: str
    organization_id: str
    organization_name: str | None
    site_id: str | None
    site_name: str | None
    roles: tuple[str, ...]
    email: str | None
    display_name: str | None
    is_active: bool


def _iso_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _normalize_roles(raw_roles: Any) -> tuple[str, ...]:
    if isinstance(raw_roles, str):
        items = [raw_roles]
    elif isinstance(raw_roles, list):
        items = [str(item) for item in raw_roles]
    else:
        items = []
    normalized = []
    seen: set[str] = set()
    for item in items:
        value = str(item or "").strip().lower()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    if not normalized:
        normalized.append("internal_ro")
    return tuple(normalized)


def _coerce_actor(body: dict[str, Any], *, auth_source: Literal["atlas_bearer", "ursa_token"]) -> ActorContext:
    user_id = str(body.get("user_id") or body.get("subject") or body.get("sub") or "").strip()
    atlas_tenant_id = str(body.get("atlas_tenant_id") or body.get("tenant_id") or "").strip()
    if not user_id:
        raise AuthError("Atlas identity response missing user_id")
    if not atlas_tenant_id:
        raise AuthError("Atlas identity response missing atlas_tenant_id")
    return ActorContext(
        user_id=user_id,
        atlas_tenant_id=atlas_tenant_id,
        roles=_normalize_roles(body.get("roles")),
        email=str(body.get("email") or "").strip() or None,
        display_name=str(body.get("display_name") or body.get("name") or "").strip() or None,
        organization=(
            str(
                body.get("organization_name")
                or body.get("organization")
                or body.get("org")
                or ""
            ).strip()
            or None
        ),
        site=str(body.get("site_name") or body.get("site") or "").strip() or None,
        auth_source=auth_source,
    )


class AtlasIdentityClient:
    def __init__(
        self,
        *,
        base_url: str,
        internal_api_key: str | None = None,
        verify_ssl: bool = True,
        timeout_seconds: float = 10.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = str(base_url or "").strip().rstrip("/")
        self.internal_api_key = str(internal_api_key or "").strip()
        self.verify_ssl = verify_ssl
        self.timeout_seconds = timeout_seconds
        self.client = client

    def _http_client(self) -> tuple[httpx.Client, bool]:
        if self.client is not None:
            return self.client, False
        return (
            httpx.Client(timeout=self.timeout_seconds, verify=self.verify_ssl),
            True,
        )

    def resolve_access_token(self, access_token: str) -> ActorContext:
        token = str(access_token or "").strip()
        if not token:
            raise AuthError("Bearer token is required")
        client, close_client = self._http_client()
        try:
            response = client.get(
                f"{self.base_url}/auth/me",
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {token}",
                },
            )
        except httpx.HTTPError as exc:
            raise AuthError(f"Atlas identity lookup failed: {exc}") from exc
        finally:
            if close_client:
                client.close()
        if response.status_code >= 400:
            raise AuthError(f"Atlas identity lookup returned {response.status_code}: {response.text}")
        return _coerce_actor(dict(response.json()), auth_source="atlas_bearer")

    def resolve_user(self, user_id: str) -> ActorContext:
        if not self.internal_api_key:
            raise AuthError("ATLAS_INTERNAL_API_KEY is required for Ursa token validation")
        target = str(user_id or "").strip()
        if not target:
            raise AuthError("user_id is required")
        client, close_client = self._http_client()
        try:
            response = client.get(
                f"{self.base_url}/internal/users/{target}/context",
                headers={
                    "Accept": "application/json",
                    "X-API-Key": self.internal_api_key,
                },
            )
        except httpx.HTTPError as exc:
            raise AuthError(f"Atlas user lookup failed: {exc}") from exc
        finally:
            if close_client:
                client.close()
        if response.status_code >= 400:
            raise AuthError(f"Atlas user lookup returned {response.status_code}: {response.text}")
        return _coerce_actor(dict(response.json()), auth_source="ursa_token")

    def list_users(
        self,
        *,
        tenant_id: str | None = None,
        search: str | None = None,
        active_only: bool = True,
        limit: int = 50,
        skip: int = 0,
    ) -> list[AtlasUserDirectoryEntry]:
        if not self.internal_api_key:
            raise AuthError("ATLAS_INTERNAL_API_KEY is required for Atlas admin user search")
        params: dict[str, Any] = {
            "active_only": "true" if active_only else "false",
            "limit": max(1, min(int(limit or 50), 200)),
            "skip": max(0, int(skip or 0)),
        }
        if str(tenant_id or "").strip():
            params["tenant_id"] = str(tenant_id).strip()
        if str(search or "").strip():
            params["search"] = str(search).strip()

        client, close_client = self._http_client()
        try:
            response = client.get(
                f"{self.base_url}/internal/users",
                headers={
                    "Accept": "application/json",
                    "X-API-Key": self.internal_api_key,
                },
                params=params,
            )
        except httpx.HTTPError as exc:
            raise AuthError(f"Atlas user search failed: {exc}") from exc
        finally:
            if close_client:
                client.close()
        if response.status_code >= 400:
            raise AuthError(f"Atlas user search returned {response.status_code}: {response.text}")
        body = dict(response.json())
        items = body.get("items")
        if not isinstance(items, list):
            raise AuthError("Atlas user search response missing items")
        results: list[AtlasUserDirectoryEntry] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            results.append(
                AtlasUserDirectoryEntry(
                    user_id=str(item.get("user_id") or "").strip(),
                    atlas_tenant_id=str(item.get("tenant_id") or "").strip(),
                    organization_id=str(item.get("organization_id") or "").strip(),
                    organization_name=str(item.get("organization_name") or "").strip() or None,
                    site_id=str(item.get("site_id") or "").strip() or None,
                    site_name=str(item.get("site_name") or "").strip() or None,
                    roles=_normalize_roles(item.get("roles")),
                    email=str(item.get("email") or "").strip() or None,
                    display_name=str(item.get("display_name") or "").strip() or None,
                    is_active=bool(item.get("is_active", True)),
                )
            )
        return results


class UserTokenService:
    def __init__(
        self,
        *,
        backend: TapDBBackend,
        identity_client: AtlasIdentityClient,
        resource_store: ResourceStore | None = None,
    ) -> None:
        self.backend = backend
        self.identity_client = identity_client
        self.resource_store = resource_store

    @staticmethod
    def generate_plaintext_token() -> str:
        return USER_TOKEN_PREFIX + secrets.token_hex(32)

    @staticmethod
    def hash_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def display_prefix(token: str) -> str:
        return f"{token[:12]}..."

    @staticmethod
    def _revision_sort_key(instance: Any) -> int:
        payload = from_json_addl(instance)
        return int(payload.get("revision_no") or 0)

    def _latest_revision(self, session, token_instance) -> Any | None:
        revisions = self.backend.list_children(
            session,
            parent=token_instance,
            relationship_type="revision",
        )
        if not revisions:
            return None
        revisions.sort(key=self._revision_sort_key, reverse=True)
        return revisions[0]

    def _token_record(self, session, token_instance) -> UserTokenRecord:
        token_payload = from_json_addl(token_instance)
        revision = self._latest_revision(session, token_instance)
        if revision is None:
            raise AuthError(f"token missing revision: {token_instance.euid}")
        revision_payload = from_json_addl(revision)
        return UserTokenRecord(
            token_euid=str(token_instance.euid),
            owner_user_id=str(token_payload.get("owner_user_id") or ""),
            token_name=str(token_instance.name or token_payload.get("token_name") or ""),
            token_prefix=str(token_payload.get("token_prefix") or ""),
            scope=str(token_payload.get("scope") or "internal_ro"),
            status=str(revision_payload.get("status") or TOKEN_STATUS_ACTIVE),
            expires_at=str(revision_payload.get("expires_at") or ""),
            created_at=str(token_payload.get("created_at") or utc_now_iso()),
            updated_at=str(revision_payload.get("created_at") or token_payload.get("updated_at") or utc_now_iso()),
            created_by=str(token_payload.get("created_by") or "").strip() or None,
            last_used_at=str(revision_payload.get("last_used_at") or "").strip() or None,
            revoked_at=str(revision_payload.get("revoked_at") or "").strip() or None,
            note=str(revision_payload.get("note") or "").strip() or None,
            client_registration_euid=str(token_payload.get("client_registration_euid") or "").strip() or None,
        )

    def create_token(
        self,
        *,
        actor: ActorContext,
        owner_user_id: str,
        token_name: str,
        scope: str,
        expires_in_days: int = 30,
        note: str | None = None,
        client_registration_euid: str | None = None,
    ) -> tuple[UserTokenRecord, str]:
        owner = str(owner_user_id or "").strip()
        if not owner:
            raise AuthError("owner_user_id is required")
        if owner != actor.user_id and not actor.is_admin:
            raise AuthError("Cannot create tokens for another user")
        if client_registration_euid and not actor.is_admin:
            raise AuthError("Client-bound tokens require admin privileges")
        expires_at = (
            datetime.now(UTC) + timedelta(days=max(1, min(int(expires_in_days or 30), 3650)))
        ).isoformat().replace("+00:00", "Z")
        plaintext = self.generate_plaintext_token()
        token_hash = self.hash_token(plaintext)
        token_prefix = self.display_prefix(plaintext)
        created_at = _iso_now()

        with self.backend.session_scope(commit=True) as session:
            token = self.backend.create_instance(
                session,
                USER_TOKEN_TEMPLATE,
                token_name.strip(),
                json_addl={
                    "owner_user_id": owner,
                    "token_name": token_name.strip(),
                    "token_prefix": token_prefix,
                    "scope": str(scope or "internal_ro").strip().lower() or "internal_ro",
                    "created_by": actor.user_id,
                    "created_at": created_at,
                    "updated_at": created_at,
                    "client_registration_euid": str(client_registration_euid or "").strip() or None,
                },
                bstatus=TOKEN_STATUS_ACTIVE,
            )
            revision = self.backend.create_instance(
                session,
                USER_TOKEN_REVISION_TEMPLATE,
                f"revision:{token.euid}:1",
                json_addl={
                    "token_euid": str(token.euid),
                    "token_hash": token_hash,
                    "revision_no": 1,
                    "status": TOKEN_STATUS_ACTIVE,
                    "expires_at": expires_at,
                    "last_used_at": None,
                    "revoked_at": None,
                    "note": note,
                    "created_by": actor.user_id,
                    "created_at": created_at,
                },
                bstatus=TOKEN_STATUS_ACTIVE,
            )
            self.backend.create_lineage(
                session,
                parent=token,
                child=revision,
                relationship_type="revision",
            )
            return self._token_record(session, token), plaintext

    def list_tokens(self, *, actor: ActorContext, owner_user_id: str | None = None) -> list[UserTokenRecord]:
        target_owner = None if owner_user_id is None else str(owner_user_id).strip()
        if target_owner is None:
            target_owner = actor.user_id
        if target_owner != actor.user_id and not actor.is_admin:
            raise AuthError("Cannot list another user's tokens")
        with self.backend.session_scope(commit=False) as session:
            if actor.is_admin and target_owner == "*":
                tokens = self.backend.list_instances_by_template(
                    session,
                    template_code=USER_TOKEN_TEMPLATE,
                    limit=500,
                )
            else:
                tokens = self.backend.list_instances_by_property(
                    session,
                    template_code=USER_TOKEN_TEMPLATE,
                    key="owner_user_id",
                    value=target_owner,
                    limit=500,
                )
            return [self._token_record(session, token) for token in tokens]

    def revoke_token(self, *, actor: ActorContext, token_euid: str, note: str | None = None) -> UserTokenRecord:
        with self.backend.session_scope(commit=True) as session:
            token = self.backend.find_instance_by_euid(
                session,
                template_code=USER_TOKEN_TEMPLATE,
                value=token_euid,
                for_update=True,
            )
            if token is None:
                raise KeyError(f"token not found: {token_euid}")
            token_payload = from_json_addl(token)
            owner_user_id = str(token_payload.get("owner_user_id") or "")
            if owner_user_id != actor.user_id and not actor.is_admin:
                raise AuthError("Cannot revoke another user's token")
            latest = self._latest_revision(session, token)
            if latest is None:
                raise AuthError(f"token missing revision: {token_euid}")
            latest_payload = from_json_addl(latest)
            if str(latest_payload.get("status") or "") == TOKEN_STATUS_REVOKED:
                return self._token_record(session, token)
            revision_no = int(latest_payload.get("revision_no") or 0) + 1
            created_at = _iso_now()
            revision = self.backend.create_instance(
                session,
                USER_TOKEN_REVISION_TEMPLATE,
                f"revision:{token.euid}:{revision_no}",
                json_addl={
                    "token_euid": str(token.euid),
                    "token_hash": str(latest_payload.get("token_hash") or ""),
                    "revision_no": revision_no,
                    "status": TOKEN_STATUS_REVOKED,
                    "expires_at": str(latest_payload.get("expires_at") or ""),
                    "last_used_at": str(latest_payload.get("last_used_at") or "").strip() or None,
                    "revoked_at": created_at,
                    "note": note or "revoked",
                    "created_by": actor.user_id,
                    "created_at": created_at,
                },
                bstatus=TOKEN_STATUS_REVOKED,
            )
            self.backend.create_lineage(
                session,
                parent=token,
                child=revision,
                relationship_type="revision",
            )
            return self._token_record(session, token)

    def validate_token(self, plaintext_token: str) -> TokenValidationResult:
        token_value = str(plaintext_token or "").strip()
        if not token_value.startswith(USER_TOKEN_PREFIX):
            raise AuthError("Invalid Ursa token prefix")
        token_hash = self.hash_token(token_value)
        with self.backend.session_scope(commit=False) as session:
            revision = self.backend.find_instance_by_external_id(
                session,
                template_code=USER_TOKEN_REVISION_TEMPLATE,
                key="token_hash",
                value=token_hash,
            )
            if revision is None:
                raise AuthError("Token not found")
            parents = self.backend.list_parents(
                session,
                child=revision,
                relationship_type="revision",
            )
            if not parents:
                raise AuthError("Token parent not found")
            token = parents[0]
            record = self._token_record(session, token)
        if not hmac.compare_digest(self.hash_token(token_value), token_hash):
            raise AuthError("Token hash mismatch")
        if record.status == TOKEN_STATUS_REVOKED:
            raise AuthError("Token is revoked")
        expires_at = str(record.expires_at or "").strip()
        if expires_at:
            expires_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if expires_dt <= datetime.now(UTC):
                raise AuthError("Token is expired")
        actor = self.identity_client.resolve_user(record.owner_user_id)
        actor = ActorContext(
            user_id=actor.user_id,
            atlas_tenant_id=actor.atlas_tenant_id,
            roles=actor.roles,
            email=actor.email,
            display_name=actor.display_name,
            organization=actor.organization,
            site=actor.site,
            auth_source="ursa_token",
            token_euid=record.token_euid,
            token_scope=record.scope,
            client_registration_euid=record.client_registration_euid,
        )
        return TokenValidationResult(actor=actor, token=record)

    def log_usage(
        self,
        *,
        token_euid: str,
        actor_user_id: str,
        endpoint: str,
        http_method: str,
        response_status: int,
        ip_address: str | None,
        user_agent: str | None,
        request_metadata: dict[str, Any] | None,
    ) -> None:
        with self.backend.session_scope(commit=True) as session:
            token = self.backend.find_instance_by_euid(
                session,
                template_code=USER_TOKEN_TEMPLATE,
                value=token_euid,
                for_update=True,
            )
            if token is None:
                return
            latest = self._latest_revision(session, token)
            if latest is not None:
                latest_payload = from_json_addl(latest)
                revision_no = int(latest_payload.get("revision_no") or 0) + 1
                created_at = _iso_now()
                revision = self.backend.create_instance(
                    session,
                    USER_TOKEN_REVISION_TEMPLATE,
                    f"revision:{token.euid}:{revision_no}",
                    json_addl={
                        "token_euid": str(token.euid),
                        "token_hash": str(latest_payload.get("token_hash") or ""),
                        "revision_no": revision_no,
                        "status": str(latest_payload.get("status") or TOKEN_STATUS_ACTIVE),
                        "expires_at": str(latest_payload.get("expires_at") or ""),
                        "last_used_at": created_at,
                        "revoked_at": str(latest_payload.get("revoked_at") or "").strip() or None,
                        "note": "usage_logged",
                        "created_by": actor_user_id,
                        "created_at": created_at,
                    },
                    bstatus=str(latest_payload.get("status") or TOKEN_STATUS_ACTIVE),
                )
                self.backend.create_lineage(
                    session,
                    parent=token,
                    child=revision,
                    relationship_type="revision",
                )
            usage = self.backend.create_instance(
                session,
                USER_TOKEN_USAGE_TEMPLATE,
                f"usage:{token_euid}:{_iso_now()}",
                json_addl={
                    "token_euid": token_euid,
                    "actor_user_id": actor_user_id,
                    "endpoint": endpoint,
                    "http_method": http_method,
                    "response_status": int(response_status),
                    "ip_address": ip_address,
                    "user_agent": user_agent,
                    "request_metadata": dict(request_metadata or {}),
                    "created_at": utc_now_iso(),
                },
                bstatus="LOGGED",
            )
            self.backend.create_lineage(
                session,
                parent=token,
                child=usage,
                relationship_type="usage",
            )

    def list_usage(self, *, actor: ActorContext, token_euid: str) -> list[UserTokenUsageRecord]:
        with self.backend.session_scope(commit=False) as session:
            token = self.backend.find_instance_by_euid(
                session,
                template_code=USER_TOKEN_TEMPLATE,
                value=token_euid,
            )
            if token is None:
                raise KeyError(f"token not found: {token_euid}")
            record = self._token_record(session, token)
            if record.owner_user_id != actor.user_id and not actor.is_admin:
                raise AuthError("Cannot inspect another user's token usage")
            usages = self.backend.list_children(
                session,
                parent=token,
                relationship_type="usage",
            )
            results: list[UserTokenUsageRecord] = []
            for usage in usages:
                payload = from_json_addl(usage)
                results.append(
                    UserTokenUsageRecord(
                        usage_euid=str(usage.euid),
                        token_euid=token_euid,
                        actor_user_id=str(payload.get("actor_user_id") or ""),
                        endpoint=str(payload.get("endpoint") or ""),
                        http_method=str(payload.get("http_method") or ""),
                        response_status=int(payload.get("response_status") or 0),
                        ip_address=str(payload.get("ip_address") or "").strip() or None,
                        user_agent=str(payload.get("user_agent") or "").strip() or None,
                        request_metadata=dict(payload.get("request_metadata") or {}),
                        created_at=str(payload.get("created_at") or utc_now_iso()),
                    )
                )
            results.sort(key=lambda row: row.created_at, reverse=True)
            return results
