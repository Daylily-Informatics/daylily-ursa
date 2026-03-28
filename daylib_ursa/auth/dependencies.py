from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any
from urllib.parse import quote

import boto3
from botocore.exceptions import ClientError
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from daylily_cognito.jwks import JWKSCache
from daylily_cognito.tokens import verify_jwt_claims

from daylib_ursa.auth.rbac import Permission, Role, can_write, has_permission, has_role, is_read_only

security = HTTPBearer(auto_error=False)


class AuthError(RuntimeError):
    """Raised when Ursa auth operations fail."""


class WebAuthRedirect(RuntimeError):
    """Raised by web dependencies to redirect users to the login page."""

    def __init__(self, redirect_url: str) -> None:
        super().__init__(redirect_url)
        self.redirect_url = redirect_url


@dataclass
class CurrentUser:
    """Authenticated user context."""

    sub: str
    email: str
    name: str | None
    tenant_id: uuid.UUID
    roles: list[str]
    auth_source: str = "cognito"
    token_euid: str | None = None
    token_scope: str | None = None
    client_registration_euid: str | None = None
    organization: str | None = None
    site: str | None = None

    @property
    def id(self) -> str:
        return self.sub

    @property
    def user_id(self) -> str:
        return self.sub

    @property
    def sub_uuid(self) -> uuid.UUID | None:
        try:
            return uuid.UUID(self.sub)
        except (TypeError, ValueError):
            return None

    @property
    def display_name(self) -> str | None:
        return self.name

    def has_role(self, role: Role) -> bool:
        return has_role(self.roles, role)

    def has_permission(self, permission: Permission) -> bool:
        return has_permission(self.roles, permission)

    @property
    def is_internal(self) -> bool:
        return self.has_role(Role.INTERNAL_USER) or self.has_role(Role.ADMIN)

    @property
    def is_admin(self) -> bool:
        return self.has_role(Role.ADMIN)

    @property
    def is_org_admin(self) -> bool:
        return self.has_role(Role.EXTERNAL_USER_ADMIN) or self.has_role(Role.ADMIN)

    @property
    def can_write(self) -> bool:
        return can_write(self.roles)


@dataclass(frozen=True)
class AtlasUserDirectoryEntry:
    user_id: str
    tenant_id: uuid.UUID
    organization_id: str
    organization_name: str | None
    site_id: str | None
    site_name: str | None
    roles: tuple[str, ...]
    email: str | None
    display_name: str | None
    is_active: bool

def _normalize_roles(raw_roles: Any) -> list[str]:
    if isinstance(raw_roles, str):
        values = [item for item in raw_roles.split(",") if item.strip()]
    elif isinstance(raw_roles, (list, tuple, set)):
        values = [str(item) for item in raw_roles]
    else:
        values = []

    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        candidate = str(item or "").strip()
        if not candidate:
            continue
        try:
            canonical = Role(candidate.upper()).value
        except ValueError:
            canonical = candidate.upper()
        if canonical in seen:
            continue
        seen.add(canonical)
        normalized.append(canonical)
    if not normalized:
        normalized.append(Role.READ_ONLY.value)
    return normalized


def _parse_uuid(value: Any, *, label: str) -> uuid.UUID:
    raw = str(value or "").strip()
    if not raw:
        raise AuthError(f"{label} is required")
    try:
        return uuid.UUID(raw)
    except ValueError as exc:
        raise AuthError(f"{label} must be a UUID") from exc


def _claims_to_current_user(claims: dict[str, Any]) -> CurrentUser:
    sub = str(claims.get("sub") or claims.get("user_sub") or "").strip()
    if not sub:
        raise AuthError("Authentication token missing subject")

    email = str(claims.get("email") or "").strip()
    tenant_value = claims.get("tenant_id") or claims.get("custom:tenant_id")
    name = str(claims.get("name") or claims.get("display_name") or "").strip() or None
    raw_roles = claims.get("roles") or claims.get("custom:roles") or claims.get("cognito:groups")
    return CurrentUser(
        sub=sub,
        email=email,
        name=name,
        tenant_id=_parse_uuid(tenant_value, label="tenant_id"),
        roles=_normalize_roles(raw_roles),
    )


def _get_current_user_from_session(request: Request) -> CurrentUser | None:
    session = getattr(request, "session", None)
    if not session:
        return None
    if "user_sub" not in session:
        return None
    try:
        return CurrentUser(
            sub=str(session.get("user_sub") or "").strip(),
            email=str(session.get("email") or "").strip(),
            name=str(session.get("name") or "").strip() or None,
            tenant_id=_parse_uuid(session.get("tenant_id"), label="tenant_id"),
            roles=_normalize_roles(session.get("roles") or []),
            auth_source=str(session.get("auth_source") or "cognito").strip() or "cognito",
            token_euid=str(session.get("token_euid") or "").strip() or None,
            token_scope=str(session.get("token_scope") or "").strip() or None,
            client_registration_euid=str(session.get("client_registration_euid") or "").strip() or None,
            organization=str(session.get("organization") or "").strip() or None,
            site=str(session.get("site") or "").strip() or None,
        )
    except AuthError:
        session.clear()
        return None


def persist_session_user(request: Request, current_user: CurrentUser) -> None:
    if not hasattr(request, "session"):
        raise AuthError("Session middleware is not configured")
    request.session["user_sub"] = current_user.sub
    request.session["email"] = current_user.email
    request.session["name"] = current_user.name or ""
    request.session["tenant_id"] = str(current_user.tenant_id)
    request.session["roles"] = list(current_user.roles)
    request.session["auth_source"] = current_user.auth_source
    request.session["token_euid"] = current_user.token_euid or ""
    request.session["token_scope"] = current_user.token_scope or ""
    request.session["client_registration_euid"] = current_user.client_registration_euid or ""
    request.session["organization"] = current_user.organization or ""
    request.session["site"] = current_user.site or ""


def clear_session_user(request: Request) -> None:
    if hasattr(request, "session"):
        request.session.clear()


class CognitoAuthProvider:
    """Local Cognito/JWKS-backed bearer token resolver."""

    def __init__(
        self,
        *,
        user_pool_id: str,
        app_client_id: str,
        region: str,
    ) -> None:
        self.user_pool_id = str(user_pool_id or "").strip()
        self.app_client_id = str(app_client_id or "").strip()
        self.region = str(region or "").strip()
        self._jwks_cache = (
            JWKSCache(self.region, self.user_pool_id)
            if self.user_pool_id and self.region
            else None
        )

    @property
    def configured(self) -> bool:
        return bool(self.user_pool_id and self.app_client_id and self.region)

    def resolve_access_token(self, access_token: str) -> CurrentUser:
        token = str(access_token or "").strip()
        if not token:
            raise AuthError("Bearer token is required")
        if not self.configured:
            raise AuthError("Cognito authentication is not configured")
        try:
            claims = verify_jwt_claims(
                token,
                expected_client_id=self.app_client_id,
                region=self.region,
                user_pool_id=self.user_pool_id,
                cache=self._jwks_cache,
            )
        except HTTPException as exc:
            raise AuthError(str(exc.detail)) from exc
        except Exception as exc:  # pragma: no cover - best effort bridge to AuthError
            raise AuthError(f"Authentication token verification failed: {exc}") from exc
        return _claims_to_current_user(claims)


class CognitoUserDirectoryService:
    """Minimal Cognito-backed user directory for admin lookups."""

    def __init__(
        self,
        *,
        user_pool_id: str,
        region: str,
        profile: str | None = None,
    ) -> None:
        self.user_pool_id = str(user_pool_id or "").strip()
        self.region = str(region or "").strip()
        self._profile = str(profile or "").strip() or None
        self._client = None

    @property
    def configured(self) -> bool:
        return bool(self.user_pool_id and self.region)

    def _get_client(self):
        if not self.configured:
            raise AuthError("Cognito user directory is not configured")
        if self._client is None:
            session_kwargs: dict[str, Any] = {"region_name": self.region}
            if self._profile:
                session_kwargs["profile_name"] = self._profile
            self._client = boto3.Session(**session_kwargs).client("cognito-idp")
        return self._client

    @staticmethod
    def _attrs_to_dict(item: dict[str, Any]) -> dict[str, str]:
        attributes = item.get("Attributes") or []
        mapped: dict[str, str] = {}
        for attr in attributes:
            if not isinstance(attr, dict):
                continue
            name = str(attr.get("Name") or "").strip()
            value = str(attr.get("Value") or "").strip()
            if name:
                mapped[name] = value
        return mapped

    def _entry_from_user(self, item: dict[str, Any]) -> AtlasUserDirectoryEntry:
        attrs = self._attrs_to_dict(item)
        user_id = (
            str(attrs.get("sub") or "").strip()
            or str(item.get("Username") or "").strip()
        )
        tenant_id = _parse_uuid(
            attrs.get("custom:tenant_id") or attrs.get("tenant_id"),
            label="tenant_id",
        )
        roles = tuple(
            _normalize_roles(
                attrs.get("custom:roles") or attrs.get("roles") or attrs.get("custom:role") or ""
            )
        )
        display_name = (
            str(attrs.get("name") or "").strip()
            or str(attrs.get("preferred_username") or "").strip()
            or None
        )
        enabled = bool(item.get("Enabled", True))
        user_status = str(item.get("UserStatus") or "").strip().upper()
        is_active = enabled and user_status != "ARCHIVED"
        return AtlasUserDirectoryEntry(
            user_id=user_id,
            tenant_id=tenant_id,
            organization_id="",
            organization_name=None,
            site_id=None,
            site_name=None,
            roles=roles,
            email=str(attrs.get("email") or "").strip() or None,
            display_name=display_name,
            is_active=is_active,
        )

    def list_users(
        self,
        *,
        tenant_id: uuid.UUID | str | None = None,
        search: str | None = None,
        active_only: bool = True,
        limit: int = 50,
        skip: int = 0,
    ) -> list[AtlasUserDirectoryEntry]:
        if not self.configured:
            raise AuthError("Cognito user directory is not configured")
        wanted_tenant = _parse_uuid(tenant_id, label="tenant_id") if tenant_id else None
        wanted_search = str(search or "").strip().lower()
        remaining_skip = max(0, int(skip or 0))
        remaining_take = max(1, min(int(limit or 50), 200))
        results: list[AtlasUserDirectoryEntry] = []
        pagination_token: str | None = None

        while remaining_take > 0:
            kwargs: dict[str, Any] = {
                "UserPoolId": self.user_pool_id,
                "Limit": min(60, remaining_take + remaining_skip + 20),
            }
            if pagination_token:
                kwargs["PaginationToken"] = pagination_token
            try:
                response = self._get_client().list_users(**kwargs)
            except ClientError as exc:
                raise AuthError(f"Cognito user search failed: {exc}") from exc
            users = list(response.get("Users") or [])
            if not users:
                break
            for item in users:
                entry = self._entry_from_user(item)
                if active_only and not entry.is_active:
                    continue
                if wanted_tenant and entry.tenant_id != wanted_tenant:
                    continue
                searchable = " ".join(
                    part
                    for part in [entry.user_id, entry.email or "", entry.display_name or ""]
                    if part
                ).lower()
                if wanted_search and wanted_search not in searchable:
                    continue
                if remaining_skip:
                    remaining_skip -= 1
                    continue
                results.append(entry)
                remaining_take -= 1
                if remaining_take == 0:
                    break
            pagination_token = response.get("PaginationToken")
            if not pagination_token:
                break
        return results

    def get_user(self, user_id: str) -> CurrentUser | None:
        target = str(user_id or "").strip()
        if not target or not self.configured:
            return None
        for search in (target,):
            try:
                users = self.list_users(active_only=False, limit=20, search=search)
            except AuthError:
                return None
            for entry in users:
                if entry.user_id != target and (entry.email or "") != target:
                    continue
                return CurrentUser(
                    sub=entry.user_id,
                    email=entry.email or "",
                    name=entry.display_name,
                    tenant_id=entry.tenant_id,
                    roles=list(entry.roles),
                )
        return None


def _get_auth_provider(request: Request) -> CognitoAuthProvider:
    provider = getattr(request.app.state, "auth_provider", None)
    if provider is None:
        raise AuthError("Authentication provider is not configured")
    return provider


def get_current_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)] = None,
) -> CurrentUser:
    user = _get_current_user_from_session(request)
    if user:
        return user

    bearer = str(getattr(credentials, "credentials", "") or "").strip()
    if not bearer:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer authorization or authenticated session is required",
        )
    if bearer.startswith("urs_"):
        from daylib_ursa.auth.tokens import USER_TOKEN_PREFIX

        if not bearer.startswith(USER_TOKEN_PREFIX):
            raise HTTPException(status_code=401, detail="Invalid Ursa token prefix")
        service = getattr(request.app.state, "token_service", None)
        if service is None:
            raise HTTPException(status_code=503, detail="User token service is not configured")
        try:
            validated = service.validate_token(bearer)
        except AuthError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        request.state.user_token_usage = {
            "token_euid": validated.token.token_euid,
            "actor_user_id": validated.actor.sub,
        }
        validated.actor.auth_source = "ursa_token"
        validated.actor.token_euid = validated.token.token_euid
        validated.actor.token_scope = validated.token.scope
        validated.actor.client_registration_euid = validated.token.client_registration_euid
        return validated.actor

    try:
        return _get_auth_provider(request).resolve_access_token(bearer)
    except AuthError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc


async def get_current_user_web(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)] = None,
) -> CurrentUser:
    try:
        return get_current_user(request, credentials)
    except HTTPException as exc:
        if exc.status_code == status.HTTP_401_UNAUTHORIZED:
            next_path = quote(str(request.url.path or "/"), safe="/?=&")
            raise WebAuthRedirect(f"/login?next={next_path}") from exc
        raise


async def get_current_tenant(
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
) -> uuid.UUID:
    return current_user.tenant_id


def require_role(*required_roles: Role) -> Callable:
    def _require_role(
        current_user: Annotated[CurrentUser, Depends(get_current_user)],
    ) -> CurrentUser:
        if any(current_user.has_role(role) for role in required_roles):
            return current_user
        required = ", ".join(role.value for role in required_roles)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Requires one of roles: {required}",
        )

    return _require_role


def require_permission(permission: Permission) -> Callable:
    def _require_permission(
        current_user: Annotated[CurrentUser, Depends(get_current_user)],
    ) -> CurrentUser:
        if current_user.has_permission(permission):
            return current_user
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Requires permission: {permission.value}",
        )

    return _require_permission


RequireAuth = Annotated[CurrentUser, Depends(get_current_user)]
RequireAuthWeb = Annotated[CurrentUser, Depends(get_current_user_web)]
RequireInternal = Annotated[CurrentUser, Depends(require_role(Role.INTERNAL_USER, Role.ADMIN))]
RequireAdmin = Annotated[CurrentUser, Depends(require_role(Role.ADMIN))]
