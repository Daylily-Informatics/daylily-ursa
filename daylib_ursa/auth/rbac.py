from __future__ import annotations

from enum import StrEnum


class Role(StrEnum):
    READ_ONLY = "READ_ONLY"
    READ_WRITE = "READ_WRITE"
    EXTERNAL_USER = "EXTERNAL_USER"
    EXTERNAL_USER_ADMIN = "EXTERNAL_USER_ADMIN"
    INTERNAL_USER = "INTERNAL_USER"
    ADMIN = "ADMIN"


class Permission(StrEnum):
    WORKSET_CREATE = "workset:create"
    WORKSET_READ = "workset:read"
    WORKSET_UPDATE = "workset:update"
    ANALYSIS_SUBMIT = "analysis:submit"
    ANALYSIS_READ = "analysis:read"
    CLUSTER_READ = "cluster:read"
    CLUSTER_MANAGE = "cluster:manage"
    COST_READ = "cost:read"
    CROSS_TENANT_READ = "cross_tenant:read"
    USER_TOKEN_MANAGE = "user_token:manage"
    USER_DIRECTORY_READ = "user_directory:read"


ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.READ_ONLY: {
        Permission.WORKSET_READ,
        Permission.ANALYSIS_READ,
        Permission.CLUSTER_READ,
        Permission.COST_READ,
    },
    Role.READ_WRITE: {
        Permission.WORKSET_CREATE,
        Permission.WORKSET_READ,
        Permission.WORKSET_UPDATE,
        Permission.ANALYSIS_SUBMIT,
        Permission.ANALYSIS_READ,
        Permission.CLUSTER_READ,
        Permission.COST_READ,
        Permission.USER_TOKEN_MANAGE,
    },
    Role.EXTERNAL_USER: {
        Permission.WORKSET_READ,
        Permission.ANALYSIS_READ,
        Permission.USER_TOKEN_MANAGE,
    },
    Role.EXTERNAL_USER_ADMIN: {
        Permission.WORKSET_CREATE,
        Permission.WORKSET_READ,
        Permission.WORKSET_UPDATE,
        Permission.ANALYSIS_SUBMIT,
        Permission.ANALYSIS_READ,
        Permission.USER_TOKEN_MANAGE,
    },
    Role.INTERNAL_USER: {
        Permission.WORKSET_CREATE,
        Permission.WORKSET_READ,
        Permission.WORKSET_UPDATE,
        Permission.ANALYSIS_SUBMIT,
        Permission.ANALYSIS_READ,
        Permission.CLUSTER_READ,
        Permission.CLUSTER_MANAGE,
        Permission.COST_READ,
        Permission.CROSS_TENANT_READ,
        Permission.USER_TOKEN_MANAGE,
        Permission.USER_DIRECTORY_READ,
    },
    Role.ADMIN: set(Permission),
}


def _normalize_role_value(value: str) -> str:
    return str(value or "").strip().upper()


def has_permission(roles: list[str], permission: Permission) -> bool:
    for role_value in roles:
        normalized = _normalize_role_value(role_value)
        try:
            role = Role(normalized)
        except ValueError:
            continue
        if permission in ROLE_PERMISSIONS.get(role, set()):
            return True
    return False


def has_role(roles: list[str], required_role: Role) -> bool:
    normalized_required = _normalize_role_value(required_role.value)
    return any(_normalize_role_value(role) == normalized_required for role in roles)


def is_internal(roles: list[str]) -> bool:
    return has_role(roles, Role.INTERNAL_USER) or has_role(roles, Role.ADMIN)


def is_admin(roles: list[str]) -> bool:
    return has_role(roles, Role.ADMIN)


def is_read_only(roles: list[str]) -> bool:
    return has_role(roles, Role.READ_ONLY)


def can_write(roles: list[str]) -> bool:
    return not is_read_only(roles)


def can_access_tenant(user_roles: list[str], user_tenant_id: str, target_tenant_id: str) -> bool:
    if has_permission(user_roles, Permission.CROSS_TENANT_READ):
        return True
    return str(user_tenant_id or "").strip() == str(target_tenant_id or "").strip()
