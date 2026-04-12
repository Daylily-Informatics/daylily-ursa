"""TapDB admin mount integration for the Ursa FastAPI app."""

from __future__ import annotations

import hmac
import importlib
import logging
import os
from pathlib import Path
from typing import Callable

from fastapi import FastAPI
from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from daylib_ursa.config import Settings
from daylib_ursa.integrations.tapdb_runtime import _resolve_tapdb_config_path

LOGGER = logging.getLogger("daylily.tapdb_mount")
_URSA_TAPDB_SCOPE_USER_KEY = "ursa_tapdb_user"


def _embedded_tapdb_admin_user() -> dict[str, object]:
    return {
        "uid": 0,
        "username": "ursa-mounted-tapdb-admin@localhost",
        "email": "ursa-mounted-tapdb-admin@localhost",
        "display_name": "Ursa Mounted TapDB Admin",
        "role": "admin",
        "is_active": True,
        "require_password_change": False,
    }


def _configure_embedded_tapdb_auth(admin_main_module, admin_auth_module) -> None:
    if getattr(admin_main_module, "_ursa_embedded_auth_configured", False):
        return

    async def _get_current_user(request):
        user = request.scope.get(_URSA_TAPDB_SCOPE_USER_KEY)
        if isinstance(user, dict):
            return user
        return None

    admin_auth_module.get_current_user = _get_current_user
    admin_main_module.get_current_user = _get_current_user
    setattr(admin_main_module, "_ursa_embedded_auth_configured", True)


def _load_tapdb_admin_app(
    *,
    tapdb_env: str,
    config_path: str,
    client_id: str,
    database_name: str,
) -> ASGIApp:
    """Load the TapDB admin FastAPI app lazily."""
    resolved_env = str(tapdb_env or "").strip().lower()
    if not resolved_env:
        raise RuntimeError("TapDB admin mount requires an explicit tapdb_env.")
    if not str(config_path or "").strip():
        raise RuntimeError("TapDB admin mount requires an explicit tapdb_config_path.")
    resolved_config_path = Path(config_path).expanduser().resolve()

    from daylily_tapdb.cli import context as tapdb_context
    from daylily_tapdb.cli import db_config as tapdb_db_config

    original_active_env_name = tapdb_context.active_env_name
    original_get_config_path = tapdb_db_config.get_config_path
    original_get_db_config_for_env = tapdb_db_config.get_db_config_for_env
    original_get_admin_settings_for_env = tapdb_db_config.get_admin_settings_for_env

    def _active_env_name(default: str = "dev") -> str:
        _ = default
        return resolved_env

    def _get_config_path(**_kwargs):
        return resolved_config_path

    def _get_db_config_for_env(env_name: str, **_kwargs):
        effective_env = str(env_name or resolved_env).strip().lower() or resolved_env
        return original_get_db_config_for_env(
            effective_env,
            config_path=resolved_config_path,
            client_id=client_id,
            database_name=database_name,
        )

    def _get_admin_settings_for_env(env_name: str, **_kwargs):
        effective_env = str(env_name or resolved_env).strip().lower() or resolved_env
        return original_get_admin_settings_for_env(
            effective_env,
            config_path=resolved_config_path,
            client_id=client_id,
            database_name=database_name,
        )

    tapdb_context.active_env_name = _active_env_name
    tapdb_db_config.get_config_path = _get_config_path
    tapdb_db_config.get_db_config_for_env = _get_db_config_for_env
    tapdb_db_config.get_admin_settings_for_env = _get_admin_settings_for_env
    try:
        cognito_module = importlib.import_module("admin.cognito")
        db_pool_module = importlib.import_module("admin.db_pool")
        module = importlib.import_module("admin.main")
        auth_module = importlib.import_module("admin.auth")
    finally:
        tapdb_context.active_env_name = original_active_env_name
        tapdb_db_config.get_config_path = original_get_config_path
        tapdb_db_config.get_db_config_for_env = original_get_db_config_for_env
        tapdb_db_config.get_admin_settings_for_env = original_get_admin_settings_for_env

    for loaded_module in (cognito_module, db_pool_module, module):
        if hasattr(loaded_module, "active_env_name"):
            loaded_module.active_env_name = _active_env_name
        if hasattr(loaded_module, "get_config_path"):
            loaded_module.get_config_path = _get_config_path
        if hasattr(loaded_module, "get_db_config_for_env"):
            loaded_module.get_db_config_for_env = _get_db_config_for_env
        if hasattr(loaded_module, "get_admin_settings_for_env"):
            loaded_module.get_admin_settings_for_env = _get_admin_settings_for_env
    if hasattr(module, "APP_ENV"):
        module.APP_ENV = resolved_env
    if hasattr(module, "IS_PROD"):
        module.IS_PROD = resolved_env == "prod"

    _configure_embedded_tapdb_auth(module, auth_module)
    tapdb_app = getattr(module, "app", None)
    if tapdb_app is None:
        raise RuntimeError("admin.main does not export an 'app'")
    return tapdb_app


class UrsaTapdbAdminGate:
    """ASGI guard that requires Ursa internal API key."""

    def __init__(self, app: ASGIApp, *, internal_api_key: str) -> None:
        self._app = app
        self._internal_api_key = str(internal_api_key or "")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self._app(scope, receive, send)
            return

        if not self._is_authenticated(scope):
            await self._deny_not_authenticated(scope, receive, send)
            return

        forward_scope = dict(scope)
        forward_scope[_URSA_TAPDB_SCOPE_USER_KEY] = _embedded_tapdb_admin_user()
        await self._app(forward_scope, receive, send)

    def _is_authenticated(self, scope: Scope) -> bool:
        connection = HTTPConnection(scope)
        provided = str(connection.headers.get("x-api-key") or "")
        expected = str(self._internal_api_key or "")
        if not provided or not expected:
            return False
        return hmac.compare_digest(provided, expected)

    async def _deny_not_authenticated(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 4401})
            return
        response = JSONResponse({"detail": "Invalid or missing API key"}, status_code=401)
        await response(scope, receive, send)


def _normalize_mount_path(raw_path: str) -> str:
    path = str(raw_path or "").strip()
    if not path:
        raise RuntimeError("URSA_TAPDB_MOUNT_PATH must not be empty")
    if not path.startswith("/"):
        raise RuntimeError("URSA_TAPDB_MOUNT_PATH must start with '/'")
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return path


def mount_tapdb_admin(
    app: FastAPI,
    settings: Settings,
    *,
    loader: Callable[..., ASGIApp] | None = None,
) -> None:
    """Mount TapDB admin under Ursa with API-key gating."""
    if not settings.ursa_tapdb_mount_enabled:
        LOGGER.info("TapDB mount disabled by URSA_TAPDB_MOUNT_ENABLED")
        return

    mount_path = _normalize_mount_path(settings.ursa_tapdb_mount_path)
    if any(getattr(route, "path", None) == mount_path for route in app.routes):
        raise RuntimeError(f"Cannot mount TapDB: path already in use: {mount_path}")

    tapdb_config_path = str(
        getattr(settings, "tapdb_config_path", "") or os.environ.get("TAPDB_CONFIG_PATH") or ""
    ).strip()
    if not tapdb_config_path:
        tapdb_config_path = (
            _resolve_tapdb_config_path(
                namespace=settings.tapdb_database_name,
                client_id=settings.tapdb_client_id,
                config_path="",
            )
            or ""
        )

    resolved_loader = loader or _load_tapdb_admin_app
    try:
        tapdb_app = resolved_loader(
            tapdb_env=settings.tapdb_env,
            config_path=tapdb_config_path,
            client_id=settings.tapdb_client_id,
            database_name=settings.tapdb_database_name,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to import TapDB admin app for mounted mode. "
            "Install daylily-tapdb admin extras or disable URSA_TAPDB_MOUNT_ENABLED."
        ) from exc

    guarded_app = UrsaTapdbAdminGate(
        tapdb_app,
        internal_api_key=settings.ursa_internal_api_key,
    )
    app.mount(mount_path, guarded_app, name="ursa-tapdb-admin")
    LOGGER.info("Mounted TapDB admin at %s", mount_path)
