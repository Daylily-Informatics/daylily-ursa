"""TapDB admin mount integration for the Ursa FastAPI app."""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any, Callable

from fastapi import FastAPI
from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse, RedirectResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from daylib_ursa.config import Settings
from daylib_ursa.portal_auth import PORTAL_SESSION_COOKIE_NAME, decode_portal_session

LOGGER = logging.getLogger("daylily.tapdb_mount")


def _force_mounted_auth_bypass_env() -> None:
    """Force TapDB into mounted mode without TapDB-local auth."""
    os.environ["TAPDB_ADMIN_DISABLE_AUTH"] = "true"
    os.environ["TAPDB_ADMIN_DISABLED_USER_ROLE"] = "admin"
    os.environ["TAPDB_ADMIN_DISABLED_USER_EMAIL"] = "ursa-mounted-tapdb-admin@localhost"
    os.environ["TAPDB_ADMIN_SHARED_AUTH"] = "false"


def _load_tapdb_admin_app() -> ASGIApp:
    """Load the TapDB admin FastAPI app lazily."""
    module = importlib.import_module("admin.main")
    tapdb_app = getattr(module, "app", None)
    if tapdb_app is None:
        raise RuntimeError("admin.main does not export an 'app'")
    return tapdb_app


class UrsaTapdbAdminGate:
    """ASGI guard that requires an authenticated Ursa admin session."""

    def __init__(self, app: ASGIApp, *, session_secret_key: str) -> None:
        self._app = app
        self._session_secret_key = session_secret_key

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self._app(scope, receive, send)
            return

        identity = self._resolve_identity(scope)
        if not bool(identity.get("logged_in")):
            await self._deny_not_authenticated(scope, receive, send)
            return
        if not bool(identity.get("is_admin")):
            await self._deny_not_admin(scope, receive, send)
            return

        await self._app(scope, receive, send)

    def _resolve_identity(self, scope: Scope) -> dict[str, Any]:
        connection = HTTPConnection(scope)
        raw = connection.cookies.get(PORTAL_SESSION_COOKIE_NAME)
        if not raw:
            return {}
        session = decode_portal_session(self._session_secret_key, raw) or {}
        return session if isinstance(session, dict) else {}

    async def _deny_not_authenticated(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 4401})
            return
        response = RedirectResponse(url="/portal/login", status_code=307)
        await response(scope, receive, send)

    async def _deny_not_admin(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 4403})
            return
        response = JSONResponse({"detail": "Admin access required"}, status_code=403)
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
    loader: Callable[[], ASGIApp] | None = None,
) -> None:
    """Mount TapDB admin under Ursa with Ursa-admin-only gating."""
    if not settings.ursa_tapdb_mount_enabled:
        LOGGER.info("TapDB mount disabled by URSA_TAPDB_MOUNT_ENABLED")
        return

    mount_path = _normalize_mount_path(settings.ursa_tapdb_mount_path)
    if any(getattr(route, "path", None) == mount_path for route in app.routes):
        raise RuntimeError(f"Cannot mount TapDB: path already in use: {mount_path}")

    _force_mounted_auth_bypass_env()
    resolved_loader = loader or _load_tapdb_admin_app
    try:
        tapdb_app = resolved_loader()
    except Exception as exc:
        raise RuntimeError(
            "Failed to import TapDB admin app for mounted mode. "
            "Install daylily-tapdb admin extras or disable URSA_TAPDB_MOUNT_ENABLED."
        ) from exc

    guarded_app = UrsaTapdbAdminGate(tapdb_app, session_secret_key=settings.session_secret_key)
    app.mount(mount_path, guarded_app, name="ursa-tapdb-admin")
    LOGGER.info("Mounted TapDB admin at %s", mount_path)
