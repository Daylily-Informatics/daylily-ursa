"""Portal authentication/session helpers for Cognito Hosted UI flows."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode

import httpx
from itsdangerous import BadSignature, URLSafeSerializer

PORTAL_SESSION_COOKIE_NAME = "ursa_portal_session"
PORTAL_SESSION_MAX_AGE_SECONDS = 60 * 60 * 8


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_domain(domain: str) -> str:
    normalized = domain.strip()
    if not normalized.startswith(("http://", "https://")):
        normalized = f"https://{normalized}"
    return normalized.rstrip("/")


def _session_serializer(secret_key: str) -> URLSafeSerializer:
    return URLSafeSerializer(secret_key, salt="ursa-portal-session")


def encode_portal_session(secret_key: str, payload: dict[str, Any]) -> str:
    return _session_serializer(secret_key).dumps(payload)


def decode_portal_session(secret_key: str, token: str) -> dict[str, Any] | None:
    try:
        loaded = _session_serializer(secret_key).loads(token)
    except BadSignature:
        return None
    if not isinstance(loaded, dict):
        return None
    return loaded


def decode_id_token_claims(id_token: str) -> dict[str, Any]:
    """Decode JWT claims without signature verification.

    The ID token arrives from Cognito's token endpoint over TLS.
    """
    if not id_token or "." not in id_token:
        return {}
    try:
        parts = id_token.split(".")
        if len(parts) < 2:
            return {}
        payload = parts[1]
        payload += "=" * ((4 - len(payload) % 4) % 4)
        decoded = base64.urlsafe_b64decode(payload.encode("utf-8")).decode("utf-8")
        claims = json.loads(decoded)
        return claims if isinstance(claims, dict) else {}
    except Exception:
        return {}


def _normalize_groups(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


def derive_identity(
    *,
    claims: dict[str, Any],
    userinfo: dict[str, Any],
    default_customer_id: str,
) -> dict[str, Any]:
    email = str(
        userinfo.get("email")
        or claims.get("email")
        or claims.get("preferred_username")
        or ""
    ).strip()
    name = str(
        userinfo.get("name")
        or claims.get("name")
        or claims.get("given_name")
        or ""
    ).strip()
    customer_id = str(
        claims.get("custom:customer_id")
        or claims.get("customer_id")
        or userinfo.get("customer_id")
        or default_customer_id
    ).strip()
    if not customer_id and email and "@" in email:
        customer_id = email.split("@", 1)[0].lower().replace("_", "-")
    customer_name = str(
        claims.get("custom:customer_name")
        or claims.get("customer_name")
        or userinfo.get("customer_name")
        or name
        or customer_id
    ).strip()
    s3_bucket = str(
        claims.get("custom:s3_bucket")
        or claims.get("s3_bucket")
        or userinfo.get("s3_bucket")
        or ""
    ).strip()
    billing_account_id = str(
        claims.get("custom:billing_account_id")
        or claims.get("billing_account_id")
        or userinfo.get("billing_account_id")
        or ""
    ).strip()
    cost_center = str(
        claims.get("custom:cost_center")
        or claims.get("cost_center")
        or userinfo.get("cost_center")
        or ""
    ).strip()

    max_worksets_raw = (
        claims.get("custom:max_concurrent_worksets")
        or claims.get("max_concurrent_worksets")
        or userinfo.get("max_concurrent_worksets")
        or 10
    )
    try:
        max_concurrent_worksets = int(max_worksets_raw)
    except (TypeError, ValueError):
        max_concurrent_worksets = 10

    groups = _normalize_groups(claims.get("cognito:groups"))
    is_admin = any(group.lower() in {"admin", "ursa-admin", "lsmc-admin"} for group in groups)
    custom_is_admin = str(claims.get("custom:is_admin") or "").strip().lower()
    if custom_is_admin in {"1", "true", "yes"}:
        is_admin = True

    return {
        "logged_in": True,
        "issued_at": _utc_now_iso(),
        "user_email": email,
        "user_sub": str(userinfo.get("sub") or claims.get("sub") or "").strip(),
        "user_name": name,
        "groups": groups,
        "is_admin": is_admin,
        "customer_id": customer_id or default_customer_id,
        "customer_name": customer_name or (customer_id or default_customer_id),
        "s3_bucket": s3_bucket,
        "billing_account_id": billing_account_id,
        "cost_center": cost_center,
        "max_concurrent_worksets": max_concurrent_worksets,
    }


async def exchange_code_for_tokens(
    *,
    domain: str,
    code: str,
    client_id: str,
    redirect_uri: str,
    client_secret: str | None = None,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    token_url = f"{_normalize_domain(domain)}/oauth2/token"
    data = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "code": code,
        "redirect_uri": redirect_uri,
    }
    if client_secret:
        data["client_secret"] = client_secret

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(
            token_url,
            data=urlencode(data),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


async def fetch_userinfo(
    *,
    domain: str,
    access_token: str,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    if not access_token:
        return {}
    userinfo_url = f"{_normalize_domain(domain)}/oauth2/userInfo"
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.get(
            userinfo_url,
            headers={"Authorization": f"Bearer {access_token}"},
        )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}
