"""Authenticated HTTPS client for Ursa <-> Dewey artifact operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


class DeweyClientError(RuntimeError):
    """Raised when Dewey artifact API requests fail."""


def _require_https_url(value: str, *, field_name: str) -> str:
    normalized = str(value or "").strip().rstrip("/")
    if not normalized:
        raise DeweyClientError(f"{field_name} is required")
    if not normalized.startswith("https://"):
        raise DeweyClientError(f"{field_name} must use an absolute https:// URL")
    return normalized


@dataclass
class DeweyClient:
    base_url: str
    token: str
    timeout_seconds: float = 10.0
    verify_ssl: bool = True
    client: httpx.Client | None = None

    def _headers(self) -> dict[str, str]:
        token = str(self.token or "").strip()
        if not token:
            raise DeweyClientError("Dewey API bearer token is required")
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

    def resolve_artifact(self, artifact_euid: str) -> dict[str, Any]:
        url = f"{_require_https_url(self.base_url, field_name='Dewey base URL')}/api/v1/resolve/artifact"
        payload = {"artifact_euid": str(artifact_euid or "").strip()}
        if not payload["artifact_euid"]:
            raise DeweyClientError("artifact_euid is required")
        client = self.client or httpx.Client(timeout=self.timeout_seconds, verify=self.verify_ssl)
        close_client = self.client is None
        try:
            response = client.post(url, json=payload, headers=self._headers())
        except httpx.HTTPError as exc:
            raise DeweyClientError(f"Dewey resolve failed: {exc}") from exc
        finally:
            if close_client:
                client.close()
        if response.status_code >= 400:
            raise DeweyClientError(f"Dewey resolve returned {response.status_code}: {response.text}")
        body: dict[str, Any] = response.json()
        if not str(body.get("storage_uri") or "").strip():
            raise DeweyClientError("Dewey resolve response missing storage_uri")
        if not str(body.get("artifact_type") or "").strip():
            raise DeweyClientError("Dewey resolve response missing artifact_type")
        return body

    def register_artifact(
        self,
        *,
        artifact_type: str,
        storage_uri: str,
        metadata: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> str:
        url = f"{_require_https_url(self.base_url, field_name='Dewey base URL')}/api/v1/artifacts/import"
        payload = {
            "artifact_type": str(artifact_type or "").strip(),
            "storage_uri": str(storage_uri or "").strip(),
            "metadata": dict(metadata or {}),
        }
        if not payload["artifact_type"]:
            raise DeweyClientError("artifact_type is required")
        if not payload["storage_uri"]:
            raise DeweyClientError("storage_uri is required")
        headers = self._headers()
        clean_idempotency = str(idempotency_key or "").strip()
        if clean_idempotency:
            headers["Idempotency-Key"] = clean_idempotency
        client = self.client or httpx.Client(timeout=self.timeout_seconds, verify=self.verify_ssl)
        close_client = self.client is None
        try:
            response = client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise DeweyClientError(f"Dewey register failed: {exc}") from exc
        finally:
            if close_client:
                client.close()
        if response.status_code >= 400:
            raise DeweyClientError(
                f"Dewey register returned {response.status_code}: {response.text}"
            )
        body: dict[str, Any] = response.json()
        artifact_euid = str(body.get("artifact_euid") or "").strip()
        if not artifact_euid:
            raise DeweyClientError("Dewey register response missing artifact_euid")
        return artifact_euid
