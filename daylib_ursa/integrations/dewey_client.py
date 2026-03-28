"""Ursa -> Dewey artifact integration client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


class DeweyClientError(RuntimeError):
    """Raised on Dewey integration failures."""


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

    def _headers(self, *, idempotency_key: str | None = None) -> dict[str, str]:
        token = str(self.token or "").strip()
        if not token:
            raise DeweyClientError("Dewey API bearer token is required")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        clean_key = str(idempotency_key or "").strip()
        if clean_key:
            headers["Idempotency-Key"] = clean_key
        return headers

    def _http_client(self) -> tuple[httpx.Client, bool]:
        if self.client is not None:
            return self.client, False
        return (
            httpx.Client(timeout=self.timeout_seconds, verify=self.verify_ssl),
            True,
        )

    def get_artifact(self, artifact_euid: str) -> dict[str, Any]:
        target = str(artifact_euid or "").strip()
        if not target:
            raise DeweyClientError("artifact_euid is required")
        url = f"{_require_https_url(self.base_url, field_name='Dewey base URL')}/api/v1/artifacts/{target}"
        client, close_client = self._http_client()
        try:
            response = client.get(url, headers=self._headers())
        except httpx.HTTPError as exc:
            raise DeweyClientError(f"Dewey artifact lookup failed: {exc}") from exc
        finally:
            if close_client:
                client.close()
        if response.status_code >= 400:
            raise DeweyClientError(
                f"Dewey artifact lookup returned {response.status_code}: {response.text}"
            )
        payload = response.json()
        if not isinstance(payload, dict):
            raise DeweyClientError("Dewey artifact response was not a JSON object")
        return payload

    def resolve_artifact(self, artifact_euid: str) -> dict[str, Any]:
        url = f"{_require_https_url(self.base_url, field_name='Dewey base URL')}/api/v1/resolve/artifact"
        payload = {"artifact_euid": str(artifact_euid or "").strip()}
        if not payload["artifact_euid"]:
            raise DeweyClientError("artifact_euid is required")
        client, close_client = self._http_client()
        try:
            response = client.post(url, json=payload, headers=self._headers())
        except httpx.HTTPError as exc:
            raise DeweyClientError(f"Dewey resolve failed: {exc}") from exc
        finally:
            if close_client:
                client.close()
        if response.status_code >= 400:
            raise DeweyClientError(
                f"Dewey resolve returned {response.status_code}: {response.text}"
            )
        body = response.json()
        if not isinstance(body, dict):
            raise DeweyClientError("Dewey resolve response was not a JSON object")
        if not str(body.get("storage_uri") or "").strip():
            raise DeweyClientError("Dewey resolve response missing storage_uri")
        if not str(body.get("artifact_type") or "").strip():
            raise DeweyClientError("Dewey resolve response missing artifact_type")
        return body

    def resolve_artifact_set(self, artifact_set_euid: str) -> dict[str, Any]:
        url = (
            f"{_require_https_url(self.base_url, field_name='Dewey base URL')}"
            "/api/v1/resolve/artifact-set"
        )
        payload = {"artifact_set_euid": str(artifact_set_euid or "").strip()}
        if not payload["artifact_set_euid"]:
            raise DeweyClientError("artifact_set_euid is required")
        client, close_client = self._http_client()
        try:
            response = client.post(url, json=payload, headers=self._headers())
        except httpx.HTTPError as exc:
            raise DeweyClientError(f"Dewey artifact-set resolve failed: {exc}") from exc
        finally:
            if close_client:
                client.close()
        if response.status_code >= 400:
            raise DeweyClientError(
                f"Dewey artifact-set resolve returned {response.status_code}: {response.text}"
            )
        body = response.json()
        if not isinstance(body, dict):
            raise DeweyClientError("Dewey artifact-set resolve response was not a JSON object")
        if not str(body.get("artifact_set_euid") or "").strip():
            raise DeweyClientError("Dewey artifact-set resolve response missing artifact_set_euid")
        return body

    def import_artifact(
        self,
        *,
        artifact_type: str,
        storage_uri: str,
        metadata: dict[str, Any] | None,
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
        client, close_client = self._http_client()
        try:
            response = client.post(
                url,
                json=payload,
                headers=self._headers(idempotency_key=idempotency_key),
            )
        except httpx.HTTPError as exc:
            raise DeweyClientError(f"Dewey artifact import failed: {exc}") from exc
        finally:
            if close_client:
                client.close()
        if response.status_code >= 400:
            raise DeweyClientError(
                f"Dewey artifact import returned {response.status_code}: {response.text}"
            )
        body = response.json()
        if not isinstance(body, dict):
            raise DeweyClientError("Dewey artifact import response was not a JSON object")
        artifact_euid = str(body.get("artifact_euid") or "").strip()
        if not artifact_euid:
            raise DeweyClientError("Dewey artifact import response missing artifact_euid")
        return artifact_euid

    def register_artifact(
        self,
        *,
        artifact_type: str,
        storage_uri: str,
        metadata: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> str:
        return self.import_artifact(
            artifact_type=artifact_type,
            storage_uri=storage_uri,
            metadata=metadata,
            idempotency_key=idempotency_key,
        )
