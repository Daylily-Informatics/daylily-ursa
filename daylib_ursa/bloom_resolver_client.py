"""HTTP client for Bloom run-index resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from daylib_ursa.analysis_store import RunResolution


class BloomResolverError(RuntimeError):
    """Raised when Bloom resolution fails."""


@dataclass
class BloomResolverClient:
    base_url: str
    token: str | None = None
    timeout_seconds: float = 10.0
    client: httpx.Client | None = None

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def resolve_run_index(self, run_euid: str, index_string: str) -> RunResolution:
        url = f"{self.base_url.rstrip('/')}/api/v1/external/atlas/beta/runs/{run_euid}/resolve"
        client = self.client or httpx.Client(timeout=self.timeout_seconds)
        close_client = self.client is None
        try:
            response = client.get(
                url,
                params={"index_string": index_string},
                headers=self._headers(),
            )
        except httpx.HTTPError as exc:
            raise BloomResolverError(f"Bloom resolver request failed: {exc}") from exc
        finally:
            if close_client:
                client.close()

        if response.status_code >= 400:
            raise BloomResolverError(
                f"Bloom resolver returned {response.status_code}: {response.text}"
            )

        body: dict[str, Any] = response.json()
        required = (
            "run_euid",
            "index_string",
            "atlas_tenant_id",
            "atlas_order_euid",
            "atlas_test_order_euid",
            "source_euid",
        )
        missing = [key for key in required if not str(body.get(key) or "").strip()]
        if missing:
            raise BloomResolverError(
                f"Bloom resolver response missing required fields: {', '.join(missing)}"
            )

        return RunResolution(
            run_euid=str(body["run_euid"]),
            index_string=str(body["index_string"]),
            atlas_tenant_id=str(body["atlas_tenant_id"]),
            atlas_order_euid=str(body["atlas_order_euid"]),
            atlas_test_order_euid=str(body["atlas_test_order_euid"]),
            source_euid=str(body["source_euid"]),
        )
