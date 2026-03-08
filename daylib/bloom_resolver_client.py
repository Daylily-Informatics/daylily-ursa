"""HTTP client for Bloom sequenced-assignment resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from daylib.analysis_store import RunResolution


class BloomResolverError(RuntimeError):
    """Raised when Bloom resolution fails."""


@dataclass
class BloomResolverClient:
    base_url: str
    token: str | None = None
    timeout_seconds: float = 10.0
    verify_ssl: bool = True
    client: httpx.Client | None = None

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def resolve_run_assignment(
        self,
        run_euid: str,
        flowcell_id: str,
        lane: str,
        library_barcode: str,
    ) -> RunResolution:
        url = (
            f"{self.base_url.rstrip('/')}"
            f"/api/v1/external/atlas/beta/runs/{run_euid}/resolve"
        )
        client = self.client or httpx.Client(
            timeout=self.timeout_seconds,
            verify=self.verify_ssl,
        )
        close_client = self.client is None
        try:
            response = client.get(
                url,
                params={
                    "flowcell_id": flowcell_id,
                    "lane": lane,
                    "library_barcode": library_barcode,
                },
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
            "flowcell_id",
            "lane",
            "library_barcode",
            "sequenced_library_assignment_euid",
            "atlas_tenant_id",
            "atlas_trf_euid",
            "atlas_test_euid",
            "atlas_test_process_item_euid",
        )
        missing = [key for key in required if not str(body.get(key) or "").strip()]
        if missing:
            raise BloomResolverError(
                f"Bloom resolver response missing required fields: {', '.join(missing)}"
            )

        return RunResolution(
            run_euid=str(body["run_euid"]),
            flowcell_id=str(body["flowcell_id"]),
            lane=str(body["lane"]),
            library_barcode=str(body["library_barcode"]),
            sequenced_library_assignment_euid=str(body["sequenced_library_assignment_euid"]),
            atlas_tenant_id=str(body["atlas_tenant_id"]),
            atlas_trf_euid=str(body["atlas_trf_euid"]),
            atlas_test_euid=str(body["atlas_test_euid"]),
            atlas_test_process_item_euid=str(body["atlas_test_process_item_euid"]),
        )
