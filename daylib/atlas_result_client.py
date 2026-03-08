"""HTTP client for Ursa result return into Atlas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


class AtlasResultClientError(RuntimeError):
    """Raised when Atlas result return fails."""


@dataclass(frozen=True)
class AtlasResultArtifact:
    artifact_type: str
    storage_uri: str
    filename: str
    mime_type: str | None = None
    checksum_sha256: str | None = None
    size_bytes: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class AtlasResultClient:
    base_url: str
    api_key: str
    timeout_seconds: float = 10.0
    client: httpx.Client | None = None

    def return_analysis_result(
        self,
        *,
        atlas_tenant_id: str,
        atlas_trf_euid: str,
        atlas_test_euid: str,
        atlas_test_process_item_euid: str,
        analysis_euid: str,
        run_euid: str,
        sequenced_library_assignment_euid: str,
        flowcell_id: str,
        lane: str,
        library_barcode: str,
        analysis_type: str,
        result_status: str,
        review_state: str,
        result_payload: dict[str, Any],
        artifacts: list[AtlasResultArtifact],
        idempotency_key: str,
    ) -> dict[str, Any]:
        url = (
            f"{self.base_url.rstrip('/')}"
            f"/api/integrations/ursa/v1/process-items/{atlas_test_process_item_euid}/analysis-results"
        )
        payload = {
            "atlas_tenant_id": atlas_tenant_id,
            "atlas_trf_euid": atlas_trf_euid,
            "atlas_test_euid": atlas_test_euid,
            "atlas_test_process_item_euid": atlas_test_process_item_euid,
            "analysis_euid": analysis_euid,
            "run_euid": run_euid,
            "sequenced_library_assignment_euid": sequenced_library_assignment_euid,
            "flowcell_id": flowcell_id,
            "lane": lane,
            "library_barcode": library_barcode,
            "analysis_type": analysis_type,
            "result_status": result_status,
            "review_state": review_state,
            "result_payload": result_payload,
            "source_system": "daylily-ursa",
            "artifacts": [
                {
                    "artifact_type": artifact.artifact_type,
                    "storage_uri": artifact.storage_uri,
                    "filename": artifact.filename,
                    "mime_type": artifact.mime_type,
                    "checksum_sha256": artifact.checksum_sha256,
                    "size_bytes": artifact.size_bytes,
                    "metadata": artifact.metadata or {},
                }
                for artifact in artifacts
            ],
        }
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
            "Idempotency-Key": idempotency_key,
        }
        client = self.client or httpx.Client(timeout=self.timeout_seconds)
        close_client = self.client is None
        try:
            response = client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise AtlasResultClientError(f"Atlas result return failed: {exc}") from exc
        finally:
            if close_client:
                client.close()
        if response.status_code >= 400:
            raise AtlasResultClientError(
                f"Atlas result return returned {response.status_code}: {response.text}"
            )
        return dict(response.json())
