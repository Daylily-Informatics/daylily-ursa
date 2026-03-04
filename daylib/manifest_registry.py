"""Manifest storage backed by TapDB graph objects."""

from __future__ import annotations

import base64
import datetime as dt
import gzip
import hashlib
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from daylib.tapdb_graph import TapDBBackend, from_json_addl, utc_now_iso


class ManifestTooLargeError(ValueError):
    """Raised when encoded manifest content is too large."""


@dataclass(frozen=True)
class SavedManifest:
    manifest_id: str
    customer_id: str
    name: Optional[str]
    description: Optional[str]
    created_at: str
    sample_count: int
    tsv_sha256: str
    tsv_gzip_b64: str
    manifest_euid: Optional[str] = None

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "customer_id": self.customer_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "sample_count": self.sample_count,
            "tsv_sha256": self.tsv_sha256,
            "manifest_euid": self.manifest_euid,
        }


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _generate_manifest_id(now: Optional[dt.datetime] = None) -> str:
    now = now or dt.datetime.now(dt.timezone.utc)
    ts = now.strftime("%Y%m%dT%H%M%S%fZ")
    return f"m-{ts}-{uuid.uuid4().hex[:10]}"


def _gzip_b64_encode(text: str) -> str:
    return base64.b64encode(gzip.compress(text.encode("utf-8"))).decode("ascii")


def _gzip_b64_decode(b64: str) -> str:
    return gzip.decompress(base64.b64decode(b64.encode("ascii"))).decode("utf-8")


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _estimate_sample_count(tsv_content: str) -> int:
    lines = [ln for ln in tsv_content.splitlines() if ln.strip()]
    if not lines:
        return 0
    header = lines[0]
    if "SAMPLE_ID" in header and "RUN_ID" in header:
        return max(0, len(lines) - 1)
    return len(lines)


def parse_tsv_to_samples(tsv_content: str) -> List[Dict[str, Any]]:
    lines = [ln for ln in tsv_content.splitlines() if ln.strip()]
    if not lines:
        return []

    headers = [h.strip() for h in lines[0].split("\t")]
    idx = {h.upper(): i for i, h in enumerate(headers)}
    out: List[Dict[str, Any]] = []

    for line in lines[1:]:
        cols = line.split("\t")
        sample = {
            "sample_id": cols[idx.get("SAMPLE_ID", -1)] if 0 <= idx.get("SAMPLE_ID", -1) < len(cols) else "",
            "r1_file": cols[idx.get("R1_FQ", -1)] if 0 <= idx.get("R1_FQ", -1) < len(cols) else "",
            "r2_file": cols[idx.get("R2_FQ", -1)] if 0 <= idx.get("R2_FQ", -1) < len(cols) else "",
            "run_id": cols[idx.get("RUN_ID", -1)] if 0 <= idx.get("RUN_ID", -1) < len(cols) else "",
            "status": "pending",
        }
        for field, col in (
            ("sample_type", "SAMPLE_TYPE"),
            ("lib_prep", "LIB_PREP"),
            ("external_sample_id", "EXTERNAL_SAMPLE_ID"),
        ):
            ci = idx.get(col, -1)
            if 0 <= ci < len(cols):
                sample[field] = cols[ci]
        if sample["sample_id"]:
            out.append(sample)

    return out


class ManifestRegistry:
    """Manifest registry persisted as TapDB content instances."""

    MANIFEST_TEMPLATE = "content/manifest/stage-samples/1.0/"
    CUSTOMER_TEMPLATE = "actor/customer/account/1.0/"

    def __init__(
        self,
    ):
        self.backend = TapDBBackend(app_username="ursa-manifest")

    def bootstrap(self) -> None:
        with self.backend.session_scope(commit=True) as session:
            self.backend.ensure_templates(session)

    def _customer_instance(self, session, customer_id: str):
        customer = self.backend.find_instance_by_external_id(
            session,
            template_code=self.CUSTOMER_TEMPLATE,
            key="customer_id",
            value=customer_id,
        )
        if customer is None:
            customer = self.backend.create_instance(
                session,
                template_code=self.CUSTOMER_TEMPLATE,
                name=customer_id,
                json_addl={
                    "customer_id": customer_id,
                    "customer_name": customer_id,
                    "email": "",
                    "s3_bucket": "",
                    "created_at": utc_now_iso(),
                    "updated_at": utc_now_iso(),
                },
                bstatus="active",
            )
        return customer

    def save_manifest(
        self,
        *,
        customer_id: str,
        tsv_content: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> SavedManifest:
        manifest_id = _generate_manifest_id()
        created_at = _utc_now_iso()
        sample_count = _estimate_sample_count(tsv_content)
        tsv_sha256 = _sha256_hex(tsv_content)
        tsv_gzip_b64 = _gzip_b64_encode(tsv_content)

        if len(tsv_gzip_b64) > 340_000:
            raise ManifestTooLargeError("Manifest payload exceeds safe object size")

        payload = {
            "manifest_id": manifest_id,
            "customer_id": customer_id,
            "name": name,
            "description": description,
            "created_at": created_at,
            "sample_count": sample_count,
            "tsv_sha256": tsv_sha256,
            "tsv_gzip_b64": tsv_gzip_b64,
            "updated_at": created_at,
        }

        with self.backend.session_scope(commit=True) as session:
            customer = self._customer_instance(session, customer_id)
            manifest = self.backend.create_instance(
                session,
                template_code=self.MANIFEST_TEMPLATE,
                name=name or manifest_id,
                json_addl=payload,
                bstatus="active",
            )
            self.backend.create_lineage(
                session,
                parent=customer,
                child=manifest,
                relationship_type="owns",
                name=f"{customer_id}:owns:{manifest_id}",
            )
            payload["manifest_euid"] = manifest.euid

        return SavedManifest(
            manifest_id=manifest_id,
            customer_id=customer_id,
            name=name,
            description=description,
            created_at=created_at,
            sample_count=sample_count,
            tsv_sha256=tsv_sha256,
            tsv_gzip_b64=tsv_gzip_b64,
            manifest_euid=payload.get("manifest_euid"),
        )

    def list_customer_manifests(self, customer_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        with self.backend.session_scope(commit=False) as session:
            customer = self.backend.find_instance_by_external_id(
                session,
                template_code=self.CUSTOMER_TEMPLATE,
                key="customer_id",
                value=customer_id,
            )
            if customer is None:
                return []
            rows = self.backend.get_customer_owned(
                session,
                customer=customer,
                template_code=self.MANIFEST_TEMPLATE,
                relationship_type="owns",
                limit=max(1, min(limit, 5000)),
            )
            out: List[Dict[str, Any]] = []
            for row in rows:
                payload = from_json_addl(row)
                out.append(
                    {
                        "manifest_id": payload.get("manifest_id"),
                        "customer_id": payload.get("customer_id"),
                        "name": payload.get("name"),
                        "description": payload.get("description"),
                        "created_at": payload.get("created_at"),
                        "sample_count": int(payload.get("sample_count", 0) or 0),
                        "tsv_sha256": payload.get("tsv_sha256"),
                        "manifest_euid": row.euid,
                    }
                )
            return out

    def get_manifest(self, *, customer_id: str, manifest_id: str) -> Optional[SavedManifest]:
        with self.backend.session_scope(commit=False) as session:
            row = self.backend.find_instance_by_external_id(
                session,
                template_code=self.MANIFEST_TEMPLATE,
                key="manifest_id",
                value=manifest_id,
            )
            if row is None:
                return None
            payload = from_json_addl(row)
            if payload.get("customer_id") != customer_id:
                return None
            return SavedManifest(
                manifest_id=payload["manifest_id"],
                customer_id=payload["customer_id"],
                name=payload.get("name"),
                description=payload.get("description"),
                created_at=payload.get("created_at", _utc_now_iso()),
                sample_count=int(payload.get("sample_count", 0) or 0),
                tsv_sha256=payload.get("tsv_sha256", ""),
                tsv_gzip_b64=payload.get("tsv_gzip_b64", ""),
                manifest_euid=row.euid,
            )

    def get_manifest_tsv(self, *, customer_id: str, manifest_id: str) -> Optional[str]:
        saved = self.get_manifest(customer_id=customer_id, manifest_id=manifest_id)
        if saved is None:
            return None
        return _gzip_b64_decode(saved.tsv_gzip_b64)
