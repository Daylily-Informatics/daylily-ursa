"""DynamoDB-backed storage for re-usable stage_samples.tsv manifests.

Manifests are customer-scoped and store the TSV payload (gzip+base64) alongside
lightweight metadata for listing and selection.

Design goals:
- Efficient list by customer_id (partition key)
- Stable lookup by (customer_id, manifest_id)
- Store TSV content in DynamoDB (subject to 400KB item limit)
"""

from __future__ import annotations

import base64
import datetime as dt
import gzip
import hashlib
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Attr, Key


LOGGER = logging.getLogger("daylily.manifest_registry")


class ManifestTooLargeError(ValueError):
    """Raised when the TSV content cannot fit into a DynamoDB item."""


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

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "customer_id": self.customer_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "sample_count": self.sample_count,
            "tsv_sha256": self.tsv_sha256,
        }


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _generate_manifest_id(now: Optional[dt.datetime] = None) -> str:
    """Generate a lexicographically time-sortable manifest id."""
    now = now or dt.datetime.now(dt.timezone.utc)
    ts = now.strftime("%Y%m%dT%H%M%S%fZ")
    return f"m-{ts}-{uuid.uuid4().hex[:10]}"


def _gzip_b64_encode(text: str) -> str:
    raw = text.encode("utf-8")
    gz = gzip.compress(raw)
    return base64.b64encode(gz).decode("ascii")


def _gzip_b64_decode(b64: str) -> str:
    gz = base64.b64decode(b64.encode("ascii"))
    raw = gzip.decompress(gz)
    return raw.decode("utf-8")


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _estimate_sample_count(tsv_content: str) -> int:
    """Count TSV rows excluding the header (best-effort)."""
    lines = [ln for ln in tsv_content.splitlines() if ln.strip()]
    if not lines:
        return 0
    # If the first line looks like a header row containing SAMPLE_ID, skip it.
    header = lines[0]
    if "SAMPLE_ID" in header and "RUN_ID" in header:
        return max(0, len(lines) - 1)
    return len(lines)


def parse_tsv_to_samples(tsv_content: str) -> List[Dict[str, Any]]:
    """Parse stage_samples.tsv content into a list of sample dicts.

    Expected columns include: RUN_ID, SAMPLE_ID, R1_FQ, R2_FQ (at minimum).
    Returns normalized sample dicts suitable for workset metadata['samples'].
    """
    lines = [ln for ln in tsv_content.splitlines() if ln.strip()]
    if not lines:
        return []

    # Parse header
    header_line = lines[0]
    headers = [h.strip() for h in header_line.split("\t")]

    # Build column index map (case-insensitive)
    col_idx = {}
    for i, h in enumerate(headers):
        col_idx[h.upper()] = i

    samples = []
    for line in lines[1:]:  # Skip header
        cols = line.split("\t")
        if len(cols) < 2:
            continue  # Skip malformed lines

        # Extract sample data using known column names
        sample = {
            "sample_id": cols[col_idx.get("SAMPLE_ID", -1)] if col_idx.get("SAMPLE_ID", -1) >= 0 and col_idx.get("SAMPLE_ID", -1) < len(cols) else "",
            "r1_file": cols[col_idx.get("R1_FQ", -1)] if col_idx.get("R1_FQ", -1) >= 0 and col_idx.get("R1_FQ", -1) < len(cols) else "",
            "r2_file": cols[col_idx.get("R2_FQ", -1)] if col_idx.get("R2_FQ", -1) >= 0 and col_idx.get("R2_FQ", -1) < len(cols) else "",
            "run_id": cols[col_idx.get("RUN_ID", -1)] if col_idx.get("RUN_ID", -1) >= 0 and col_idx.get("RUN_ID", -1) < len(cols) else "",
            "status": "pending",
        }

        # Add optional fields if present
        for field, tsv_col in [
            ("sample_type", "SAMPLE_TYPE"),
            ("lib_prep", "LIB_PREP"),
            ("external_sample_id", "EXTERNAL_SAMPLE_ID"),
        ]:
            idx = col_idx.get(tsv_col, -1)
            if idx >= 0 and idx < len(cols):
                sample[field] = cols[idx]

        # Only add if we have at least sample_id
        if sample["sample_id"]:
            samples.append(sample)

    return samples


class ManifestRegistry:
    """DynamoDB-backed manifest registry."""

    def __init__(
        self,
        table_name: str = "daylily-manifests",
        region: str = "us-west-2",
        profile: Optional[str] = None,
    ):
        session_kwargs = {"region_name": region}
        if profile:
            session_kwargs["profile_name"] = profile
        session = boto3.Session(**session_kwargs)
        self.dynamodb = session.resource("dynamodb")
        self.table_name = table_name
        self.table = self.dynamodb.Table(table_name)
        self._hash_key_name = "customer_id"
        self._range_key_name: Optional[str] = "manifest_id"
        self._schema_loaded = False

        LOGGER.info("ManifestRegistry bound to table: %s", self.table.table_name)
        assert hasattr(self.table, "table_name")

    def _detect_table_schema(self) -> None:
        """Detect key schema from DynamoDB and cache result.

        Supports both legacy schema (customer_id + manifest_id) and pk-style tables.
        Falls back to legacy schema when table metadata is unavailable.
        """
        if self._schema_loaded:
            return

        hash_key = "customer_id"
        range_key: Optional[str] = "manifest_id"

        try:
            desc = self.dynamodb.meta.client.describe_table(TableName=self.table_name)
            table_desc = desc.get("Table", {}) if isinstance(desc, dict) else {}
            key_schema = table_desc.get("KeySchema", []) if isinstance(table_desc, dict) else []
            if isinstance(key_schema, list):
                for element in key_schema:
                    if not isinstance(element, dict):
                        continue
                    attr_name = str(element.get("AttributeName") or "").strip()
                    key_type = str(element.get("KeyType") or "").strip().upper()
                    if key_type == "HASH" and attr_name:
                        hash_key = attr_name
                    elif key_type == "RANGE" and attr_name:
                        range_key = attr_name
                if not any(
                    isinstance(element, dict) and str(element.get("KeyType", "")).upper() == "RANGE"
                    for element in key_schema
                ):
                    range_key = None
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") != "ResourceNotFoundException":
                LOGGER.warning("Failed to inspect manifest table schema, using legacy defaults: %s", e)
        except Exception as e:  # pragma: no cover - defensive
            LOGGER.warning("Unexpected error inspecting manifest table schema, using legacy defaults: %s", e)

        self._hash_key_name = hash_key
        self._range_key_name = range_key
        self._schema_loaded = True
        LOGGER.info(
            "Manifest table schema detected: hash=%s range=%s",
            self._hash_key_name,
            self._range_key_name or "(none)",
        )

    def _build_primary_key(self, customer_id: str, manifest_id: str) -> Dict[str, str]:
        self._detect_table_schema()
        if self._hash_key_name == "customer_id" and self._range_key_name == "manifest_id":
            return {"customer_id": customer_id, "manifest_id": manifest_id}
        if self._hash_key_name == "pk" and self._range_key_name == "sk":
            return {"pk": f"manifest#{customer_id}", "sk": manifest_id}
        if self._hash_key_name == "pk" and self._range_key_name is None:
            return {"pk": f"manifest#{customer_id}#{manifest_id}"}
        key = {self._hash_key_name: customer_id}
        if self._range_key_name:
            key[self._range_key_name] = manifest_id
        return key

    def create_table_if_not_exists(self) -> None:
        try:
            self.table.load()
            LOGGER.info("Manifest table %s already exists", self.table_name)
            self._schema_loaded = False
            self._detect_table_schema()
            return
        except ClientError as e:
            if e.response["Error"]["Code"] != "ResourceNotFoundException":
                raise

        LOGGER.info("Creating manifest table %s", self.table_name)
        table = self.dynamodb.create_table(
            TableName=self.table_name,
            KeySchema=[
                {"AttributeName": "customer_id", "KeyType": "HASH"},
                {"AttributeName": "manifest_id", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "customer_id", "AttributeType": "S"},
                {"AttributeName": "manifest_id", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        table.wait_until_exists()
        LOGGER.info("Manifest table %s created successfully", self.table_name)
        self._schema_loaded = False
        self._detect_table_schema()

    def save_manifest(
        self,
        *,
        customer_id: str,
        tsv_content: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> SavedManifest:
        if not customer_id or not customer_id.strip():
            raise ValueError("customer_id is required")
        if not tsv_content or not tsv_content.strip():
            raise ValueError("tsv_content is required")

        created_at = _utc_now_iso()
        manifest_id = _generate_manifest_id()
        sample_count = _estimate_sample_count(tsv_content)

        tsv_sha256 = _sha256_hex(tsv_content)
        tsv_gzip_b64 = _gzip_b64_encode(tsv_content)

        # DynamoDB hard limit is 400KB per item; keep a conservative ceiling.
        if len(tsv_gzip_b64.encode("utf-8")) > 350_000:
            raise ManifestTooLargeError(
                "Manifest TSV is too large to store in DynamoDB. "
                "Consider splitting into multiple manifests."
            )

        item: Dict[str, Any] = {
            "customer_id": customer_id,
            "manifest_id": manifest_id,
            "created_at": created_at,
            "name": name or "",
            "description": description or "",
            "sample_count": int(sample_count),
            "tsv_sha256": tsv_sha256,
            "tsv_gzip_b64": tsv_gzip_b64,
            "schema_version": 1,
            "entity_type": "manifest",
        }
        item.update(self._build_primary_key(customer_id, manifest_id))

        self._detect_table_schema()
        if self._range_key_name:
            condition_expr = (
                f"attribute_not_exists({self._hash_key_name}) AND "
                f"attribute_not_exists({self._range_key_name})"
            )
        else:
            condition_expr = f"attribute_not_exists({self._hash_key_name})"

        try:
            self.table.put_item(
                Item=item,
                ConditionExpression=condition_expr,
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                # Extremely unlikely given UUID randomness; surface as conflict.
                raise ValueError("Manifest id collision; please retry") from e
            raise

        LOGGER.info(
            "Saved manifest %s for customer %s (samples=%d)",
            manifest_id,
            customer_id,
            sample_count,
        )

        return SavedManifest(
            manifest_id=manifest_id,
            customer_id=customer_id,
            name=name,
            description=description,
            created_at=created_at,
            sample_count=sample_count,
            tsv_sha256=tsv_sha256,
            tsv_gzip_b64=tsv_gzip_b64,
        )

    def list_customer_manifests(self, customer_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        """List manifests (metadata only) for a customer."""
        if not customer_id or not customer_id.strip():
            raise ValueError("customer_id is required")

        self._detect_table_schema()
        items: List[Dict[str, Any]] = []

        try:
            if self._hash_key_name == "customer_id":
                resp = self.table.query(
                    KeyConditionExpression=Key("customer_id").eq(customer_id),
                    Limit=limit,
                    ScanIndexForward=False,  # newest-first due to time-sortable manifest_id
                )
                items = resp.get("Items", [])
            elif self._hash_key_name == "pk" and self._range_key_name == "sk":
                resp = self.table.query(
                    KeyConditionExpression=Key("pk").eq(f"manifest#{customer_id}"),
                    Limit=limit,
                    ScanIndexForward=False,
                )
                items = resp.get("Items", [])
            else:
                # Fallback for pk-only and other schemas without customer queryability.
                # Filter by stored customer_id attribute and manifest entity type.
                scan_kwargs: Dict[str, Any] = {
                    "FilterExpression": Attr("customer_id").eq(customer_id)
                    & Attr("entity_type").eq("manifest"),
                }
                last_evaluated_key = None
                while len(items) < limit:
                    if last_evaluated_key:
                        scan_kwargs["ExclusiveStartKey"] = last_evaluated_key
                    resp = self.table.scan(**scan_kwargs)
                    page_items = resp.get("Items", [])
                    items.extend(page_items)
                    last_evaluated_key = resp.get("LastEvaluatedKey")
                    if not last_evaluated_key:
                        break
                items = items[:limit]
        except ClientError as e:
            LOGGER.error("Failed to list manifests for customer %s: %s", customer_id, str(e))
            raise

        items = sorted(
            items,
            key=lambda it: str(it.get("created_at") or it.get("manifest_id") or ""),
            reverse=True,
        )
        result = []
        for it in items:
            result.append(
                {
                    "manifest_id": it.get("manifest_id"),
                    "customer_id": it.get("customer_id"),
                    "name": it.get("name") or None,
                    "description": it.get("description") or None,
                    "created_at": it.get("created_at"),
                    "sample_count": int(it.get("sample_count") or 0),
                }
            )
        return result

    def get_manifest(self, *, customer_id: str, manifest_id: str) -> Optional[SavedManifest]:
        if not customer_id or not customer_id.strip():
            raise ValueError("customer_id is required")
        if not manifest_id or not manifest_id.strip():
            raise ValueError("manifest_id is required")

        key = self._build_primary_key(customer_id, manifest_id)
        try:
            resp = self.table.get_item(Key=key)
        except ClientError as e:
            LOGGER.error(
                "Failed to get manifest %s for customer %s: %s",
                manifest_id,
                customer_id,
                str(e),
            )
            raise

        item = resp.get("Item")
        if not item:
            return None

        return SavedManifest(
            manifest_id=item["manifest_id"],
            customer_id=item["customer_id"],
            name=item.get("name") or None,
            description=item.get("description") or None,
            created_at=item.get("created_at") or "",
            sample_count=int(item.get("sample_count") or 0),
            tsv_sha256=item.get("tsv_sha256") or "",
            tsv_gzip_b64=item.get("tsv_gzip_b64") or "",
        )

    def get_manifest_tsv(self, *, customer_id: str, manifest_id: str) -> Optional[str]:
        m = self.get_manifest(customer_id=customer_id, manifest_id=manifest_id)
        if not m:
            return None
        return _gzip_b64_decode(m.tsv_gzip_b64)
