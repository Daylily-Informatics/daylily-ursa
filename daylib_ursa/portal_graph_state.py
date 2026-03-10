"""TapDB graph-backed portal state for bucket, manifest, and workset services.

This module enforces EUID-only object identifiers and lineage-based relationships.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from daylib_ursa.s3_bucket_validator import LinkedBucketManager
from daylib_ursa.s3_utils import RegionAwareS3Client
from daylib_ursa.tapdb_graph import TapDBBackend, from_json_addl, utc_now_iso


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _stable_key(*parts: str) -> str:
    data = "::".join(str(p or "").strip() for p in parts)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:32]


def _format_bytes_human(size_bytes: int) -> str:
    value = float(max(0, int(size_bytes)))
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while value >= 1024.0 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(value)} {units[idx]}"
    return f"{value:.2f} {units[idx]}"


class GraphPortalState:
    """Portal persistence/service adapter backed by TapDB instances + lineage."""

    CUSTOMER_TEMPLATE = "actor/customer/account/1.0/"
    BUCKET_TEMPLATE = "data/storage/s3-bucket-link/1.0/"
    MANIFEST_TEMPLATE = "data/manifest/stage-samples/1.0/"
    WORKSET_TEMPLATE = "workflow/workset/analysis-request/1.0/"
    PRICING_RUN_TEMPLATE = "data/pricing/capture-run/1.0/"
    PRICING_POINT_TEMPLATE = "data/pricing/capture-point/1.0/"

    def __init__(
        self,
        *,
        region: str = "us-west-2",
        profile: str | None = None,
        backend: TapDBBackend | None = None,
    ) -> None:
        self.region = region
        self.profile = profile
        self.backend = backend or TapDBBackend(app_username="ursa-portal")
        self.bucket_manager = LinkedBucketManager(region=region, profile=profile)
        self.s3 = RegionAwareS3Client(default_region=region, profile=profile)

        with self.backend.session_scope(commit=True) as session:
            self.backend.ensure_templates(session)

    # ---------------------------------------------------------------------
    # Shared helpers
    # ---------------------------------------------------------------------

    def _customer(self, session: Any, customer_id: str, *, create: bool = True) -> Any:
        key = str(customer_id or "").strip() or "default-customer"
        row = self.backend.find_instance_by_euid(
            session,
            template_code=self.CUSTOMER_TEMPLATE,
            value=key,
        )
        if row is None:
            row = self.backend.find_instance_by_external_id(
                session,
                template_code=self.CUSTOMER_TEMPLATE,
                key="customer_id",
                value=key,
            )
        if row is not None or not create:
            return row

        now = utc_now_iso()
        return self.backend.create_instance(
            session,
            template_code=self.CUSTOMER_TEMPLATE,
            name=key,
            json_addl={
                "customer_id": key,
                "customer_name": key,
                "created_at": now,
                "updated_at": now,
            },
            bstatus="active",
        )

    @staticmethod
    def _customer_external_id(customer_row: Any) -> str:
        payload = from_json_addl(customer_row)
        external_id = str(payload.get("customer_id") or "").strip()
        if external_id:
            return external_id
        return str(customer_row.euid)

    def _ensure_owned(self, session: Any, customer_row: Any, child_row: Any) -> bool:
        parents = self.backend.list_parents(session, child=child_row, relationship_type="owns")
        return any(parent.uid == customer_row.uid for parent in parents)

    def _resolve_owned(
        self,
        session: Any,
        *,
        customer_row: Any,
        template_code: str,
        object_id: str,
    ) -> Any | None:
        row = self.backend.find_instance_by_euid(
            session,
            template_code=template_code,
            value=str(object_id or "").strip(),
        )
        if row is None:
            return None
        if not self._ensure_owned(session, customer_row, row):
            return None
        return row

    # ---------------------------------------------------------------------
    # Manifests
    # ---------------------------------------------------------------------

    def _to_manifest_payload(self, row: Any) -> dict[str, Any]:
        payload = from_json_addl(row)
        return {
            "manifest_id": str(row.euid),
            "name": payload.get("name") or row.name,
            "description": payload.get("description"),
            "tsv_content": payload.get("tsv_content") or "",
            "sample_count": int(payload.get("sample_count") or 0),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
        }

    def list_manifests(self, *, customer_id: str, limit: int = 200) -> list[dict[str, Any]]:
        with self.backend.session_scope(commit=False) as session:
            customer = self._customer(session, customer_id, create=True)
            rows = self.backend.get_customer_owned(
                session,
                customer=customer,
                template_code=self.MANIFEST_TEMPLATE,
                relationship_type="owns",
                limit=max(limit, 1),
            )
            return [self._to_manifest_payload(row) for row in rows]

    def create_manifest(
        self,
        *,
        customer_id: str,
        tsv_content: str,
        name: str | None,
        description: str | None = None,
    ) -> dict[str, Any]:
        lines = [line for line in str(tsv_content or "").splitlines() if line.strip()]
        sample_count = max(len(lines) - 1, 0)
        with self.backend.session_scope(commit=True) as session:
            customer = self._customer(session, customer_id, create=True)
            row = self.backend.create_instance(
                session,
                template_code=self.MANIFEST_TEMPLATE,
                name=str(name or "").strip() or f"manifest-{utc_now_iso()}",
                json_addl={
                    "external_key": _stable_key(customer_id, str(name or ""), secrets.token_hex(4)),
                    "name": str(name or "").strip() or None,
                    "description": str(description or "").strip() or None,
                    "tsv_content": str(tsv_content or ""),
                    "sample_count": sample_count,
                    "created_at": utc_now_iso(),
                    "updated_at": utc_now_iso(),
                },
            )
            self.backend.create_lineage(session, parent=customer, child=row, relationship_type="owns")
            return self._to_manifest_payload(row)

    def get_manifest(self, *, customer_id: str, manifest_id: str) -> dict[str, Any] | None:
        with self.backend.session_scope(commit=False) as session:
            customer = self._customer(session, customer_id, create=True)
            row = self._resolve_owned(
                session,
                customer_row=customer,
                template_code=self.MANIFEST_TEMPLATE,
                object_id=manifest_id,
            )
            if row is None:
                return None
            return self._to_manifest_payload(row)

    # ---------------------------------------------------------------------
    # Buckets + S3 operations
    # ---------------------------------------------------------------------

    @staticmethod
    def _normalize_s3_prefix(value: str | None) -> str:
        prefix = str(value or "").strip().lstrip("/")
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        return prefix

    def _to_bucket_payload(self, row: Any) -> dict[str, Any]:
        payload = from_json_addl(row)
        return {
            "bucket_id": str(row.euid),
            "bucket_name": payload.get("bucket_name"),
            "bucket_type": payload.get("bucket_type", "primary"),
            "display_name": payload.get("display_name") or payload.get("bucket_name"),
            "description": payload.get("description"),
            "is_validated": bool(payload.get("is_validated", False)),
            "validation_timestamp": payload.get("validation_timestamp"),
            "can_read": bool(payload.get("can_read", False)),
            "can_write": bool(payload.get("can_write", False)),
            "can_list": bool(payload.get("can_list", False)),
            "region": payload.get("region"),
            "prefix_restriction": payload.get("prefix_restriction"),
            "read_only": bool(payload.get("read_only", False)),
            "linked_at": payload.get("linked_at"),
            "updated_at": payload.get("updated_at"),
            "external_bucket_key": payload.get("bucket_id"),
        }

    def list_buckets(self, *, customer_id: str) -> list[dict[str, Any]]:
        with self.backend.session_scope(commit=False) as session:
            customer = self._customer(session, customer_id, create=True)
            rows = self.backend.get_customer_owned(
                session,
                customer=customer,
                template_code=self.BUCKET_TEMPLATE,
                relationship_type="owns",
                limit=1000,
            )
            return [self._to_bucket_payload(row) for row in rows]

    def get_bucket(self, *, customer_id: str, bucket_id: str) -> dict[str, Any] | None:
        with self.backend.session_scope(commit=False) as session:
            customer = self._customer(session, customer_id, create=True)
            row = self._resolve_owned(
                session,
                customer_row=customer,
                template_code=self.BUCKET_TEMPLATE,
                object_id=bucket_id,
            )
            if row is None:
                return None
            return self._to_bucket_payload(row)

    def validate_bucket_name(self, bucket_name: str) -> dict[str, Any]:
        result = self.bucket_manager.validator.validate_bucket(bucket_name)
        output = result.to_dict()
        output["remediation_steps"] = self.bucket_manager.validator.get_setup_instructions(
            bucket_name,
            result,
        )
        return output

    def link_bucket(
        self,
        *,
        customer_id: str,
        bucket_name: str,
        bucket_type: str,
        display_name: str | None,
        description: str | None,
        prefix_restriction: str | None,
        read_only: bool,
        validate: bool,
    ) -> dict[str, Any]:
        linked, _validation = self.bucket_manager.link_bucket(
            customer_id=customer_id,
            bucket_name=bucket_name,
            bucket_type=bucket_type,
            display_name=display_name,
            description=description,
            prefix_restriction=prefix_restriction,
            read_only=read_only,
            validate=validate,
        )
        lookup = str(linked.bucket_euid or linked.bucket_id)
        with self.backend.session_scope(commit=False) as session:
            row = self.backend.find_instance_by_euid(
                session,
                template_code=self.BUCKET_TEMPLATE,
                value=lookup,
            )
            if row is None:
                raise RuntimeError("linked bucket instance not found")
            return self._to_bucket_payload(row)

    def update_bucket(
        self,
        *,
        customer_id: str,
        bucket_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any] | None:
        with self.backend.session_scope(commit=True) as session:
            customer = self._customer(session, customer_id, create=True)
            row = self._resolve_owned(
                session,
                customer_row=customer,
                template_code=self.BUCKET_TEMPLATE,
                object_id=bucket_id,
            )
            if row is None:
                return None
            payload = from_json_addl(row)
            payload.update({k: v for k, v in updates.items() if v is not None})
            payload["updated_at"] = utc_now_iso()
            self.backend.update_instance_json(session, row, payload)
            return self._to_bucket_payload(row)

    def unlink_bucket(self, *, customer_id: str, bucket_id: str) -> bool:
        with self.backend.session_scope(commit=True) as session:
            customer = self._customer(session, customer_id, create=True)
            row = self._resolve_owned(
                session,
                customer_row=customer,
                template_code=self.BUCKET_TEMPLATE,
                object_id=bucket_id,
            )
            if row is None:
                return False
            row.is_deleted = True
            row.bstatus = "deleted"
            session.flush()
            return True

    def revalidate_bucket(self, *, customer_id: str, bucket_id: str) -> dict[str, Any] | None:
        bucket = self.get_bucket(customer_id=customer_id, bucket_id=bucket_id)
        if bucket is None:
            return None
        result = self.bucket_manager.validator.validate_bucket(str(bucket.get("bucket_name") or ""))
        updated = self.update_bucket(
            customer_id=customer_id,
            bucket_id=bucket_id,
            updates={
                "is_validated": True,
                "validation_timestamp": utc_now_iso(),
                "can_read": result.can_read,
                "can_write": result.can_write,
                "can_list": result.can_list,
                "region": result.region,
            },
        )
        if updated is None:
            return None
        return {
            **updated,
            "is_valid": bool(result.is_valid),
            "errors": list(result.errors),
            "warnings": list(result.warnings),
        }

    def browse_bucket(
        self,
        *,
        customer_id: str,
        bucket_id: str,
        prefix: str | None = None,
        max_keys: int = 200,
    ) -> dict[str, Any]:
        bucket = self.get_bucket(customer_id=customer_id, bucket_id=bucket_id)
        if bucket is None:
            raise KeyError("bucket not found")

        bucket_name = str(bucket.get("bucket_name") or "")
        if not bucket_name:
            raise ValueError("bucket missing bucket_name")

        effective_prefix = str(prefix or "")
        restriction = str(bucket.get("prefix_restriction") or "")
        if restriction:
            if not effective_prefix.startswith(restriction):
                effective_prefix = restriction

        response = self.s3.list_objects_v2(
            Bucket=bucket_name,
            Prefix=effective_prefix,
            Delimiter="/",
            MaxKeys=max_keys,
        )

        items: list[dict[str, Any]] = []
        for folder in response.get("CommonPrefixes") or []:
            key = str(folder.get("Prefix") or "")
            items.append(
                {
                    "key": key,
                    "name": key.rstrip("/").split("/")[-1] + "/",
                    "is_folder": True,
                }
            )

        for obj in response.get("Contents") or []:
            key = str(obj.get("Key") or "")
            if key.endswith("/"):
                continue
            name = key.split("/")[-1]
            suffix = Path(name).suffix.lstrip(".").lower()
            items.append(
                {
                    "key": key,
                    "name": name,
                    "is_folder": False,
                    "size_bytes": int(obj.get("Size") or 0),
                    "last_modified": str(obj.get("LastModified") or ""),
                    "file_format": suffix,
                }
            )

        breadcrumbs = [{"name": "Root", "prefix": ""}]
        if effective_prefix:
            parts = [part for part in effective_prefix.strip("/").split("/") if part]
            acc = ""
            for part in parts:
                acc = f"{acc}{part}/"
                breadcrumbs.append({"name": part, "prefix": acc})

        return {
            "bucket_id": bucket_id,
            "bucket_name": bucket_name,
            "current_prefix": effective_prefix,
            "items": items,
            "breadcrumbs": breadcrumbs,
        }

    def discover_bucket_files(
        self,
        *,
        customer_id: str,
        bucket_id: str,
        prefix: str | None,
        max_files: int,
        file_formats: list[str] | None,
    ) -> dict[str, Any]:
        bucket = self.get_bucket(customer_id=customer_id, bucket_id=bucket_id)
        if bucket is None:
            raise KeyError("bucket not found")

        bucket_name = str(bucket.get("bucket_name") or "")
        effective_prefix = str(prefix or "")
        allowed = {str(fmt).strip().lower() for fmt in (file_formats or []) if str(fmt).strip()}

        discovered: list[dict[str, Any]] = []
        continuation_token: str | None = None
        remaining = max(1, min(int(max_files or 1000), 5000))

        while remaining > 0:
            kwargs: dict[str, Any] = {
                "Bucket": bucket_name,
                "Prefix": effective_prefix,
                "MaxKeys": min(1000, remaining),
            }
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token
            response = self.s3.list_objects_v2(**kwargs)
            objects = response.get("Contents") or []
            for obj in objects:
                key = str(obj.get("Key") or "")
                if key.endswith("/"):
                    continue
                suffix = Path(key).suffix.lstrip(".").lower()
                if allowed and suffix not in allowed:
                    continue
                uri = f"s3://{bucket_name}/{key}"
                discovered.append(
                    {
                        "key": key,
                        "file_size_bytes": int(obj.get("Size") or 0),
                        "last_modified": str(obj.get("LastModified") or ""),
                        "detected_format": suffix,
                        "s3_uri": uri,
                    }
                )
                remaining -= 1
                if remaining <= 0:
                    break
            if remaining <= 0:
                break
            if not response.get("IsTruncated"):
                break
            continuation_token = str(response.get("NextContinuationToken") or "") or None
            if not continuation_token:
                break

        return {
            "files": discovered,
            "total_files": len(discovered),
        }

    def discover_samples(self, *, customer_id: str, max_files: int = 2000) -> dict[str, Any]:
        buckets = self.list_buckets(customer_id=customer_id)
        if not buckets:
            return {"samples": []}
        primary = next((item for item in buckets if item.get("bucket_type") == "primary"), buckets[0])
        discovered = self.discover_bucket_files(
            customer_id=customer_id,
            bucket_id=str(primary.get("bucket_id") or ""),
            prefix=str(primary.get("prefix_restriction") or ""),
            max_files=max_files,
            file_formats=["fastq", "fq", "gz"],
        )

        grouped: dict[str, dict[str, Any]] = {}
        for item in discovered.get("files") or []:
            key = str(item.get("key") or "")
            name = Path(key).name
            if not re.search(r"\.(fastq|fq)(\.gz)?$", name, flags=re.IGNORECASE):
                continue
            sample_guess = re.sub(r"(_R?[12].*)$", "", name, flags=re.IGNORECASE)
            sample = grouped.setdefault(sample_guess, {"sample_id": sample_guess, "fastq_r1": "", "fastq_r2": ""})
            uri = f"s3://{primary.get('bucket_name')}/{key}"
            if re.search(r"_R?1", name, flags=re.IGNORECASE):
                sample["fastq_r1"] = uri
            elif re.search(r"_R?2", name, flags=re.IGNORECASE):
                sample["fastq_r2"] = uri
            else:
                if not sample.get("fastq_r1"):
                    sample["fastq_r1"] = uri

        return {"samples": list(grouped.values())}

    # ---------------------------------------------------------------------
    # Worksets
    # ---------------------------------------------------------------------

    def _to_workset_payload(self, session: Any, row: Any) -> dict[str, Any]:
        payload = from_json_addl(row)
        manifests = self.backend.list_children(session, parent=row, relationship_type="uses_manifest")
        manifest_id = str(manifests[0].euid) if manifests else None
        return {
            "workset_id": str(row.euid),
            "customer_id": str(payload.get("customer_id") or ""),
            "workset_name": str(payload.get("workset_name") or ""),
            "pipeline_type": str(payload.get("pipeline_type") or ""),
            "reference_genome": str(payload.get("reference_genome") or ""),
            "manifest_id": manifest_id,
            "sample_count": int(payload.get("sample_count") or 0),
            "priority": str(payload.get("priority") or "normal"),
            "workset_type": str(payload.get("workset_type") or "ruo"),
            "notification_email": payload.get("notification_email"),
            "enable_qc": bool(payload.get("enable_qc", True)),
            "archive_results": bool(payload.get("archive_results", True)),
            "state": str(payload.get("state") or "ready"),
            "preferred_cluster": payload.get("preferred_cluster"),
            "cluster_name": payload.get("cluster_name"),
            "cluster_region": payload.get("cluster_region"),
            "target_region": payload.get("target_region"),
            "cluster_create_job_id": payload.get("cluster_create_job_id"),
            "message": payload.get("message"),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
        }

    def create_workset(
        self,
        *,
        customer_id: str,
        payload: Dict[str, Any],
        state: str,
        cluster_name: Optional[str],
        cluster_region: Optional[str],
        target_region: Optional[str],
        cluster_create_job_id: Optional[str],
        message: Optional[str],
    ) -> Dict[str, Any]:
        manifest_id = str(payload.get("manifest_id") or "").strip()
        manifest_tsv = str(payload.get("manifest_tsv_content") or "").strip()

        with self.backend.session_scope(commit=True) as session:
            customer = self._customer(session, customer_id, create=True)
            workset_row = self.backend.create_instance(
                session,
                template_code=self.WORKSET_TEMPLATE,
                name=str(payload.get("workset_name") or "").strip() or "workset",
                json_addl={
                    "external_key": _stable_key(customer_id, str(payload.get("workset_name") or ""), secrets.token_hex(4)),
                    "customer_id": customer_id,
                    "workset_name": str(payload.get("workset_name") or "").strip(),
                    "pipeline_type": str(payload.get("pipeline_type") or "").strip(),
                    "reference_genome": str(payload.get("reference_genome") or "").strip(),
                    "sample_count": int(payload.get("sample_count") or 0),
                    "priority": str(payload.get("priority") or "normal"),
                    "workset_type": str(payload.get("workset_type") or "ruo"),
                    "notification_email": payload.get("notification_email"),
                    "enable_qc": bool(payload.get("enable_qc", True)),
                    "archive_results": bool(payload.get("archive_results", True)),
                    "state": state,
                    "preferred_cluster": payload.get("preferred_cluster"),
                    "cluster_name": cluster_name,
                    "cluster_region": cluster_region,
                    "target_region": target_region,
                    "cluster_create_job_id": cluster_create_job_id,
                    "message": message,
                    "created_at": utc_now_iso(),
                    "updated_at": utc_now_iso(),
                },
            )
            self.backend.create_lineage(
                session,
                parent=customer,
                child=workset_row,
                relationship_type="owns",
            )

            manifest_row = None
            if manifest_id:
                manifest_row = self._resolve_owned(
                    session,
                    customer_row=customer,
                    template_code=self.MANIFEST_TEMPLATE,
                    object_id=manifest_id,
                )
                if manifest_row is None:
                    raise ValueError(f"Manifest not found for customer: {manifest_id}")
            elif manifest_tsv:
                created_manifest = self.create_manifest(
                    customer_id=customer_id,
                    tsv_content=manifest_tsv,
                    name=str(payload.get("workset_name") or "workset-manifest"),
                )
                manifest_row = self.backend.find_instance_by_euid(
                    session,
                    template_code=self.MANIFEST_TEMPLATE,
                    value=str(created_manifest["manifest_id"]),
                )

            if manifest_row is not None:
                self.backend.create_lineage(
                    session,
                    parent=workset_row,
                    child=manifest_row,
                    relationship_type="uses_manifest",
                )

            return self._to_workset_payload(session, workset_row)

    def update_workset_cluster_assignment(
        self,
        *,
        workset_id: str,
        cluster_name: str,
        cluster_region: str,
        state: str = "ready",
        message: Optional[str] = None,
    ) -> None:
        with self.backend.session_scope(commit=True) as session:
            row = self.backend.find_instance_by_euid(
                session,
                template_code=self.WORKSET_TEMPLATE,
                value=workset_id,
                for_update=True,
            )
            if row is None:
                return
            payload = from_json_addl(row)
            payload["cluster_name"] = cluster_name
            payload["cluster_region"] = cluster_region
            payload["state"] = state
            payload["message"] = message
            payload["updated_at"] = utc_now_iso()
            self.backend.update_instance_json(session, row, payload)

    def list_pending_worksets(self, *, target_region: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.backend.session_scope(commit=False) as session:
            rows = self.backend.list_instances_by_template(
                session,
                template_code=self.WORKSET_TEMPLATE,
                limit=10000,
            )
            out: list[dict[str, Any]] = []
            for row in rows:
                payload = self._to_workset_payload(session, row)
                if payload.get("state") != "pending_cluster_creation":
                    continue
                if target_region and payload.get("target_region") != target_region:
                    continue
                out.append(payload)
            return out

    def list_worksets(
        self,
        customer_id: str,
        *,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        with self.backend.session_scope(commit=False) as session:
            customer = self._customer(session, customer_id, create=True)
            rows = self.backend.get_customer_owned(
                session,
                customer=customer,
                template_code=self.WORKSET_TEMPLATE,
                relationship_type="owns",
                limit=max(limit, 1),
            )
            worksets = [self._to_workset_payload(session, row) for row in rows]
            if status:
                return [item for item in worksets if item.get("state") == status]
            return worksets

    def get_workset(self, customer_id: str, workset_id: str) -> Optional[Dict[str, Any]]:
        with self.backend.session_scope(commit=False) as session:
            customer = self._customer(session, customer_id, create=True)
            row = self._resolve_owned(
                session,
                customer_row=customer,
                template_code=self.WORKSET_TEMPLATE,
                object_id=workset_id,
            )
            if row is None:
                return None
            return self._to_workset_payload(session, row)

    def get_dashboard_stats(self, customer_id: str) -> Dict[str, Any]:
        worksets = self.list_worksets(customer_id, limit=10000)
        counts: dict[str, int] = {}
        dewey_artifacts = 0
        for item in worksets:
            state = str(item.get("state") or "")
            counts[state] = counts.get(state, 0) + 1
            dewey_artifacts += max(int(item.get("sample_count") or 0), 0)

        dewey_size_bytes = 0
        dewey_size_gb = dewey_size_bytes / (1024**3)
        dewey_storage_cost = dewey_size_gb * 0.023
        compute_cost = float(counts.get("complete", 0)) * 5.0

        return {
            "in_progress_worksets": counts.get("running", 0),
            "ready_worksets": counts.get("ready", 0) + counts.get("pending_cluster_creation", 0),
            "completed_worksets": counts.get("complete", 0),
            "error_worksets": counts.get("error", 0),
            "cost_this_month": round(compute_cost + dewey_storage_cost, 4),
            "compute_cost_usd": round(compute_cost, 4),
            "dewey_artifacts": dewey_artifacts,
            "dewey_artifact_size_gb": round(dewey_size_gb, 4),
            "dewey_storage_cost_usd": round(dewey_storage_cost, 4),
            "dewey_storage_human": _format_bytes_human(dewey_size_bytes),
        }

    def get_activity_series(self, customer_id: str, *, days: int = 30) -> Dict[str, Any]:
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=max(days - 1, 0))
        labels: List[str] = []
        submitted: List[int] = []
        completed: List[int] = []
        failed: List[int] = []

        worksets = self.list_worksets(customer_id, limit=20000)
        grouped: dict[str, dict[str, int]] = {}
        for item in worksets:
            created = _parse_iso(str(item.get("created_at") or ""))
            if created is None:
                continue
            key = created.date().isoformat()
            state = str(item.get("state") or "")
            entry = grouped.setdefault(key, {"submitted": 0, "completed": 0, "failed": 0})
            entry["submitted"] += 1
            if state == "complete":
                entry["completed"] += 1
            if state == "error":
                entry["failed"] += 1

        for offset in range(days):
            current_day = start_date + timedelta(days=offset)
            key = current_day.isoformat()
            row = grouped.get(key, {})
            labels.append(current_day.strftime("%b %d"))
            submitted.append(int(row.get("submitted", 0)))
            completed.append(int(row.get("completed", 0)))
            failed.append(int(row.get("failed", 0)))

        return {
            "labels": labels,
            "datasets": {
                "submitted": submitted,
                "completed": completed,
                "failed": failed,
            },
        }

    # ---------------------------------------------------------------------
    # Pricing (EUID-backed)
    # ---------------------------------------------------------------------

    def _to_pricing_run(self, row: Any) -> dict[str, Any]:
        payload = from_json_addl(row)
        return {
            "run_id": str(row.euid),
            "trigger": payload.get("trigger"),
            "requested_by": payload.get("requested_by"),
            "status": payload.get("status"),
            "error": payload.get("error"),
            "created_at": payload.get("created_at"),
            "started_at": payload.get("started_at"),
            "completed_at": payload.get("completed_at"),
            "snapshot_captured_at": payload.get("snapshot_captured_at"),
        }

    def create_pricing_run(self, *, trigger: str, requested_by: Optional[str]) -> Dict[str, Any]:
        with self.backend.session_scope(commit=True) as session:
            row = self.backend.create_instance(
                session,
                template_code=self.PRICING_RUN_TEMPLATE,
                name=f"pricing-{trigger}-{utc_now_iso()}",
                json_addl={
                    "external_key": _stable_key(trigger, str(requested_by or ""), secrets.token_hex(4)),
                    "trigger": trigger,
                    "requested_by": requested_by,
                    "status": "queued",
                    "created_at": utc_now_iso(),
                    "started_at": None,
                    "completed_at": None,
                    "snapshot_captured_at": None,
                    "error": None,
                },
            )
            return self._to_pricing_run(row)

    def get_active_pricing_run(self) -> Dict[str, Any] | None:
        runs = self.list_pricing_runs(limit=50)
        for run in runs:
            if run.get("status") in {"queued", "running"}:
                return run
        return None

    def mark_pricing_run_running(self, run_id: str) -> None:
        with self.backend.session_scope(commit=True) as session:
            row = self.backend.find_instance_by_euid(
                session,
                template_code=self.PRICING_RUN_TEMPLATE,
                value=run_id,
                for_update=True,
            )
            if row is None:
                return
            payload = from_json_addl(row)
            payload["status"] = "running"
            payload["started_at"] = utc_now_iso()
            payload["updated_at"] = utc_now_iso()
            self.backend.update_instance_json(session, row, payload)

    def mark_pricing_run_failed(self, run_id: str, error: str) -> None:
        with self.backend.session_scope(commit=True) as session:
            row = self.backend.find_instance_by_euid(
                session,
                template_code=self.PRICING_RUN_TEMPLATE,
                value=run_id,
                for_update=True,
            )
            if row is None:
                return
            payload = from_json_addl(row)
            payload["status"] = "failed"
            payload["error"] = str(error or "")
            payload["completed_at"] = utc_now_iso()
            payload["updated_at"] = utc_now_iso()
            self.backend.update_instance_json(session, row, payload)

    def save_pricing_snapshot(self, run_id: str, snapshot: Dict[str, Any]) -> None:
        points = list(snapshot.get("points") or [])
        with self.backend.session_scope(commit=True) as session:
            run_row = self.backend.find_instance_by_euid(
                session,
                template_code=self.PRICING_RUN_TEMPLATE,
                value=run_id,
                for_update=True,
            )
            if run_row is None:
                return
            run_payload = from_json_addl(run_row)
            run_payload["status"] = "completed"
            run_payload["completed_at"] = utc_now_iso()
            run_payload["snapshot_captured_at"] = snapshot.get("captured_at")
            run_payload["error"] = None
            run_payload["updated_at"] = utc_now_iso()
            self.backend.update_instance_json(session, run_row, run_payload)

            for point in points:
                point_row = self.backend.create_instance(
                    session,
                    template_code=self.PRICING_POINT_TEMPLATE,
                    name=f"pricing-point-{point.get('region')}-{point.get('partition')}",
                    json_addl={
                        "captured_at": point.get("captured_at"),
                        "region": point.get("region"),
                        "availability_zone": point.get("availability_zone"),
                        "partition": point.get("partition"),
                        "instance_type": point.get("instance_type"),
                        "vcpu_count": int(point.get("vcpu_count") or 0),
                        "hourly_spot_price": float(point.get("hourly_spot_price") or 0.0),
                        "vcpu_cost_per_hour": float(point.get("vcpu_cost_per_hour") or 0.0),
                        "cluster_config_path": snapshot.get("cluster_config_path"),
                        "created_at": utc_now_iso(),
                        "updated_at": utc_now_iso(),
                    },
                )
                self.backend.create_lineage(
                    session,
                    parent=run_row,
                    child=point_row,
                    relationship_type="has_point",
                )

    def last_successful_pricing_capture(self) -> str | None:
        runs = self.list_pricing_runs(limit=200)
        for run in runs:
            if run.get("status") == "completed" and run.get("snapshot_captured_at"):
                return str(run["snapshot_captured_at"])
        return None

    def list_pricing_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self.backend.session_scope(commit=False) as session:
            rows = self.backend.list_instances_by_template(
                session,
                template_code=self.PRICING_RUN_TEMPLATE,
                limit=max(limit, 1),
            )
            runs = [self._to_pricing_run(row) for row in rows]
            runs.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
            return runs[:limit]

    def get_pricing_snapshot_payload(
        self,
        *,
        region: Optional[str] = None,
        partitions: Optional[Iterable[str]] = None,
        from_ts: Optional[str] = None,
        to_ts: Optional[str] = None,
    ) -> Dict[str, Any]:
        requested_partitions = {str(value).strip() for value in (partitions or []) if str(value).strip()}
        with self.backend.session_scope(commit=False) as session:
            rows = self.backend.list_instances_by_template(
                session,
                template_code=self.PRICING_POINT_TEMPLATE,
                limit=50000,
            )
            points = [from_json_addl(row) for row in rows]

        filtered: list[dict[str, Any]] = []
        for point in points:
            if region and str(point.get("region") or "") != region:
                continue
            if requested_partitions and str(point.get("partition") or "") not in requested_partitions:
                continue
            captured = str(point.get("captured_at") or "")
            if from_ts and captured < from_ts:
                continue
            if to_ts and captured > to_ts:
                continue
            filtered.append(point)

        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for point in filtered:
            captured_at = str(point.get("captured_at") or "")
            point_region = str(point.get("region") or "")
            key = (captured_at, point_region)
            snapshot = grouped.setdefault(
                key,
                {
                    "captured_at": captured_at,
                    "region": point_region,
                    "partitions": {},
                },
            )
            partition = str(point.get("partition") or "")
            az = str(point.get("availability_zone") or "")
            part = snapshot["partitions"].setdefault(partition, {"partition": partition, "availability_zones": {}})
            zone = part["availability_zones"].setdefault(az, {"availability_zone": az, "points": []})
            zone["points"].append(
                {
                    "instance_type": point.get("instance_type"),
                    "vcpu_count": point.get("vcpu_count"),
                    "hourly_spot_price": point.get("hourly_spot_price"),
                    "vcpu_cost_per_hour": point.get("vcpu_cost_per_hour"),
                }
            )

        snapshots: list[dict[str, Any]] = []
        for _, snapshot in sorted(grouped.items()):
            partitions_payload: list[dict[str, Any]] = []
            for partition_name, partition in sorted(snapshot["partitions"].items()):
                zones_payload: list[dict[str, Any]] = []
                for zone_name, zone in sorted(partition["availability_zones"].items()):
                    costs = [float(item.get("vcpu_cost_per_hour") or 0.0) for item in zone["points"]]
                    if costs:
                        sorted_costs = sorted(costs)
                        median = sorted_costs[len(sorted_costs) // 2]
                        box = {
                            "min": min(sorted_costs),
                            "q1": sorted_costs[max(0, len(sorted_costs) // 4)],
                            "median": median,
                            "q3": sorted_costs[min(len(sorted_costs) - 1, (len(sorted_costs) * 3) // 4)],
                            "max": max(sorted_costs),
                        }
                    else:
                        box = {"min": 0.0, "q1": 0.0, "median": 0.0, "q3": 0.0, "max": 0.0}
                    zones_payload.append(
                        {
                            "availability_zone": zone_name,
                            "points": zone["points"],
                            "box": box,
                        }
                    )
                partitions_payload.append({"partition": partition_name, "availability_zones": zones_payload})
            snapshots.append(
                {
                    "captured_at": snapshot["captured_at"],
                    "region": snapshot["region"],
                    "partitions": partitions_payload,
                }
            )

        latest_cheapest: list[dict[str, Any]] = []
        latest_by_region_partition: dict[tuple[str, str], dict[str, Any]] = {}
        for snapshot in snapshots:
            captured_at = str(snapshot.get("captured_at") or "")
            point_region = str(snapshot.get("region") or "")
            for partition in snapshot.get("partitions") or []:
                partition_name = str(partition.get("partition") or "")
                for zone in partition.get("availability_zones") or []:
                    median = float(((zone.get("box") or {}).get("median") or 0.0))
                    key = (point_region, partition_name)
                    current = latest_by_region_partition.get(key)
                    candidate = {
                        "captured_at": captured_at,
                        "region": point_region,
                        "partition": partition_name,
                        "availability_zone": zone.get("availability_zone"),
                        "median_vcpu_cost_per_hour": median,
                    }
                    if current is None:
                        latest_by_region_partition[key] = candidate
                        continue
                    if captured_at > str(current.get("captured_at") or ""):
                        latest_by_region_partition[key] = candidate
                        continue
                    if captured_at == str(current.get("captured_at") or "") and median < float(
                        current.get("median_vcpu_cost_per_hour") or 0.0
                    ):
                        latest_by_region_partition[key] = candidate

        for _, value in sorted(latest_by_region_partition.items()):
            latest_cheapest.append(value)

        return {
            "snapshots": snapshots,
            "latest_cheapest_az": latest_cheapest,
            "runs": self.list_pricing_runs(limit=5),
        }

    # ---------------------------------------------------------------------
    # Usage APIs (customer facing)
    # ---------------------------------------------------------------------

    def get_usage_summary(self, customer_id: str) -> dict[str, Any]:
        stats = self.get_dashboard_stats(customer_id)
        storage_gb = float(stats.get("dewey_artifact_size_gb") or 0.0)
        storage_cost = float(stats.get("dewey_storage_cost_usd") or 0.0)
        compute_cost = float(stats.get("compute_cost_usd") or 0.0)
        total = round(storage_cost + compute_cost, 4)
        worksets = self.list_worksets(customer_id, limit=2000)
        active = len([item for item in worksets if item.get("state") in {"ready", "running"}])
        return {
            "total_cost": total,
            "cost_change": 0.0,
            "compute_cost_usd": round(compute_cost, 4),
            "storage_cost_usd": round(storage_cost, 4),
            "transfer_cost_usd": 0.0,
            "transfer_intra_region_cost_usd": 0.0,
            "transfer_cross_region_cost_usd": 0.0,
            "transfer_internet_cost_usd": 0.0,
            "transfer_intra_region_gb": 0.0,
            "transfer_cross_region_gb": 0.0,
            "transfer_internet_gb": 0.0,
            "transfer_intra_region_rate_per_gb": 0.0,
            "transfer_cross_region_rate_per_gb": 0.02,
            "transfer_internet_rate_per_gb": 0.09,
            "workset_storage_human": str(stats.get("dewey_storage_human") or "0 B"),
            "storage_gb": round(storage_gb, 4),
            "active_worksets": active,
        }

    def get_usage_details(self, customer_id: str) -> list[dict[str, Any]]:
        details: list[dict[str, Any]] = []
        worksets = self.list_worksets(customer_id, limit=5000)
        for workset in worksets:
            state = str(workset.get("state") or "")
            if state not in {"complete", "running", "ready", "pending_cluster_creation"}:
                continue
            created = str(workset.get("created_at") or utc_now_iso())
            cost = 5.0 if state == "complete" else 0.5
            details.append(
                {
                    "date": created[:10],
                    "type": "Compute",
                    "workset_id": workset.get("workset_id"),
                    "workset_label": workset.get("workset_name"),
                    "quantity": 1,
                    "unit": "run",
                    "cost": cost,
                    "is_actual": False,
                }
            )

        total_bytes = 0
        if total_bytes > 0:
            storage_gb = total_bytes / (1024**3)
            details.append(
                {
                    "date": utc_now_iso()[:10],
                    "type": "Storage",
                    "workset_id": None,
                    "workset_label": "Dewey artifacts",
                    "quantity": round(storage_gb, 4),
                    "unit": "GB",
                    "cost": round(storage_gb * 0.023, 4),
                    "is_actual": False,
                }
            )

        details.sort(key=lambda item: str(item.get("date") or ""), reverse=True)
        return details

    def get_cost_history(self, customer_id: str, *, days: int = 30) -> dict[str, Any]:
        details = self.get_usage_details(customer_id)
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=max(days - 1, 0))

        by_day: dict[str, float] = {}
        for item in details:
            day = str(item.get("date") or "")
            by_day[day] = by_day.get(day, 0.0) + float(item.get("cost") or 0.0)

        labels: list[str] = []
        costs: list[float] = []
        for offset in range(days):
            current_day = start_date + timedelta(days=offset)
            key = current_day.isoformat()
            labels.append(current_day.strftime("%b %d"))
            costs.append(round(by_day.get(key, 0.0), 4))

        return {"labels": labels, "costs": costs}

    def get_cost_breakdown(self, customer_id: str) -> dict[str, Any]:
        usage = self.get_usage_summary(customer_id)
        compute = float(usage.get("compute_cost_usd") or 0.0)
        storage = float(usage.get("storage_cost_usd") or 0.0)
        transfer = float(usage.get("transfer_cost_usd") or 0.0)
        categories = ["Compute", "Storage", "Transfer"]
        values = [round(compute, 4), round(storage, 4), round(transfer, 4)]
        return {"categories": categories, "values": values}
