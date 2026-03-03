"""S3 bucket validation and linked-bucket management using TapDB graph storage."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

from daylib.tapdb_graph import TapDBBackend, from_json_addl, utc_now_iso


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class BucketValidationResult:
    bucket_name: str
    exists: bool = False
    accessible: bool = False
    can_read: bool = False
    can_write: bool = False
    can_list: bool = False
    region: Optional[str] = None
    owner_account: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    remediation_steps: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.exists and self.accessible and self.can_read and self.can_list

    @property
    def is_fully_configured(self) -> bool:
        return self.is_valid and self.can_write

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bucket_name": self.bucket_name,
            "exists": self.exists,
            "accessible": self.accessible,
            "can_read": self.can_read,
            "can_write": self.can_write,
            "can_list": self.can_list,
            "region": self.region,
            "owner_account": self.owner_account,
            "is_valid": self.is_valid,
            "is_fully_configured": self.is_fully_configured,
            "errors": self.errors,
            "warnings": self.warnings,
            "remediation_steps": self.remediation_steps,
        }


@dataclass
class LinkedBucket:
    bucket_id: str
    customer_id: str
    bucket_name: str
    bucket_type: str = "primary"
    display_name: Optional[str] = None
    description: Optional[str] = None
    is_validated: bool = False
    validation_timestamp: Optional[str] = None
    can_read: bool = False
    can_write: bool = False
    can_list: bool = False
    region: Optional[str] = None
    prefix_restriction: Optional[str] = None
    read_only: bool = False
    linked_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)
    bucket_euid: Optional[str] = None


class S3BucketValidator:
    """Validate S3 bucket configuration and permissions."""

    def __init__(self, region: str = "us-west-2", profile: Optional[str] = None):
        session_kwargs: Dict[str, Any] = {"region_name": region}
        if profile:
            session_kwargs["profile_name"] = profile
        session = boto3.Session(**session_kwargs)
        self.s3 = session.client("s3")
        self.sts = session.client("sts")
        self.region = region
        try:
            self.account_id = self.sts.get_caller_identity().get("Account")
        except Exception:
            self.account_id = None

    def validate_bucket(self, bucket_name: str, test_prefix: str = "daylily-validation-test/") -> BucketValidationResult:
        if bucket_name.startswith("s3://"):
            bucket_name = bucket_name[5:]
        result = BucketValidationResult(bucket_name=bucket_name)

        try:
            self.s3.head_bucket(Bucket=bucket_name)
            result.exists = True
            result.accessible = True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code == "404":
                result.errors.append(f"Bucket '{bucket_name}' does not exist")
            elif code == "403":
                result.exists = True
                result.errors.append(f"Access denied to bucket '{bucket_name}'")
            else:
                result.errors.append(f"Bucket check failed: {exc}")
            return result

        try:
            location = self.s3.get_bucket_location(Bucket=bucket_name)
            result.region = location.get("LocationConstraint") or "us-east-1"
        except ClientError:
            result.warnings.append("Could not determine bucket region")

        result.can_list = self._test_list(bucket_name, result)
        result.can_read = self._test_read(bucket_name, result)
        result.can_write = self._test_write(bucket_name, test_prefix, result)
        return result

    def _test_list(self, bucket_name: str, result: BucketValidationResult) -> bool:
        try:
            self.s3.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
            return True
        except ClientError as exc:
            result.errors.append(f"Cannot list bucket: {exc}")
            return False

    def _test_read(self, bucket_name: str, result: BucketValidationResult) -> bool:
        try:
            resp = self.s3.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
            items = resp.get("Contents") or []
            if items:
                self.s3.head_object(Bucket=bucket_name, Key=items[0]["Key"])
            return True
        except ClientError as exc:
            result.warnings.append(f"Read check uncertain: {exc}")
            return True

    def _test_write(self, bucket_name: str, test_prefix: str, result: BucketValidationResult) -> bool:
        key = f"{test_prefix.rstrip('/')}/daylily-permission-test.txt"
        try:
            self.s3.put_object(Bucket=bucket_name, Key=key, Body=b"daylily")
            self.s3.delete_object(Bucket=bucket_name, Key=key)
            return True
        except ClientError as exc:
            result.warnings.append(f"Write not available: {exc}")
            return False

    def generate_customer_bucket_policy(
        self,
        bucket_name: str,
        daylily_account_id: str,
        daylily_role_arn: Optional[str] = None,
    ) -> Dict[str, Any]:
        principal = {"AWS": daylily_role_arn} if daylily_role_arn else {"AWS": f"arn:aws:iam::{daylily_account_id}:root"}
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "DaylilyReadAccess",
                    "Effect": "Allow",
                    "Principal": principal,
                    "Action": ["s3:GetObject", "s3:GetObjectVersion", "s3:GetObjectTagging"],
                    "Resource": f"arn:aws:s3:::{bucket_name}/*",
                },
                {
                    "Sid": "DaylilyListAccess",
                    "Effect": "Allow",
                    "Principal": principal,
                    "Action": ["s3:ListBucket"],
                    "Resource": f"arn:aws:s3:::{bucket_name}",
                },
                {
                    "Sid": "DaylilyWriteResults",
                    "Effect": "Allow",
                    "Principal": principal,
                    "Action": ["s3:PutObject", "s3:PutObjectTagging", "s3:DeleteObject"],
                    "Resource": f"arn:aws:s3:::{bucket_name}/results/*",
                },
            ],
        }

    def get_setup_instructions(self, bucket_name: str, validation_result: BucketValidationResult) -> List[str]:
        steps: List[str] = []
        if not validation_result.exists:
            steps.append(f"Create bucket: aws s3 mb s3://{bucket_name}")
        if validation_result.exists and not validation_result.can_list:
            steps.append("Grant list permission on the bucket")
        if validation_result.exists and not validation_result.can_read:
            steps.append("Grant read permission for objects")
        if validation_result.exists and not validation_result.can_write:
            steps.append("Grant write permission for results prefix")
        return steps


class LinkedBucketManager:
    """Manage customer-linked buckets in TapDB graph storage."""

    CUSTOMER_TEMPLATE = "actor/customer/account/1.0/"
    BUCKET_TEMPLATE = "data/storage/s3-bucket-link/1.0/"

    def __init__(
        self,
        table_name: str = "tapdb-linked-buckets",
        region: str = "us-west-2",
        profile: Optional[str] = None,
    ):
        self.table_name = table_name
        self.region = region
        self.profile = profile
        self.backend = TapDBBackend(app_username="ursa-linked-bucket")
        self.validator = S3BucketValidator(region=region, profile=profile)

    def create_table_if_not_exists(self) -> None:
        with self.backend.session_scope(commit=True) as session:
            self.backend.ensure_templates(session)

    @staticmethod
    def _bucket_id(customer_id: str, bucket_name: str) -> str:
        return hashlib.sha256(f"{customer_id}:{bucket_name}".encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _to_bucket(payload: Dict[str, Any]) -> LinkedBucket:
        return LinkedBucket(
            bucket_id=payload["bucket_id"],
            customer_id=payload["customer_id"],
            bucket_name=payload["bucket_name"],
            bucket_type=payload.get("bucket_type", "primary"),
            display_name=payload.get("display_name"),
            description=payload.get("description"),
            is_validated=bool(payload.get("is_validated", False)),
            validation_timestamp=payload.get("validation_timestamp"),
            can_read=bool(payload.get("can_read", False)),
            can_write=bool(payload.get("can_write", False)),
            can_list=bool(payload.get("can_list", False)),
            region=payload.get("region"),
            prefix_restriction=payload.get("prefix_restriction"),
            read_only=bool(payload.get("read_only", False)),
            linked_at=payload.get("linked_at", _utc_now_iso()),
            updated_at=payload.get("updated_at", _utc_now_iso()),
            bucket_euid=payload.get("euid"),
        )

    def _ensure_customer(self, session, customer_id: str):
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
            )
        return customer

    def link_bucket(
        self,
        customer_id: str,
        bucket_name: str,
        *,
        bucket_type: str = "primary",
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        prefix_restriction: Optional[str] = None,
        read_only: bool = False,
        validation_result: Optional[BucketValidationResult] = None,
    ) -> LinkedBucket:
        bucket_id = self._bucket_id(customer_id, bucket_name)
        now = utc_now_iso()
        payload: Dict[str, Any] = {
            "bucket_id": bucket_id,
            "customer_id": customer_id,
            "bucket_name": bucket_name,
            "bucket_type": bucket_type,
            "display_name": display_name,
            "description": description,
            "prefix_restriction": prefix_restriction,
            "read_only": bool(read_only),
            "linked_at": now,
            "updated_at": now,
            "is_validated": False,
            "validation_timestamp": None,
            "can_read": False,
            "can_write": False,
            "can_list": False,
            "region": None,
        }

        if validation_result is not None:
            payload.update(
                {
                    "is_validated": True,
                    "validation_timestamp": now,
                    "can_read": validation_result.can_read,
                    "can_write": validation_result.can_write,
                    "can_list": validation_result.can_list,
                    "region": validation_result.region,
                }
            )

        with self.backend.session_scope(commit=True) as session:
            customer = self._ensure_customer(session, customer_id)
            existing = self.backend.find_instance_by_external_id(
                session,
                template_code=self.BUCKET_TEMPLATE,
                key="bucket_id",
                value=bucket_id,
            )
            if existing is None:
                bucket_row = self.backend.create_instance(
                    session,
                    template_code=self.BUCKET_TEMPLATE,
                    name=display_name or bucket_name,
                    json_addl=payload,
                    bstatus="active",
                )
                self.backend.create_lineage(session, parent=customer, child=bucket_row, relationship_type="owns")
                payload["bucket_euid"] = bucket_row.euid
            else:
                self.backend.update_instance_json(session, existing, payload)
                payload["bucket_euid"] = existing.euid

        return self._to_bucket(payload)

    def get_linked_bucket(self, bucket_id: str) -> Optional[LinkedBucket]:
        with self.backend.session_scope(commit=False) as session:
            row = self.backend.find_instance_by_external_id(
                session,
                template_code=self.BUCKET_TEMPLATE,
                key="bucket_id",
                value=bucket_id,
            )
            if row is None:
                return None
            return self._to_bucket(from_json_addl(row))

    def get_bucket(self, bucket_id: str) -> Optional[LinkedBucket]:
        return self.get_linked_bucket(bucket_id)

    def list_customer_buckets(self, customer_id: str) -> List[LinkedBucket]:
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
                template_code=self.BUCKET_TEMPLATE,
                relationship_type="owns",
                limit=10000,
            )
            return [self._to_bucket(from_json_addl(row)) for row in rows]

    def unlink_bucket(self, bucket_id: str) -> bool:
        with self.backend.session_scope(commit=True) as session:
            row = self.backend.find_instance_by_external_id(
                session,
                template_code=self.BUCKET_TEMPLATE,
                key="bucket_id",
                value=bucket_id,
            )
            if row is None:
                return False
            row.is_deleted = True
            row.bstatus = "deleted"
            session.flush()
            return True

    def update_bucket(self, bucket_id: str, **updates: Any) -> Optional[LinkedBucket]:
        with self.backend.session_scope(commit=True) as session:
            row = self.backend.find_instance_by_external_id(
                session,
                template_code=self.BUCKET_TEMPLATE,
                key="bucket_id",
                value=bucket_id,
            )
            if row is None:
                return None
            payload = from_json_addl(row)
            payload.update({k: v for k, v in updates.items() if v is not None})
            payload["updated_at"] = utc_now_iso()
            self.backend.update_instance_json(session, row, payload)
            return self._to_bucket(payload)

    def revalidate_bucket(
        self,
        bucket_id: str,
    ) -> tuple[Optional[LinkedBucket], Optional[BucketValidationResult]]:
        bucket = self.get_bucket(bucket_id)
        if bucket is None:
            return None, None

        validation = self.validator.validate_bucket(bucket.bucket_name)
        updated = self.update_bucket(
            bucket_id,
            is_validated=True,
            validation_timestamp=utc_now_iso(),
            can_read=validation.can_read,
            can_write=validation.can_write,
            can_list=validation.can_list,
            region=validation.region,
        )
        return updated, validation
