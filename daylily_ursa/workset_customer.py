"""Customer onboarding and management backed by TapDB graph objects."""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
import secrets
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

from daylily_ursa.tapdb_graph import TapDBBackend, from_json_addl, utc_now_iso

LOGGER = logging.getLogger("daylily.workset_customer")


@dataclass
class CustomerConfig:
    """Customer configuration."""

    customer_id: str
    customer_name: str
    email: str
    s3_bucket: str
    max_concurrent_worksets: int = 5
    max_storage_gb: int = 1000
    billing_account_id: Optional[str] = None
    cost_center: Optional[str] = None
    is_admin: bool = False
    bucket_region: Optional[str] = None
    api_tokens: List[Dict[str, Any]] = field(default_factory=list)
    customer_euid: Optional[str] = None


class CustomerManager:
    """Manage customer accounts and resources using TapDB graph objects."""

    CUSTOMER_TEMPLATE = "actor/customer/account/1.0/"

    def __init__(
        self,
        region: str,
        profile: Optional[str] = None,
        bucket_prefix: str = "daylily-customer",
    ):
        session_kwargs: Dict[str, Any] = {"region_name": region}
        if profile:
            session_kwargs["profile_name"] = profile

        self._session = boto3.Session(**session_kwargs)
        self.s3 = self._session.client("s3")
        self.region = region
        self.bucket_prefix = bucket_prefix
        self.profile = profile
        self.backend = TapDBBackend(app_username="ursa-customer")

    def bootstrap(self) -> None:
        """Ensure TapDB templates exist for customer graph objects."""
        with self.backend.session_scope(commit=True) as session:
            self.backend.ensure_templates(session)

    @staticmethod
    def _hash_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _token_id(name: str) -> str:
        return hashlib.sha256(f"{name}:{utc_now_iso()}".encode("utf-8")).hexdigest()[:12]

    def _generate_customer_id(self, customer_name: str) -> str:
        name_part = customer_name.lower().replace(" ", "-")[:20]
        random_part = secrets.token_hex(4)
        return f"{name_part}-{random_part}"

    def _create_customer_bucket(
        self,
        bucket_name: str,
        customer_id: str,
        cost_center: Optional[str],
        bucket_region: Optional[str] = None,
    ) -> None:
        target_region = bucket_region or self.region
        s3_client = self._session.client("s3", region_name=target_region)

        try:
            if target_region == "us-east-1":
                s3_client.create_bucket(Bucket=bucket_name)
            else:
                s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={"LocationConstraint": target_region},
                )
            tags = [
                {"Key": "Customer", "Value": customer_id},
                {"Key": "ManagedBy", "Value": "daylily-ursa"},
            ]
            if cost_center:
                tags.append({"Key": "CostCenter", "Value": cost_center})
            s3_client.put_bucket_tagging(Bucket=bucket_name, Tagging={"TagSet": tags})
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code not in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
                raise

    @staticmethod
    def _to_config(payload: Dict[str, Any]) -> CustomerConfig:
        return CustomerConfig(
            customer_id=payload["customer_id"],
            customer_name=payload.get("customer_name") or payload.get("name") or payload["customer_id"],
            email=payload.get("email", ""),
            s3_bucket=payload.get("s3_bucket", ""),
            max_concurrent_worksets=int(payload.get("max_concurrent_worksets", 5)),
            max_storage_gb=int(payload.get("max_storage_gb", 1000)),
            billing_account_id=payload.get("billing_account_id"),
            cost_center=payload.get("cost_center"),
            is_admin=bool(payload.get("is_admin", False)),
            bucket_region=payload.get("bucket_region"),
            api_tokens=list(payload.get("api_tokens", [])),
            customer_euid=payload.get("euid"),
        )

    def _get_customer_instance(self, customer_id: str):
        with self.backend.session_scope(commit=False) as session:
            return self.backend.find_instance_by_external_id(
                session,
                template_code=self.CUSTOMER_TEMPLATE,
                key="customer_id",
                value=customer_id,
            )

    def onboard_customer(
        self,
        customer_name: str,
        email: str,
        max_concurrent_worksets: int = 5,
        max_storage_gb: int = 1000,
        billing_account_id: Optional[str] = None,
        cost_center: Optional[str] = None,
        custom_s3_bucket: Optional[str] = None,
        bucket_region: Optional[str] = None,
    ) -> CustomerConfig:
        existing = self.get_customer_by_email(email)
        if existing is not None:
            return existing

        customer_id = self._generate_customer_id(customer_name)
        effective_bucket_region = bucket_region or self.region

        if custom_s3_bucket:
            bucket_name = custom_s3_bucket
        else:
            bucket_name = f"{self.bucket_prefix}-{customer_id}"
            self._create_customer_bucket(bucket_name, customer_id, cost_center, effective_bucket_region)

        payload: Dict[str, Any] = {
            "customer_id": customer_id,
            "customer_name": customer_name,
            "email": email,
            "s3_bucket": bucket_name,
            "max_concurrent_worksets": max_concurrent_worksets,
            "max_storage_gb": max_storage_gb,
            "billing_account_id": billing_account_id,
            "cost_center": cost_center,
            "bucket_region": effective_bucket_region,
            "is_admin": False,
            "api_tokens": [],
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
        }

        with self.backend.session_scope(commit=True) as session:
            self.backend.ensure_templates(session)
            self.backend.create_instance(
                session,
                template_code=self.CUSTOMER_TEMPLATE,
                name=customer_name,
                json_addl=payload,
                bstatus="active",
            )

        return self._to_config(payload)

    def _list_customer_payloads(self) -> List[Dict[str, Any]]:
        with self.backend.session_scope(commit=False) as session:
            rows = self.backend.list_instances_by_template(
                session,
                template_code=self.CUSTOMER_TEMPLATE,
                limit=10000,
            )
            return [from_json_addl(row) for row in rows]

    def get_customer_config(self, customer_id: str) -> Optional[CustomerConfig]:
        with self.backend.session_scope(commit=False) as session:
            row = self.backend.find_instance_by_external_id(
                session,
                template_code=self.CUSTOMER_TEMPLATE,
                key="customer_id",
                value=customer_id,
            )
            if row is None:
                return None
            return self._to_config(from_json_addl(row))

    def get_customer_by_email(self, email: str) -> Optional[CustomerConfig]:
        email_norm = email.strip().lower()
        for payload in self._list_customer_payloads():
            if str(payload.get("email", "")).strip().lower() == email_norm:
                return self._to_config(payload)
        return None

    def list_customers(self) -> List[CustomerConfig]:
        return [self._to_config(p) for p in self._list_customer_payloads()]

    def update_customer(
        self,
        customer_id: str,
        **updates: Any,
    ) -> Optional[CustomerConfig]:
        with self.backend.session_scope(commit=True) as session:
            row = self.backend.find_instance_by_external_id(
                session,
                template_code=self.CUSTOMER_TEMPLATE,
                key="customer_id",
                value=customer_id,
            )
            if row is None:
                return None
            payload = from_json_addl(row)
            payload.update({k: v for k, v in updates.items() if v is not None})
            payload["updated_at"] = utc_now_iso()
            row.name = payload.get("customer_name", row.name)
            self.backend.update_instance_json(session, row, payload)
            return self._to_config(payload)

    def set_admin_status(self, customer_id: str, is_admin: bool) -> bool:
        updated = self.update_customer(customer_id, is_admin=bool(is_admin))
        return updated is not None

    def get_customer_usage(self, customer_id: str) -> Dict[str, Any]:
        with self.backend.session_scope(commit=False) as session:
            customer = self.backend.find_instance_by_external_id(
                session,
                template_code=self.CUSTOMER_TEMPLATE,
                key="customer_id",
                value=customer_id,
            )
            if customer is None:
                return {
                    "customer_id": customer_id,
                    "workset_count": 0,
                    "active_worksets": 0,
                    "storage_gb": 0.0,
                    "max_storage_gb": 0,
                    "max_concurrent_worksets": 0,
                }
            worksets = self.backend.get_customer_owned(
                session,
                customer=customer,
                template_code="workflow/workset/analysis/1.0/",
                relationship_type="owns",
                limit=100000,
            )
            active_states = {"ready", "in_progress", "retrying"}
            active = [ws for ws in worksets if (ws.bstatus or "") in active_states]
            storage_bytes = 0
            for ws in worksets:
                payload = dict(ws.json_addl or {})
                metrics = payload.get("storage_metrics") or {}
                storage_bytes += int(metrics.get("results_storage_bytes", 0) or 0)
            cfg = self._to_config(from_json_addl(customer))
            return {
                "customer_id": customer_id,
                "workset_count": len(worksets),
                "active_worksets": len(active),
                "storage_gb": round(storage_bytes / (1024 ** 3), 3),
                "max_storage_gb": cfg.max_storage_gb,
                "max_concurrent_worksets": cfg.max_concurrent_worksets,
            }

    def list_api_tokens(self, customer_id: str) -> List[Dict[str, Any]]:
        cfg = self.get_customer_config(customer_id)
        if cfg is None:
            return []
        tokens = []
        for item in cfg.api_tokens:
            view = dict(item)
            view.pop("token_hash", None)
            tokens.append(view)
        return tokens

    def add_api_token(self, customer_id: str, name: str, expiry_days: int) -> Dict[str, Any]:
        raw_token = secrets.token_urlsafe(32)
        now = dt.datetime.now(dt.timezone.utc)
        expires_at = (now + dt.timedelta(days=max(expiry_days, 1))).isoformat().replace("+00:00", "Z")
        token_id = self._token_id(name)
        token_rec = {
            "id": token_id,
            "name": name,
            "token_hash": self._hash_token(raw_token),
            "created_at": now.isoformat().replace("+00:00", "Z"),
            "expires_at": expires_at,
            "revoked": False,
        }
        cfg = self.get_customer_config(customer_id)
        if cfg is None:
            raise ValueError("Customer not found")
        tokens = list(cfg.api_tokens)
        tokens.append(token_rec)
        updated = self.update_customer(customer_id, api_tokens=tokens)
        if updated is None:
            raise ValueError("Failed to update customer token set")
        return {
            "id": token_id,
            "name": name,
            "token": raw_token,
            "created_at": token_rec["created_at"],
            "expires_at": expires_at,
        }

    def revoke_api_token(self, customer_id: str, token_id: str) -> bool:
        cfg = self.get_customer_config(customer_id)
        if cfg is None:
            return False
        changed = False
        tokens = []
        for rec in cfg.api_tokens:
            item = dict(rec)
            if item.get("id") == token_id and not bool(item.get("revoked", False)):
                item["revoked"] = True
                changed = True
            tokens.append(item)
        if not changed:
            return False
        return self.update_customer(customer_id, api_tokens=tokens) is not None

    def get_customer_by_api_key(self, api_key: str) -> Optional[CustomerConfig]:
        hashed = self._hash_token(api_key)
        now = dt.datetime.now(dt.timezone.utc)
        for cfg in self.list_customers():
            for token in cfg.api_tokens:
                if token.get("token_hash") != hashed:
                    continue
                if bool(token.get("revoked", False)):
                    continue
                expires_raw = token.get("expires_at")
                if expires_raw:
                    try:
                        expires_at = dt.datetime.fromisoformat(str(expires_raw).replace("Z", "+00:00"))
                        if expires_at <= now:
                            continue
                    except ValueError:
                        continue
                return cfg
        return None
