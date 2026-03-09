"""Customer onboarding helpers for post-auth portal entry."""

from __future__ import annotations

import re
import secrets
from typing import Any

import boto3

from daylib_ursa.config import Settings
from daylib_ursa.s3_bucket_validator import LinkedBucketManager


class OnboardingError(RuntimeError):
    """Raised when required post-login onboarding cannot be completed."""


def _normalize_bucket_name(raw: str) -> str:
    bucket = str(raw or "").strip()
    if bucket.startswith("s3://"):
        bucket = bucket[5:]
    if "/" in bucket:
        bucket = bucket.split("/", 1)[0]
    return bucket.strip().lower()


def _slug_customer_id(customer_id: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", customer_id.lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug or "customer"


def _generate_bucket_name(customer_id: str, *, account_id: str, region: str) -> str:
    customer_slug = _slug_customer_id(customer_id)
    account_tail = (account_id or "000000")[-6:]
    suffix = secrets.token_hex(3)
    prefix = f"ursa-{region}-{account_tail}-"
    max_customer_len = max(1, 63 - len(prefix) - len(suffix) - 1)
    customer_part = customer_slug[:max_customer_len].strip("-") or "customer"
    return f"{prefix}{customer_part}-{suffix}".lower()


def _resolve_account_id(*, profile: str | None, region: str) -> str:
    kwargs: dict[str, Any] = {"region_name": region}
    if profile:
        kwargs["profile_name"] = profile
    session = boto3.Session(**kwargs)
    sts = session.client("sts", region_name=region)
    return str(sts.get_caller_identity().get("Account") or "")


def _create_bucket(*, bucket_name: str, profile: str | None, region: str) -> None:
    kwargs: dict[str, Any] = {"region_name": region}
    if profile:
        kwargs["profile_name"] = profile
    session = boto3.Session(**kwargs)
    s3 = session.client("s3", region_name=region)
    create_args: dict[str, Any] = {"Bucket": bucket_name}
    if region != "us-east-1":
        create_args["CreateBucketConfiguration"] = {"LocationConstraint": region}
    s3.create_bucket(**create_args)
    try:
        s3.put_bucket_tagging(
            Bucket=bucket_name,
            Tagging={
                "TagSet": [
                    {"Key": "ManagedBy", "Value": "Ursa"},
                    {"Key": "ProvisionedBy", "Value": "PortalOnboarding"},
                ]
            },
        )
    except Exception:
        # Tagging failures should not block onboarding if bucket exists.
        pass


def ensure_customer_onboarding(*, identity: dict[str, Any], settings: Settings) -> dict[str, Any]:
    """Ensure authenticated user has required customer/bucket onboarding."""
    updated = dict(identity)
    external_customer_id = str(updated.get("customer_id") or settings.ursa_portal_default_customer_id).strip()
    if not external_customer_id:
        raise OnboardingError("Missing customer_id for portal onboarding")
    updated["customer_id"] = external_customer_id
    updated.setdefault("customer_name", external_customer_id)

    region = settings.get_effective_region()
    manager = LinkedBucketManager(region=region, profile=settings.aws_profile)
    manager.bootstrap()

    existing_buckets = manager.list_customer_buckets(external_customer_id)
    if existing_buckets:
        primary = next((bucket for bucket in existing_buckets if bucket.bucket_type == "primary"), None)
        chosen = primary or existing_buckets[0]
        updated["s3_bucket"] = chosen.bucket_name
        canonical_customer_id = (
            manager.resolve_customer_euid(external_customer_id)
            if hasattr(manager, "resolve_customer_euid")
            else external_customer_id
        )
        if canonical_customer_id != external_customer_id:
            updated["legacy_customer_id"] = external_customer_id
        updated["customer_id"] = canonical_customer_id
        return updated

    requested_bucket = _normalize_bucket_name(str(updated.get("s3_bucket") or ""))
    if requested_bucket:
        manager.link_bucket(external_customer_id, requested_bucket, bucket_type="primary", validate=False)
        updated["s3_bucket"] = requested_bucket
        canonical_customer_id = (
            manager.resolve_customer_euid(external_customer_id)
            if hasattr(manager, "resolve_customer_euid")
            else external_customer_id
        )
        if canonical_customer_id != external_customer_id:
            updated["legacy_customer_id"] = external_customer_id
        updated["customer_id"] = canonical_customer_id
        return updated

    account_id = _resolve_account_id(profile=settings.aws_profile, region=region)
    last_error: Exception | None = None
    for _ in range(5):
        candidate = _generate_bucket_name(external_customer_id, account_id=account_id, region=region)
        try:
            _create_bucket(bucket_name=candidate, profile=settings.aws_profile, region=region)
            manager.link_bucket(external_customer_id, candidate, bucket_type="primary", validate=False)
            updated["s3_bucket"] = candidate
            canonical_customer_id = (
                manager.resolve_customer_euid(external_customer_id)
                if hasattr(manager, "resolve_customer_euid")
                else external_customer_id
            )
            if canonical_customer_id != external_customer_id:
                updated["legacy_customer_id"] = external_customer_id
            updated["customer_id"] = canonical_customer_id
            return updated
        except Exception as exc:
            last_error = exc
            if "BucketAlreadyExists" in str(exc) or "BucketAlreadyOwnedByYou" in str(exc):
                continue
            break

    raise OnboardingError(
        f"Failed to provision onboarding bucket for customer '{external_customer_id}': {last_error}"
    )
