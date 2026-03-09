from __future__ import annotations

from types import SimpleNamespace

from daylib_ursa.config import Settings
from daylib_ursa.portal_onboarding import ensure_customer_onboarding


def _settings() -> Settings:
    return Settings(
        cors_origins="*",
        ursa_internal_api_key="test-key",
        bloom_base_url="https://bloom.example",
        atlas_base_url="https://atlas.example",
        aws_profile="lsmc",
    )


def test_onboarding_uses_existing_primary_bucket(monkeypatch):
    class FakeManager:
        def __init__(self, region: str, profile: str | None = None):
            self.region = region
            self.profile = profile

        def bootstrap(self) -> None:
            return None

        def list_customer_buckets(self, customer_id: str):
            return [SimpleNamespace(bucket_name="existing-bucket", bucket_type="primary")]

    monkeypatch.setattr("daylib_ursa.portal_onboarding.LinkedBucketManager", FakeManager)
    identity = {
        "customer_id": "lsmc-main",
        "customer_name": "LSMC Main",
        "user_email": "jmajor@lsmc.bio",
        "logged_in": True,
    }
    updated = ensure_customer_onboarding(identity=identity, settings=_settings())
    assert updated["s3_bucket"] == "existing-bucket"


def test_onboarding_canonicalizes_customer_id_to_euid(monkeypatch):
    class FakeManager:
        def __init__(self, region: str, profile: str | None = None):
            self.region = region
            self.profile = profile

        def bootstrap(self) -> None:
            return None

        def resolve_customer_euid(self, customer_id: str) -> str:
            assert customer_id == "default-customer"
            return "actor-customer-euid-1234"

        def list_customer_buckets(self, customer_id: str):
            return [SimpleNamespace(bucket_name="existing-bucket", bucket_type="primary")]

    monkeypatch.setattr("daylib_ursa.portal_onboarding.LinkedBucketManager", FakeManager)
    identity = {
        "customer_id": "default-customer",
        "customer_name": "Default Customer",
        "user_email": "jmajor@lsmc.bio",
        "logged_in": True,
    }
    updated = ensure_customer_onboarding(identity=identity, settings=_settings())
    assert updated["customer_id"] == "actor-customer-euid-1234"
    assert updated["legacy_customer_id"] == "default-customer"
    assert updated["s3_bucket"] == "existing-bucket"


def test_onboarding_prefers_existing_primary_bucket_over_identity_bucket(monkeypatch):
    class FakeManager:
        def __init__(self, region: str, profile: str | None = None):
            self.region = region
            self.profile = profile

        def bootstrap(self) -> None:
            return None

        def list_customer_buckets(self, customer_id: str):
            return [SimpleNamespace(bucket_name="linked-primary", bucket_type="primary")]

    monkeypatch.setattr("daylib_ursa.portal_onboarding.LinkedBucketManager", FakeManager)
    identity = {
        "customer_id": "lsmc-main",
        "customer_name": "LSMC Main",
        "user_email": "jmajor@lsmc.bio",
        "s3_bucket": "identity-only-bucket",
        "logged_in": True,
    }
    updated = ensure_customer_onboarding(identity=identity, settings=_settings())
    assert updated["s3_bucket"] == "linked-primary"


def test_onboarding_links_identity_bucket_when_no_linked_bucket(monkeypatch):
    events: list[tuple[str, str]] = []

    class FakeManager:
        def __init__(self, region: str, profile: str | None = None):
            self.region = region
            self.profile = profile

        def bootstrap(self) -> None:
            return None

        def list_customer_buckets(self, customer_id: str):
            return []

        def link_bucket(self, customer_id: str, bucket_name: str, **kwargs):
            events.append(("link", bucket_name))

    monkeypatch.setattr("daylib_ursa.portal_onboarding.LinkedBucketManager", FakeManager)
    identity = {
        "customer_id": "lsmc-main",
        "customer_name": "LSMC Main",
        "user_email": "jmajor@lsmc.bio",
        "s3_bucket": "s3://Provided-Bucket/path/ignored",
        "logged_in": True,
    }
    updated = ensure_customer_onboarding(identity=identity, settings=_settings())
    assert updated["s3_bucket"] == "provided-bucket"
    assert events == [("link", "provided-bucket")]


def test_onboarding_provisions_bucket_when_missing(monkeypatch):
    events: list[tuple[str, str]] = []

    class FakeManager:
        def __init__(self, region: str, profile: str | None = None):
            self.region = region
            self.profile = profile

        def bootstrap(self) -> None:
            events.append(("bootstrap", "ok"))

        def list_customer_buckets(self, customer_id: str):
            return []

        def link_bucket(self, customer_id: str, bucket_name: str, **kwargs):
            events.append(("link", bucket_name))

    monkeypatch.setattr("daylib_ursa.portal_onboarding.LinkedBucketManager", FakeManager)
    monkeypatch.setattr("daylib_ursa.portal_onboarding._resolve_account_id", lambda profile, region: "123456789012")
    monkeypatch.setattr("daylib_ursa.portal_onboarding._generate_bucket_name", lambda customer_id, account_id, region: "ursa-auto-generated-bucket")
    monkeypatch.setattr(
        "daylib_ursa.portal_onboarding._create_bucket",
        lambda bucket_name, profile, region: events.append(("create", bucket_name)),
    )

    identity = {
        "customer_id": "lsmc-main",
        "customer_name": "LSMC Main",
        "user_email": "jmajor@lsmc.bio",
        "logged_in": True,
    }
    updated = ensure_customer_onboarding(identity=identity, settings=_settings())
    assert updated["s3_bucket"] == "ursa-auto-generated-bucket"
    assert ("create", "ursa-auto-generated-bucket") in events
    assert ("link", "ursa-auto-generated-bucket") in events


def test_onboarding_provisioning_is_idempotent(monkeypatch):
    events: list[tuple[str, str]] = []
    linked_buckets: list[str] = []

    class FakeManager:
        def __init__(self, region: str, profile: str | None = None):
            self.region = region
            self.profile = profile

        def bootstrap(self) -> None:
            return None

        def list_customer_buckets(self, customer_id: str):
            return [
                SimpleNamespace(bucket_name=bucket_name, bucket_type="primary")
                for bucket_name in linked_buckets
            ]

        def link_bucket(self, customer_id: str, bucket_name: str, **kwargs):
            linked_buckets.append(bucket_name)
            events.append(("link", bucket_name))

    monkeypatch.setattr("daylib_ursa.portal_onboarding.LinkedBucketManager", FakeManager)
    monkeypatch.setattr("daylib_ursa.portal_onboarding._resolve_account_id", lambda profile, region: "123456789012")
    monkeypatch.setattr(
        "daylib_ursa.portal_onboarding._generate_bucket_name",
        lambda customer_id, account_id, region: "ursa-auto-generated-bucket",
    )
    monkeypatch.setattr(
        "daylib_ursa.portal_onboarding._create_bucket",
        lambda bucket_name, profile, region: events.append(("create", bucket_name)),
    )

    identity = {
        "customer_id": "lsmc-main",
        "customer_name": "LSMC Main",
        "user_email": "jmajor@lsmc.bio",
        "logged_in": True,
    }
    first = ensure_customer_onboarding(identity=identity, settings=_settings())
    second = ensure_customer_onboarding(identity=identity, settings=_settings())

    assert first["s3_bucket"] == "ursa-auto-generated-bucket"
    assert second["s3_bucket"] == "ursa-auto-generated-bucket"
    assert events.count(("create", "ursa-auto-generated-bucket")) == 1
    assert events.count(("link", "ursa-auto-generated-bucket")) == 1
