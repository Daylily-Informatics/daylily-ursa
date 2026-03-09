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
