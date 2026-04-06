from __future__ import annotations

import uuid

import pytest

from daylib_ursa.auth import AuthError
from daylib_ursa.auth import dependencies as auth_dependencies
from daylib_ursa.config import Settings


def test_claims_to_current_user_maps_canonical_cognito_groups() -> None:
    user = auth_dependencies._claims_to_current_user(
        {
            "sub": "user-123",
            "email": "ursa@example.com",
            "custom:customer_id": "00000000-0000-0000-0000-000000000001",
            "cognito:groups": ["platform-admin"],
        }
    )

    assert user.sub == "user-123"
    assert user.email == "ursa@example.com"
    assert user.tenant_id == uuid.UUID("00000000-0000-0000-0000-000000000001")
    assert user.roles == ["ADMIN"]
    assert user.is_admin is True


def test_cognito_auth_provider_accepts_id_token_claims(monkeypatch) -> None:
    monkeypatch.setattr(
        auth_dependencies,
        "decode_jwt_unverified",
        lambda _token: {"token_use": "id"},
    )
    monkeypatch.setattr(
        auth_dependencies.CognitoAuthProvider,
        "_verify_id_token_claims",
        lambda self, _token, access_token=None: {
            "sub": "user-123",
            "email": "ursa@example.com",
            "aud": "client-123",
            "custom:customer_id": "00000000-0000-0000-0000-000000000001",
            "cognito:groups": ["ursa-admin"],
        },
    )

    provider = auth_dependencies.CognitoAuthProvider(
        user_pool_id="pool-123",
        app_client_id="client-123",
        region="us-west-2",
    )

    user = provider.resolve_access_token("token-value")

    assert user.tenant_id == uuid.UUID("00000000-0000-0000-0000-000000000001")
    assert user.roles == ["ADMIN"]


def test_cognito_auth_provider_passes_paired_access_token_for_id_token_at_hash(monkeypatch) -> None:
    from jose import jwt

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        auth_dependencies,
        "decode_jwt_unverified",
        lambda _token: {"token_use": "id"},
    )
    monkeypatch.setattr(jwt, "get_unverified_header", lambda _token: {"kid": "kid-123"})

    def _decode(
        token,
        key,
        algorithms=None,
        options=None,
        audience=None,
        issuer=None,
        subject=None,
        access_token=None,
    ):
        captured["token"] = token
        captured["key"] = key
        captured["algorithms"] = algorithms
        captured["issuer"] = issuer
        captured["access_token"] = access_token
        return {
            "sub": "user-123",
            "email": "ursa@example.com",
            "aud": "client-123",
            "custom:customer_id": "00000000-0000-0000-0000-000000000001",
            "cognito:groups": ["ursa-admin"],
        }

    monkeypatch.setattr(jwt, "decode", _decode)

    provider = auth_dependencies.CognitoAuthProvider(
        user_pool_id="pool-123",
        app_client_id="client-123",
        region="us-west-2",
    )
    provider._jwks_cache = type("_Cache", (), {"get_key": lambda self, kid: f"key-for-{kid}"})()

    user = provider.resolve_access_token(
        "id-token-value",
        paired_access_token="access-token-value",
    )

    assert user.roles == ["ADMIN"]
    assert captured["token"] == "id-token-value"
    assert captured["access_token"] == "access-token-value"
    assert captured["issuer"] == "https://cognito-idp.us-west-2.amazonaws.com/pool-123"


def test_claims_to_current_user_maps_external_admin_group() -> None:
    user = auth_dependencies._claims_to_current_user(
        {
            "sub": "user-123",
            "email": "ursa@example.com",
            "custom:customer_id": "00000000-0000-0000-0000-000000000001",
            "cognito:groups": ["ursa-external-admin"],
        }
    )

    assert user.roles == ["EXTERNAL_USER_ADMIN"]


def test_settings_whitelist_domains_default_to_base_four() -> None:
    settings = Settings(
        ursa_internal_output_bucket="ursa-internal",
        cognito_domain="https://auth.example.com",
        cognito_app_client_id="client-1",
        cognito_callback_url="https://localhost:8914/auth/callback",
        cognito_logout_url="https://localhost:8914/login",
    )

    assert settings.get_whitelist_domains() == [
        "lsmc.com",
        "lsmc.bio",
        "lsmc.life",
        "daylilyinformatics.com",
    ]
    assert settings.is_domain_whitelisted("user@lsmc.bio") is True
    assert settings.is_domain_whitelisted("user@gmail.com") is False


def test_cognito_auth_provider_rejects_id_token_with_wrong_audience(monkeypatch) -> None:
    monkeypatch.setattr(
        auth_dependencies,
        "decode_jwt_unverified",
        lambda _token: {"token_use": "id"},
    )
    monkeypatch.setattr(
        auth_dependencies.CognitoAuthProvider,
        "_verify_id_token_claims",
        lambda self, _token, access_token=None: (_ for _ in ()).throw(
            AuthError("Invalid token audience")
        ),
    )

    provider = auth_dependencies.CognitoAuthProvider(
        user_pool_id="pool-123",
        app_client_id="client-123",
        region="us-west-2",
    )

    with pytest.raises(AuthError, match="Invalid token audience"):
        provider.resolve_access_token("token-value")


def test_user_directory_uses_cognito_groups_as_role_source() -> None:
    class _Client:
        def list_users(self, **_kwargs):
            return {
                "Users": [
                    {
                        "Username": "user-123",
                        "Enabled": True,
                        "UserStatus": "CONFIRMED",
                        "Attributes": [
                            {"Name": "sub", "Value": "user-123"},
                            {"Name": "email", "Value": "ursa@example.com"},
                            {
                                "Name": "custom:customer_id",
                                "Value": "00000000-0000-0000-0000-000000000001",
                            },
                            {"Name": "custom:roles", "Value": "READ_ONLY"},
                        ],
                    }
                ]
            }

        def admin_list_groups_for_user(self, **_kwargs):
            return {
                "Groups": [
                    {"GroupName": "platform-admin"},
                ]
            }

    directory = auth_dependencies.CognitoUserDirectoryService(
        user_pool_id="pool-123",
        region="us-west-2",
    )
    directory._client = _Client()

    users = directory.list_users(active_only=False, limit=10)

    assert len(users) == 1
    assert users[0].roles == ("ADMIN",)
