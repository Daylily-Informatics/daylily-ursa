from __future__ import annotations

import uuid

import pytest

from daylib_ursa.auth import AuthError
from daylib_ursa.auth import dependencies as auth_dependencies


def test_claims_to_current_user_accepts_customer_id_and_groups() -> None:
    user = auth_dependencies._claims_to_current_user(
        {
            "sub": "user-123",
            "email": "ursa@example.com",
            "custom:customer_id": "00000000-0000-0000-0000-000000000001",
            "cognito:groups": ["admin"],
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
        lambda self, _token: {
            "sub": "user-123",
            "email": "ursa@example.com",
            "aud": "client-123",
            "custom:customer_id": "00000000-0000-0000-0000-000000000001",
            "cognito:groups": ["admin"],
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


def test_cognito_auth_provider_rejects_id_token_with_wrong_audience(monkeypatch) -> None:
    monkeypatch.setattr(
        auth_dependencies,
        "decode_jwt_unverified",
        lambda _token: {"token_use": "id"},
    )
    monkeypatch.setattr(
        auth_dependencies.CognitoAuthProvider,
        "_verify_id_token_claims",
        lambda self, _token: (_ for _ in ()).throw(AuthError("Invalid token audience")),
    )

    provider = auth_dependencies.CognitoAuthProvider(
        user_pool_id="pool-123",
        app_client_id="client-123",
        region="us-west-2",
    )

    with pytest.raises(AuthError, match="Invalid token audience"):
        provider.resolve_access_token("token-value")
