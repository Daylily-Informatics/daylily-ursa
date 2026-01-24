"""Tests for portal account/security UX endpoints (Phase 0.5)."""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from daylib.workset_api import create_app
from daylib.workset_state_db import WorksetStateDB


def _make_state_db() -> MagicMock:
    return MagicMock(spec=WorksetStateDB)


def test_change_password_challenge_rejects_short_password() -> None:
    cognito_auth = MagicMock()
    cognito_auth.authenticate.return_value = {"challenge": "NEW_PASSWORD_REQUIRED", "session": "sess"}

    customer_manager = MagicMock()
    customer = MagicMock()
    customer.customer_id = "cust1"
    customer.is_admin = False
    customer_manager.get_customer_by_email.return_value = customer

    app = create_app(
        state_db=_make_state_db(),
        enable_auth=True,
        cognito_auth=cognito_auth,
        customer_manager=customer_manager,
    )
    client = TestClient(app)

    login = client.post(
        "/portal/login",
        data={"email": "user@example.com", "password": "temp"},
        follow_redirects=False,
    )
    assert login.status_code == 302
    assert login.headers["location"].startswith("/portal/change-password")

    resp = client.post(
        "/portal/change-password",
        data={"new_password": "1234567", "confirm_password": "1234567"},
        follow_redirects=False,
    )
    assert resp.status_code == 302
    assert "Password+must+be+at+least+8+characters" in resp.headers["location"]
    cognito_auth.respond_to_new_password_challenge.assert_not_called()


def test_reset_password_rejects_short_password() -> None:
    cognito_auth = MagicMock()
    customer_manager = MagicMock()

    app = create_app(
        state_db=_make_state_db(),
        enable_auth=True,
        cognito_auth=cognito_auth,
        customer_manager=customer_manager,
    )
    client = TestClient(app)

    resp = client.post(
        "/portal/reset-password",
        data={"email": "user@example.com", "code": "123456", "password": "1234567", "confirm_password": "1234567"},
        follow_redirects=False,
    )
    assert resp.status_code == 302
    assert "Password+must+be+at+least+8+characters" in resp.headers["location"]
    cognito_auth.confirm_forgot_password.assert_not_called()


def test_api_change_password_returns_user_friendly_error_on_short_password() -> None:
    cognito_auth = MagicMock()
    cognito_auth.authenticate.return_value = {"access_token": "at", "id_token": "it"}

    customer_manager = MagicMock()
    customer = MagicMock()
    customer.customer_id = "cust1"
    customer.is_admin = False
    customer_manager.get_customer_by_email.return_value = customer

    app = create_app(
        state_db=_make_state_db(),
        enable_auth=True,
        cognito_auth=cognito_auth,
        customer_manager=customer_manager,
    )
    client = TestClient(app)

    client.post(
        "/portal/login",
        data={"email": "user@example.com", "password": "pass"},
        follow_redirects=False,
    )

    resp = client.post(
        "/api/v1/auth/change-password",
        json={"current_password": "old", "new_password": "1234567"},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Password must be at least 8 characters"


def test_api_tokens_endpoints_work_for_authenticated_session() -> None:
    customer_manager = MagicMock()
    customer = MagicMock()
    customer.customer_id = "cust1"
    customer.is_admin = False
    customer_manager.get_customer_by_email.return_value = customer

    customer_manager.list_api_tokens.return_value = [
        {"id": "t1", "name": "Token 1", "created_at": "2024-01-01T00:00:00Z", "expires_at": None, "revoked": False},
    ]
    customer_manager.add_api_token.return_value = {
        "secret": "sekret",
        "token": {"id": "t2", "name": "New", "created_at": "2024-01-01T00:00:00Z", "expires_at": None},
    }
    customer_manager.revoke_api_token.return_value = True

    app = create_app(
        state_db=_make_state_db(),
        enable_auth=False,
        customer_manager=customer_manager,
    )
    client = TestClient(app)

    client.post(
        "/portal/login",
        data={"email": "user@example.com", "password": "pass"},
        follow_redirects=False,
    )

    listed = client.get("/api/v1/auth/tokens")
    assert listed.status_code == 200
    assert listed.json()[0]["id"] == "t1"

    created = client.post("/api/v1/auth/tokens", json={"name": "New", "expiry_days": 0})
    assert created.status_code == 200
    assert created.json()["token"] == "sekret"

    revoked = client.delete("/api/v1/auth/tokens/t2")
    assert revoked.status_code == 200
    assert revoked.json()["revoked"] is True
