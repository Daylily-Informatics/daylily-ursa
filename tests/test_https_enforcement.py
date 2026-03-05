"""Transport security tests for HTTPS-only enforcement."""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from daylily_ursa.config import get_settings_for_testing
from daylily_ursa.workset_api import create_app


def _create_test_app(**setting_overrides):
    settings = get_settings_for_testing(**setting_overrides)
    return create_app(
        state_db=MagicMock(),
        enable_auth=False,
        settings=settings,
    )


def test_http_requests_are_rejected_with_426():
    app = _create_test_app()
    client = TestClient(app, base_url="http://testserver")

    response = client.get("/")

    assert response.status_code == 426
    assert response.json() == {"detail": "HTTPS is required"}


def test_https_requests_are_allowed_and_include_hsts():
    app = _create_test_app()
    client = TestClient(app, base_url="https://testserver")

    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["strict-transport-security"] == "max-age=31536000; includeSubDomains"


def test_forwarded_https_from_trusted_proxy_is_allowed():
    app = _create_test_app(https_trusted_proxy_ips="testclient")
    client = TestClient(app, base_url="http://testserver")

    response = client.get("/", headers={"X-Forwarded-Proto": "https"})

    assert response.status_code == 200


def test_forwarded_https_from_untrusted_proxy_is_rejected():
    app = _create_test_app()
    client = TestClient(app, base_url="http://testserver")

    response = client.get("/", headers={"X-Forwarded-Proto": "https"})

    assert response.status_code == 426


def test_session_cookie_is_secure_only():
    app = _create_test_app()
    client = TestClient(app, base_url="https://testserver")

    response = client.post(
        "/portal/login",
        data={"email": "admin@example.com", "password": "irrelevant"},
        follow_redirects=False,
    )

    assert response.status_code == 302
    set_cookie = response.headers.get("set-cookie", "")
    assert "secure" in set_cookie.lower()
