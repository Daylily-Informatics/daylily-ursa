from __future__ import annotations

import base64
import json
import uuid
from types import SimpleNamespace

from fastapi.testclient import TestClient

from daylib_ursa.auth.dependencies import CognitoAuthProvider
from daylib_ursa.auth import CurrentUser
from daylib_ursa.config import clear_settings_cache, get_settings, get_settings_for_testing
from daylib_ursa import gui_app
from daylib_ursa.gui_app import mount_gui


def test_get_settings_reads_cognito_and_deployment_from_yaml_not_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("URSA_DEPLOYMENT_CODE", "local")
    config_dir = tmp_path / ".config" / "ursa-local"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "ursa-config-local.yaml"
    config_path.write_text(
        """
aws_profile: lsmc
ursa_internal_output_bucket: ursa-internal
tapdb_client_id: ursa
tapdb_database_name: daylily-ursa
tapdb_env: dev
api_host: 127.0.0.1
api_port: 8913
ursa_portal_default_customer_id: 77777777-7777-7777-7777-777777777777
bloom_base_url: https://localhost:8912
bloom_verify_ssl: true
atlas_base_url: https://localhost:8915
atlas_verify_ssl: true
dewey_enabled: false
dewey_base_url: https://localhost:8914
dewey_api_token: dewey-dev-token
dewey_verify_ssl: true
cognito_user_pool_id: yaml-pool
cognito_app_client_id: yaml-client
cognito_app_client_secret: yaml-secret
cognito_domain: yaml.auth.us-west-2.amazoncognito.com
cognito_region: us-west-2
cognito_callback_url: https://localhost:8914/auth/callback
cognito_logout_url: https://localhost:8914/login
deployment:
  name: staging
  color: "#124e78"
  is_production: false
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr("daylib_ursa.ursa_config._global_config", None)
    monkeypatch.setenv("COGNITO_USER_POOL_ID", "env-pool")
    monkeypatch.setenv("COGNITO_APP_CLIENT_ID", "env-client")
    monkeypatch.setenv("COGNITO_DOMAIN", "env.example.com")
    monkeypatch.setenv("COGNITO_REGION", "eu-west-1")
    monkeypatch.setenv("URSA_INTERNAL_OUTPUT_BUCKET", "ursa-internal")

    clear_settings_cache()
    settings = get_settings()

    assert settings.cognito_user_pool_id == "yaml-pool"
    assert settings.cognito_app_client_id == "yaml-client"
    assert settings.cognito_app_client_secret == "yaml-secret"
    assert settings.cognito_domain == "yaml.auth.us-west-2.amazoncognito.com"
    assert settings.cognito_region == "us-west-2"
    assert settings.cognito_callback_url == "https://localhost:8914/auth/callback"
    assert settings.cognito_logout_url == "https://localhost:8914/login"
    assert settings.dewey_api_token == "dewey-dev-token"
    assert settings.ursa_portal_default_customer_id == "77777777-7777-7777-7777-777777777777"
    assert settings.deployment == {
        "name": "staging",
        "color": "#124e78",
        "is_production": False,
    }


def _app_with_gui(settings):
    from fastapi import FastAPI
    from starlette.middleware.sessions import SessionMiddleware

    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key="test-secret")
    app.state.settings = settings
    app.state.identity_client = SimpleNamespace(resolve_access_token=lambda _token: None)
    app.state.auth_provider = SimpleNamespace(
        resolve_access_token=lambda _token, **_kwargs: CurrentUser(
            sub="user-123",
            email="user@example.com",
            name="Ursa User",
            tenant_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
            roles=["ADMIN"],
        )
    )
    mount_gui(app)
    return app


def _login_user(
    monkeypatch,
    client,
    *,
    email: str = "user@example.com",
    sub: str = "user-123",
    name: str = "Ursa User",
    roles: list[str] | None = None,
) -> None:
    monkeypatch.setattr(gui_app.secrets, "token_urlsafe", lambda _n: "state-123")

    def _resolve_access_token(_token: str, **_kwargs):
        return CurrentUser(
            sub=sub,
            email=email,
            name=name,
            tenant_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
            roles=roles or ["ADMIN"],
        )

    client.app.state.auth_provider = SimpleNamespace(resolve_access_token=_resolve_access_token)
    monkeypatch.setattr(gui_app, "build_authorization_url", lambda **_kwargs: "https://example.auth/login")
    monkeypatch.setattr(
        gui_app,
        "exchange_authorization_code",
        lambda **_kwargs: {
            "id_token": "id-token-123",
            "access_token": "access-token-456",
        },
    )
    login_response = client.get("/auth/login?next=/usage", follow_redirects=False)
    assert login_response.status_code == 303
    callback_response = client.get(
        "/auth/callback?code=auth-code&state=state-123",
        follow_redirects=False,
    )
    assert callback_response.status_code == 303
    assert callback_response.headers["location"] == "/usage"


def _decode_session_cookie(client: TestClient) -> dict[str, object]:
    cookie_name = next(iter(client.cookies.keys()))
    payload = client.cookies[cookie_name].split(".", 1)[0]
    return json.loads(base64.b64decode(payload))


def test_login_page_renders_banner_and_favicon():
    settings = get_settings_for_testing(
        ursa_internal_output_bucket="ursa-internal",
        deployment_name="staging",
        deployment_color="#124e78",
        deployment_is_production=False,
        cognito_domain="ursa.auth.us-west-2.amazoncognito.com",
        cognito_app_client_id="client-123",
        cognito_callback_url="https://localhost:8913/auth/callback",
        cognito_logout_url="https://localhost:8913/login",
    )
    client = TestClient(_app_with_gui(settings))

    response = client.get("/login")

    assert response.status_code == 200
    assert "STAGING" in response.text
    assert "#124e78" in response.text
    assert "/ui/static/favicon.svg" in response.text
    assert "Sign In with Cognito" in response.text
    assert "/auth/login?next=/" in response.text


def test_auth_login_redirects_to_cognito(monkeypatch):
    settings = get_settings_for_testing(
        ursa_internal_output_bucket="ursa-internal",
        cognito_domain="ursa.auth.us-west-2.amazoncognito.com",
        cognito_app_client_id="client-123",
        cognito_callback_url="https://localhost:8913/auth/callback",
        cognito_logout_url="https://localhost:8913/login",
    )
    client = TestClient(_app_with_gui(settings))

    monkeypatch.setattr(gui_app.secrets, "token_urlsafe", lambda _n: "state-123")
    monkeypatch.setattr(gui_app, "build_authorization_url", lambda **_kwargs: "https://example.auth/login")
    response = client.get("/auth/login?next=/usage", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "https://example.auth/login"


def test_auth_callback_persists_session_and_redirects(monkeypatch):
    settings = get_settings_for_testing(
        ursa_internal_output_bucket="ursa-internal",
        cognito_domain="ursa.auth.us-west-2.amazoncognito.com",
        cognito_app_client_id="client-123",
        cognito_callback_url="https://localhost:8913/auth/callback",
        cognito_logout_url="https://localhost:8913/login",
    )
    client = TestClient(_app_with_gui(settings))
    monkeypatch.setattr(gui_app.secrets, "token_urlsafe", lambda _n: "state-123")
    monkeypatch.setattr(gui_app, "build_authorization_url", lambda **_kwargs: "https://example.auth/login")
    client.get("/auth/login?next=/usage", follow_redirects=False)
    monkeypatch.setattr(gui_app, "exchange_authorization_code", lambda **_kwargs: {"id_token": "token-123"})
    response = client.get("/auth/callback?code=auth-code&state=state-123", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/usage"

    login_page = client.get("/login?next=/usage", follow_redirects=False)
    assert login_page.status_code == 303
    assert login_page.headers["location"] == "/usage"


def test_two_browser_sessions_keep_distinct_users(monkeypatch):
    settings = get_settings_for_testing(
        ursa_internal_output_bucket="ursa-internal",
        cognito_domain="ursa.auth.us-west-2.amazoncognito.com",
        cognito_app_client_id="client-123",
        cognito_callback_url="https://localhost:8913/auth/callback",
        cognito_logout_url="https://localhost:8913/login",
    )
    app = _app_with_gui(settings)

    with TestClient(app) as client_a, TestClient(app) as client_b:
        _login_user(
            monkeypatch,
            client_a,
            email="operator-a@example.com",
            sub="sub-a",
            name="Operator A",
        )
        _login_user(
            monkeypatch,
            client_b,
            email="operator-b@example.com",
            sub="sub-b",
            name="Operator B",
        )

        session_a = _decode_session_cookie(client_a)
        session_b = _decode_session_cookie(client_b)

        assert session_a["email"] == "operator-a@example.com"
        assert session_a["user_sub"] == "sub-a"
        assert session_b["email"] == "operator-b@example.com"
        assert session_b["user_sub"] == "sub-b"
        assert session_a["email"] != session_b["email"]
        assert session_a["user_sub"] != session_b["user_sub"]
        assert client_a.get("/login?next=/usage", follow_redirects=False).status_code == 303
        assert client_b.get("/login?next=/usage", follow_redirects=False).status_code == 303


def test_logout_from_one_session_does_not_clear_the_other(monkeypatch):
    settings = get_settings_for_testing(
        ursa_internal_output_bucket="ursa-internal",
        cognito_domain="ursa.auth.us-west-2.amazoncognito.com",
        cognito_app_client_id="client-123",
        cognito_callback_url="https://localhost:8913/auth/callback",
        cognito_logout_url="https://localhost:8913/login",
    )
    app = _app_with_gui(settings)

    with TestClient(app) as client_a, TestClient(app) as client_b:
        _login_user(
            monkeypatch,
            client_a,
            email="shared@example.com",
            sub="sub-shared",
            name="Shared User",
        )
        _login_user(
            monkeypatch,
            client_b,
            email="shared@example.com",
            sub="sub-shared",
            name="Shared User",
        )

        logout = client_a.get("/auth/logout", follow_redirects=False)
        assert logout.status_code == 303
        assert logout.headers["location"].startswith("https://ursa.auth.us-west-2.amazoncognito.com/logout")

        assert client_a.get("/login?next=/usage", follow_redirects=False).status_code == 200
        assert client_b.get("/login?next=/usage", follow_redirects=False).status_code == 303
        assert _decode_session_cookie(client_b)["email"] == "shared@example.com"


def test_auth_callback_passes_paired_access_token_for_id_token_verification(monkeypatch):
    settings = get_settings_for_testing(
        ursa_internal_output_bucket="ursa-internal",
        cognito_user_pool_id="pool-123",
        cognito_region="us-west-2",
        cognito_domain="ursa.auth.us-west-2.amazoncognito.com",
        cognito_app_client_id="client-123",
        cognito_callback_url="https://localhost:8913/auth/callback",
        cognito_logout_url="https://localhost:8913/login",
    )
    from fastapi import FastAPI
    from starlette.middleware.sessions import SessionMiddleware
    from daylib_ursa.auth import dependencies as auth_dependencies

    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key="test-secret")
    app.state.settings = settings
    app.state.identity_client = SimpleNamespace(resolve_access_token=lambda _token: None)
    app.state.auth_provider = CognitoAuthProvider(
        user_pool_id="pool-123",
        app_client_id="client-123",
        region="us-west-2",
    )
    mount_gui(app)
    client = TestClient(app)

    monkeypatch.setattr(gui_app.secrets, "token_urlsafe", lambda _n: "state-123")
    monkeypatch.setattr(gui_app, "build_authorization_url", lambda **_kwargs: "https://example.auth/login")
    monkeypatch.setattr(
        gui_app,
        "exchange_authorization_code",
        lambda **_kwargs: {
            "id_token": "id-token-123",
            "access_token": "access-token-456",
        },
    )
    captured: dict[str, str | None] = {}
    monkeypatch.setattr(
        auth_dependencies,
        "decode_jwt_unverified",
        lambda _token: {"token_use": "id"},
    )

    def _verify_id_token_claims(self, token: str, *, access_token: str | None = None):
        captured["token"] = token
        captured["access_token"] = access_token
        return {
            "sub": "user-123",
            "email": "user@example.com",
            "aud": "client-123",
            "custom:customer_id": "11111111-1111-1111-1111-111111111111",
            "cognito:groups": ["ursa-admin"],
        }

    monkeypatch.setattr(CognitoAuthProvider, "_verify_id_token_claims", _verify_id_token_claims)

    client.get("/auth/login?next=/usage", follow_redirects=False)
    response = client.get("/auth/callback?code=auth-code&state=state-123", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/usage"
    assert captured == {
        "token": "id-token-123",
        "access_token": "access-token-456",
    }


def test_favicon_route_redirects_to_svg():
    client = TestClient(_app_with_gui(get_settings_for_testing(ursa_internal_output_bucket="ursa-internal")))

    response = client.get("/favicon.ico", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/ui/static/favicon.svg"
