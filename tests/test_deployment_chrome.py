from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient

from daylib.config import clear_settings_cache, get_settings, get_settings_for_testing
from daylib.workset_api import create_app


def test_get_settings_reads_cognito_and_deployment_from_yaml_not_env(tmp_path, monkeypatch):
    config_path = tmp_path / "ursa-config.yaml"
    config_path.write_text(
        """
aws_profile: lsmc
cognito_user_pool_id: yaml-pool
cognito_app_client_id: yaml-client
cognito_app_client_secret: yaml-secret
cognito_domain: yaml.auth.us-west-2.amazoncognito.com
cognito_region: us-west-2
cognito_callback_url: https://localhost:8914/auth/callback
cognito_logout_url: https://localhost:8914/portal/login
deployment:
  name: staging
  color: "#124e78"
  is_production: false
""".strip()
    )

    monkeypatch.setattr("daylib.ursa_config.DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr("daylib.ursa_config.LEGACY_CONFIG_PATHS", [])
    monkeypatch.setattr("daylib.ursa_config._global_config", None)
    monkeypatch.setenv("COGNITO_USER_POOL_ID", "env-pool")
    monkeypatch.setenv("COGNITO_APP_CLIENT_ID", "env-client")
    monkeypatch.setenv("COGNITO_DOMAIN", "env.example.com")
    monkeypatch.setenv("COGNITO_REGION", "eu-west-1")

    clear_settings_cache()
    settings = get_settings()

    assert settings.cognito_user_pool_id == "yaml-pool"
    assert settings.cognito_app_client_id == "yaml-client"
    assert settings.cognito_app_client_secret == "yaml-secret"
    assert settings.cognito_domain == "yaml.auth.us-west-2.amazoncognito.com"
    assert settings.cognito_region == "us-west-2"
    assert settings.cognito_callback_url == "https://localhost:8914/auth/callback"
    assert settings.cognito_logout_url == "https://localhost:8914/portal/login"
    assert settings.deployment == {
        "name": "staging",
        "color": "#124e78",
        "is_production": False,
    }

    clear_settings_cache()
    monkeypatch.setattr("daylib.ursa_config._global_config", None)


def test_login_page_renders_banner_and_favicon(mock_state_db):
    settings = get_settings_for_testing(
        deployment_name="staging",
        deployment_color="#124e78",
        deployment_is_production=False,
    )

    app = create_app(
        state_db=mock_state_db,
        enable_auth=False,
        settings=settings,
    )
    client = TestClient(app)

    response = client.get("/portal/login")

    assert response.status_code == 200
    assert b"STAGING" in response.content
    assert b"#124e78" in response.content
    assert b"/static/favicon.svg" in response.content


def test_hosted_ui_login_redirect_uses_yaml_callback_url(mock_state_db):
    settings = get_settings_for_testing(
        enable_auth=True,
        cognito_app_client_id="test-client-id",
        cognito_domain="example.auth.us-west-2.amazoncognito.com",
        cognito_callback_url="https://localhost:8914/auth/callback",
        cognito_logout_url="https://localhost:8914/portal/login",
    )

    app = create_app(
        state_db=mock_state_db,
        enable_auth=True,
        cognito_auth=object(),
        settings=settings,
    )
    client = TestClient(app)

    response = client.get("/portal/login?sso=1", follow_redirects=False)

    assert response.status_code == 302
    parsed = urlparse(response.headers["location"])
    query = parse_qs(parsed.query)
    assert query["redirect_uri"] == ["https://localhost:8914/auth/callback"]


def test_favicon_route_redirects_to_svg(mock_state_db):
    app = create_app(state_db=mock_state_db, enable_auth=False)
    client = TestClient(app)

    response = client.get("/favicon.ico", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/static/favicon.svg"
