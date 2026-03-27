from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from daylib_ursa.config import clear_settings_cache, get_settings, get_settings_for_testing
from daylib_ursa.gui_app import mount_gui


def test_get_settings_reads_cognito_and_deployment_from_yaml_not_env(tmp_path, monkeypatch):
    config_path = tmp_path / "ursa-config.yaml"
    config_path.write_text(
        """
aws_profile: lsmc
ursa_internal_output_bucket: ursa-internal
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

    monkeypatch.setattr("daylib_ursa.ursa_config.DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr("daylib_ursa.ursa_config.LEGACY_CONFIG_PATHS", [])
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
    mount_gui(app)
    return app


def test_login_page_renders_banner_and_favicon():
    settings = get_settings_for_testing(
        ursa_internal_output_bucket="ursa-internal",
        deployment_name="staging",
        deployment_color="#124e78",
        deployment_is_production=False,
    )
    client = TestClient(_app_with_gui(settings))

    response = client.get("/login")

    assert response.status_code == 200
    assert "STAGING" in response.text
    assert "#124e78" in response.text
    assert "/ui/static/favicon.svg" in response.text


def test_favicon_route_redirects_to_svg():
    client = TestClient(_app_with_gui(get_settings_for_testing(ursa_internal_output_bucket="ursa-internal")))

    response = client.get("/favicon.ico", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/ui/static/favicon.svg"
