from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import typer

def _read_project_scripts(pyproject: Path) -> dict[str, str]:
    """Parse the [project.scripts] table from pyproject.toml.

    Avoids adding a TOML parser dependency; this project keeps that section
    simple (string literals with no inline tables).
    """

    scripts: dict[str, str] = {}
    in_section = False
    for raw_line in pyproject.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_section = line == "[project.scripts]"
            continue
        if not in_section:
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        scripts[key] = value
    return scripts


def test_console_script_entrypoints_are_importable_and_callable():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    scripts = _read_project_scripts(pyproject)
    assert scripts, "No [project.scripts] entries found in pyproject.toml"

    for name, target in scripts.items():
        assert ":" in target, f"Console script {name!r} has unexpected target: {target!r}"
        module_name, func_name = target.split(":", 1)
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
        assert callable(func), f"Console script {name!r} target is not callable: {target!r}"


def test_ursa_server_start_uses_packaged_entrypoint(monkeypatch):
    from daylib.cli import server as server_mod
    import daylib.ursa_config as ursa_config_mod

    class DummyUrsaConfig:
        aws_profile = "test-profile"
        is_configured = True
        cognito_user_pool_id = None
        cognito_app_client_id = None
        cognito_region = None

        def get_allowed_regions(self):
            return ["us-west-2"]

        def get_effective_dynamo_db_region(self):
            return "us-west-2"

    monkeypatch.setenv("AWS_PROFILE", "test-profile")
    monkeypatch.setattr(ursa_config_mod, "get_ursa_config", lambda reload=False: DummyUrsaConfig())
    monkeypatch.setattr(server_mod, "_ensure_dir", lambda: None)
    monkeypatch.setattr(server_mod, "_get_pid", lambda: None)
    monkeypatch.setattr(server_mod, "_source_env_file", lambda: False)
    monkeypatch.setattr(server_mod, "_resolve_https_cert_paths", lambda host: ("/tmp/cert.pem", "/tmp/key.pem"))

    captured: dict[str, object] = {}

    def _fake_run(cmd, cwd=None, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["kwargs"] = kwargs
        class _DummyCompletedProcess:
            def __init__(self, returncode: int):
                self.returncode = returncode

        return _DummyCompletedProcess(0)

    monkeypatch.setattr(server_mod.subprocess, "run", _fake_run)

    server_mod.start(
        port=1234,
        host="127.0.0.1",
        auth=False,
        reload=False,
        background=False,
    )

    cmd = captured.get("cmd")
    assert isinstance(cmd, list)
    assert cmd[:3] == [sys.executable, "-m", "daylib.workset_api_cli"]
    assert not any("bin/daylily-workset-api" in str(part) for part in cmd)

    kwargs = captured.get("kwargs")
    assert isinstance(kwargs, dict)
    assert isinstance(kwargs.get("env"), dict)
    assert kwargs["env"].get("DAYLILY_ENABLE_AUTH") == "false"


def test_validate_cognito_oauth_uris_detects_port_mismatch():
    from daylib.cli import server as server_mod

    errors = server_mod._validate_cognito_oauth_uris(
        app_client={
            "ClientName": "ursa",
            "AllowedOAuthFlowsUserPoolClient": True,
            "CallbackURLs": ["https://localhost:9999/auth/callback"],
            "LogoutURLs": ["https://localhost:9999/portal/login"],
            "DefaultRedirectURI": "https://localhost:9999/auth/callback",
        },
        expected_callback_url="https://localhost:8914/auth/callback",
        expected_logout_url="https://localhost:8914/portal/login",
        expected_port=8914,
        runtime_host="localhost",
    )

    assert any("port mismatch" in err for err in errors)


def test_validate_cognito_oauth_uris_accepts_matching_port():
    from daylib.cli import server as server_mod

    errors = server_mod._validate_cognito_oauth_uris(
        app_client={
            "ClientName": "ursa",
            "AllowedOAuthFlowsUserPoolClient": True,
            "CallbackURLs": ["https://localhost:8914/auth/callback"],
            "LogoutURLs": ["https://localhost:8914/portal/login"],
            "DefaultRedirectURI": "https://localhost:8914/auth/callback",
        },
        expected_callback_url="https://localhost:8914/auth/callback",
        expected_logout_url="https://localhost:8914/portal/login",
        expected_port=8914,
        runtime_host="localhost",
    )

    assert errors == []


def test_ursa_server_start_auth_fails_when_cognito_uri_ports_mismatch(monkeypatch):
    from daylib.cli import server as server_mod
    import daylib.ursa_config as ursa_config_mod

    class DummyUrsaConfig:
        aws_profile = "test-profile"
        is_configured = True
        cognito_user_pool_id = "pool-id"
        cognito_app_client_id = "client-id"
        cognito_region = "us-west-2"
        cognito_app_client_secret = None
        cognito_domain = "example.auth.us-west-2.amazoncognito.com"

        def get_allowed_regions(self):
            return ["us-west-2"]

        def get_effective_dynamo_db_region(self):
            return "us-west-2"

    monkeypatch.setenv("AWS_PROFILE", "test-profile")
    monkeypatch.setattr(ursa_config_mod, "get_ursa_config", lambda reload=False: DummyUrsaConfig())
    monkeypatch.setattr(server_mod, "_ensure_dir", lambda: None)
    monkeypatch.setattr(server_mod, "_get_pid", lambda: None)
    monkeypatch.setattr(server_mod, "_source_env_file", lambda: False)
    monkeypatch.setattr(server_mod, "_resolve_https_cert_paths", lambda host: ("/tmp/cert.pem", "/tmp/key.pem"))
    monkeypatch.setattr(server_mod, "_require_auth_dependencies", lambda: None)
    monkeypatch.setattr(
        server_mod,
        "_describe_cognito_app_client",
        lambda **kwargs: {
            "ClientName": "ursa",
            "AllowedOAuthFlowsUserPoolClient": True,
            "CallbackURLs": ["https://localhost:9999/auth/callback"],
            "LogoutURLs": ["https://localhost:9999/portal/login"],
            "DefaultRedirectURI": "https://localhost:9999/auth/callback",
        },
    )

    called = {"run": False}

    def _fake_run(cmd, cwd=None, **kwargs):
        called["run"] = True
        class _DummyCompletedProcess:
            def __init__(self, returncode: int):
                self.returncode = returncode
        return _DummyCompletedProcess(0)

    monkeypatch.setattr(server_mod.subprocess, "run", _fake_run)

    with pytest.raises(typer.Exit):
        server_mod.start(
            port=8914,
            host="127.0.0.1",
            auth=True,
            reload=False,
            background=False,
        )

    assert called["run"] is False


def test_validate_cognito_oauth_uris_rejects_default_redirect_not_in_callback_urls():
    from daylib.cli import server as server_mod

    errors = server_mod._validate_cognito_oauth_uris(
        app_client={
            "ClientName": "ursa",
            "AllowedOAuthFlowsUserPoolClient": True,
            "CallbackURLs": ["https://localhost:8914/auth/callback"],
            "LogoutURLs": ["https://localhost:8914/portal/login"],
            "DefaultRedirectURI": "https://localhost:8914/other/callback",
        },
        expected_callback_url="https://localhost:8914/auth/callback",
        expected_logout_url="https://localhost:8914/portal/login",
        expected_port=8914,
        runtime_host="localhost",
    )

    assert any("DefaultRedirectURI is not in CallbackURLs" in err for err in errors)


def test_validate_cognito_oauth_uris_rejects_client_name_mismatch():
    from daylib.cli import server as server_mod

    errors = server_mod._validate_cognito_oauth_uris(
        app_client={
            "ClientName": "not-ursa",
            "AllowedOAuthFlowsUserPoolClient": True,
            "CallbackURLs": ["https://localhost:8914/auth/callback"],
            "LogoutURLs": ["https://localhost:8914/portal/login"],
            "DefaultRedirectURI": "https://localhost:8914/auth/callback",
        },
        expected_callback_url="https://localhost:8914/auth/callback",
        expected_logout_url="https://localhost:8914/portal/login",
        expected_port=8914,
        runtime_host="localhost",
    )

    assert any("name mismatch" in err for err in errors)
