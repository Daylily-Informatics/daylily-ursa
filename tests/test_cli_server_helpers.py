"""Coverage-focused tests for Ursa server CLI helper logic."""

from __future__ import annotations

import builtins
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer

from daylib_ursa.cli import server as server_cli


def test_require_auth_dependencies_raises_when_jose_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "jose":
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(typer.Exit) as exc:
        server_cli._require_auth_dependencies()

    assert exc.value.exit_code == 1


def test_resolve_https_cert_paths_requires_both_env_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("URSA_SSL_CERT_FILE", "/tmp/cert.pem")
    monkeypatch.delenv("URSA_SSL_KEY_FILE", raising=False)

    with pytest.raises(typer.Exit) as exc:
        server_cli._resolve_https_cert_paths("localhost")

    assert exc.value.exit_code == 1


def test_resolve_https_cert_paths_uses_env_files_when_present(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    cert.write_text("cert", encoding="utf-8")
    key.write_text("key", encoding="utf-8")

    monkeypatch.setenv("URSA_SSL_CERT_FILE", str(cert))
    monkeypatch.setenv("URSA_SSL_KEY_FILE", str(key))

    assert server_cli._resolve_https_cert_paths("localhost") == (str(cert), str(key))


def test_resolve_https_cert_paths_fails_without_mkcert(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cert = tmp_path / "auto-cert.pem"
    key = tmp_path / "auto-key.pem"
    monkeypatch.setattr(server_cli, "DEFAULT_SSL_CERT_FILE", cert)
    monkeypatch.setattr(server_cli, "DEFAULT_SSL_KEY_FILE", key)
    monkeypatch.delenv("URSA_SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("URSA_SSL_KEY_FILE", raising=False)
    monkeypatch.setattr(server_cli.shutil, "which", lambda _name: None)

    with pytest.raises(typer.Exit) as exc:
        server_cli._resolve_https_cert_paths("localhost")

    assert exc.value.exit_code == 1


def test_resolve_https_cert_paths_generates_with_mkcert(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cert = tmp_path / "auto-cert.pem"
    key = tmp_path / "auto-key.pem"
    monkeypatch.setattr(server_cli, "DEFAULT_SSL_CERT_FILE", cert)
    monkeypatch.setattr(server_cli, "DEFAULT_SSL_KEY_FILE", key)
    monkeypatch.delenv("URSA_SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("URSA_SSL_KEY_FILE", raising=False)
    monkeypatch.setattr(server_cli.shutil, "which", lambda _name: "/usr/local/bin/mkcert")

    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(args)
        if "-cert-file" in args and "-key-file" in args:
            cert_path = Path(args[args.index("-cert-file") + 1])
            key_path = Path(args[args.index("-key-file") + 1])
            cert_path.write_text("cert", encoding="utf-8")
            key_path.write_text("key", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(server_cli.subprocess, "run", fake_run)

    result = server_cli._resolve_https_cert_paths("0.0.0.0")
    assert result == (str(cert), str(key))
    assert any("-install" in call for call in calls)


def test_uri_helpers_cover_normalization_and_ports() -> None:
    assert server_cli._runtime_oauth_host("0.0.0.0") == "localhost"
    assert server_cli._runtime_oauth_host("example.com") == "example.com"
    assert server_cli._default_port_for_scheme("https") == 443
    assert server_cli._default_port_for_scheme("http") == 80
    assert server_cli._default_port_for_scheme("ftp") is None
    assert server_cli._normalize_uri("https://example.com/path/") == "https://example.com/path"
    assert server_cli._uri_port("https://example.com") == 443
    assert server_cli._uri_port("https://example.com:444") == 444


def test_validate_uri_list_ports_flags_invalid_and_mismatch() -> None:
    errors = server_cli._validate_uri_list_ports(
        uris=[
            "notaurl",
            "ftp://localhost/callback",
            "https://localhost:9999/auth/callback",
            "https://example.com/auth/callback",
        ],
        label="CallbackURLs",
        expected_port=8914,
        runtime_host="localhost",
    )
    assert any("invalid URI" in error for error in errors)
    assert any("unsupported URI scheme" in error for error in errors)
    assert any("port mismatch" in error for error in errors)


def test_validate_cognito_oauth_uris_reports_mismatch() -> None:
    app_client = {
        "ClientName": "wrong-name",
        "AllowedOAuthFlowsUserPoolClient": False,
        "CallbackURLs": ["https://localhost:8912/auth/callback"],
        "LogoutURLs": ["https://localhost:8912/"],
        "DefaultRedirectURI": "https://localhost:8912/auth/callback",
    }
    errors = server_cli._validate_cognito_oauth_uris(
        app_client=app_client,
        expected_callback_url="https://localhost:8914/auth/callback",
        expected_logout_url="https://localhost:8914/",
        expected_port=8914,
        runtime_host="localhost",
    )
    assert any("name mismatch" in error for error in errors)
    assert any("OAuth2 flows enabled" in error for error in errors)
    assert any("Expected callback URI is not configured" in error for error in errors)


def test_require_cognito_configuration_sets_env_from_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = SimpleNamespace(
        cognito_user_pool_id="pool",
        cognito_app_client_id="client",
        cognito_region="us-west-2",
        cognito_domain="example.auth.us-west-2.amazoncognito.com",
    )
    for key in (
        "COGNITO_USER_POOL_ID",
        "COGNITO_APP_CLIENT_ID",
        "COGNITO_REGION",
        "COGNITO_DOMAIN",
    ):
        monkeypatch.delenv(key, raising=False)

    server_cli._require_cognito_configuration(cfg)

    assert os.environ["COGNITO_USER_POOL_ID"] == "pool"
    assert os.environ["COGNITO_APP_CLIENT_ID"] == "client"
    assert os.environ["COGNITO_REGION"] == "us-west-2"
    assert "amazoncognito.com" in os.environ["COGNITO_DOMAIN"]


def test_require_cognito_configuration_raises_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = SimpleNamespace(
        cognito_user_pool_id="",
        cognito_app_client_id="",
        cognito_region="",
        cognito_domain="",
    )
    for key in (
        "COGNITO_USER_POOL_ID",
        "COGNITO_APP_CLIENT_ID",
        "COGNITO_REGION",
        "COGNITO_DOMAIN",
    ):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(typer.Exit) as exc:
        server_cli._require_cognito_configuration(cfg)

    assert exc.value.exit_code == 1


def test_get_pid_clears_non_ursa_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pid_file = tmp_path / "server.pid"
    pid_file.write_text("4242", encoding="utf-8")
    monkeypatch.setattr(server_cli, "PID_FILE", pid_file)
    monkeypatch.setattr(server_cli.os, "kill", lambda _pid, _sig: None)
    monkeypatch.setattr(
        server_cli.subprocess,
        "check_output",
        lambda *_args, **_kwargs: "python -m something_else\n",
    )

    assert server_cli._get_pid() is None
    assert not pid_file.exists()


def test_source_env_file_reads_key_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("X=1\nY='two'\n# comment\n", encoding="utf-8")
    monkeypatch.setattr(server_cli.Path, "cwd", lambda: tmp_path)
    monkeypatch.delenv("X", raising=False)
    monkeypatch.delenv("Y", raising=False)

    assert server_cli._source_env_file() is True
    assert os.environ["X"] == "1"
    assert os.environ["Y"] == "two"


def test_stop_handles_missing_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_cli, "_get_pid", lambda: None)
    server_cli.stop()


def test_stop_permission_error_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_cli, "_get_pid", lambda: 999)

    def fake_kill(_pid: int, _sig: int) -> None:
        raise PermissionError

    monkeypatch.setattr(server_cli.os, "kill", fake_kill)

    with pytest.raises(typer.Exit) as exc:
        server_cli.stop()

    assert exc.value.exit_code == 1

