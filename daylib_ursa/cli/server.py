"""Server management commands for the Ursa beta analysis API."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer
import boto3
from cli_core_yo.certs import resolve_https_certs, shared_dayhoff_certs_dir
from cli_core_yo.oauth import runtime_oauth_host, validate_cognito_app_client
from cli_core_yo.server import (
    display_host,
    latest_log,
    list_logs,
    new_log_path,
    stop_pid,
    write_pid,
)

from daylib_ursa.cli._registry_v2 import (
    REQUIRED,
    REQUIRED_LONG_RUNNING,
    REQUIRED_MUTATING,
    REQUIRED_MUTATING_LONG_RUNNING,
    register_group_commands,
)
from daylib_ursa.config import get_settings
from daylib_ursa.config import DEFAULT_API_PORT
from daylib_ursa.integrations.tapdb_runtime import export_database_url_for_target
from daylib_ursa.ursa_config import get_config_dir
from cli_core_yo import output as cli_output

if TYPE_CHECKING:
    from cli_core_yo.registry import CommandRegistry
    from cli_core_yo.spec import CliSpec

server_app = typer.Typer(help="API server management commands")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_CERTS_DIR = PROJECT_ROOT / "certs"

REQUIRED_COGNITO_APP_CLIENT_NAME = "ursa"


def _config_dir() -> Path:
    return get_config_dir()


def _log_dir() -> Path:
    return _config_dir() / "logs"


def _pid_file() -> Path:
    return _config_dir() / "server.pid"


def _runtime_meta_file() -> Path:
    return _config_dir() / "server-meta.json"


def _shared_dayhoff_certs_dir() -> Path:
    from daylib_ursa.ursa_config import _resolve_deployment_code

    return shared_dayhoff_certs_dir(_resolve_deployment_code())


def _option_default(value, fallback):
    return fallback if isinstance(value, typer.models.OptionInfo) else value


def _resolved_server_host_port(
    *,
    port: int | None = None,
    host: str | None = None,
) -> tuple[str, int]:
    settings = get_settings()
    resolved_port = int(
        port
        if port is not None
        else os.environ.get(
            "URSA_RUNTIME__PORT",
            getattr(settings, "api_port", os.environ.get("URSA_PORT", DEFAULT_API_PORT)),
        )
    )
    resolved_host = str(
        host
        if host is not None
        else os.environ.get(
            "URSA_RUNTIME__HOST",
            getattr(settings, "api_host", os.environ.get("URSA_HOST", "0.0.0.0")),
        )
    )
    return resolved_host, resolved_port


def _require_auth_dependencies() -> None:
    """Fail fast if auth is requested but optional auth deps aren't installed."""

    try:
        import jose  # noqa: F401
    except ImportError:
        cli_output.error(" Authentication requested but python-jose is not installed")
        cli_output.print_rich(
            "   Refresh the runtime env with: [cyan]conda env update -f environment.yaml --prune[/cyan]"
        )
        raise typer.Exit(1)


def _ensure_dir():
    """Ensure deployment-scoped Ursa runtime directories exist."""
    _config_dir().mkdir(parents=True, exist_ok=True)
    _log_dir().mkdir(parents=True, exist_ok=True)


def _runtime_scheme() -> str:
    meta_file = _runtime_meta_file()
    if not meta_file.exists():
        return "https"
    try:
        payload = json.loads(meta_file.read_text(encoding="utf-8"))
    except Exception:
        return "https"
    if isinstance(payload, dict):
        ssl_enabled = payload.get("ssl_enabled")
        if isinstance(ssl_enabled, bool):
            return "https" if ssl_enabled else "http"
    return "https"


def _write_runtime_meta(*, ssl_enabled: bool) -> None:
    _runtime_meta_file().write_text(
        json.dumps({"ssl_enabled": ssl_enabled}, sort_keys=True),
        encoding="utf-8",
    )


def _clear_runtime_meta() -> None:
    _runtime_meta_file().unlink(missing_ok=True)


def _https_san_hosts(host: str) -> tuple[str, ...]:
    san_hosts = ["localhost", "127.0.0.1", "::1"]
    if host not in ("0.0.0.0", "::", "127.0.0.1", "localhost"):
        san_hosts.insert(0, host)
    return tuple(san_hosts)


def _resolve_https_cert_paths(
    host: str,
    *,
    cert: str | None = None,
    key: str | None = None,
) -> tuple[str, str]:
    shared_dir = _shared_dayhoff_certs_dir()
    env_cert = str(os.environ.get("SSL_CERT_FILE", "")).strip()
    env_key = str(os.environ.get("SSL_KEY_FILE", "")).strip()
    if bool(env_cert) != bool(env_key):
        cli_output.error(" SSL_CERT_FILE and SSL_KEY_FILE must be set together")
        raise typer.Exit(1)
    try:
        resolved = resolve_https_certs(
            cert_path=cert,
            key_path=key,
            shared_certs_dir=shared_dir,
            fallback_certs_dir=REPO_CERTS_DIR,
            hosts=_https_san_hosts(host),
        )
    except SystemExit as exc:
        cli_output.print_rich(f"[red]✗[/red]  {exc}")
        raise typer.Exit(1) from exc

    return str(resolved.cert_path), str(resolved.key_path)


def _describe_cognito_app_client(
    *,
    profile: str,
    region: str,
    user_pool_id: str,
    app_client_id: str,
) -> dict:
    """Fetch Cognito app-client configuration."""
    session = boto3.Session(profile_name=profile, region_name=region)
    cognito = session.client("cognito-idp")
    response = cognito.describe_user_pool_client(
        UserPoolId=user_pool_id,
        ClientId=app_client_id,
    )
    return dict(response.get("UserPoolClient") or {})


def _require_cognito_configuration(ursa_config) -> dict[str, str]:
    """Require Cognito configuration from YAML config without exporting env vars."""
    field_map = {
        "cognito_user_pool_id": "Cognito user pool ID",
        "cognito_app_client_id": "Cognito app client ID",
        "cognito_region": "Cognito region",
        "cognito_domain": "Cognito domain",
        "cognito_callback_url": "Cognito callback URL",
        "cognito_logout_url": "Cognito logout URL",
    }
    missing: list[str] = []
    resolved: dict[str, str] = {}
    for attr_name, label in field_map.items():
        value = str(getattr(ursa_config, attr_name, "") or "").strip()
        if value:
            resolved[attr_name] = value
        else:
            missing.append(label)
    if missing:
        cli_output.error(" Authentication is mandatory but Cognito config is missing")
        cli_output.print_rich("   Missing YAML fields: [cyan]" + ", ".join(missing) + "[/cyan]")
        raise typer.Exit(1)
    return resolved


def _get_pid() -> Optional[int]:
    """Get the running server PID if exists."""
    pid_file = _pid_file()
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
            cmdline = subprocess.check_output(
                ["ps", "-p", str(pid), "-o", "command="],
                text=True,
            ).strip()
            if "daylib_ursa.workset_api_cli" not in cmdline:
                pid_file.unlink(missing_ok=True)
                return None
            return pid
        except (ValueError, ProcessLookupError, PermissionError, subprocess.SubprocessError):
            pid_file.unlink(missing_ok=True)
    return None


def _run_cognito_uri_check(
    port: int,
    host: str,
    aws_profile: str,
    cognito_config: dict[str, str],
) -> None:
    """Validate Cognito app-client callback/logout URIs match YAML configuration."""
    user_pool_id = cognito_config["cognito_user_pool_id"]
    app_client_id = cognito_config["cognito_app_client_id"]
    region = cognito_config["cognito_region"]
    try:
        app_client = _describe_cognito_app_client(
            profile=aws_profile,
            region=region,
            user_pool_id=user_pool_id,
            app_client_id=app_client_id,
        )
    except Exception as exc:
        cli_output.print_rich(f"[yellow]⚠[/yellow]  Could not fetch Cognito app client: {exc}")
        return

    oauth_host = runtime_oauth_host(host)
    expected_callback = cognito_config["cognito_callback_url"]
    expected_logout = cognito_config["cognito_logout_url"]
    errors = validate_cognito_app_client(
        app_client=app_client,
        expected_callback_url=expected_callback,
        expected_logout_url=expected_logout,
        expected_port=port,
        runtime_host=oauth_host,
        expected_client_name=REQUIRED_COGNITO_APP_CLIENT_NAME,
    )
    if errors:
        cli_output.print_rich("[yellow]⚠[/yellow]  Cognito URI validation warnings:")
        for err in errors:
            cli_output.print_rich(f"   • {err}")
        cli_output.print_rich(f"   Server is starting on port [cyan]{port}[/cyan]")
        cli_output.print_rich("   Use [dim]--no-check-cognito-uris[/dim] to skip\n")


@server_app.command("start")
def start(
    port: int | None = typer.Option(None, "--port", "-p", help="Port to run the server on"),
    host: str | None = typer.Option(None, "--host", "-h", help="Host to bind to"),
    ssl: bool = typer.Option(True, "--ssl/--no-ssl", help="Serve over HTTPS"),
    cert: str | None = typer.Option(None, "--cert", help="Path to TLS certificate file"),
    key: str | None = typer.Option(None, "--key", help="Path to TLS private key file"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload (foreground)"),
    background: bool = typer.Option(
        True, "--background/--foreground", "-b/-f", help="Run in background"
    ),
    check_cognito_uris: bool = typer.Option(
        True,
        "--check-cognito-uris/--no-check-cognito-uris",
        help="Validate Cognito callback/logout URI ports before startup",
    ),
):
    """Start the Ursa beta analysis API server."""
    port = _option_default(port, None)
    host = _option_default(host, None)
    ssl = _option_default(ssl, True)
    cert = _option_default(cert, None)
    key = _option_default(key, None)
    reload = _option_default(reload, False)
    background = _option_default(background, True)
    check_cognito_uris = _option_default(check_cognito_uris, True)

    _ensure_dir()

    settings = get_settings()
    host, port = _resolved_server_host_port(port=port, host=host)

    # Check if already running
    pid = _get_pid()
    if pid:
        cli_output.print_rich(f"[yellow]⚠[/yellow]  Server already running (PID {pid})")
        protocol = "https" if _runtime_scheme() == "https" else "http"
        cli_output.print_rich(f"   URL: [cyan]{protocol}://{host}:{port}[/cyan]")
        return

    if not ssl and (cert or key):
        cli_output.error(" --cert and --key cannot be used with --no-ssl")
        raise typer.Exit(1)

    # Resolve AWS profile from env or config when explicitly provided.
    from daylib_ursa.ursa_config import get_config_file_path, get_ursa_config

    ursa_config = get_ursa_config()

    aws_profile = os.environ.get("AWS_PROFILE") or ursa_config.aws_profile
    if aws_profile and not os.environ.get("AWS_PROFILE"):
        os.environ["AWS_PROFILE"] = aws_profile

    _require_auth_dependencies()
    cognito_config = _require_cognito_configuration(ursa_config)

    if check_cognito_uris:
        _run_cognito_uri_check(port, host, aws_profile or "default", cognito_config)

    aws_region = (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or (ursa_config.get_allowed_regions()[0] if ursa_config.is_configured else "us-west-2")
    )

    protocol = "https" if ssl else "http"
    if ssl:
        cert, key = _resolve_https_cert_paths(host, cert=cert, key=key)

    # Check config file for region configuration
    if not ursa_config.is_configured:
        config_file_path = get_config_file_path()
        cli_output.print_rich(f"[yellow]⚠[/yellow]  No regions configured in {config_file_path}")
        cli_output.print_rich("   Cluster discovery requires region definitions.")
        cli_output.print_rich(f"   Create [cyan]{config_file_path}[/cyan] with:")
        cli_output.print_rich("")
        cli_output.print_rich("[dim]   regions:")
        cli_output.print_rich("     - us-west-2")
        cli_output.print_rich("     - us-east-1[/dim]")
    else:
        regions = ursa_config.get_allowed_regions()
        cli_output.print_rich(
            f"[green]✓[/green]  Ursa config loaded: [cyan]{len(regions)} regions[/cyan]"
        )

    # Build command (package-safe: uses module execution, not repo-relative bin/)
    cmd = [
        sys.executable,
        "-m",
        "daylib_ursa.workset_api_cli",
        "--host",
        host,
        "--port",
        str(port),
        "--region",
        aws_region,
        "--bootstrap-tapdb",
        "--ssl" if ssl else "--no-ssl",
    ]
    if cert:
        cmd.extend(["--cert", cert])
    if key:
        cmd.extend(["--key", key])
    if aws_profile:
        cmd.extend(["--profile", aws_profile])

    # Set up environment
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("TAPDB_"):
            env.pop(key, None)
    env["MERIDIAN_DOMAIN_CODE"] = "R"
    env["TAPDB_APP_CODE"] = "R"
    env["PYTHONUNBUFFERED"] = "1"
    env["ENABLE_AUTH"] = "true"

    env["DATABASE_BACKEND"] = settings.database_backend
    env["DATABASE_TARGET"] = settings.database_target
    if settings.database_backend == "tapdb":
        env["DATABASE_URL"] = export_database_url_for_target(
            target=settings.database_target,
            profile=aws_profile,
            region=aws_region,
            client_id=settings.tapdb_client_id,
            namespace=settings.tapdb_database_name,
            tapdb_env=settings.tapdb_env,
        )

    if reload:
        cmd.append("--reload")
        background = False  # Reload requires foreground
        cli_output.print_rich("[dim]Auto-reload enabled (foreground mode)[/dim]")

    if background:
        log_file = new_log_path(_log_dir())
        log_f = open(log_file, "w", buffering=1)  # Line-buffered

        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=PROJECT_ROOT,
            env=env,
        )

        time.sleep(2)
        if proc.poll() is not None:
            _clear_runtime_meta()
            log_f.close()
            cli_output.error(" Server failed to start. Check logs:")
            cli_output.print_rich(f"   [dim]{log_file}[/dim]")
            # Show last few lines of error
            if log_file.exists():
                content = log_file.read_text().strip()
                if content:
                    cli_output.print_rich("\n[dim]--- Last error ---[/dim]")
                    for line in content.split("\n")[-10:]:
                        cli_output.print_rich(f"   {line}")
            raise typer.Exit(1)

        write_pid(_pid_file(), proc.pid)
        _write_runtime_meta(ssl_enabled=ssl)
        cli_output.print_rich(f"[green]✓[/green]  Server started (PID {proc.pid})")
        cli_output.print_rich(f"   URL: [cyan]{protocol}://{host}:{port}[/cyan]")
        cli_output.print_rich(f"   Logs: [dim]{log_file}[/dim]")
    else:
        _write_runtime_meta(ssl_enabled=ssl)
        cli_output.print_rich(
            f"[green]✓[/green]  Starting server on [cyan]{protocol}://{host}:{port}[/cyan]"
        )
        cli_output.print_rich("   Press Ctrl+C to stop\n")
        try:
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
            if result.returncode != 0:
                _clear_runtime_meta()
                raise typer.Exit(result.returncode)
        except KeyboardInterrupt:
            _clear_runtime_meta()
            cli_output.print_rich("\n[yellow]⚠[/yellow]  Server stopped")
        else:
            _clear_runtime_meta()


@server_app.command("stop")
def stop():
    """Stop the Ursa API server."""
    stopped, msg = stop_pid(_pid_file())
    if stopped:
        _clear_runtime_meta()
        cli_output.print_rich(f"[green]✓[/green]  {msg}")
    elif "Permission" in msg:
        cli_output.print_rich(f"[red]✗[/red]  {msg}")
        raise typer.Exit(1)
    else:
        cli_output.print_rich(f"[yellow]⚠[/yellow]  {msg}")


@server_app.command("status")
def status():
    """Check the status of the Ursa beta analysis API server."""
    pid = _get_pid()
    if pid:
        host, port = _resolved_server_host_port()
        log_file = latest_log(_log_dir())
        dh = display_host(host)
        protocol = _runtime_scheme()
        cli_output.print_rich(f"[green]●[/green]  Server is [green]running[/green] (PID {pid})")
        cli_output.print_rich(f"   URL: [cyan]{protocol}://{dh}:{port}[/cyan]")
        if log_file:
            cli_output.print_rich(f"   Logs: [dim]{log_file}[/dim]")
    else:
        cli_output.print_rich("[dim]○[/dim]  Server is [dim]not running[/dim]")


@server_app.command("logs")
def logs(
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines to show"),
    all_logs: bool = typer.Option(False, "--all", "-a", help="List all log files"),
):
    """View and follow Ursa API server logs (Ctrl+C to stop)."""
    if all_logs:
        log_entries = list_logs(_log_dir())
        if not log_entries:
            cli_output.print_rich("[yellow]⚠[/yellow]  No log files found.")
            return
        cli_output.print_rich(f"[bold]Server log files ({len(log_entries)}):[/bold]")
        for lf in log_entries[:20]:
            size = lf.stat().st_size
            cli_output.print_rich(f"  {lf.name}  [dim]({size:,} bytes)[/dim]")
        return

    log_file = latest_log(_log_dir())
    if not log_file:
        cli_output.print_rich("[yellow]⚠[/yellow]  No log file found. Start the server first.")
        return

    cli_output.print_rich(f"[dim]Following {log_file.name} (Ctrl+C to stop)[/dim]\n")
    try:
        subprocess.run(["tail", "-f", "-n", str(lines), str(log_file)])
    except KeyboardInterrupt:
        cli_output.print_rich("\n")


@server_app.command("restart")
def restart(
    port: int | None = typer.Option(None, "--port", "-p", help="Port to run the server on"),
    host: str | None = typer.Option(None, "--host", "-h", help="Host to bind to"),
    ssl: bool = typer.Option(True, "--ssl/--no-ssl", help="Serve over HTTPS"),
    cert: str | None = typer.Option(None, "--cert", help="Path to TLS certificate file"),
    key: str | None = typer.Option(None, "--key", help="Path to TLS private key file"),
):
    """Restart the Ursa API server."""
    port = _option_default(port, None)
    host = _option_default(host, None)
    ssl = _option_default(ssl, True)
    cert = _option_default(cert, None)
    key = _option_default(key, None)
    stop()
    time.sleep(1)
    start(port=port, host=host, ssl=ssl, cert=cert, key=key, reload=False, background=True)


def register(registry: CommandRegistry, spec: CliSpec) -> None:
    """cli-core-yo plugin: register server command group."""
    _ = spec
    register_group_commands(
        registry,
        "server",
        "API server management",
        [
            ("start", start, REQUIRED_MUTATING_LONG_RUNNING),
            ("stop", stop, REQUIRED_MUTATING),
            ("status", status, REQUIRED),
            ("logs", logs, REQUIRED_LONG_RUNNING),
            ("restart", restart, REQUIRED_MUTATING_LONG_RUNNING),
        ],
    )
