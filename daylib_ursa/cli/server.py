"""Server management commands for the Ursa beta analysis API."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer
import boto3
from cli_core_yo.oauth import runtime_oauth_host, validate_cognito_app_client
from cli_core_yo.server import (
    display_host,
    latest_log,
    list_logs,
    new_log_path,
    source_env_file,
    stop_pid,
    write_pid,
)
from rich.console import Console

from daylib_ursa.config import get_settings
from daylib_ursa.config import DEFAULT_API_PORT
from daylib_ursa.integrations.tapdb_runtime import export_database_url_for_target
from daylib_ursa.ursa_config import get_config_dir

if TYPE_CHECKING:
    from cli_core_yo.registry import CommandRegistry
    from cli_core_yo.spec import CliSpec

server_app = typer.Typer(help="API server management commands")
console = Console()
PROJECT_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_COGNITO_APP_CLIENT_NAME = "ursa"


def _config_dir() -> Path:
    return get_config_dir()


def _log_dir() -> Path:
    return _config_dir() / "logs"


def _pid_file() -> Path:
    return _config_dir() / "server.pid"


def _cert_dir() -> Path:
    return _config_dir() / "certs"


def _default_ssl_cert_file() -> Path:
    return _cert_dir() / "cert.pem"


def _default_ssl_key_file() -> Path:
    return _cert_dir() / "key.pem"


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
        console.print("[red]✗[/red]  Authentication requested but python-jose is not installed")
        console.print('   Install with: [cyan]python -m pip install -e ".[auth]"[/cyan]')
        raise typer.Exit(1)


def _ensure_dir():
    """Ensure deployment-scoped Ursa runtime directories exist."""
    _config_dir().mkdir(parents=True, exist_ok=True)
    _log_dir().mkdir(parents=True, exist_ok=True)
    _cert_dir().mkdir(parents=True, exist_ok=True)


def _resolve_https_cert_paths(host: str) -> tuple[str, str]:
    """Resolve or generate HTTPS certificate/key paths.

    Prefers explicit env vars:
    - URSA_SSL_CERT_FILE
    - URSA_SSL_KEY_FILE

    If not provided, auto-generates localhost certs via mkcert.
    """
    cert_from_env = os.environ.get("URSA_SSL_CERT_FILE")
    key_from_env = os.environ.get("URSA_SSL_KEY_FILE")

    if cert_from_env or key_from_env:
        if not cert_from_env or not key_from_env:
            console.print("[red]✗[/red]  Both URSA_SSL_CERT_FILE and URSA_SSL_KEY_FILE must be set")
            raise typer.Exit(1)
        cert_path = Path(cert_from_env).expanduser()
        key_path = Path(key_from_env).expanduser()
        if not cert_path.exists() or not key_path.exists():
            console.print("[red]✗[/red]  HTTPS certificate file(s) not found")
            console.print(f"   cert: [dim]{cert_path}[/dim]")
            console.print(f"   key:  [dim]{key_path}[/dim]")
            raise typer.Exit(1)
        return str(cert_path), str(key_path)

    cert_path = _default_ssl_cert_file()
    key_path = _default_ssl_key_file()
    if cert_path.exists() and key_path.exists():
        return str(cert_path), str(key_path)

    mkcert_bin = shutil.which("mkcert")
    if not mkcert_bin:
        console.print("[red]✗[/red]  HTTPS is required but mkcert is not installed")
        console.print("   Install mkcert and retry, or set URSA_SSL_CERT_FILE / URSA_SSL_KEY_FILE")
        raise typer.Exit(1)

    # Install local root CA (idempotent), then generate a localhost cert.
    subprocess.run(
        [mkcert_bin, "-install"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    san_hosts = ["localhost", "127.0.0.1", "::1"]
    if host not in ("0.0.0.0", "::", "127.0.0.1", "localhost"):
        san_hosts.insert(0, host)

    try:
        subprocess.run(
            [
                mkcert_bin,
                "-cert-file",
                str(cert_path),
                "-key-file",
                str(key_path),
                *san_hosts,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        console.print("[red]✗[/red]  Failed to generate HTTPS certs with mkcert")
        raise typer.Exit(1)

    return str(cert_path), str(key_path)


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
        console.print("[red]✗[/red]  Authentication is mandatory but Cognito config is missing")
        console.print("   Missing YAML fields: [cyan]" + ", ".join(missing) + "[/cyan]")
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
        console.print(f"[yellow]⚠[/yellow]  Could not fetch Cognito app client: {exc}")
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
        console.print("[yellow]⚠[/yellow]  Cognito URI validation warnings:")
        for err in errors:
            console.print(f"   • {err}")
        console.print(f"   Server is starting on port [cyan]{port}[/cyan]")
        console.print("   Use [dim]--no-check-cognito-uris[/dim] to skip\n")


@server_app.command("start")
def start(
    port: int | None = typer.Option(None, "--port", "-p", help="Port to run the server on"),
    host: str | None = typer.Option(None, "--host", "-h", help="Host to bind to"),
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
    _ensure_dir()

    # Source .env file
    if source_env_file(PROJECT_ROOT / ".env"):
        console.print("[dim]Loaded .env file[/dim]")

    settings = get_settings()
    host, port = _resolved_server_host_port(port=port, host=host)

    # Check if already running
    pid = _get_pid()
    if pid:
        console.print(f"[yellow]⚠[/yellow]  Server already running (PID {pid})")
        console.print(f"   URL: [cyan]https://{host}:{port}[/cyan]")
        return

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

    ssl_certfile, ssl_keyfile = _resolve_https_cert_paths(host)

    # Check config file for region configuration
    if not ursa_config.is_configured:
        config_file_path = get_config_file_path()
        console.print(f"[yellow]⚠[/yellow]  No regions configured in {config_file_path}")
        console.print("   Cluster discovery requires region definitions.")
        console.print(f"   Create [cyan]{config_file_path}[/cyan] with:")
        console.print("")
        console.print("[dim]   regions:")
        console.print("     - us-west-2")
        console.print("     - us-east-1[/dim]")
    else:
        regions = ursa_config.get_allowed_regions()
        console.print(f"[green]✓[/green]  Ursa config loaded: [cyan]{len(regions)} regions[/cyan]")

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
        "--ssl-certfile",
        ssl_certfile,
        "--ssl-keyfile",
        ssl_keyfile,
    ]
    if aws_profile:
        cmd.extend(["--profile", aws_profile])

    # Set up environment
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("TAPDB_"):
            env.pop(key, None)
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
        console.print("[dim]Auto-reload enabled (foreground mode)[/dim]")

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
            log_f.close()
            console.print("[red]✗[/red]  Server failed to start. Check logs:")
            console.print(f"   [dim]{log_file}[/dim]")
            # Show last few lines of error
            if log_file.exists():
                content = log_file.read_text().strip()
                if content:
                    console.print("\n[dim]--- Last error ---[/dim]")
                    for line in content.split("\n")[-10:]:
                        console.print(f"   {line}")
            raise typer.Exit(1)

        write_pid(_pid_file(), proc.pid)
        console.print(f"[green]✓[/green]  Server started (PID {proc.pid})")
        console.print(f"   URL: [cyan]https://{host}:{port}[/cyan]")
        console.print(f"   Logs: [dim]{log_file}[/dim]")
    else:
        console.print(f"[green]✓[/green]  Starting server on [cyan]https://{host}:{port}[/cyan]")
        console.print("   Press Ctrl+C to stop\n")
        try:
            result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
            if result.returncode != 0:
                raise typer.Exit(result.returncode)
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠[/yellow]  Server stopped")


@server_app.command("stop")
def stop():
    """Stop the Ursa API server."""
    stopped, msg = stop_pid(_pid_file())
    if stopped:
        console.print(f"[green]✓[/green]  {msg}")
    elif "Permission" in msg:
        console.print(f"[red]✗[/red]  {msg}")
        raise typer.Exit(1)
    else:
        console.print(f"[yellow]⚠[/yellow]  {msg}")


@server_app.command("status")
def status():
    """Check the status of the Ursa beta analysis API server."""
    pid = _get_pid()
    if pid:
        host, port = _resolved_server_host_port()
        log_file = latest_log(_log_dir())
        dh = display_host(host)
        console.print(f"[green]●[/green]  Server is [green]running[/green] (PID {pid})")
        console.print(f"   URL: [cyan]https://{dh}:{port}[/cyan]")
        if log_file:
            console.print(f"   Logs: [dim]{log_file}[/dim]")
    else:
        console.print("[dim]○[/dim]  Server is [dim]not running[/dim]")


@server_app.command("logs")
def logs(
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines to show"),
    all_logs: bool = typer.Option(False, "--all", "-a", help="List all log files"),
):
    """View and follow Ursa API server logs (Ctrl+C to stop)."""
    if all_logs:
        log_entries = list_logs(_log_dir())
        if not log_entries:
            console.print("[yellow]⚠[/yellow]  No log files found.")
            return
        console.print(f"[bold]Server log files ({len(log_entries)}):[/bold]")
        for lf in log_entries[:20]:
            size = lf.stat().st_size
            console.print(f"  {lf.name}  [dim]({size:,} bytes)[/dim]")
        return

    log_file = latest_log(_log_dir())
    if not log_file:
        console.print("[yellow]⚠[/yellow]  No log file found. Start the server first.")
        return

    console.print(f"[dim]Following {log_file.name} (Ctrl+C to stop)[/dim]\n")
    try:
        subprocess.run(["tail", "-f", "-n", str(lines), str(log_file)])
    except KeyboardInterrupt:
        console.print("\n")


@server_app.command("restart")
def restart(
    port: int | None = typer.Option(None, "--port", "-p", help="Port to run the server on"),
    host: str | None = typer.Option(None, "--host", "-h", help="Host to bind to"),
):
    """Restart the Ursa API server."""
    stop()
    time.sleep(1)
    start(port=port, host=host, reload=False, background=True)


def register(registry: CommandRegistry, spec: CliSpec) -> None:
    """cli-core-yo plugin: register server command group."""
    _ = spec
    registry.add_typer_app(None, server_app, "server", "API server management")
