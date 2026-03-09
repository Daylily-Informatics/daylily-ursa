"""Server management commands for the Ursa beta analysis API."""

import os
import signal
import subprocess
import sys
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import typer
import boto3
from rich.console import Console

server_app = typer.Typer(help="API server management commands")
console = Console()

# PID and log file locations (XDG Base Directory: ~/.config/ursa/)
CONFIG_DIR = Path.home() / ".config" / "ursa"
LOG_DIR = CONFIG_DIR / "logs"
PID_FILE = CONFIG_DIR / "server.pid"
CERT_DIR = CONFIG_DIR / "certs"
DEFAULT_SSL_CERT_FILE = CERT_DIR / "ursa-localhost.pem"
DEFAULT_SSL_KEY_FILE = CERT_DIR / "ursa-localhost-key.pem"
_LOCAL_OAUTH_HOSTS = {"localhost", "127.0.0.1", "::1"}
REQUIRED_COGNITO_APP_CLIENT_NAME = "ursa"


def _require_auth_dependencies() -> None:
    """Fail fast if auth is requested but optional auth deps aren't installed."""

    try:
        import jose  # noqa: F401
    except ImportError:
        console.print("[red]✗[/red]  Authentication requested but python-jose is not installed")
        console.print('   Install with: [cyan]python -m pip install -e ".[auth]"[/cyan]')
        raise typer.Exit(1)


def _ensure_dir():
    """Ensure ~/.config/ursa directories exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CERT_DIR.mkdir(parents=True, exist_ok=True)


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

    cert_path = DEFAULT_SSL_CERT_FILE
    key_path = DEFAULT_SSL_KEY_FILE
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


def _runtime_oauth_host(host: str) -> str:
    """Resolve runtime callback host for browser-facing URLs."""
    if host in ("0.0.0.0", "::"):
        return "localhost"
    return host


def _default_port_for_scheme(scheme: str) -> Optional[int]:
    """Return implicit port for known URI schemes."""
    if scheme == "https":
        return 443
    if scheme == "http":
        return 80
    return None


def _normalize_uri(uri: str) -> str:
    """Normalize URI for reliable comparison."""
    parsed = urlparse(uri.strip())
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    normalized = parsed._replace(path=path, params="", query="", fragment="")
    return normalized.geturl()


def _uri_port(uri: str) -> Optional[int]:
    """Resolve explicit or implicit URI port."""
    parsed = urlparse(uri.strip())
    if not parsed.scheme or not parsed.netloc:
        return None
    return parsed.port or _default_port_for_scheme(parsed.scheme.lower())


def _is_local_oauth_uri(uri: str, runtime_host: str) -> bool:
    """Return True when URI points to local runtime host."""
    parsed = urlparse(uri.strip())
    hostname = (parsed.hostname or "").lower()
    return hostname in _LOCAL_OAUTH_HOSTS or hostname == runtime_host.lower()


def _validate_uri_list_ports(
    *,
    uris: list[str],
    label: str,
    expected_port: int,
    runtime_host: str,
) -> list[str]:
    """Validate URI structure and port alignment for local endpoints."""
    errors: list[str] = []
    for raw_uri in uris:
        uri = raw_uri.strip()
        parsed = urlparse(uri)
        if not parsed.scheme or not parsed.netloc:
            errors.append(f"{label} contains invalid URI: {uri}")
            continue
        if parsed.scheme not in {"http", "https"}:
            errors.append(f"{label} contains unsupported URI scheme: {uri}")
            continue
        if _is_local_oauth_uri(uri, runtime_host):
            uri_port = _uri_port(uri)
            if uri_port != expected_port:
                errors.append(
                    f"{label} URI port mismatch for local endpoint: {uri} "
                    f"(expected port {expected_port})"
                )
    return errors


def _validate_cognito_oauth_uris(
    *,
    app_client: dict,
    expected_callback_url: str,
    expected_logout_url: str,
    expected_port: int,
    runtime_host: str,
    expected_client_name: str = REQUIRED_COGNITO_APP_CLIENT_NAME,
) -> list[str]:
    """Validate Cognito app-client OAuth URLs against runtime expectations."""
    errors: list[str] = []
    actual_client_name = str(app_client.get("ClientName") or "").strip()
    callback_urls = [str(u) for u in (app_client.get("CallbackURLs") or []) if u]
    logout_urls = [str(u) for u in (app_client.get("LogoutURLs") or []) if u]
    default_redirect_uri = str(app_client.get("DefaultRedirectURI") or "").strip()

    if not actual_client_name:
        errors.append("Cognito app client has no ClientName configured")
    elif actual_client_name != expected_client_name:
        errors.append(
            "Cognito app client name mismatch: "
            f"found '{actual_client_name}', expected '{expected_client_name}'"
        )
    if not app_client.get("AllowedOAuthFlowsUserPoolClient", False):
        errors.append("Cognito app client does not have OAuth2 flows enabled")
    if not callback_urls:
        errors.append("Cognito app client has no CallbackURLs configured")
    if not logout_urls:
        errors.append("Cognito app client has no LogoutURLs configured")

    errors.extend(
        _validate_uri_list_ports(
            uris=callback_urls,
            label="CallbackURLs",
            expected_port=expected_port,
            runtime_host=runtime_host,
        )
    )
    errors.extend(
        _validate_uri_list_ports(
            uris=logout_urls,
            label="LogoutURLs",
            expected_port=expected_port,
            runtime_host=runtime_host,
        )
    )
    if default_redirect_uri:
        errors.extend(
            _validate_uri_list_ports(
                uris=[default_redirect_uri],
                label="DefaultRedirectURI",
                expected_port=expected_port,
                runtime_host=runtime_host,
            )
        )

    normalized_callbacks = {_normalize_uri(u) for u in callback_urls}
    normalized_logouts = {_normalize_uri(u) for u in logout_urls}
    normalized_expected_callback = _normalize_uri(expected_callback_url)
    normalized_expected_logout = _normalize_uri(expected_logout_url)
    normalized_default_redirect = (
        _normalize_uri(default_redirect_uri) if default_redirect_uri else ""
    )

    if normalized_expected_callback not in normalized_callbacks:
        errors.append(
            "Expected callback URI is not configured in Cognito app client: "
            f"{expected_callback_url}"
        )
    if normalized_expected_logout not in normalized_logouts:
        errors.append(
            f"Expected logout URI is not configured in Cognito app client: {expected_logout_url}"
        )
    if default_redirect_uri and normalized_default_redirect not in normalized_callbacks:
        errors.append(
            f"Cognito app client DefaultRedirectURI is not in CallbackURLs: {default_redirect_uri}"
        )

    errors.extend(
        _validate_uri_list_ports(
            uris=[expected_callback_url, expected_logout_url],
            label="Configured OAuth URI",
            expected_port=expected_port,
            runtime_host=runtime_host,
        )
    )
    return errors


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


def _require_cognito_configuration(ursa_config) -> None:
    """Require Cognito configuration and project it into env vars."""
    field_map = {
        "COGNITO_USER_POOL_ID": "cognito_user_pool_id",
        "COGNITO_APP_CLIENT_ID": "cognito_app_client_id",
        "COGNITO_REGION": "cognito_region",
        "COGNITO_DOMAIN": "cognito_domain",
    }
    missing: list[str] = []
    for env_key, attr_name in field_map.items():
        if not os.environ.get(env_key):
            value = getattr(ursa_config, attr_name, None)
            if value:
                os.environ[env_key] = str(value)
            else:
                missing.append(env_key)
    if missing:
        console.print("[red]✗[/red]  Authentication is mandatory but Cognito config is missing")
        console.print("   Missing: [cyan]" + ", ".join(missing) + "[/cyan]")
        raise typer.Exit(1)


def _get_log_file() -> Path:
    """Get timestamped log file path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"server_{ts}.log"


def _get_latest_log() -> Optional[Path]:
    """Get the most recent log file."""
    logs = sorted(LOG_DIR.glob("server_*.log"), reverse=True)
    return logs[0] if logs else None


def _get_pid() -> Optional[int]:
    """Get the running server PID if exists."""
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)
            cmdline = subprocess.check_output(
                ["ps", "-p", str(pid), "-o", "command="],
                text=True,
            ).strip()
            if "daylib_ursa.workset_api_cli" not in cmdline:
                PID_FILE.unlink(missing_ok=True)
                return None
            return pid
        except (ValueError, ProcessLookupError, PermissionError, subprocess.SubprocessError):
            PID_FILE.unlink(missing_ok=True)
    return None


def _source_env_file() -> bool:
    """Source .env file if it exists."""
    env_file = Path.cwd() / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    # Remove quotes if present
                    value = value.strip().strip('"').strip("'")
                    os.environ[key.strip()] = value
        return True
    return False


@server_app.command("start")
def start(
    port: int = typer.Option(8914, "--port", "-p", help="Port to run the server on"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload (foreground)"),
    background: bool = typer.Option(
        True, "--background/--foreground", "-b/-f", help="Run in background"
    ),
):
    """Start the Ursa beta analysis API server."""
    _ensure_dir()

    # Source .env file
    if _source_env_file():
        console.print("[dim]Loaded .env file[/dim]")

    # Override with env vars if set
    port = int(os.environ.get("URSA_PORT", port))
    host = os.environ.get("URSA_HOST", host)

    # Check if already running
    pid = _get_pid()
    if pid:
        console.print(f"[yellow]⚠[/yellow]  Server already running (PID {pid})")
        console.print(f"   URL: [cyan]https://{host}:{port}[/cyan]")
        return

    # Check AWS_PROFILE (from env or config file)
    from daylib_ursa.ursa_config import get_ursa_config, DEFAULT_CONFIG_PATH

    ursa_config = get_ursa_config()

    aws_profile = os.environ.get("AWS_PROFILE") or ursa_config.aws_profile
    if not aws_profile:
        console.print("[red]✗[/red]  AWS_PROFILE not set")
        console.print("   Set via environment: [cyan]export AWS_PROFILE=your-profile[/cyan]")
        console.print(f"   Or in config file:   [cyan]{DEFAULT_CONFIG_PATH}[/cyan]")
        raise typer.Exit(1)

    # Set in environment for subprocess and boto3
    if not os.environ.get("AWS_PROFILE"):
        os.environ["AWS_PROFILE"] = aws_profile

    _require_auth_dependencies()
    _require_cognito_configuration(ursa_config)

    aws_region = (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or (ursa_config.get_allowed_regions()[0] if ursa_config.is_configured else "us-west-2")
    )

    ssl_certfile, ssl_keyfile = _resolve_https_cert_paths(host)

    # Check config file for region configuration
    if not ursa_config.is_configured:
        console.print(f"[yellow]⚠[/yellow]  No regions configured in {DEFAULT_CONFIG_PATH}")
        console.print("   Cluster discovery requires region definitions.")
        console.print(f"   Create [cyan]{DEFAULT_CONFIG_PATH}[/cyan] with:")
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
        "--profile",
        aws_profile,
        "--region",
        aws_region,
        "--bootstrap-tapdb",
        "--ssl-certfile",
        ssl_certfile,
        "--ssl-keyfile",
        ssl_keyfile,
    ]

    # Set up environment
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["ENABLE_AUTH"] = "true"

    if reload:
        cmd.append("--reload")
        background = False  # Reload requires foreground
        console.print("[dim]Auto-reload enabled (foreground mode)[/dim]")

    if background:
        log_file = _get_log_file()
        log_f = open(log_file, "w", buffering=1)  # Line-buffered

        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=Path.cwd(),
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

        PID_FILE.write_text(str(proc.pid))
        console.print(f"[green]✓[/green]  Server started (PID {proc.pid})")
        console.print(f"   URL: [cyan]https://{host}:{port}[/cyan]")
        console.print(f"   Logs: [dim]{log_file}[/dim]")
    else:
        console.print(f"[green]✓[/green]  Starting server on [cyan]https://{host}:{port}[/cyan]")
        console.print("   Press Ctrl+C to stop\n")
        try:
            result = subprocess.run(cmd, cwd=Path.cwd(), env=env)
            if result.returncode != 0:
                raise typer.Exit(result.returncode)
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠[/yellow]  Server stopped")


@server_app.command("stop")
def stop():
    """Stop the Ursa API server."""
    pid = _get_pid()
    if not pid:
        console.print("[yellow]⚠[/yellow]  No server running")
        return

    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(10):
            time.sleep(0.5)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
        else:
            os.kill(pid, signal.SIGKILL)

        PID_FILE.unlink(missing_ok=True)
        console.print(f"[green]✓[/green]  Server stopped (was PID {pid})")
    except ProcessLookupError:
        PID_FILE.unlink(missing_ok=True)
        console.print("[yellow]⚠[/yellow]  Server was not running")
    except PermissionError:
        console.print(f"[red]✗[/red]  Permission denied stopping PID {pid}")
        raise typer.Exit(1)


@server_app.command("status")
def status():
    """Check the status of the Ursa beta analysis API server."""
    pid = _get_pid()
    if pid:
        port = os.environ.get("URSA_PORT", "8914")
        host = os.environ.get("URSA_HOST", "0.0.0.0")
        log_file = _get_latest_log()
        console.print(f"[green]●[/green]  Server is [green]running[/green] (PID {pid})")
        console.print(f"   URL: [cyan]https://{host}:{port}[/cyan]")
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
        log_files = sorted(LOG_DIR.glob("server_*.log"), reverse=True)
        if not log_files:
            console.print("[yellow]⚠[/yellow]  No log files found.")
            return
        console.print(f"[bold]Server log files ({len(log_files)}):[/bold]")
        for lf in log_files[:20]:
            size = lf.stat().st_size
            console.print(f"  {lf.name}  [dim]({size:,} bytes)[/dim]")
        return

    log_file = _get_latest_log()
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
    port: int = typer.Option(8914, "--port", "-p", help="Port to run the server on"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
):
    """Restart the Ursa API server."""
    stop()
    time.sleep(1)
    start(port=port, host=host, reload=False, background=True)
