"""GUI/Portal management commands for Ursa CLI."""

import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

gui_app = typer.Typer(help="GUI/Portal management commands")
console = Console()

# PID and log file locations (XDG Base Directory: ~/.config/ursa/)
CONFIG_DIR = Path.home() / ".config" / "ursa"
LOG_DIR = CONFIG_DIR / "logs"
CERTS_DIR = CONFIG_DIR / "certs"
PID_FILE = CONFIG_DIR / "server.pid"


def _require_auth_dependencies() -> None:
    """Fail fast if auth is requested but optional auth deps aren't installed."""

    try:
        import jose  # noqa: F401
    except ImportError:
        console.print("[red]✗[/red]  Authentication requested but python-jose is not installed")
        console.print("   Install with: [cyan]python -m pip install -e \".[auth]\"[/cyan]")
        raise typer.Exit(1)


def _ensure_dir():
    """Ensure ~/.config/ursa directories exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


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
            return pid
        except (ValueError, ProcessLookupError, PermissionError):
            PID_FILE.unlink(missing_ok=True)
    return None





@gui_app.command("start")
def start(
    port: int = typer.Option(8001, "--port", "-p", help="Port to run the server on"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    auth: bool = typer.Option(True, "--auth/--no-auth", help="Enable Cognito authentication"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload (foreground)"),
    background: bool = typer.Option(True, "--background/--foreground", "-b/-f", help="Run in background"),
    insecure_http: bool = typer.Option(False, "--insecure-http", help="Use HTTP instead of HTTPS (not recommended)"),
    cert: Optional[str] = typer.Option(None, "--cert", help="Path to SSL certificate file"),
    key: Optional[str] = typer.Option(None, "--key", help="Path to SSL private key file"),
):
    """Start the Ursa API server with HTTPS (default).

    By default, the server uses HTTPS with auto-generated self-signed certificates.
    Certificates are stored in ~/.config/ursa/certs/ and created automatically
    if they don't exist.

    For production, use --cert and --key to specify your own certificates.
    Use --insecure-http only for debugging (not recommended).

    Examples:
        ursa gui start                    # HTTPS with auto-generated cert
        ursa gui start --insecure-http    # HTTP only (shows warning)
        ursa gui start --cert /path/to/cert.pem --key /path/to/key.pem
    """
    _ensure_dir()

    # Configuration loaded from ~/.config/ursa/ursa-config.yaml via UrsaConfig
    # Environment variables can override YAML values

    # Override with env vars if set
    port = int(os.environ.get("URSA_PORT", port))
    host = os.environ.get("URSA_HOST", host)

    # Check if already running
    pid = _get_pid()
    if pid:
        console.print(f"[yellow]⚠[/yellow]  Server already running (PID {pid})")
        console.print(f"   URL: [cyan]http://{host}:{port}[/cyan]")
        return

    # Check AWS_PROFILE (from env or config file)
    from daylib.ursa_config import get_ursa_config, DEFAULT_CONFIG_PATH
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

    # Handle SSL/HTTPS configuration
    use_https = not insecure_http
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None

    # Check for certificate path environment variable overrides
    cert = cert or os.environ.get("URSA_SSL_CERT_PATH")
    key = key or os.environ.get("URSA_SSL_KEY_PATH")

    if use_https:
        from daylib.cli.certs import (
            DEFAULT_CERT_PATH,
            DEFAULT_KEY_PATH,
            certs_exist,
            generate_self_signed_cert,
            get_cert_info,
        )

        if cert and key:
            # Custom certificate paths provided
            ssl_cert_path = cert
            ssl_key_path = key
            if not Path(cert).exists():
                console.print(f"[red]✗[/red]  Certificate not found: {cert}")
                raise typer.Exit(1)
            if not Path(key).exists():
                console.print(f"[red]✗[/red]  Private key not found: {key}")
                raise typer.Exit(1)
            console.print(f"[green]✓[/green]  Using custom SSL certificate")
        else:
            # Use default certificate location
            ssl_cert_path = str(DEFAULT_CERT_PATH)
            ssl_key_path = str(DEFAULT_KEY_PATH)

            if not certs_exist():
                console.print("[dim]Generating self-signed certificate for HTTPS...[/dim]")
                generate_self_signed_cert()
                console.print("[green]✓[/green]  Self-signed certificate generated")

            # Check certificate validity
            info = get_cert_info()
            if info and info["is_expired"]:
                console.print("[yellow]⚠[/yellow]  Certificate expired, regenerating...")
                generate_self_signed_cert()
                info = get_cert_info()

            if info:
                console.print(f"[green]✓[/green]  HTTPS enabled (cert expires in {info['days_until_expiry']} days)")
    else:
        console.print("[yellow]⚠[/yellow]  [bold]INSECURE MODE:[/bold] Using HTTP instead of HTTPS")
        console.print("   [dim]This is not recommended for anything other than debugging[/dim]")

    # Build command (package-safe: uses module execution, not repo-relative bin/)
    cmd = [
        sys.executable,
        "-m",
        "daylib.workset_api_cli",
        "--host",
        host,
        "--port",
        str(port),
        "--profile",
        aws_profile,
        "--create-table",
    ]

    # Set up environment
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # Set auth env var for the API server
    if auth:
        _require_auth_dependencies()
        env["DAYLILY_ENABLE_AUTH"] = "true"
        # Pass Cognito config from ursa config to environment if not already set
        if ursa_config.cognito_user_pool_id and not os.environ.get("COGNITO_USER_POOL_ID"):
            env["COGNITO_USER_POOL_ID"] = ursa_config.cognito_user_pool_id
        if ursa_config.cognito_app_client_id and not os.environ.get("COGNITO_APP_CLIENT_ID"):
            env["COGNITO_APP_CLIENT_ID"] = ursa_config.cognito_app_client_id
        if ursa_config.cognito_region and not os.environ.get("COGNITO_REGION"):
            env["COGNITO_REGION"] = ursa_config.cognito_region

        missing: list[str] = []
        if not env.get("COGNITO_USER_POOL_ID"):
            missing.append("COGNITO_USER_POOL_ID")
        if not (env.get("COGNITO_APP_CLIENT_ID") or env.get("COGNITO_CLIENT_ID")):
            missing.append("COGNITO_APP_CLIENT_ID")
        if not env.get("COGNITO_REGION"):
            missing.append("COGNITO_REGION")
        if missing:
            console.print("[red]✗[/red]  Authentication enabled but Cognito config is missing")
            console.print("   Missing: [cyan]" + ", ".join(missing) + "[/cyan]")
            console.print("   Set via environment variables or in your Ursa config file")
            raise typer.Exit(1)
        console.print("[green]✓[/green]  Authentication ENABLED")
    else:
        env["DAYLILY_ENABLE_AUTH"] = "false"
        console.print("[yellow]⚠[/yellow]  Authentication DISABLED")

    # Pass SSL configuration to the API server
    if use_https and ssl_cert_path and ssl_key_path:
        env["URSA_SSL_CERT_PATH"] = ssl_cert_path
        env["URSA_SSL_KEY_PATH"] = ssl_key_path

    # Determine protocol for URL display
    protocol = "https" if use_https else "http"

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
        console.print(f"   URL: [cyan]{protocol}://{host}:{port}[/cyan]")
        console.print(f"   Portal: [cyan]{protocol}://{host}:{port}/portal[/cyan]")
        console.print(f"   Logs: [dim]{log_file}[/dim]")
        if use_https:
            console.print("")
            console.print("[dim]Note: Browsers will show a warning for self-signed certificates.[/dim]")
            console.print("[dim]      Click 'Advanced' → 'Proceed' to continue.[/dim]")
    else:
        console.print(f"[green]✓[/green]  Starting server on [cyan]{protocol}://{host}:{port}[/cyan]")
        console.print("   Press Ctrl+C to stop")
        if use_https:
            console.print("[dim]   Note: Accept the self-signed certificate warning in your browser[/dim]")
        console.print("")
        try:
            result = subprocess.run(cmd, cwd=Path.cwd(), env=env)
            if result.returncode != 0:
                raise typer.Exit(result.returncode)
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠[/yellow]  Server stopped")


@gui_app.command("stop")
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


@gui_app.command("status")
def status():
    """Check the status of the Ursa API server."""
    pid = _get_pid()
    if pid:
        port = os.environ.get("URSA_PORT", "8001")
        host = os.environ.get("URSA_HOST", "0.0.0.0")
        log_file = _get_latest_log()

        # Determine protocol based on certificate availability
        from daylib.cli.certs import certs_exist
        ssl_cert = os.environ.get("URSA_SSL_CERT_PATH")
        ssl_key = os.environ.get("URSA_SSL_KEY_PATH")
        use_https = (ssl_cert and ssl_key) or certs_exist()
        protocol = "https" if use_https else "http"

        console.print(f"[green]●[/green]  Server is [green]running[/green] (PID {pid})")
        console.print(f"   URL: [cyan]{protocol}://{host}:{port}[/cyan]")
        if log_file:
            console.print(f"   Logs: [dim]{log_file}[/dim]")
    else:
        console.print("[dim]○[/dim]  Server is [dim]not running[/dim]")


@gui_app.command("logs")
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


@gui_app.command("restart")
def restart(
    port: int = typer.Option(8001, "--port", "-p", help="Port to run the server on"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    auth: bool = typer.Option(True, "--auth/--no-auth", help="Enable Cognito authentication"),
    insecure_http: bool = typer.Option(False, "--insecure-http", help="Use HTTP instead of HTTPS (not recommended)"),
    cert: Optional[str] = typer.Option(None, "--cert", help="Path to SSL certificate file"),
    key: Optional[str] = typer.Option(None, "--key", help="Path to SSL private key file"),
):
    """Restart the Ursa API server."""
    stop()
    time.sleep(1)
    start(
        port=port,
        host=host,
        auth=auth,
        reload=False,
        background=True,
        insecure_http=insecure_http,
        cert=cert,
        key=key,
    )


@gui_app.command("generate-cert")
def generate_cert(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing certificates"),
    days: int = typer.Option(365, "--days", "-d", help="Certificate validity in days"),
    cn: str = typer.Option("localhost", "--cn", help="Common Name for the certificate"),
):
    """Generate self-signed SSL certificate for HTTPS.

    Creates a self-signed certificate and private key in ~/.config/ursa/certs/.
    These are used automatically when starting the server with HTTPS.

    Examples:
        ursa gui generate-cert
        ursa gui generate-cert --force --days 730
        ursa gui generate-cert --cn myhost.local
    """
    from daylib.cli.certs import (
        DEFAULT_CERT_PATH,
        DEFAULT_KEY_PATH,
        DEFAULT_SANS,
        certs_exist,
        generate_self_signed_cert,
        get_cert_info,
    )

    if certs_exist() and not force:
        console.print("[yellow]⚠[/yellow]  Certificates already exist:")
        console.print(f"   Certificate: [dim]{DEFAULT_CERT_PATH}[/dim]")
        console.print(f"   Private key: [dim]{DEFAULT_KEY_PATH}[/dim]")
        console.print("   Use [cyan]--force[/cyan] to regenerate")
        return

    console.print("[dim]Generating self-signed certificate...[/dim]")
    cert_path, key_path = generate_self_signed_cert(
        cn=cn,
        days=days,
        sans=DEFAULT_SANS,
    )

    info = get_cert_info(cert_path)
    if info:
        console.print(f"[green]✓[/green]  Certificate generated successfully")
        console.print(f"   Certificate: [cyan]{cert_path}[/cyan]")
        console.print(f"   Private key: [cyan]{key_path}[/cyan]")
        console.print(f"   Common Name: [dim]{cn}[/dim]")
        console.print(f"   SANs: [dim]{', '.join(info['sans'])}[/dim]")
        console.print(f"   Valid until: [dim]{info['not_valid_after'].strftime('%Y-%m-%d')}[/dim] ({info['days_until_expiry']} days)")
        console.print(f"   Fingerprint: [dim]{info['fingerprint_sha256'][:47]}...[/dim]")
        console.print("")
        console.print("[yellow]Note:[/yellow] Browsers will show a warning for self-signed certificates.")
        console.print("      This is normal for development. Click 'Advanced' → 'Proceed' to continue.")


@gui_app.command("cert-info")
def cert_info(
    cert_path: Optional[str] = typer.Option(None, "--cert", help="Path to certificate file"),
):
    """Display information about the current SSL certificate.

    Shows details about the certificate including expiration date,
    fingerprint, and Subject Alternative Names (SANs).
    """
    from daylib.cli.certs import DEFAULT_CERT_PATH, get_cert_info

    path = Path(cert_path) if cert_path else DEFAULT_CERT_PATH
    info = get_cert_info(path)

    if not info:
        console.print(f"[yellow]⚠[/yellow]  No certificate found at: [dim]{path}[/dim]")
        console.print("   Run [cyan]ursa gui generate-cert[/cyan] to create one")
        return

    table = Table(title="SSL Certificate Information", show_header=False)
    table.add_column("Property", style="bold")
    table.add_column("Value")

    table.add_row("Path", info["path"])
    table.add_row("Subject", info["subject"])
    table.add_row("Issuer", info["issuer"])
    table.add_row("SANs", ", ".join(info["sans"]))
    table.add_row("Valid From", info["not_valid_before"].strftime("%Y-%m-%d %H:%M:%S UTC"))
    table.add_row("Valid Until", info["not_valid_after"].strftime("%Y-%m-%d %H:%M:%S UTC"))

    if info["is_expired"]:
        table.add_row("Status", "[red]EXPIRED[/red]")
    elif info["days_until_expiry"] < 30:
        table.add_row("Status", f"[yellow]Expires in {info['days_until_expiry']} days[/yellow]")
    else:
        table.add_row("Status", f"[green]Valid[/green] ({info['days_until_expiry']} days remaining)")

    table.add_row("SHA-256 Fingerprint", info["fingerprint_sha256"])

    console.print(table)
