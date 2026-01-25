"""Server management commands for Ursa CLI."""

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

server_app = typer.Typer(help="API server management commands")
console = Console()

# PID and log file locations
CONFIG_DIR = Path.home() / ".ursa"
LOG_DIR = CONFIG_DIR / "logs"
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
    """Ensure .ursa directories exist."""
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
    port: int = typer.Option(8001, "--port", "-p", help="Port to run the server on"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    auth: bool = typer.Option(True, "--auth/--no-auth", help="Enable Cognito authentication"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload (foreground)"),
    background: bool = typer.Option(True, "--background/--foreground", "-b/-f", help="Run in background"),
):
    """Start the Ursa API server."""
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
        console.print(f"   URL: [cyan]http://{host}:{port}[/cyan]")
        console.print(f"   Portal: [cyan]http://{host}:{port}/portal[/cyan]")
        console.print(f"   Logs: [dim]{log_file}[/dim]")
    else:
        console.print(f"[green]✓[/green]  Starting server on [cyan]http://{host}:{port}[/cyan]")
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
    """Check the status of the Ursa API server."""
    pid = _get_pid()
    if pid:
        port = os.environ.get("URSA_PORT", "8001")
        host = os.environ.get("URSA_HOST", "0.0.0.0")
        log_file = _get_latest_log()
        console.print(f"[green]●[/green]  Server is [green]running[/green] (PID {pid})")
        console.print(f"   URL: [cyan]http://{host}:{port}[/cyan]")
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
    port: int = typer.Option(8001, "--port", "-p", help="Port to run the server on"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    auth: bool = typer.Option(True, "--auth/--no-auth", help="Enable Cognito authentication"),
):
    """Restart the Ursa API server."""
    stop()
    time.sleep(1)
    start(port=port, host=host, auth=auth, reload=False, background=True)

