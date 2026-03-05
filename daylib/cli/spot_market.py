"""Spot market tracker management commands for Ursa CLI."""

from __future__ import annotations

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

spot_market_app = typer.Typer(help="Spot market tracker daemon management commands")
console = Console()

CONFIG_DIR = Path.home() / ".ursa" / "spot-market"
DAEMON_LOG_DIR = CONFIG_DIR / "daemon-logs"
PID_FILE = CONFIG_DIR / "spot-market.pid"


def _ensure_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    DAEMON_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _get_log_file() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DAEMON_LOG_DIR / f"spot_market_{ts}.log"


def _get_latest_log() -> Optional[Path]:
    logs = sorted(DAEMON_LOG_DIR.glob("spot_market_*.log"), reverse=True)
    return logs[0] if logs else None


def _get_pid() -> Optional[int]:
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)
            return pid
        except (ValueError, ProcessLookupError, PermissionError):
            PID_FILE.unlink(missing_ok=True)
    return None


@spot_market_app.command("start")
def start(
    background: bool = typer.Option(True, "--background/--foreground", "-b/-f", help="Run in background"),
):
    """Start the spot market tracker daemon."""
    _ensure_dir()

    pid = _get_pid()
    if pid:
        console.print(f"[yellow]⚠[/yellow]  Spot market daemon already running (PID {pid})")
        return

    cmd = [sys.executable, "-m", "daylib.spot_market.daemon"]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    if background:
        log_file = _get_log_file()
        log_f = open(log_file, "w", buffering=1)
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=Path.cwd(),
            env=env,
        )

        time.sleep(1)
        if proc.poll() is not None:
            log_f.close()
            console.print("[red]✗[/red]  Spot market daemon failed to start. Check logs:")
            console.print(f"   [dim]{log_file}[/dim]")
            raise typer.Exit(1)

        PID_FILE.write_text(str(proc.pid))
        console.print(f"[green]✓[/green]  Spot market daemon started (PID {proc.pid})")
        console.print(f"   Logs: [dim]{log_file}[/dim]")
    else:
        console.print("[green]✓[/green]  Starting spot market daemon (foreground)")
        console.print("   Press Ctrl+C to stop\n")
        try:
            subprocess.run(cmd, cwd=Path.cwd(), env=env)
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠[/yellow]  Spot market daemon stopped")


@spot_market_app.command("stop")
def stop():
    """Stop the spot market tracker daemon."""
    pid = _get_pid()
    if not pid:
        console.print("[yellow]⚠[/yellow]  No spot market daemon running")
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
        console.print(f"[green]✓[/green]  Spot market daemon stopped (was PID {pid})")
    except ProcessLookupError:
        PID_FILE.unlink(missing_ok=True)
        console.print("[yellow]⚠[/yellow]  Spot market daemon was not running")
    except PermissionError:
        console.print(f"[red]✗[/red]  Permission denied stopping PID {pid}")
        raise typer.Exit(1)


@spot_market_app.command("status")
def status():
    """Check the status of the spot market tracker daemon."""
    pid = _get_pid()
    if pid:
        log_file = _get_latest_log()
        console.print(f"[green]●[/green]  Spot market daemon is [green]running[/green] (PID {pid})")
        if log_file:
            console.print(f"   Logs: [dim]{log_file}[/dim]")
    else:
        console.print("[dim]○[/dim]  Spot market daemon is [dim]not running[/dim]")


@spot_market_app.command("logs")
def logs(
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines to show"),
    all_logs: bool = typer.Option(False, "--all", "-a", help="List all log files"),
):
    """View and follow spot market daemon logs (Ctrl+C to stop)."""
    _ensure_dir()

    if all_logs:
        log_files = sorted(DAEMON_LOG_DIR.glob("spot_market_*.log"), reverse=True)
        if not log_files:
            console.print("[yellow]⚠[/yellow]  No log files found.")
            return
        console.print(f"[bold]Spot market log files ({len(log_files)}):[/bold]")
        for lf in log_files[:20]:
            size = lf.stat().st_size
            console.print(f"  {lf.name}  [dim]({size:,} bytes)[/dim]")
        return

    log_file = _get_latest_log()
    if not log_file:
        console.print("[yellow]⚠[/yellow]  No log file found. Start the daemon first.")
        return

    console.print(f"[dim]Following {log_file.name} (Ctrl+C to stop)[/dim]\n")
    try:
        subprocess.run(["tail", "-f", "-n", str(lines), str(log_file)])
    except KeyboardInterrupt:
        console.print("\n")

