"""Monitor management commands for Ursa CLI."""

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

monitor_app = typer.Typer(help="Workset monitor management commands")
console = Console()

# PID and log file locations
CONFIG_DIR = Path.home() / ".ursa"
LOG_DIR = CONFIG_DIR / "logs"
PID_FILE = CONFIG_DIR / "monitor.pid"


def _ensure_dir():
    """Ensure .ursa directories exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _get_log_file() -> Path:
    """Get timestamped log file path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"monitor_{ts}.log"


def _get_latest_log() -> Optional[Path]:
    """Get the most recent log file."""
    logs = sorted(LOG_DIR.glob("monitor_*.log"), reverse=True)
    return logs[0] if logs else None


def _get_pid() -> Optional[int]:
    """Get the running monitor PID if exists."""
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)
            return pid
        except (ValueError, ProcessLookupError, PermissionError):
            PID_FILE.unlink(missing_ok=True)
    return None


def _find_monitor_script() -> Path:
    """Find the daylily-workset-monitor script."""
    # Check in bin/ relative to package
    pkg_bin = Path(__file__).parent.parent.parent / "bin" / "daylily-workset-monitor"
    if pkg_bin.exists():
        return pkg_bin
    # Check in current directory
    cwd_bin = Path.cwd() / "bin" / "daylily-workset-monitor"
    if cwd_bin.exists():
        return cwd_bin
    # Fallback to PATH
    return Path("daylily-workset-monitor")


def _find_default_config() -> Optional[Path]:
    """Find default monitor config file."""
    search_paths = [
        Path.cwd() / "config" / "workset-monitor-config.yaml",
        Path.cwd() / "config" / "daylily-workset-monitor.yaml",
        CONFIG_DIR / "monitor-config.yaml",
    ]
    for p in search_paths:
        if p.exists():
            return p
    return None


@monitor_app.command("start")
def start(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to monitor config YAML"),
    once: bool = typer.Option(False, "--once", help="Run single iteration and exit"),
    verbose: bool = typer.Option(True, "--verbose/--quiet", "-v/-q", help="Enable verbose logging (default: on)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Do not mutate S3 or execute commands"),
    background: bool = typer.Option(True, "--background/--foreground", "-b/-f", help="Run in background"),
    enable_dynamodb: bool = typer.Option(True, "--enable-dynamodb/--no-dynamodb", help="Enable DynamoDB state tracking (default: on)"),
    dynamodb_table: str = typer.Option("daylily-worksets", "--dynamodb-table", help="DynamoDB table name"),
    parallel: Optional[int] = typer.Option(None, "--parallel", "-p", help="Maximum number of worksets to run in parallel (overrides config file)"),
):
    """Start the workset monitor daemon."""
    _ensure_dir()

    # Check if already running
    pid = _get_pid()
    if pid:
        console.print(f"[yellow]⚠[/yellow]  Monitor already running (PID {pid})")
        return

    # Find config file
    if config is None:
        config = _find_default_config()

    if config is None or not config.exists():
        console.print("[red]✗[/red]  No monitor config file found")
        console.print("   Provide one with: [cyan]ursa monitor start --config path/to/config.yaml[/cyan]")
        console.print("   Or create: [cyan]~/.ursa/monitor-config.yaml[/cyan]")
        console.print("   Or create: [cyan]./config/workset-monitor-config.yaml[/cyan]")
        raise typer.Exit(1)

    # Check ursa-config.yaml for region configuration
    from daylib.ursa_config import get_ursa_config, DEFAULT_CONFIG_PATH
    ursa_config = get_ursa_config()
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

    # Find monitor script
    monitor_script = _find_monitor_script()

    # Build command
    cmd = [sys.executable, str(monitor_script), str(config)]
    if once:
        cmd.append("--once")
    if verbose:
        cmd.append("--verbose")
    if dry_run:
        cmd.append("--dry-run")
    if enable_dynamodb:
        cmd.extend(["--enable-dynamodb", "--dynamodb-table", dynamodb_table])
    if parallel is not None:
        cmd.extend(["--parallel", str(parallel)])

    if background:
        log_file = _get_log_file()
        log_f = open(log_file, "w", buffering=1)  # Line-buffered

        # Set PYTHONUNBUFFERED for immediate output
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

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
            console.print("[red]✗[/red]  Monitor failed to start. Check logs:")
            console.print(f"   [dim]{log_file}[/dim]")
            if log_file.exists():
                content = log_file.read_text().strip()
                if content:
                    console.print("\n[dim]--- Last error ---[/dim]")
                    for line in content.split("\n")[-10:]:
                        console.print(f"   {line}")
            raise typer.Exit(1)

        PID_FILE.write_text(str(proc.pid))
        console.print(f"[green]✓[/green]  Monitor started (PID {proc.pid})")
        console.print(f"   Config: [dim]{config}[/dim]")
        console.print(f"   Logs: [dim]{log_file}[/dim]")
    else:
        console.print("[green]✓[/green]  Starting monitor")
        console.print(f"   Config: [dim]{config}[/dim]")
        console.print("   Press Ctrl+C to stop\n")
        try:
            subprocess.run(cmd, cwd=Path.cwd())
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠[/yellow]  Monitor stopped")


@monitor_app.command("stop")
def stop():
    """Stop the workset monitor daemon."""
    pid = _get_pid()
    if not pid:
        console.print("[yellow]⚠[/yellow]  No monitor running")
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
        console.print(f"[green]✓[/green]  Monitor stopped (was PID {pid})")
    except ProcessLookupError:
        PID_FILE.unlink(missing_ok=True)
        console.print("[yellow]⚠[/yellow]  Monitor was not running")
    except PermissionError:
        console.print(f"[red]✗[/red]  Permission denied stopping PID {pid}")
        raise typer.Exit(1)


@monitor_app.command("status")
def status():
    """Check the status of the workset monitor."""
    pid = _get_pid()
    if pid:
        log_file = _get_latest_log()
        console.print(f"[green]●[/green]  Monitor is [green]running[/green] (PID {pid})")
        if log_file:
            console.print(f"   Logs: [dim]{log_file}[/dim]")
    else:
        console.print("[dim]○[/dim]  Monitor is [dim]not running[/dim]")


@monitor_app.command("logs")
def logs(
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines to show"),
    all_logs: bool = typer.Option(False, "--all", "-a", help="List all log files"),
):
    """View and follow workset monitor logs (Ctrl+C to stop)."""
    if all_logs:
        log_files = sorted(LOG_DIR.glob("monitor_*.log"), reverse=True)
        if not log_files:
            console.print("[yellow]⚠[/yellow]  No log files found.")
            return
        console.print(f"[bold]Monitor log files ({len(log_files)}):[/bold]")
        for lf in log_files[:20]:
            size = lf.stat().st_size
            console.print(f"  {lf.name}  [dim]({size:,} bytes)[/dim]")
        return

    log_file = _get_latest_log()
    if not log_file:
        console.print("[yellow]⚠[/yellow]  No log file found. Start the monitor first.")
        return

    console.print(f"[dim]Following {log_file.name} (Ctrl+C to stop)[/dim]\n")
    try:
        subprocess.run(["tail", "-f", "-n", str(lines), str(log_file)])
    except KeyboardInterrupt:
        console.print("\n")


@monitor_app.command("grep")
def grep_logs(
    pattern: str = typer.Argument(..., help="Pattern to search for in logs"),
    all_logs: bool = typer.Option(False, "--all", "-a", help="Search all log files, not just the latest"),
    ignore_case: bool = typer.Option(True, "--ignore-case/--case-sensitive", "-i/-s", help="Case-insensitive search (default)"),
    context: int = typer.Option(0, "--context", "-C", help="Lines of context before and after match"),
    count: bool = typer.Option(False, "--count", "-c", help="Only show count of matching lines"),
):
    """Search monitor logs for a pattern.

    Examples:
        ursa monitor grep 'error'
        ursa monitor grep 'workset-123' --all
        ursa monitor grep 'failed' -C 3
    """
    if all_logs:
        log_files = sorted(LOG_DIR.glob("monitor_*.log"), reverse=True)
    else:
        latest = _get_latest_log()
        log_files = [latest] if latest else []

    if not log_files:
        console.print("[yellow]⚠[/yellow]  No log files found.")
        return

    # Build grep command
    grep_cmd = ["grep", "--color=always"]
    if ignore_case:
        grep_cmd.append("-i")
    if context > 0:
        grep_cmd.extend(["-C", str(context)])
    if count:
        grep_cmd.append("-c")
    if len(log_files) > 1:
        grep_cmd.append("-H")  # Show filename for multiple files

    grep_cmd.append(pattern)
    grep_cmd.extend(str(lf) for lf in log_files)

    console.print(f"[dim]Searching {len(log_files)} log file(s) for '{pattern}'[/dim]\n")

    result = subprocess.run(grep_cmd)

    if result.returncode == 1:
        console.print(f"\n[yellow]⚠[/yellow]  No matches found for '{pattern}'")

