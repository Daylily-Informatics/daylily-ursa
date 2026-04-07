"""Monitor management commands for Ursa CLI."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer
from daylib_ursa.ursa_config import get_config_dir, _resolve_deployment_code
from cli_core_yo import output as cli_output

if TYPE_CHECKING:
    from cli_core_yo.registry import CommandRegistry
    from cli_core_yo.spec import CliSpec

monitor_app = typer.Typer(help="Workset monitor management commands")


def _config_dir() -> Path:
    return get_config_dir()


def _log_dir() -> Path:
    return _config_dir() / "logs"


def _pid_file() -> Path:
    return _config_dir() / "monitor.pid"


def _default_monitor_config_path() -> Path:
    return _config_dir() / f"monitor-config-{_resolve_deployment_code()}.yaml"


def _ensure_dir():
    """Ensure XDG-style Ursa monitor directories exist."""
    _config_dir().mkdir(parents=True, exist_ok=True)
    _log_dir().mkdir(parents=True, exist_ok=True)


def _get_log_file() -> Path:
    """Get timestamped log file path."""
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return _log_dir() / f"monitor_{ts}.log"


def _get_latest_log() -> Optional[Path]:
    """Get the most recent log file."""
    logs = sorted(_log_dir().glob("monitor_*.log"), reverse=True)
    return logs[0] if logs else None


def _get_pid() -> Optional[int]:
    """Get the running monitor PID if exists."""
    pid_file = _pid_file()
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
            return pid
        except (ValueError, ProcessLookupError, PermissionError):
            pid_file.unlink(missing_ok=True)
    return None


def _find_default_config() -> Optional[Path]:
    """Find default monitor config file."""
    search_paths = [
        _default_monitor_config_path(),
        Path.cwd() / "config" / "workset-monitor-config.yaml",
        Path.cwd() / "config" / "daylily-workset-monitor.yaml",
    ]
    for p in search_paths:
        if p.exists():
            return p
    return None


@monitor_app.command("start")
def start(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to monitor config YAML"
    ),
    once: bool = typer.Option(False, "--once", help="Run single iteration and exit"),
    verbose: bool = typer.Option(
        True, "--verbose/--quiet", "-v/-q", help="Enable verbose logging (default: on)"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Do not mutate S3 or execute commands"),
    background: bool = typer.Option(
        True, "--background/--foreground", "-b/-f", help="Run in background"
    ),
    enable_tapdb: bool = typer.Option(
        True, "--enable-tapdb/--no-tapdb", help="Enable TapDB state tracking (default: on)"
    ),
    parallel: Optional[int] = typer.Option(
        None,
        "--parallel",
        "-p",
        help="Maximum number of worksets to run in parallel (overrides config file)",
    ),
):
    """Start the workset monitor daemon."""
    _ensure_dir()

    # Check if already running
    pid = _get_pid()
    if pid:
        cli_output.print_rich(f"[yellow]⚠[/yellow]  Monitor already running (PID {pid})")
        return

    # Find config file
    if config is None:
        config = _find_default_config()

    if config is None or not config.exists():
        cli_output.error(" No monitor config file found")
        cli_output.print_rich(
            "   Provide one with: [cyan]ursa monitor start --config path/to/monitor-config.yaml[/cyan]"
        )
        cli_output.print_rich(f"   Or create: [cyan]{_default_monitor_config_path()}[/cyan]")
        raise typer.Exit(1)

    # Check deployment-scoped Ursa config for region configuration
    from daylib_ursa.ursa_config import get_config_file_path, get_ursa_config

    ursa_config = get_ursa_config()
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

    # Build command
    cmd = [sys.executable, "-m", "daylib_ursa.workset_monitor_cli", str(config)]
    if once:
        cmd.append("--once")
    if verbose:
        cmd.append("--verbose")
    if dry_run:
        cmd.append("--dry-run")
    if enable_tapdb:
        cmd.append("--enable-tapdb")
    if parallel is not None:
        cmd.extend(["--parallel", str(parallel)])

    if background:
        log_file = _get_log_file()
        log_f = open(log_file, "w", buffering=1)  # Line-buffered
        pid_file = _pid_file()

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
            cli_output.error(" Monitor failed to start. Check logs:")
            cli_output.print_rich(f"   [dim]{log_file}[/dim]")
            if log_file.exists():
                content = log_file.read_text().strip()
                if content:
                    cli_output.print_rich("\n[dim]--- Last error ---[/dim]")
                    for line in content.split("\n")[-10:]:
                        cli_output.print_rich(f"   {line}")
            raise typer.Exit(1)

        pid_file.write_text(str(proc.pid))
        cli_output.print_rich(f"[green]✓[/green]  Monitor started (PID {proc.pid})")
        cli_output.print_rich(f"   Config: [dim]{config}[/dim]")
        cli_output.print_rich(f"   Logs: [dim]{log_file}[/dim]")
    else:
        cli_output.success(" Starting monitor")
        cli_output.print_rich(f"   Config: [dim]{config}[/dim]")
        cli_output.print_rich("   Press Ctrl+C to stop\n")
        try:
            subprocess.run(cmd, cwd=Path.cwd())
        except KeyboardInterrupt:
            cli_output.print_rich("\n[yellow]⚠[/yellow]  Monitor stopped")


@monitor_app.command("stop")
def stop():
    """Stop the workset monitor daemon."""
    pid = _get_pid()
    if not pid:
        cli_output.print_rich("[yellow]⚠[/yellow]  No monitor running")
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

        _pid_file().unlink(missing_ok=True)
        cli_output.print_rich(f"[green]✓[/green]  Monitor stopped (was PID {pid})")
    except ProcessLookupError:
        _pid_file().unlink(missing_ok=True)
        cli_output.print_rich("[yellow]⚠[/yellow]  Monitor was not running")
    except PermissionError:
        cli_output.print_rich(f"[red]✗[/red]  Permission denied stopping PID {pid}")
        raise typer.Exit(1)


@monitor_app.command("status")
def status():
    """Check the status of the workset monitor."""
    from cli_core_yo import ccyo_out
    from cli_core_yo.runtime import get_context

    pid = _get_pid()
    log_file = _get_latest_log() if pid else None
    data = {
        "running": pid is not None,
        "pid": pid,
        "log_file": str(log_file) if log_file else None,
    }

    if get_context().json_mode:
        ccyo_out.emit_json(data)
        return

    if pid:
        cli_output.print_rich(f"[green]●[/green]  Monitor is [green]running[/green] (PID {pid})")
        if log_file:
            cli_output.print_rich(f"   Logs: [dim]{log_file}[/dim]")
    else:
        cli_output.print_rich("[dim]○[/dim]  Monitor is [dim]not running[/dim]")


@monitor_app.command("logs")
def logs(
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines to show"),
    all_logs: bool = typer.Option(False, "--all", "-a", help="List all log files"),
):
    """View and follow workset monitor logs (Ctrl+C to stop)."""
    if all_logs:
        log_files = sorted(_log_dir().glob("monitor_*.log"), reverse=True)
        if not log_files:
            cli_output.print_rich("[yellow]⚠[/yellow]  No log files found.")
            return
        cli_output.print_rich(f"[bold]Monitor log files ({len(log_files)}):[/bold]")
        for lf in log_files[:20]:
            size = lf.stat().st_size
            cli_output.print_rich(f"  {lf.name}  [dim]({size:,} bytes)[/dim]")
        return

    log_file = _get_latest_log()
    if not log_file:
        cli_output.print_rich("[yellow]⚠[/yellow]  No log file found. Start the monitor first.")
        return

    cli_output.print_rich(f"[dim]Following {log_file.name} (Ctrl+C to stop)[/dim]\n")
    try:
        subprocess.run(["tail", "-f", "-n", str(lines), str(log_file)])
    except KeyboardInterrupt:
        cli_output.print_rich("\n")


@monitor_app.command("grep")
def grep_logs(
    pattern: str = typer.Argument(..., help="Pattern to search for in logs"),
    all_logs: bool = typer.Option(
        False, "--all", "-a", help="Search all log files, not just the latest"
    ),
    ignore_case: bool = typer.Option(
        True, "--ignore-case/--case-sensitive", "-i/-s", help="Case-insensitive search (default)"
    ),
    context: int = typer.Option(
        0, "--context", "-C", help="Lines of context before and after match"
    ),
    count: bool = typer.Option(False, "--count", "-c", help="Only show count of matching lines"),
):
    """Search monitor logs for a pattern.

    Examples:
        ursa monitor grep 'error'
        ursa monitor grep 'workset-123' --all
        ursa monitor grep 'failed' -C 3
    """
    if all_logs:
        log_files = sorted(_log_dir().glob("monitor_*.log"), reverse=True)
    else:
        latest = _get_latest_log()
        log_files = [latest] if latest else []

    if not log_files:
        cli_output.print_rich("[yellow]⚠[/yellow]  No log files found.")
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

    cli_output.print_rich(f"[dim]Searching {len(log_files)} log file(s) for '{pattern}'[/dim]\n")

    result = subprocess.run(grep_cmd)

    if result.returncode == 1:
        cli_output.print_rich(f"\n[yellow]⚠[/yellow]  No matches found for '{pattern}'")


def register(registry: CommandRegistry, spec: CliSpec) -> None:
    """cli-core-yo plugin: register monitor command group."""
    _ = spec
    registry.add_typer_app(None, monitor_app, "monitor", "Workset monitor management")
