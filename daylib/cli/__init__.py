"""Ursa CLI - Workset Management CLI using Typer."""

import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from daylib.cli.server import server_app
from daylib.cli.monitor import monitor_app
from daylib.cli.aws import aws_app
from daylib.cli.cognito import cognito_app
from daylib.cli.test import test_app
from daylib.cli.env import env_app

console = Console()

# Commands that skip AWS validation
_SKIP_AWS_VALIDATION = {"version", "test", "env"}


def _validate_aws_env() -> None:
    """Validate AWS environment variables.

    Checks:
    - AWS_PROFILE is set and not '' or 'default'
    - AWS_REGION and REGION are both set
    - AWS_REGION and REGION match
    """
    errors = []

    # Check AWS_PROFILE
    aws_profile = os.environ.get("AWS_PROFILE", "")
    if not aws_profile:
        errors.append("AWS_PROFILE is not set. Set it with: [cyan]export AWS_PROFILE=your-profile[/cyan]")
    elif aws_profile == "default":
        errors.append("AWS_PROFILE is 'default'. Use a named profile: [cyan]export AWS_PROFILE=your-profile[/cyan]")

    # Check AWS_REGION and REGION
    aws_region = os.environ.get("AWS_REGION", "")
    region = os.environ.get("REGION", "")

    if not aws_region:
        errors.append("AWS_REGION is not set. Set it with: [cyan]export AWS_REGION=us-west-2[/cyan]")
    if not region:
        errors.append("REGION is not set. Set it with: [cyan]export REGION=us-west-2[/cyan]")

    if aws_region and region and aws_region != region:
        errors.append(
            f"AWS_REGION ({aws_region}) and REGION ({region}) do not match. "
            "Set both to the same value."
        )

    if errors:
        console.print("[red bold]✗ AWS Environment Error[/red bold]\n")
        for err in errors:
            console.print(f"  • {err}")
        console.print()
        raise typer.Exit(1)


def _aws_callback(ctx: typer.Context) -> None:
    """Callback to validate AWS environment before commands."""
    # Get the command being invoked
    invoked = ctx.invoked_subcommand

    # Skip validation for certain commands
    if invoked in _SKIP_AWS_VALIDATION:
        return

    _validate_aws_env()


app = typer.Typer(
    name="ursa",
    help="Ursa - Workset Management CLI",
    add_completion=True,
    no_args_is_help=True,
    callback=_aws_callback,
)

# Register subcommand groups
app.add_typer(server_app, name="server", help="API server management")
app.add_typer(monitor_app, name="monitor", help="Workset monitor management")
app.add_typer(aws_app, name="aws", help="AWS resource management")
app.add_typer(cognito_app, name="cognito", help="Cognito authentication management")
app.add_typer(test_app, name="test", help="Testing and code quality")
app.add_typer(env_app, name="env", help="Environment and configuration")


@app.command("version")
def version():
    """Show Ursa version."""
    try:
        from daylib import __version__
        console.print(f"ursa [cyan]{__version__}[/cyan]")
    except ImportError:
        console.print("ursa [cyan]dev[/cyan]")


@app.command("info")
def info():
    """Show Ursa configuration and status."""
    from rich.table import Table

    table = Table(title="Ursa Info")
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    # Version
    try:
        from daylib import __version__
        table.add_row("Version", __version__)
    except ImportError:
        table.add_row("Version", "dev")

    # Python
    table.add_row("Python", sys.version.split()[0])

    # AWS Profile
    table.add_row("AWS Profile", os.environ.get("AWS_PROFILE", "[dim]not set[/dim]"))

    # AWS Region
    table.add_row("AWS Region", os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))

    # Config dir
    config_dir = Path.home() / ".ursa"
    table.add_row("Config Dir", str(config_dir))

    # Check if server is running
    pid_file = config_dir / "server.pid"
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            import os as os_mod
            os_mod.kill(pid, 0)
            table.add_row("API Server", f"[green]Running[/green] (PID {pid})")
        except (ValueError, ProcessLookupError, PermissionError):
            table.add_row("API Server", "[dim]Stopped[/dim]")
    else:
        table.add_row("API Server", "[dim]Stopped[/dim]")

    # Check if monitor is running
    monitor_pid_file = config_dir / "monitor.pid"
    if monitor_pid_file.exists():
        try:
            pid = int(monitor_pid_file.read_text().strip())
            import os as os_mod
            os_mod.kill(pid, 0)
            table.add_row("Monitor", f"[green]Running[/green] (PID {pid})")
        except (ValueError, ProcessLookupError, PermissionError):
            table.add_row("Monitor", "[dim]Stopped[/dim]")
    else:
        table.add_row("Monitor", "[dim]Stopped[/dim]")

    console.print(table)


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    raise SystemExit(main())

