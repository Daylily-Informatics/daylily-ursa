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
_SKIP_AWS_VALIDATION = {"version", "test", "env", "info"}


def _validate_aws_env() -> None:
    """Validate AWS environment is configured.

    Checks for AWS_PROFILE from either:
    - AWS_PROFILE environment variable
    - aws_profile in ~/.config/ursa/ursa-config.yaml

    Regions are configured in ~/.config/ursa/ursa-config.yaml and explicitly passed to AWS API calls.
    """
    from daylib.ursa_config import get_ursa_config

    errors = []
    ursa_config = get_ursa_config()

    # Check AWS_PROFILE - either from env or config
    aws_profile_env = os.environ.get("AWS_PROFILE", "")
    aws_profile_config = ursa_config.aws_profile

    if not aws_profile_env and not aws_profile_config:
        errors.append(
            "AWS_PROFILE is not set.\n"
            "   Set via environment: [cyan]export AWS_PROFILE=your-profile[/cyan]\n"
            "   Or in config file:   [cyan]~/.config/ursa/ursa-config.yaml[/cyan] → [dim]aws_profile: your-profile[/dim]"
        )
    elif aws_profile_env == "default":
        errors.append("AWS_PROFILE is 'default'. Use a named profile: [cyan]export AWS_PROFILE=your-profile[/cyan]")

    # If we have a profile from config but not env, set it in environment for boto3/AWS CLI
    if not aws_profile_env and aws_profile_config:
        os.environ["AWS_PROFILE"] = aws_profile_config

    if errors:
        console.print("[red bold]✗ AWS Configuration Error[/red bold]\n")
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


def _format_value_with_source(value: Optional[str], source: str) -> str:
    """Format a config value with its source indicator."""
    if not value:
        return "[dim]not set[/dim]"

    source_style = {
        "env": "[green](env)[/green]",
        "config": "[blue](config)[/blue]",
        "not set": "",
    }
    return f"{value} {source_style.get(source, '')}"


@app.command("info")
def info():
    """Show Ursa configuration and status."""
    from rich.table import Table
    from daylib.ursa_config import get_ursa_config, DEFAULT_CONFIG_PATH

    ursa_config = get_ursa_config()

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

    # Config file (XDG Base Directory: ~/.config/ursa/)
    config_dir = Path.home() / ".config" / "ursa"
    if ursa_config.config_path and ursa_config.config_path.exists():
        config_status = f"[green]{ursa_config.config_path}[/green]"
        if ursa_config.from_legacy_path:
            config_status += " [yellow](legacy path)[/yellow]"
    else:
        config_status = f"[yellow]not found[/yellow] [dim]({DEFAULT_CONFIG_PATH})[/dim]"
    table.add_row("Config File", config_status)

    # AWS Profile (show source)
    aws_profile = ursa_config.get_effective_aws_profile()
    profile_source = ursa_config.get_value_source("aws_profile")
    table.add_row("AWS Profile", _format_value_with_source(aws_profile, profile_source))

    # Cognito Region (show source)
    cognito_region = ursa_config.get_effective_cognito_region()
    cognito_source = ursa_config.get_value_source("cognito_region")
    table.add_row("Cognito Region", _format_value_with_source(cognito_region, cognito_source))

    # Cognito User Pool ID (show source)
    pool_id = os.environ.get("COGNITO_USER_POOL_ID") or ursa_config.cognito_user_pool_id
    pool_source = ursa_config.get_value_source("cognito_user_pool_id")
    if pool_id:
        # Truncate for display
        display_pool = pool_id[:20] + "..." if len(pool_id) > 20 else pool_id
        table.add_row("Cognito Pool ID", _format_value_with_source(display_pool, pool_source))
    else:
        table.add_row("Cognito Pool ID", "[dim]not set[/dim]")

    # Regions config
    if ursa_config.is_configured:
        regions = ursa_config.get_allowed_regions()
        table.add_row("Scan Regions", f"[green]{len(regions)}[/green] [dim]({', '.join(regions)})[/dim]")
        table.add_row("Bucket Source", "[dim]cluster tags (aws-parallelcluster-monitor-bucket)[/dim]")
    else:
        table.add_row("Scan Regions", "[yellow]none configured[/yellow]")

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

    # Show warnings/suggestions
    if not ursa_config.is_configured:
        console.print()
        console.print("[yellow]⚠[/yellow]  No regions configured")
        console.print(f"   Create config: [cyan]{DEFAULT_CONFIG_PATH}[/cyan]")
        console.print("   See example:   [cyan]config/ursa-config.example.yaml[/cyan]")


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    raise SystemExit(main())

