"""Ursa CLI for beta analysis service operations."""

from importlib.metadata import PackageNotFoundError, version as package_version
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from daylib.cli.env import env_app
from daylib.cli.quality import quality_app
from daylib.cli.server import logs as server_logs
from daylib.cli.server import server_app
from daylib.cli.test import test_app

console = Console()

# Commands that skip AWS validation
_SKIP_AWS_VALIDATION = {"version", "test", "env", "config", "quality", "info", "doctor", "logs"}


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
    help="Ursa beta analysis service CLI",
    add_completion=True,
    no_args_is_help=True,
    callback=_aws_callback,
)

# Register subcommand groups
app.add_typer(server_app, name="server", help="API server management")
app.add_typer(test_app, name="test", help="Testing and code quality")
app.add_typer(env_app, name="env", help="Environment and configuration")
app.add_typer(env_app, name="config", help="Configuration and environment")
app.add_typer(quality_app, name="quality", help="Linting, formatting, and checks")


@app.command("version")
def version(
    ecosystem: bool = typer.Option(False, "--ecosystem", help="Show cross-repo version compatibility matrix"),
):
    """Show Ursa version and optional ecosystem compatibility matrix."""
    try:
        console.print(f"ursa [cyan]{package_version('daylily-ursa')}[/cyan]")
    except PackageNotFoundError:
        console.print("ursa [cyan]dev[/cyan]")

    if ecosystem:
        _show_ecosystem_versions()


def _show_ecosystem_versions() -> None:
    """Display the cross-repo version compatibility matrix."""
    import json
    from rich.table import Table

    # Locate ecosystem-versions.json relative to the project root
    project_root = Path(__file__).resolve().parent.parent.parent
    versions_path = project_root / "config" / "ecosystem-versions.json"

    if not versions_path.exists():
        console.print("[red]✗[/red] Ecosystem versions file not found: config/ecosystem-versions.json")
        return

    try:
        data = json.loads(versions_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        console.print(f"[red]✗[/red] Failed to read ecosystem versions: {exc}")
        return

    console.print()

    # Component table
    comp_table = Table(title="Ecosystem Components", title_style="bold cyan")
    comp_table.add_column("Component", style="cyan")
    comp_table.add_column("Repository", style="dim")
    comp_table.add_column("Current Version", style="green")

    for name, info in data.get("components", {}).items():
        comp_table.add_row(name, info.get("repo", ""), info.get("current", "unknown"))

    console.print(comp_table)
    console.print()

    # Tested combinations table
    combos = data.get("tested_combinations", [])
    if combos:
        combo_table = Table(title="Tested Combinations", title_style="bold cyan")
        combo_table.add_column("Date", style="dim")
        combo_table.add_column("Ursa")
        combo_table.add_column("Ephemeral Cluster")
        combo_table.add_column("Omics Analysis")
        combo_table.add_column("Cognito")
        combo_table.add_column("TapDB")
        combo_table.add_column("Omics Refs")
        combo_table.add_column("Notes", style="dim")

        for combo in combos:
            combo_table.add_row(
                combo.get("date", ""),
                combo.get("ursa", ""),
                combo.get("ephemeral_cluster", ""),
                combo.get("omics_analysis", ""),
                combo.get("cognito", ""),
                combo.get("tapdb", ""),
                combo.get("omics_references", ""),
                combo.get("notes", ""),
            )

        console.print(combo_table)

    console.print()
    console.print(f"[dim]Schema v{data.get('schema_version', '?')} · Last updated: {data.get('last_updated', '?')}[/dim]")


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

    # Regions config
    if ursa_config.is_configured:
        regions = ursa_config.get_allowed_regions()
        table.add_row("Scan Regions", f"[green]{len(regions)}[/green] [dim]({', '.join(regions)})[/dim]")
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

    from daylib.config import get_settings

    settings = get_settings()
    table.add_row("Bloom URL", settings.bloom_base_url)
    table.add_row("Atlas URL", settings.atlas_base_url)

    console.print(table)

    # Show warnings/suggestions
    if not ursa_config.is_configured:
        console.print()
        console.print("[yellow]⚠[/yellow]  No regions configured")
        console.print(f"   Create config: [cyan]{DEFAULT_CONFIG_PATH}[/cyan]")
        console.print("   See example:   [cyan]config/ursa-config.example.yaml[/cyan]")


@app.command("doctor")
def doctor():
    """Show a quick configuration and dependency health check."""
    from daylib.cli.env import status as env_status

    env_status()


@app.command("logs")
def logs(
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines to show"),
    all_logs: bool = typer.Option(False, "--all", "-a", help="List all log files"),
):
    """Show Ursa server logs."""
    server_logs(lines=lines, all_logs=all_logs)


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    raise SystemExit(main())
