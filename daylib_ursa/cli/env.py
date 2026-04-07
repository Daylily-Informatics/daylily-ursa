"""Environment and configuration commands for Ursa CLI."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.table import Table
from cli_core_yo import output as cli_output

if TYPE_CHECKING:
    from cli_core_yo.registry import CommandRegistry
    from cli_core_yo.spec import CliSpec

env_app = typer.Typer(help="Environment and configuration commands")


@env_app.command("validate")
def validate():
    """Validate Ursa configuration file and report issues."""
    from daylib_ursa.ursa_config import get_config_file_path, validate_config_file

    config_path = get_config_file_path()
    if not config_path.exists():
        cli_output.print_rich(f"[yellow]⚠[/yellow]  Config file not found: {config_path}")
        cli_output.print_rich("   Create one with: [cyan]ursa config generate[/cyan]")
        raise typer.Exit(1)

    is_valid, errors, warnings = validate_config_file(config_path)
    if errors:
        cli_output.print_rich(f"[red]Found {len(errors)} error(s):[/red]")
        for err in errors:
            cli_output.print_rich(f"  [red]•[/red] {err}")
    if warnings:
        cli_output.print_rich(f"[yellow]Found {len(warnings)} warning(s):[/yellow]")
        for w in warnings:
            cli_output.print_rich(f"  [yellow]•[/yellow] {w}")
    if is_valid and not warnings:
        cli_output.print_rich(f"[green]✓[/green]  Configuration is valid: {config_path}")
    if not is_valid:
        raise typer.Exit(1)


@env_app.command("status")
def status():
    """Check system status (conda, AWS, dependencies, tooling)."""
    import sys as _sys

    from cli_core_yo import ccyo_out
    from cli_core_yo.runtime import get_context
    from daylib_ursa.ursa_config import get_config_file_path

    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    aws_profile = os.environ.get("AWS_PROFILE", "")
    aws_region = os.environ.get("AWS_REGION", "")
    config_path = get_config_file_path()
    aws_region_source = "env" if aws_region else None
    if not aws_region:
        try:
            from daylib_ursa.ursa_config import get_ursa_config

            ursa_config = get_ursa_config()
            if ursa_config.aws_region:
                aws_region = ursa_config.aws_region
                aws_region_source = "config"
        except ValueError:
            pass
    data = {
        "python_version": _sys.version.split()[0],
        "conda_env": conda_env or None,
        "aws_profile": aws_profile or None,
        "aws_region": aws_region or None,
        "aws_region_source": aws_region_source,
        "config_path": str(config_path),
        "config_exists": config_path.exists(),
    }

    if get_context().json_mode:
        ccyo_out.emit_json(data)
        return

    table = Table(title="System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status")

    table.add_row("Python", data["python_version"])
    table.add_row(
        "Conda Env", f"[green]{conda_env}[/green]" if conda_env else "[yellow]Not active[/yellow]"
    )
    table.add_row(
        "AWS Profile", f"[green]{aws_profile}[/green]" if aws_profile else "[red]Not set[/red]"
    )
    if aws_region:
        table.add_row("AWS Region", f"[green]{aws_region}[/green] ({aws_region_source})")
    else:
        table.add_row("AWS Region", "[yellow]Not configured[/yellow]")
    table.add_row(
        "Config file",
        f"[green]{config_path}[/green]"
        if config_path.exists()
        else f"[yellow]{config_path} (not found)[/yellow]",
    )

    cli_output.print_rich(table)

    # Show key dependencies
    cli_output.print_rich("\n[bold]Key Dependencies:[/bold]")
    deps = [
        "boto3",
        "fastapi",
        "uvicorn",
        "pydantic",
        "typer",
        "rich",
        "sqlalchemy",
        "httpx",
    ]
    for dep in deps:
        try:
            mod = __import__(dep)
            version = getattr(mod, "__version__", "?")
            cli_output.print_rich(f"  [green]✓[/green] {dep} ({version})")
        except ImportError:
            cli_output.print_rich(f"  [red]✗[/red] {dep} (not installed)")

    cli_output.print_rich("\n[bold]CLI Tools:[/bold]")
    tools = ["aws", "pcluster", "jq", "yq", "rclone", "parallel", "fd", "psql", "node"]
    for tool in tools:
        path = shutil.which(tool)
        if path:
            cli_output.print_rich(f"  [green]✓[/green] {tool} ({path})")
        else:
            cli_output.print_rich(f"  [red]✗[/red] {tool} (not found)")


@env_app.command("clean")
def clean():
    """Remove cached files and build artifacts."""
    project_root = Path.cwd()

    patterns = [
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "*.egg-info",
        "dist",
        "build",
        "htmlcov",
        ".coverage",
    ]

    removed = 0
    for pattern in patterns:
        for path in project_root.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                cli_output.print_rich(f"  [dim]Removed {path}[/dim]")
            else:
                path.unlink()
                cli_output.print_rich(f"  [dim]Removed {path}[/dim]")
            removed += 1

    if removed:
        cli_output.print_rich(f"\n[green]✓[/green]  Cleaned {removed} items")
    else:
        cli_output.print_rich("[dim]Nothing to clean[/dim]")


def register(registry: CommandRegistry, spec: CliSpec) -> None:
    """cli-core-yo plugin: extend the built-in env group."""
    _ = spec
    registry.add_command("env", "validate", validate, help_text="Validate Ursa configuration file.")
    registry.add_command(
        "env", "clean", clean, help_text="Remove cached files and build artifacts."
    )
