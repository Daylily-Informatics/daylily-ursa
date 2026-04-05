"""Environment and configuration commands for Ursa CLI."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from cli_core_yo.registry import CommandRegistry
    from cli_core_yo.spec import CliSpec

env_app = typer.Typer(help="Environment and configuration commands")
console = Console()


@env_app.command("validate")
def validate():
    """Validate Ursa configuration file and report issues."""
    from daylib_ursa.ursa_config import get_config_file_path, validate_config_file

    config_path = get_config_file_path()
    if not config_path.exists():
        console.print(f"[yellow]⚠[/yellow]  Config file not found: {config_path}")
        console.print("   Create one with: [cyan]ursa config generate[/cyan]")
        raise typer.Exit(1)

    is_valid, errors, warnings = validate_config_file(config_path)
    if errors:
        console.print(f"[red]Found {len(errors)} error(s):[/red]")
        for err in errors:
            console.print(f"  [red]•[/red] {err}")
    if warnings:
        console.print(f"[yellow]Found {len(warnings)} warning(s):[/yellow]")
        for w in warnings:
            console.print(f"  [yellow]•[/yellow] {w}")
    if is_valid and not warnings:
        console.print(f"[green]✓[/green]  Configuration is valid: {config_path}")
    if not is_valid:
        raise typer.Exit(1)


@env_app.command("status")
def status():
    """Check system status (conda, AWS, dependencies, tooling)."""
    table = Table(title="System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status")

    # Python
    import sys

    table.add_row("Python", f"{sys.version.split()[0]}")

    # Conda environment
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env:
        table.add_row("Conda Env", f"[green]{conda_env}[/green]")
    else:
        table.add_row("Conda Env", "[yellow]Not active[/yellow]")

    # AWS Profile
    aws_profile = os.environ.get("AWS_PROFILE", "")
    if aws_profile:
        table.add_row("AWS Profile", f"[green]{aws_profile}[/green]")
    else:
        table.add_row("AWS Profile", "[red]Not set[/red]")

    # AWS Region (from AWS_REGION or ursa config)
    aws_region = os.environ.get("AWS_REGION", "")
    if aws_region:
        table.add_row("AWS Region", f"[green]{aws_region}[/green] (env)")
    else:
        # Try to get from ursa config
        try:
            from daylib_ursa.ursa_config import get_ursa_config

            ursa_config = get_ursa_config()
            if ursa_config.aws_region:
                table.add_row("AWS Region", f"[green]{ursa_config.aws_region}[/green] (config)")
            else:
                table.add_row("AWS Region", "[yellow]Not configured[/yellow]")
        except Exception:
            table.add_row("AWS Region", "[yellow]Not configured[/yellow]")

    # .env file
    env_file = Path.cwd() / ".env"
    if env_file.exists():
        table.add_row(".env file", "[green]Found[/green]")
    else:
        table.add_row(".env file", "[yellow]Not found[/yellow]")

    console.print(table)

    # Show key dependencies
    console.print("\n[bold]Key Dependencies:[/bold]")
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
            console.print(f"  [green]✓[/green] {dep} ({version})")
        except ImportError:
            console.print(f"  [red]✗[/red] {dep} (not installed)")

    console.print("\n[bold]CLI Tools:[/bold]")
    tools = ["aws", "pcluster", "jq", "yq", "rclone", "parallel", "fd", "psql", "node"]
    for tool in tools:
        path = shutil.which(tool)
        if path:
            console.print(f"  [green]✓[/green] {tool} ({path})")
        else:
            console.print(f"  [red]✗[/red] {tool} (not found)")


@env_app.command("generate")
def generate(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing .env file"),
):
    """Generate .env file template."""
    env_file = Path.cwd() / ".env"

    if env_file.exists() and not force:
        console.print("[yellow]⚠[/yellow]  .env file already exists")
        console.print("   Use --force to overwrite")
        return

    template = """# Ursa Configuration
# ==================
# This file is loaded by 'ursa server start' and other commands.
# For multi-region configuration, use ~/.config/ursa-<deployment>/ursa-config-<deployment>.yaml instead.

# ========== AWS Configuration ==========
# Regions are configured in ~/.config/ursa-<deployment>/ursa-config-<deployment>.yaml
# Do NOT use AWS_DEFAULT_REGION - regions must be explicit per API call

# Regions to scan for ParallelCluster instances (comma-separated)
URSA_ALLOWED_REGIONS=us-west-2

# ========== Server Configuration ==========
# Host/port and TapDB namespace should normally be configured in
# ~/.config/ursa-<deployment>/ursa-config-<deployment>.yaml, not here.
# HTTPS is required for GUI/API startup.
# Resolution order for `ursa server start`:
#   1. --cert / --key
#   2. SSL_CERT_FILE / SSL_KEY_FILE
#   3. shared Dayhoff deployment certs under ~/.local/state/dayhoff/<deploy>/certs
#   4. repo-local certs/ fallback
#   5. mkcert generation into the shared Dayhoff cert dir
# Use `ursa server start --no-ssl` to run HTTP only.

# ========== Authentication ==========
# Set to 'true' to enable Cognito authentication
ENABLE_AUTH=true

# Cognito settings (required if ENABLE_AUTH=true)
# Run 'daycog setup --client-name ursa' from daylily-cognito to create, or set manually
# COGNITO_USER_POOL_ID=us-west-2_xxxxxxxxx
# COGNITO_APP_CLIENT_ID=xxxxxxxxxxxxxxxxxxxxxxxxxx
# COGNITO_APP_CLIENT_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxx
# COGNITO_DOMAIN=your-domain-prefix.auth.us-west-2.amazoncognito.com

# ========== S3 Configuration ==========
# Secrets and one-off overrides may live here.
# Example:
# URSA_INTERNAL_OUTPUT_BUCKET=your-bucket-name

# ========== Optional ==========
# Whitelist domains for user registration (comma-separated)
# WHITELIST_DOMAINS=example.com,company.org
"""

    env_file.write_text(template)
    console.print("[green]✓[/green]  Created .env file")
    console.print("   Edit it to configure your settings")


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
                console.print(f"  [dim]Removed {path}[/dim]")
            else:
                path.unlink()
                console.print(f"  [dim]Removed {path}[/dim]")
            removed += 1

    if removed:
        console.print(f"\n[green]✓[/green]  Cleaned {removed} items")
    else:
        console.print("[dim]Nothing to clean[/dim]")


def register(registry: CommandRegistry, spec: CliSpec) -> None:
    """cli-core-yo plugin: extend the built-in env group."""
    _ = spec
    registry.add_command("env", "validate", validate, help_text="Validate Ursa configuration file.")
    registry.add_command("env", "generate", generate, help_text="Generate a local .env template.")
    registry.add_command(
        "env", "clean", clean, help_text="Remove cached files and build artifacts."
    )
