"""Environment and configuration commands for Ursa CLI."""

import os
import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

env_app = typer.Typer(help="Environment and configuration commands")
console = Console()


@env_app.command("status")
def status():
    """Check system status (conda, AWS, dependencies)."""
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
            from daylib.ursa_config import get_ursa_config
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
    deps = ["boto3", "fastapi", "uvicorn", "pydantic", "typer", "rich"]
    for dep in deps:
        try:
            mod = __import__(dep)
            version = getattr(mod, "__version__", "?")
            console.print(f"  [green]✓[/green] {dep} ({version})")
        except ImportError:
            console.print(f"  [red]✗[/red] {dep} (not installed)")


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
# For multi-region configuration, use ~/.config/ursa/ursa-config.yaml instead.

# ========== AWS Configuration ==========
# Regions are configured in ~/.config/ursa/ursa-config.yaml
# Do NOT use AWS_DEFAULT_REGION - regions must be explicit per API call

# Regions to scan for ParallelCluster instances (comma-separated)
URSA_ALLOWED_REGIONS=us-west-2

# ========== Server Configuration ==========
URSA_HOST=0.0.0.0
URSA_PORT=8001

# ========== Authentication ==========
# Set to 'true' to enable Cognito authentication
ENABLE_AUTH=false

# Cognito settings (required if ENABLE_AUTH=true)
# Run 'ursa cognito setup' to create, or set manually
# COGNITO_USER_POOL_ID=us-west-2_xxxxxxxxx
# COGNITO_APP_CLIENT_ID=xxxxxxxxxxxxxxxxxxxxxxxxxx

# ========== S3 Configuration ==========
# Default S3 bucket for workset data
# S3_BUCKET=your-bucket-name

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

