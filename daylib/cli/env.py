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

    # Ursa config file
    from daylib.ursa_config import DEFAULT_CONFIG_PATH
    if DEFAULT_CONFIG_PATH.exists():
        table.add_row("Config file", f"[green]Found[/green] ({DEFAULT_CONFIG_PATH})")
    else:
        table.add_row("Config file", f"[yellow]Not found[/yellow] ({DEFAULT_CONFIG_PATH})")

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
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing config file"),
):
    """Generate ursa-config.yaml template in ~/.config/ursa/."""
    from daylib.ursa_config import DEFAULT_CONFIG_PATH

    if DEFAULT_CONFIG_PATH.exists() and not force:
        console.print(f"[yellow]⚠[/yellow]  Config file already exists: {DEFAULT_CONFIG_PATH}")
        console.print("   Use --force to overwrite")
        return

    # Ensure parent directory exists
    DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    template = """# Ursa Configuration
# ==================
# This is the single source of configuration for Ursa.
# Environment variables can override these values.

# ========== AWS Configuration ==========
aws_profile: lsmc

# Regions to scan for ParallelCluster instances
regions:
  - us-west-2
  # - us-east-1

# DynamoDB region (for workset state, customers, etc.)
dynamo_db_region: us-west-2

# ========== Authentication (Cognito) ==========
# Run 'ursa cognito setup' to create these, or set manually
# cognito_user_pool_id: us-west-2_xxxxxxxxx
# cognito_app_client_id: xxxxxxxxxxxxxxxxxxxxxxxxxx
# cognito_region: us-west-2
# cognito_domain: your-domain

# ========== S3 Configuration ==========
# Default S3 bucket for workset data
# s3_bucket: your-bucket-name

# ========== Optional ==========
# Whitelist domains for user registration (comma-separated)
# whitelist_domains: example.com,company.org
"""

    DEFAULT_CONFIG_PATH.write_text(template)
    console.print(f"[green]✓[/green]  Created config file: {DEFAULT_CONFIG_PATH}")
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

