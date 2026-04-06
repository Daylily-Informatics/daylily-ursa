"""Test execution commands for Ursa CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console

if TYPE_CHECKING:
    from cli_core_yo.registry import CommandRegistry
    from cli_core_yo.spec import CliSpec

test_app = typer.Typer(help="Test execution commands")
console = Console()


def _get_project_root() -> Path:
    """Get the project root directory."""
    # Try to find pyproject.toml
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return cwd


@test_app.command("run")
def run(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    pattern: str = typer.Option("", "--pattern", "-k", help="Run tests matching pattern"),
    path: str = typer.Argument("tests/", help="Test path or file"),
):
    """Run the test suite."""
    project_root = _get_project_root()

    cmd = [sys.executable, "-m", "pytest", path]

    if verbose:
        cmd.append("-v")
    else:
        cmd.extend(["-q", "--tb=short"])

    if pattern:
        cmd.extend(["-k", pattern])

    console.print("[cyan]Running tests...[/cyan]")
    result = subprocess.run(cmd, cwd=project_root)
    raise typer.Exit(result.returncode)


@test_app.command("cov")
def coverage(
    html: bool = typer.Option(False, "--html", help="Generate HTML coverage report"),
):
    """Run tests with coverage report."""
    project_root = _get_project_root()

    cmd = [sys.executable, "-m", "pytest", "--cov=daylib_ursa", "--cov-report=term-missing"]

    if html:
        cmd.append("--cov-report=html")

    console.print("[cyan]Running tests with coverage...[/cyan]")
    result = subprocess.run(cmd, cwd=project_root)

    if html and result.returncode == 0:
        console.print("\n[green]✓[/green]  HTML report: [cyan]htmlcov/index.html[/cyan]")

    raise typer.Exit(result.returncode)


@test_app.command("all")
def all_checks():
    """Run the full test suite with coverage."""
    project_root = _get_project_root()

    cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short", "--cov=daylib_ursa"]

    console.print("[cyan]Running full test suite...[/cyan]")
    result = subprocess.run(cmd, cwd=project_root)
    raise typer.Exit(result.returncode)


def register(registry: CommandRegistry, spec: CliSpec) -> None:
    """cli-core-yo plugin: register test command group."""
    _ = spec
    registry.add_typer_app(None, test_app, "test", "Test execution commands")
