"""Code quality commands for Ursa CLI."""

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

quality_app = typer.Typer(help="Code quality commands")
console = Console()


def _get_project_root() -> Path:
    """Get the project root directory."""
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return cwd


@quality_app.command("lint")
def lint(
    fix: bool = typer.Option(False, "--fix", "-f", help="Auto-fix issues"),
):
    """Run the ruff linter."""
    project_root = _get_project_root()
    cmd = [sys.executable, "-m", "ruff", "check", "daylib_ursa/", "tests/"]
    if fix:
        cmd.append("--fix")
    console.print("[cyan]Running linter...[/cyan]")
    result = subprocess.run(cmd, cwd=project_root)
    raise typer.Exit(result.returncode)


@quality_app.command("format")
def format_code(
    check: bool = typer.Option(False, "--check", "-c", help="Check only, don't modify"),
):
    """Format code with ruff."""
    project_root = _get_project_root()
    cmd = [sys.executable, "-m", "ruff", "format", "daylib_ursa/", "tests/"]
    if check:
        cmd.append("--check")
    console.print(f"[cyan]{'Checking' if check else 'Formatting'} code...[/cyan]")
    result = subprocess.run(cmd, cwd=project_root)
    raise typer.Exit(result.returncode)


@quality_app.command("typecheck")
def typecheck():
    """Run mypy type checker."""
    project_root = _get_project_root()
    cmd = [sys.executable, "-m", "mypy", "daylib_ursa/"]
    console.print("[cyan]Running type checker...[/cyan]")
    result = subprocess.run(cmd, cwd=project_root)
    raise typer.Exit(result.returncode)


@quality_app.command("check")
def check():
    """Run the standard quality bundle (lint + format check + typecheck)."""
    project_root = _get_project_root()
    checks = [
        ("Lint", [sys.executable, "-m", "ruff", "check", "daylib_ursa/", "tests/"]),
        ("Format", [sys.executable, "-m", "ruff", "format", "--check", "daylib_ursa/", "tests/"]),
        ("Typecheck", [sys.executable, "-m", "mypy", "daylib_ursa/"]),
    ]
    failed = []
    for name, cmd in checks:
        console.print(f"\n[cyan]Running {name}...[/cyan]")
        result = subprocess.run(cmd, cwd=project_root)
        if result.returncode != 0:
            failed.append(name)
            console.print(f"[red]✗[/red]  {name} failed")
        else:
            console.print(f"[green]✓[/green]  {name} passed")
    if failed:
        console.print(f"\n[red]✗[/red]  {len(failed)} check(s) failed: {', '.join(failed)}")
        raise typer.Exit(1)
    else:
        console.print("\n[green]✓[/green]  All quality checks passed")


def register(registry: CommandRegistry, spec: CliSpec) -> None:
    """cli-core-yo plugin: register quality command group."""
    _ = spec
    registry.add_typer_app(None, quality_app, "quality", "Code quality commands")
