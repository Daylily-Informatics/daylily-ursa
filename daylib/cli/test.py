"""Testing and code quality commands for Ursa CLI."""

import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

test_app = typer.Typer(help="Testing and code quality commands")
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

    console.print(f"[cyan]Running tests...[/cyan]")
    result = subprocess.run(cmd, cwd=project_root)
    raise typer.Exit(result.returncode)


@test_app.command("cov")
def coverage(
    html: bool = typer.Option(False, "--html", help="Generate HTML coverage report"),
):
    """Run tests with coverage report."""
    project_root = _get_project_root()

    cmd = [sys.executable, "-m", "pytest", "--cov=daylib", "--cov-report=term-missing"]

    if html:
        cmd.append("--cov-report=html")

    console.print(f"[cyan]Running tests with coverage...[/cyan]")
    result = subprocess.run(cmd, cwd=project_root)

    if html and result.returncode == 0:
        console.print("\n[green]✓[/green]  HTML report: [cyan]htmlcov/index.html[/cyan]")

    raise typer.Exit(result.returncode)


@test_app.command("lint")
def lint(
    fix: bool = typer.Option(False, "--fix", "-f", help="Auto-fix issues"),
):
    """Run ruff linter."""
    project_root = _get_project_root()

    cmd = [sys.executable, "-m", "ruff", "check", "daylib/", "tests/"]

    if fix:
        cmd.append("--fix")

    console.print(f"[cyan]Running linter...[/cyan]")
    result = subprocess.run(cmd, cwd=project_root)
    raise typer.Exit(result.returncode)


@test_app.command("format")
def format_code(
    check: bool = typer.Option(False, "--check", "-c", help="Check only, don't modify"),
):
    """Format code with ruff."""
    project_root = _get_project_root()

    cmd = [sys.executable, "-m", "ruff", "format", "daylib/", "tests/"]

    if check:
        cmd.append("--check")

    console.print(f"[cyan]{'Checking' if check else 'Formatting'} code...[/cyan]")
    result = subprocess.run(cmd, cwd=project_root)
    raise typer.Exit(result.returncode)


@test_app.command("typecheck")
def typecheck():
    """Run mypy type checker."""
    project_root = _get_project_root()

    cmd = [sys.executable, "-m", "mypy", "daylib/"]

    console.print(f"[cyan]Running type checker...[/cyan]")
    result = subprocess.run(cmd, cwd=project_root)
    raise typer.Exit(result.returncode)


@test_app.command("all")
def all_checks():
    """Run all quality checks (lint, format check, typecheck, tests)."""
    project_root = _get_project_root()

    checks = [
        ("Lint", [sys.executable, "-m", "ruff", "check", "daylib/", "tests/"]),
        ("Format", [sys.executable, "-m", "ruff", "format", "--check", "daylib/", "tests/"]),
        ("Tests", [sys.executable, "-m", "pytest", "-q", "--tb=short"]),
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
        console.print(f"\n[green]✓[/green]  All checks passed")

