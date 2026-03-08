"""Quality command aliases for the Ursa CLI."""

from __future__ import annotations

import typer

quality_app = typer.Typer(help="Code quality commands")


@quality_app.command("lint")
def lint(
    fix: bool = typer.Option(False, "--fix", "-f", help="Auto-fix issues"),
):
    """Run the ruff linter."""
    from daylib.cli import test as test_cmds

    test_cmds.lint(fix=fix)


@quality_app.command("format")
def format_code(
    check: bool = typer.Option(False, "--check", "-c", help="Check only, don't modify"),
):
    """Format code with ruff."""
    from daylib.cli import test as test_cmds

    test_cmds.format_code(check=check)


@quality_app.command("typecheck")
def typecheck():
    """Run the type checker."""
    from daylib.cli import test as test_cmds

    test_cmds.typecheck()


@quality_app.command("check")
def check():
    """Run the standard quality bundle."""
    from daylib.cli import test as test_cmds

    test_cmds.all_checks()
