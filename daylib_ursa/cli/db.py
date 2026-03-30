"""TapDB lifecycle and Ursa overlay commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cli_core_yo.registry import CommandRegistry
    from cli_core_yo.spec import CliSpec

import os
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from daylib_ursa.analysis_store import AnalysisStore
from daylib_ursa.config import get_settings
from daylib_ursa.integrations.tapdb_runtime import (
    DEFAULT_AWS_PROFILE,
    DEFAULT_AWS_REGION,
    DEFAULT_TAPDB_CLIENT_ID,
    DEFAULT_TAPDB_DATABASE_NAME,
    TapDBRuntimeError,
    ensure_tapdb_version,
    export_database_url_for_target,
    run_tapdb_cli,
    tapdb_env_for_target,
)

console = Console()
db_app = typer.Typer(help="TapDB lifecycle and Ursa overlay commands")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_target(value: str) -> str:
    settings = get_settings()
    target = (
        (value or os.environ.get("DATABASE_TARGET") or settings.database_target).strip().lower()
    )
    if target not in {"local", "aurora"}:
        raise TapDBRuntimeError(f"Unsupported target '{target}'. Use local or aurora.")
    return target


def _resolve_runtime_options(
    *,
    target: str,
    profile: str,
    region: str,
    namespace: str,
) -> tuple[str, str, str, str]:
    settings = get_settings()
    resolved_target = _resolve_target(target)
    resolved_profile = (
        profile.strip()
        or os.environ.get("AWS_PROFILE", "")
        or str(settings.aws_profile or "").strip()
        or DEFAULT_AWS_PROFILE
    )
    resolved_region = (
        region.strip()
        or os.environ.get("AWS_REGION", "")
        or settings.get_effective_region()
        or DEFAULT_AWS_REGION
    )
    resolved_namespace = (
        namespace.strip()
        or os.environ.get("TAPDB_DATABASE_NAME", "")
        or str(settings.tapdb_database_name or "").strip()
        or DEFAULT_TAPDB_DATABASE_NAME
    )
    return resolved_target, resolved_profile, resolved_region, resolved_namespace


def _resolve_client_id() -> str:
    settings = get_settings()
    return (
        os.environ.get("TAPDB_CLIENT_ID", "").strip()
        or str(settings.tapdb_client_id or "").strip()
        or DEFAULT_TAPDB_CLIENT_ID
    )


def _apply_ursa_overlay(*, start_step: int, total_steps: int) -> None:
    console.print(f"[cyan][{start_step}/{total_steps}] Ursa TapDB overlay[/cyan]")
    store = AnalysisStore()
    store.bootstrap()
    console.print("  [green]✓[/green] Ursa TapDB overlay complete")


def _prepare_database_url(
    *,
    target: str,
    profile: str,
    region: str,
    namespace: str,
) -> str:
    db_url = export_database_url_for_target(
        target=target,
        client_id=_resolve_client_id(),
        profile=profile,
        region=region,
        namespace=namespace,
    )
    console.print(f"  [green]✓[/green] DATABASE_URL resolved via TapDB config: [dim]{db_url}[/dim]")
    return db_url


def _tapdb_bootstrap(
    *,
    target: str,
    cluster: str,
    profile: str,
    region: str,
    namespace: str,
) -> None:
    args = ["bootstrap", "local", "--no-gui"]
    if target == "aurora":
        if not cluster.strip():
            raise TapDBRuntimeError("Aurora target requires --cluster.")
        args = [
            "bootstrap",
            "aurora",
            "--cluster",
            cluster.strip(),
            "--region",
            region,
            "--no-gui",
        ]
    result = run_tapdb_cli(
        args=args,
        target=target,
        client_id=_resolve_client_id(),
        profile=profile,
        region=region,
        namespace=namespace,
        cwd=PROJECT_ROOT,
    )
    if result.stdout:
        console.print(result.stdout.rstrip())


def _tapdb_delete_database(
    *,
    target: str,
    profile: str,
    region: str,
    namespace: str,
) -> None:
    result = run_tapdb_cli(
        args=["db", "delete", tapdb_env_for_target(target), "--force"],
        target=target,
        client_id=_resolve_client_id(),
        profile=profile,
        region=region,
        namespace=namespace,
        cwd=PROJECT_ROOT,
    )
    if result.stdout:
        console.print(result.stdout.rstrip())


def _show_runtime_context(*, target: str, profile: str, region: str, namespace: str) -> None:
    console.print(f"  target: [cyan]{target}[/cyan]")
    console.print(f"  profile: [cyan]{profile}[/cyan]")
    console.print(f"  region: [cyan]{region}[/cyan]")
    console.print(f"  namespace: [cyan]{namespace}[/cyan]")
    console.print(f"  tapdb_env: [cyan]{tapdb_env_for_target(target)}[/cyan]")


@db_app.command("build")
def build(
    target: str = typer.Option(
        "",
        "--target",
        help="Build target: local or aurora (defaults from config/environment).",
    ),
    cluster: str = typer.Option(
        "",
        "--cluster",
        help="Aurora cluster identifier (required for --target aurora).",
    ),
    region: str = typer.Option(DEFAULT_AWS_REGION, "--region", help="AWS region"),
    profile: str = typer.Option(DEFAULT_AWS_PROFILE, "--profile", help="AWS profile"),
    namespace: str = typer.Option(
        DEFAULT_TAPDB_DATABASE_NAME,
        "--namespace",
        help="TapDB database namespace",
    ),
) -> None:
    """Build TapDB lifecycle + Ursa overlay."""
    ensure_tapdb_version()
    console.print(Panel.fit("[bold blue]Ursa - Database Build[/bold blue]", border_style="blue"))
    console.print()

    try:
        resolved_target, resolved_profile, resolved_region, resolved_namespace = (
            _resolve_runtime_options(
                target=target,
                profile=profile,
                region=region,
                namespace=namespace,
            )
        )
        _show_runtime_context(
            target=resolved_target,
            profile=resolved_profile,
            region=resolved_region,
            namespace=resolved_namespace,
        )
        console.print()
        console.print("[cyan][1/3] TapDB bootstrap[/cyan]")
        _tapdb_bootstrap(
            target=resolved_target,
            cluster=cluster,
            profile=resolved_profile,
            region=resolved_region,
            namespace=resolved_namespace,
        )
        console.print("[green]✓[/green] TapDB lifecycle bootstrap complete")
        console.print()
        console.print("[cyan][2/3] Resolve DATABASE_URL[/cyan]")
        _prepare_database_url(
            target=resolved_target,
            profile=resolved_profile,
            region=resolved_region,
            namespace=resolved_namespace,
        )
        console.print()
        _apply_ursa_overlay(start_step=3, total_steps=3)
    except TapDBRuntimeError as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print()
    console.print(Panel.fit("[bold green]✓ Database build complete[/bold green]", border_style="green"))


@db_app.command("seed")
def seed(
    target: str = typer.Option(
        "",
        "--target",
        help="Seed target: local or aurora (defaults from config/environment).",
    ),
    region: str = typer.Option(DEFAULT_AWS_REGION, "--region", help="AWS region"),
    profile: str = typer.Option(DEFAULT_AWS_PROFILE, "--profile", help="AWS profile"),
    namespace: str = typer.Option(
        DEFAULT_TAPDB_DATABASE_NAME,
        "--namespace",
        help="TapDB database namespace",
    ),
) -> None:
    """Apply the Ursa TapDB overlay only."""
    try:
        resolved_target, resolved_profile, resolved_region, resolved_namespace = (
            _resolve_runtime_options(
                target=target,
                profile=profile,
                region=region,
                namespace=namespace,
            )
        )
        _prepare_database_url(
            target=resolved_target,
            profile=resolved_profile,
            region=resolved_region,
            namespace=resolved_namespace,
        )
        _apply_ursa_overlay(start_step=1, total_steps=1)
    except TapDBRuntimeError as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(1) from exc
    except Exception as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(1) from exc


@db_app.command("reset")
def reset(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
    target: str = typer.Option(
        "",
        "--target",
        help="Reset target: local or aurora (defaults from config/environment).",
    ),
    cluster: str = typer.Option(
        "",
        "--cluster",
        help="Aurora cluster identifier (required for --target aurora).",
    ),
    region: str = typer.Option(DEFAULT_AWS_REGION, "--region", help="AWS region"),
    profile: str = typer.Option(DEFAULT_AWS_PROFILE, "--profile", help="AWS profile"),
    namespace: str = typer.Option(
        DEFAULT_TAPDB_DATABASE_NAME,
        "--namespace",
        help="TapDB database namespace",
    ),
) -> None:
    """Reset DB through TapDB lifecycle, then run Ursa overlay."""
    if not force:
        console.print("[yellow]⚠[/yellow] This will delete the current TapDB database target.")
        if not typer.confirm("Continue with reset?"):
            console.print("[dim]Aborted.[/dim]")
            raise typer.Exit(0)

    try:
        resolved_target, resolved_profile, resolved_region, resolved_namespace = (
            _resolve_runtime_options(
                target=target,
                profile=profile,
                region=region,
                namespace=namespace,
            )
        )
        _show_runtime_context(
            target=resolved_target,
            profile=resolved_profile,
            region=resolved_region,
            namespace=resolved_namespace,
        )
        console.print()
        console.print("[cyan][1/4] TapDB database delete[/cyan]")
        _tapdb_delete_database(
            target=resolved_target,
            profile=resolved_profile,
            region=resolved_region,
            namespace=resolved_namespace,
        )
        console.print("[green]✓[/green] TapDB delete complete")
        console.print()
        console.print("[cyan][2/4] TapDB bootstrap[/cyan]")
        _tapdb_bootstrap(
            target=resolved_target,
            cluster=cluster,
            profile=resolved_profile,
            region=resolved_region,
            namespace=resolved_namespace,
        )
        console.print("[green]✓[/green] TapDB bootstrap complete")
        console.print()
        console.print("[cyan][3/4] Resolve DATABASE_URL[/cyan]")
        _prepare_database_url(
            target=resolved_target,
            profile=resolved_profile,
            region=resolved_region,
            namespace=resolved_namespace,
        )
        console.print()
        _apply_ursa_overlay(start_step=4, total_steps=4)
    except TapDBRuntimeError as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print("[green]✓[/green] Database reset complete")


@db_app.command("nuke")
def nuke(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
    target: str = typer.Option(
        "",
        "--target",
        help="Nuke target: local or aurora (defaults from config/environment).",
    ),
    region: str = typer.Option(DEFAULT_AWS_REGION, "--region", help="AWS region"),
    profile: str = typer.Option(DEFAULT_AWS_PROFILE, "--profile", help="AWS profile"),
    namespace: str = typer.Option(
        DEFAULT_TAPDB_DATABASE_NAME,
        "--namespace",
        help="TapDB database namespace",
    ),
) -> None:
    """Delete database via TapDB-managed lifecycle only."""
    if not force:
        console.print("[bold red]☢ Ursa DB nuke (TapDB-managed) ☢[/bold red]")
        console.print("This deletes the selected TapDB database target only.")
        console.print("Ursa will not delete local pgdata or migration files.")
        if not typer.confirm("Continue?"):
            console.print("[dim]Aborted.[/dim]")
            raise typer.Exit(0)

    try:
        resolved_target, resolved_profile, resolved_region, resolved_namespace = (
            _resolve_runtime_options(
                target=target,
                profile=profile,
                region=region,
                namespace=namespace,
            )
        )
        _tapdb_delete_database(
            target=resolved_target,
            profile=resolved_profile,
            region=resolved_region,
            namespace=resolved_namespace,
        )
    except TapDBRuntimeError as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print("[green]✓[/green] TapDB database deleted")
    console.print("Rebuild with: [cyan]ursa db build[/cyan]")


def register(registry: CommandRegistry, spec: CliSpec) -> None:
    """Register the db command group."""
    registry.add_typer_app(None, db_app, "db", "TapDB lifecycle and overlay commands")
