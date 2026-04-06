"""TapDB lifecycle and Ursa overlay commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cli_core_yo.registry import CommandRegistry
    from cli_core_yo.spec import CliSpec

import typer
from cli_core_yo import output as cli_output

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

db_app = typer.Typer(help="TapDB lifecycle and Ursa overlay commands")


def _bootstrap_ursa_templates() -> None:
    store = AnalysisStore()
    store.bootstrap()


def _resolved_runtime_defaults(
    *,
    profile: str,
    region: str,
    namespace: str,
) -> tuple[str, str, str]:
    settings = get_settings()
    effective_profile = str(profile or getattr(settings, "aws_profile", "") or "").strip()
    if not effective_profile:
        effective_profile = DEFAULT_AWS_PROFILE

    effective_region = str(region or "").strip()
    if not effective_region:
        resolver = getattr(settings, "get_effective_region", None)
        if callable(resolver):
            effective_region = str(resolver() or "").strip()
    if not effective_region:
        effective_region = DEFAULT_AWS_REGION

    effective_namespace = str(
        namespace or getattr(settings, "tapdb_database_name", "") or ""
    ).strip()
    if not effective_namespace:
        effective_namespace = DEFAULT_TAPDB_DATABASE_NAME

    return effective_profile, effective_region, effective_namespace


def _validate_target(target: str) -> str:
    normalized = str(target or "").strip().lower()
    if normalized not in {"local", "aurora"}:
        raise TapDBRuntimeError("Unsupported database target. Use local or aurora.")
    return normalized


def _apply_ursa_overlay(*, start_step: int, total_steps: int) -> None:
    cli_output.print_rich(
        f"[cyan]Step {start_step}/{total_steps}:[/cyan] Applying Ursa TapDB template overlay"
    )
    _bootstrap_ursa_templates()


def _build_target(
    *,
    target: str,
    cluster: str,
    profile: str,
    region: str,
    namespace: str,
    overlay_start_step: int,
    overlay_total_steps: int,
) -> None:
    ensure_tapdb_version()
    target = _validate_target(target)
    profile, region, namespace = _resolved_runtime_defaults(
        profile=profile,
        region=region,
        namespace=namespace,
    )
    if target == "local":
        result = run_tapdb_cli(
            args=["bootstrap", "local", "--no-gui"],
            target=target,
            client_id=DEFAULT_TAPDB_CLIENT_ID,
            profile=profile,
            region=region,
            namespace=namespace,
        )
    else:
        if not cluster.strip():
            raise TapDBRuntimeError("--cluster is required for aurora target")
        result = run_tapdb_cli(
            args=[
                "bootstrap",
                "aurora",
                "--cluster",
                cluster.strip(),
                "--region",
                region,
                "--no-gui",
            ],
            target=target,
            client_id=DEFAULT_TAPDB_CLIENT_ID,
            profile=profile,
            region=region,
            namespace=namespace,
        )
    if result.stdout:
        cli_output.print_rich(result.stdout.rstrip())

    db_url = export_database_url_for_target(
        target=target,
        client_id=DEFAULT_TAPDB_CLIENT_ID,
        profile=profile,
        region=region,
        namespace=namespace,
    )
    cli_output.print_rich(f"[green]DATABASE_URL[/green] resolved: [dim]{db_url}[/dim]")
    _apply_ursa_overlay(start_step=overlay_start_step, total_steps=overlay_total_steps)
    cli_output.print_rich("[green]Ursa TapDB overlay complete[/green]")


@db_app.command("build")
def build(
    target: str = typer.Option("local", "--target", help="TapDB target: local|aurora"),
    cluster: str = typer.Option("", "--cluster", help="Aurora cluster ID for aurora target"),
    profile: str = typer.Option(DEFAULT_AWS_PROFILE, "--profile", help="AWS profile"),
    region: str = typer.Option(DEFAULT_AWS_REGION, "--region", help="AWS region"),
    namespace: str = typer.Option(
        DEFAULT_TAPDB_DATABASE_NAME, "--namespace", help="TapDB namespace"
    ),
) -> None:
    """Bootstrap TapDB runtime and apply the Ursa overlay."""
    try:
        _build_target(
            target=target,
            cluster=cluster,
            profile=profile,
            region=region,
            namespace=namespace,
            overlay_start_step=3,
            overlay_total_steps=3,
        )
    except TapDBRuntimeError as exc:
        cli_output.print_rich(f"[red]DB build failed:[/red] {exc}")
        raise typer.Exit(1) from exc


@db_app.command("seed")
def seed(
    target: str = typer.Option("local", "--target", help="TapDB target: local|aurora"),
    profile: str = typer.Option(DEFAULT_AWS_PROFILE, "--profile", help="AWS profile"),
    region: str = typer.Option(DEFAULT_AWS_REGION, "--region", help="AWS region"),
    namespace: str = typer.Option(
        DEFAULT_TAPDB_DATABASE_NAME, "--namespace", help="TapDB namespace"
    ),
) -> None:
    """Apply the Ursa TapDB template overlay only."""
    try:
        target = _validate_target(target)
        profile, region, namespace = _resolved_runtime_defaults(
            profile=profile,
            region=region,
            namespace=namespace,
        )
        db_url = export_database_url_for_target(
            target=target,
            client_id=DEFAULT_TAPDB_CLIENT_ID,
            profile=profile,
            region=region,
            namespace=namespace,
        )
        cli_output.print_rich(f"[green]DATABASE_URL[/green] resolved: [dim]{db_url}[/dim]")
        _apply_ursa_overlay(start_step=1, total_steps=1)
    except Exception as exc:
        cli_output.print_rich(f"[red]DB seed failed:[/red] {exc}")
        raise typer.Exit(1) from exc


@db_app.command("reset")
def reset(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
    target: str = typer.Option("local", "--target", help="TapDB target: local|aurora"),
    cluster: str = typer.Option("", "--cluster", help="Aurora cluster ID for aurora target"),
    profile: str = typer.Option(DEFAULT_AWS_PROFILE, "--profile", help="AWS profile"),
    region: str = typer.Option(DEFAULT_AWS_REGION, "--region", help="AWS region"),
    namespace: str = typer.Option(
        DEFAULT_TAPDB_DATABASE_NAME, "--namespace", help="TapDB namespace"
    ),
) -> None:
    """Delete and rebuild the TapDB target, then apply the Ursa overlay."""
    if not force and not typer.confirm("This will delete the current TapDB DB target. Continue?"):
        raise typer.Exit(0)

    try:
        target = _validate_target(target)
        profile, region, namespace = _resolved_runtime_defaults(
            profile=profile,
            region=region,
            namespace=namespace,
        )
        run_tapdb_cli(
            args=["db", "delete", tapdb_env_for_target(target), "--force"],
            target=target,
            client_id=DEFAULT_TAPDB_CLIENT_ID,
            profile=profile,
            region=region,
            namespace=namespace,
        )
    except TapDBRuntimeError as exc:
        cli_output.print_rich(f"[red]Delete failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    try:
        _build_target(
            target=target,
            cluster=cluster,
            profile=profile,
            region=region,
            namespace=namespace,
            overlay_start_step=4,
            overlay_total_steps=4,
        )
    except TapDBRuntimeError as exc:
        cli_output.print_rich(f"[red]DB build failed:[/red] {exc}")
        raise typer.Exit(1) from exc


@db_app.command("nuke")
def nuke(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
    target: str = typer.Option("local", "--target", help="TapDB target: local|aurora"),
    profile: str = typer.Option(DEFAULT_AWS_PROFILE, "--profile", help="AWS profile"),
    region: str = typer.Option(DEFAULT_AWS_REGION, "--region", help="AWS region"),
    namespace: str = typer.Option(
        DEFAULT_TAPDB_DATABASE_NAME, "--namespace", help="TapDB namespace"
    ),
) -> None:
    """Delete the TapDB target without rebuilding."""
    if not force and not typer.confirm("This will delete the current TapDB DB target. Continue?"):
        raise typer.Exit(0)

    try:
        target = _validate_target(target)
        profile, region, namespace = _resolved_runtime_defaults(
            profile=profile,
            region=region,
            namespace=namespace,
        )
        run_tapdb_cli(
            args=["db", "delete", tapdb_env_for_target(target), "--force"],
            target=target,
            client_id=DEFAULT_TAPDB_CLIENT_ID,
            profile=profile,
            region=region,
            namespace=namespace,
        )
    except TapDBRuntimeError as exc:
        cli_output.print_rich(f"[red]Delete failed:[/red] {exc}")
        raise typer.Exit(1) from exc


def register(registry: CommandRegistry, spec: CliSpec) -> None:
    """Register the db command group."""
    registry.add_typer_app(None, db_app, "db", "TapDB lifecycle and overlay commands")
