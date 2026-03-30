"""TapDB lifecycle and Ursa overlay commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cli_core_yo.registry import CommandRegistry
    from cli_core_yo.spec import CliSpec

import typer
from rich.console import Console

from daylib_ursa.analysis_store import AnalysisStore
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


def _bootstrap_ursa_templates() -> None:
    store = AnalysisStore()
    store.bootstrap()


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
    ensure_tapdb_version()
    try:
        if target == "local":
            result = run_tapdb_cli(
                ["bootstrap", "local", "--no-gui"],
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
                [
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
            console.print(result.stdout.rstrip())

        db_url = export_database_url_for_target(
            target=target,
            client_id=DEFAULT_TAPDB_CLIENT_ID,
            profile=profile,
            region=region,
            namespace=namespace,
        )
        console.print(f"[green]DATABASE_URL[/green] resolved: [dim]{db_url}[/dim]")
        _bootstrap_ursa_templates()
        console.print("[green]Ursa TapDB overlay complete[/green]")
    except TapDBRuntimeError as exc:
        console.print(f"[red]DB build failed:[/red] {exc}")
        raise typer.Exit(1) from exc


@db_app.command("seed")
def seed() -> None:
    """Apply the Ursa TapDB template overlay only."""
    try:
        _bootstrap_ursa_templates()
    except Exception as exc:
        console.print(f"[red]DB seed failed:[/red] {exc}")
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
        run_tapdb_cli(
            ["db", "delete", tapdb_env_for_target(target), "--force"],
            target=target,
            client_id=DEFAULT_TAPDB_CLIENT_ID,
            profile=profile,
            region=region,
            namespace=namespace,
        )
    except TapDBRuntimeError as exc:
        console.print(f"[red]Delete failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    build(target=target, cluster=cluster, profile=profile, region=region, namespace=namespace)


def register(registry: CommandRegistry, spec: CliSpec) -> None:
    """Register the db command group."""
    registry.add_typer_app(None, db_app, "db", "TapDB lifecycle and overlay commands")
