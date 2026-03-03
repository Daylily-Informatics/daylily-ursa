"""Infrastructure readiness commands for Ursa TapDB-backed services."""

from __future__ import annotations

import os
from typing import List

import typer
from rich.console import Console
from rich.table import Table

aws_app = typer.Typer(help="Infrastructure readiness commands")
console = Console()


def _effective_profile() -> str | None:
    return os.environ.get("AWS_PROFILE") or None


def _effective_region() -> str:
    from daylib.config import get_settings

    settings = get_settings()
    return settings.get_effective_region()


@aws_app.command("setup")
def setup() -> None:
    """Bootstrap TapDB templates and Ursa registries."""
    from daylib.biospecimen import BiospecimenRegistry
    from daylib.file_registry import FileRegistry
    from daylib.manifest_registry import ManifestRegistry
    from daylib.s3_bucket_validator import LinkedBucketManager
    from daylib.workset_customer import CustomerManager
    from daylib.workset_state_db import WorksetStateDB

    region = _effective_region()
    profile = _effective_profile()
    errors: List[str] = []

    console.print("[cyan]Bootstrapping Ursa TapDB resources...[/cyan]")

    components = [
        (
            "worksets",
            lambda: WorksetStateDB(
                table_name="tapdb-worksets",
                region=region,
                profile=profile,
            ).create_table_if_not_exists(),
        ),
        (
            "customers",
            lambda: CustomerManager(
                region=region,
                profile=profile,
            ).create_customer_table_if_not_exists(),
        ),
        (
            "files",
            lambda: FileRegistry(
                region=region,
                profile=profile,
            ).create_tables_if_not_exist(),
        ),
        (
            "manifests",
            lambda: ManifestRegistry(
                region=region,
                profile=profile,
            ).create_table_if_not_exists(),
        ),
        (
            "biospecimen",
            lambda: BiospecimenRegistry(
                region=region,
                profile=profile,
            ).create_tables_if_not_exist(),
        ),
        (
            "linked-buckets",
            lambda: LinkedBucketManager(
                region=region,
                profile=profile,
            ).create_table_if_not_exists(),
        ),
    ]

    for label, fn in components:
        try:
            fn()
            console.print(f"[green]✓[/green] {label}")
        except Exception as exc:  # pragma: no cover - operational path
            errors.append(f"{label}: {exc}")
            console.print(f"[red]✗[/red] {label}: {exc}")

    if errors:
        raise typer.Exit(1)

    console.print("\n[green]✓[/green] TapDB bootstrap complete")


@aws_app.command("status")
def status() -> None:
    """Show TapDB template readiness status."""
    from daylib.tapdb_graph.backend import TEMPLATE_DEFINITIONS, TapDBBackend

    backend = TapDBBackend(app_username="ursa-status")
    table = Table(title="TapDB Template Status")
    table.add_column("Template", style="cyan")
    table.add_column("Status")

    try:
        with backend.session_scope() as session:
            for spec in TEMPLATE_DEFINITIONS:
                template = backend.templates.get_template(session, spec.template_code)
                if template is None:
                    table.add_row(spec.template_code, "[red]missing[/red]")
                else:
                    table.add_row(spec.template_code, "[green]ready[/green]")
    except Exception as exc:  # pragma: no cover - operational path
        console.print(f"[red]✗[/red] Unable to check TapDB status: {exc}")
        raise typer.Exit(1)

    console.print(table)


@aws_app.command("teardown")
def teardown(
    force: bool = typer.Option(False, "--force", "-f", help="Acknowledge this non-reversible step"),
) -> None:
    """TapDB teardown is intentionally not automated from Ursa CLI."""
    if not force:
        console.print("[yellow]Teardown is disabled by default.[/yellow]")
        console.print("Use --force to print manual teardown instructions.")
        raise typer.Exit(1)

    console.print("[yellow]Manual teardown required.[/yellow]")
    console.print("1. Stop API/worker/monitor processes.")
    console.print("2. Snapshot database before changes.")
    console.print("3. Use DBA-managed SQL migration tooling to prune Ursa rows/templates.")
    console.print("4. Re-run `ursa aws setup` to re-bootstrap templates.")
