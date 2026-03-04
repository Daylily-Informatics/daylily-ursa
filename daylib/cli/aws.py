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
            lambda: WorksetStateDB().bootstrap(),
        ),
        (
            "customers",
            lambda: CustomerManager(
                region=region,
                profile=profile,
            ).bootstrap(),
        ),
        (
            "files",
            lambda: FileRegistry().bootstrap(),
        ),
        (
            "manifests",
            lambda: ManifestRegistry().bootstrap(),
        ),
        (
            "biospecimen",
            lambda: BiospecimenRegistry().bootstrap(),
        ),
        (
            "linked-buckets",
            lambda: LinkedBucketManager(
                region=region,
                profile=profile,
            ).bootstrap(),
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
    template_table = Table(title="TapDB Template Status")
    template_table.add_column("Template", style="cyan")
    template_table.add_column("Status")
    sequence_table = Table(title="TapDB Instance Sequence Status")
    sequence_table.add_column("Sequence", style="cyan")
    sequence_table.add_column("Status")
    missing_templates = False
    missing_sequences = False

    try:
        with backend.session_scope() as session:
            for spec in TEMPLATE_DEFINITIONS:
                template = backend.templates.get_template(session, spec.template_code)
                if template is None:
                    template_table.add_row(spec.template_code, "[red]missing[/red]")
                    missing_templates = True
                else:
                    template_table.add_row(spec.template_code, "[green]ready[/green]")
            missing = set(backend.get_missing_instance_sequences(session))
            for seq_name in backend.list_required_instance_sequences(session):
                if seq_name in missing:
                    sequence_table.add_row(seq_name, "[red]missing[/red]")
                    missing_sequences = True
                else:
                    sequence_table.add_row(seq_name, "[green]ready[/green]")
    except Exception as exc:  # pragma: no cover - operational path
        console.print(f"[red]✗[/red] Unable to check TapDB status: {exc}")
        raise typer.Exit(1)

    console.print(template_table)
    console.print(sequence_table)
    if missing_templates or missing_sequences:
        console.print(
            "[yellow]Remediation:[/yellow] run "
            "`ursa aws setup` then `ursa aws repair-sequences`."
        )
        raise typer.Exit(1)


@aws_app.command("repair-sequences")
def repair_sequences(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show missing sequences without applying changes",
    )
) -> None:
    """Repair missing TapDB instance-prefix sequences required by Ursa templates."""
    from daylib.tapdb_graph.backend import TapDBBackend

    backend = TapDBBackend(app_username="ursa-repair")
    try:
        with backend.session_scope(commit=not dry_run) as session:
            missing = backend.get_missing_instance_sequences(session)
            if not missing:
                console.print("[green]✓[/green] No missing TapDB instance sequences.")
                return
            console.print("[yellow]Missing TapDB instance sequences:[/yellow]")
            for seq_name in missing:
                console.print(f"  - {seq_name}")
            if dry_run:
                console.print("[cyan]Dry run only.[/cyan] No changes applied.")
                return
            backend.ensure_instance_sequences(session)
            remaining = backend.get_missing_instance_sequences(session)
    except Exception as exc:  # pragma: no cover - operational path
        console.print(f"[red]✗[/red] Unable to repair sequence state: {exc}")
        raise typer.Exit(1)

    if remaining:
        console.print("[red]✗[/red] Sequence repair incomplete:")
        for seq_name in remaining:
            console.print(f"  - {seq_name}")
        raise typer.Exit(1)

    console.print("[green]✓[/green] Repaired TapDB instance sequences.")


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
