"""Integration management commands for Ursa."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import typer
from rich.console import Console

from daylib_ursa.config import get_settings
from daylib_ursa.integrations.dewey_client import DeweyClient, DeweyClientError

if TYPE_CHECKING:
    from cli_core_yo.registry import CommandRegistry
    from cli_core_yo.spec import CliSpec


console = Console()
integrations_app = typer.Typer(help="Integration operations")
dewey_app = typer.Typer(help="Dewey integration operations")
integrations_app.add_typer(dewey_app, name="dewey")


def _require_dewey_client() -> DeweyClient:
    settings = get_settings()
    if not bool(getattr(settings, "dewey_enabled", False)):
        console.print("[red]✗[/red] Dewey integration is disabled")
        console.print("   Set [cyan]dewey_enabled=true[/cyan] in Ursa configuration")
        raise typer.Exit(1)
    base_url = str(getattr(settings, "dewey_base_url", "") or "").strip()
    token = str(getattr(settings, "dewey_api_token", "") or "").strip()
    if not base_url or not token:
        console.print("[red]✗[/red] Dewey integration is not fully configured")
        console.print(
            "   Required settings: [cyan]dewey_base_url[/cyan], [cyan]dewey_api_token[/cyan]"
        )
        raise typer.Exit(1)
    return DeweyClient(
        base_url=base_url,
        token=token,
        verify_ssl=bool(getattr(settings, "dewey_verify_ssl", True)),
        timeout_seconds=float(getattr(settings, "dewey_timeout_seconds", 10.0)),
    )


def _parse_metadata_json(metadata_json: str) -> dict[str, Any]:
    raw = str(metadata_json or "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Invalid JSON for --metadata-json: {exc}") from exc
    if not isinstance(payload, dict):
        raise typer.BadParameter("--metadata-json must decode to a JSON object")
    return payload


def _print_json(payload: dict[str, Any]) -> None:
    console.print_json(data=payload)


@dewey_app.command("resolve-artifact")
def resolve_artifact(
    artifact_euid: str = typer.Option(..., "--artifact-euid", help="Dewey artifact EUID"),
) -> None:
    """Resolve a Dewey artifact to its storage metadata."""
    client = _require_dewey_client()
    try:
        _print_json(client.resolve_artifact(artifact_euid))
    except DeweyClientError as exc:
        console.print(f"[red]✗[/red] Dewey artifact resolve failed: {exc}")
        raise typer.Exit(1) from exc


@dewey_app.command("resolve-artifact-set")
def resolve_artifact_set(
    artifact_set_euid: str = typer.Option(
        ..., "--artifact-set-euid", help="Dewey artifact-set EUID"
    ),
) -> None:
    """Resolve a Dewey artifact set and print the returned payload."""
    client = _require_dewey_client()
    try:
        _print_json(client.resolve_artifact_set(artifact_set_euid))
    except DeweyClientError as exc:
        console.print(f"[red]✗[/red] Dewey artifact-set resolve failed: {exc}")
        raise typer.Exit(1) from exc


@dewey_app.command("get-artifact")
def get_artifact(
    artifact_euid: str = typer.Option(..., "--artifact-euid", help="Dewey artifact EUID"),
) -> None:
    """Fetch an artifact directly from Dewey by EUID."""
    client = _require_dewey_client()
    try:
        _print_json(client.get_artifact(artifact_euid))
    except DeweyClientError as exc:
        console.print(f"[red]✗[/red] Dewey artifact fetch failed: {exc}")
        raise typer.Exit(1) from exc


@dewey_app.command("import-artifact")
def import_artifact(
    artifact_type: str = typer.Option(..., "--artifact-type", help="Artifact type"),
    storage_uri: str = typer.Option(..., "--storage-uri", help="Artifact storage URI"),
    metadata_json: str = typer.Option(
        "{}", "--metadata-json", help="Artifact metadata JSON object"
    ),
    idempotency_key: str = typer.Option(
        "", "--idempotency-key", help="Optional import idempotency key"
    ),
) -> None:
    """Import an artifact into Dewey and print the new artifact EUID."""
    client = _require_dewey_client()
    metadata = _parse_metadata_json(metadata_json)
    try:
        artifact_euid = client.import_artifact(
            artifact_type=artifact_type,
            storage_uri=storage_uri,
            metadata=metadata,
            idempotency_key=str(idempotency_key or "").strip() or None,
        )
    except DeweyClientError as exc:
        console.print(f"[red]✗[/red] Dewey artifact import failed: {exc}")
        raise typer.Exit(1) from exc
    console.print_json(data={"artifact_euid": artifact_euid})


def register(registry: CommandRegistry, spec: CliSpec) -> None:
    """cli-core-yo plugin: register integrations command group."""
    _ = spec
    registry.add_typer_app(None, integrations_app, "integrations", "Integration operations")
