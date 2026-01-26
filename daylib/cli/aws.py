"""AWS resource management commands for Ursa CLI."""

import os
from typing import List, Tuple

import typer
from rich.console import Console
from rich.table import Table

aws_app = typer.Typer(help="AWS resource management commands")
console = Console()

# Biospecimen tables (hardcoded defaults not in settings)
BIOSPECIMEN_TABLES = [
    "daylily-subjects",
    "daylily-biospecimens",
    "daylily-biosamples",
    "daylily-libraries",
]

# FileRegistry tables (hardcoded defaults in daylib/file_registry.py)
FILE_REGISTRY_TABLES = [
    "daylily-files",
    "daylily-filesets",
    "daylily-file-workset-usage",
]


def _check_aws_profile():
    """Check if AWS_PROFILE is set."""
    if not os.environ.get("AWS_PROFILE"):
        console.print("[red]✗[/red]  AWS_PROFILE not set")
        console.print("   Set it with: [cyan]export AWS_PROFILE=your-profile[/cyan]")
        raise typer.Exit(1)


def _get_all_table_names() -> List[str]:
    """Get all DynamoDB table names used by the system."""
    from daylib.config import get_settings

    settings = get_settings()
    return [
        settings.workset_table_name,
        settings.customer_table_name,
        settings.daylily_manifest_table,
        settings.daylily_linked_buckets_table,
    ] + FILE_REGISTRY_TABLES + BIOSPECIMEN_TABLES


def _get_core_tables_with_keys() -> List[Tuple[str, str]]:
    """Get core tables with their primary key names (for setup).

    Note: FileRegistry and BiospecimenRegistry tables have complex schemas
    and should be created via their respective create_table_if_not_exists() methods.
    This only creates simple single-key tables.
    """
    from daylib.config import get_settings

    settings = get_settings()
    return [
        (settings.workset_table_name, "workset_id"),
        (settings.customer_table_name, "customer_id"),
        (settings.daylily_linked_buckets_table, "bucket_id"),
    ]


@aws_app.command("setup")
def setup():
    """Create required AWS resources (DynamoDB tables)."""
    _check_aws_profile()

    console.print("[cyan]Creating AWS resources...[/cyan]")

    try:
        import boto3
        from daylib.config import get_settings

        settings = get_settings()
        region = settings.get_effective_region()
        dynamodb = boto3.resource("dynamodb", region_name=region)

        tables_to_create = _get_core_tables_with_keys()

        for table_name, key_name in tables_to_create:
            try:
                table = dynamodb.Table(table_name)
                table.load()
                console.print(f"[green]✓[/green]  Table exists: {table_name}")
            except dynamodb.meta.client.exceptions.ResourceNotFoundException:
                console.print(f"[cyan]►[/cyan]  Creating table: {table_name}")
                dynamodb.create_table(
                    TableName=table_name,
                    KeySchema=[{"AttributeName": key_name, "KeyType": "HASH"}],
                    AttributeDefinitions=[{"AttributeName": key_name, "AttributeType": "S"}],
                    BillingMode="PAY_PER_REQUEST",
                )
                console.print(f"[green]✓[/green]  Created table: {table_name}")

        console.print("\n[green]✓[/green]  AWS setup complete")

    except Exception as e:
        console.print(f"[red]✗[/red]  Error: {e}")
        raise typer.Exit(1)


@aws_app.command("status")
def status():
    """Check status of AWS resources."""
    _check_aws_profile()

    console.print("[cyan]Checking AWS resources...[/cyan]\n")

    try:
        import boto3
        from daylib.config import get_settings

        settings = get_settings()
        region = settings.get_effective_region()
        dynamodb = boto3.resource("dynamodb", region_name=region)

        table = Table(title="DynamoDB Tables")
        table.add_column("Table", style="cyan")
        table.add_column("Status")
        table.add_column("Items")

        tables_to_check = _get_all_table_names()

        for table_name in tables_to_check:
            try:
                tbl = dynamodb.Table(table_name)
                tbl.load()
                item_count = tbl.item_count
                table.add_row(table_name, "[green]Active[/green]", str(item_count))
            except dynamodb.meta.client.exceptions.ResourceNotFoundException:
                table.add_row(table_name, "[red]Not Found[/red]", "-")

        console.print(table)

    except Exception as e:
        console.print(f"[red]✗[/red]  Error: {e}")
        raise typer.Exit(1)


@aws_app.command("teardown")
def teardown(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Delete all DynamoDB tables created by the project."""
    _check_aws_profile()

    tables_to_delete = _get_all_table_names()

    if not force:
        console.print("[yellow]⚠[/yellow]  This will delete ALL DynamoDB tables:")
        for tbl in tables_to_delete:
            console.print(f"   • {tbl}")
        confirm = typer.confirm("\nAre you sure?")
        if not confirm:
            console.print("[dim]Cancelled[/dim]")
            return

    try:
        import boto3
        from daylib.config import get_settings

        settings = get_settings()
        region = settings.get_effective_region()
        dynamodb = boto3.resource("dynamodb", region_name=region)

        deleted = 0
        not_found = 0
        for table_name in tables_to_delete:
            try:
                tbl = dynamodb.Table(table_name)
                tbl.delete()
                console.print(f"[green]✓[/green]  Deleted table: {table_name}")
                deleted += 1
            except dynamodb.meta.client.exceptions.ResourceNotFoundException:
                console.print(f"[dim]○[/dim]  Table not found: {table_name}")
                not_found += 1

        console.print(f"\n[green]✓[/green]  AWS teardown complete (deleted: {deleted}, not found: {not_found})")

    except Exception as e:
        console.print(f"[red]✗[/red]  Error: {e}")
        raise typer.Exit(1)

