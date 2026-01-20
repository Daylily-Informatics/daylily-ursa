"""AWS resource management commands for Ursa CLI."""

import os

import typer
from rich.console import Console
from rich.table import Table

aws_app = typer.Typer(help="AWS resource management commands")
console = Console()


def _check_aws_profile():
    """Check if AWS_PROFILE is set."""
    if not os.environ.get("AWS_PROFILE"):
        console.print("[red]✗[/red]  AWS_PROFILE not set")
        console.print("   Set it with: [cyan]export AWS_PROFILE=your-profile[/cyan]")
        raise typer.Exit(1)


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

        tables_to_create = [
            (settings.workset_table_name, "workset_id"),
            (settings.customer_table_name, "customer_id"),
            (settings.daylily_file_registry_table, "file_id"),
        ]

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

        tables_to_check = [
            settings.workset_table_name,
            settings.customer_table_name,
            settings.daylily_file_registry_table,
        ]

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
    """Delete all AWS resources created by the project."""
    _check_aws_profile()

    if not force:
        console.print("[yellow]⚠[/yellow]  This will delete all DynamoDB tables!")
        confirm = typer.confirm("Are you sure?")
        if not confirm:
            console.print("[dim]Cancelled[/dim]")
            return

    try:
        import boto3
        from daylib.config import get_settings

        settings = get_settings()
        region = settings.get_effective_region()
        dynamodb = boto3.resource("dynamodb", region_name=region)

        tables_to_delete = [
            settings.workset_table_name,
            settings.customer_table_name,
            settings.daylily_file_registry_table,
        ]

        for table_name in tables_to_delete:
            try:
                tbl = dynamodb.Table(table_name)
                tbl.delete()
                console.print(f"[green]✓[/green]  Deleted table: {table_name}")
            except dynamodb.meta.client.exceptions.ResourceNotFoundException:
                console.print(f"[dim]○[/dim]  Table not found: {table_name}")

        console.print("\n[green]✓[/green]  AWS teardown complete")

    except Exception as e:
        console.print(f"[red]✗[/red]  Error: {e}")
        raise typer.Exit(1)

