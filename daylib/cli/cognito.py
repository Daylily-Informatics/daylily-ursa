"""Cognito authentication management commands for Ursa CLI."""

import os

import typer
from rich.console import Console
from rich.table import Table

cognito_app = typer.Typer(help="Cognito authentication management commands")
console = Console()


def _check_aws_profile():
    """Check if AWS_PROFILE is set."""
    if not os.environ.get("AWS_PROFILE"):
        console.print("[red]✗[/red]  AWS_PROFILE not set")
        console.print("   Set it with: [cyan]export AWS_PROFILE=your-profile[/cyan]")
        raise typer.Exit(1)


@cognito_app.command("status")
def status():
    """Check Cognito configuration status."""
    _check_aws_profile()

    console.print("[cyan]Checking Cognito configuration...[/cyan]\n")

    try:
        import boto3
        from daylib.config import get_settings

        settings = get_settings()
        region = settings.get_effective_region()
        cognito = boto3.client("cognito-idp", region_name=region)

        table = Table(title="Cognito Configuration")
        table.add_column("Property", style="cyan")
        table.add_column("Value")

        # Check env vars
        pool_id = os.environ.get("COGNITO_USER_POOL_ID", "")
        client_id = os.environ.get("COGNITO_APP_CLIENT_ID", "")

        if pool_id:
            try:
                pool = cognito.describe_user_pool(UserPoolId=pool_id)
                table.add_row("User Pool ID", pool_id)
                table.add_row("User Pool Name", pool["UserPool"]["Name"])
                table.add_row("Status", "[green]Active[/green]")
            except Exception as e:
                table.add_row("User Pool ID", pool_id)
                table.add_row("Status", f"[red]Error: {e}[/red]")
        else:
            table.add_row("User Pool ID", "[dim]Not configured[/dim]")

        if client_id:
            table.add_row("App Client ID", client_id)
        else:
            table.add_row("App Client ID", "[dim]Not configured[/dim]")

        console.print(table)

        if not pool_id or not client_id:
            console.print("\n[yellow]⚠[/yellow]  Cognito not fully configured")
            console.print("   Run: [cyan]ursa cognito setup[/cyan]")

    except Exception as e:
        console.print(f"[red]✗[/red]  Error: {e}")
        raise typer.Exit(1)


@cognito_app.command("setup")
def setup(
    pool_name: str = typer.Option("ursa-users", "--name", "-n", help="User pool name"),
    port: int = typer.Option(8001, "--port", "-p", help="Server port for callback URL"),
):
    """Create Cognito User Pool and App Client."""
    _check_aws_profile()

    console.print("[cyan]Creating Cognito resources...[/cyan]")

    try:
        import boto3
        from daylib.config import get_settings

        settings = get_settings()
        region = settings.get_effective_region()
        cognito = boto3.client("cognito-idp", region_name=region)

        # Check if pool already exists
        pools = cognito.list_user_pools(MaxResults=60)
        existing = [p for p in pools["UserPools"] if p["Name"] == pool_name]

        if existing:
            console.print(f"[yellow]⚠[/yellow]  User pool '{pool_name}' already exists")
            pool_id = existing[0]["Id"]
        else:
            # Create user pool
            pool = cognito.create_user_pool(
                PoolName=pool_name,
                AutoVerifiedAttributes=["email"],
                UsernameAttributes=["email"],
                Policies={
                    "PasswordPolicy": {
                        "MinimumLength": 8,
                        "RequireUppercase": True,
                        "RequireLowercase": True,
                        "RequireNumbers": True,
                        "RequireSymbols": False,
                    }
                },
            )
            pool_id = pool["UserPool"]["Id"]
            console.print(f"[green]✓[/green]  Created user pool: {pool_name}")

        # Create app client
        callback_url = f"http://localhost:{port}/auth/callback"
        client = cognito.create_user_pool_client(
            UserPoolId=pool_id,
            ClientName=f"{pool_name}-client",
            GenerateSecret=False,
            AllowedOAuthFlows=["code"],
            AllowedOAuthScopes=["openid", "email", "profile"],
            AllowedOAuthFlowsUserPoolClient=True,
            CallbackURLs=[callback_url],
            SupportedIdentityProviders=["COGNITO"],
        )
        client_id = client["UserPoolClient"]["ClientId"]
        console.print(f"[green]✓[/green]  Created app client: {client_id}")

        # Show configuration
        console.print("\n[green]✓[/green]  Cognito setup complete")
        console.print("\nAdd to your .env file:")
        console.print(f"   [cyan]COGNITO_USER_POOL_ID={pool_id}[/cyan]")
        console.print(f"   [cyan]COGNITO_APP_CLIENT_ID={client_id}[/cyan]")

    except Exception as e:
        console.print(f"[red]✗[/red]  Error: {e}")
        raise typer.Exit(1)


@cognito_app.command("set-admin")
def set_admin(
    email: str = typer.Option(..., "--email", "-e", prompt="User email", help="User email address"),
    grant: bool = typer.Option(True, "--grant/--revoke", "-g/-r", help="Grant or revoke admin status"),
):
    """Grant or revoke admin status for a user."""
    _check_aws_profile()

    try:
        from daylib.workset_customer import CustomerManager
        from daylib.config import get_settings

        settings = get_settings()
        region = settings.aws_default_region
        profile = os.environ.get("AWS_PROFILE")

        console.print(f"[dim]Region: {region}, Profile: {profile}[/dim]")

        manager = CustomerManager(region=region, profile=profile)

        # Use the built-in method
        success = manager.set_admin_status(email, grant)
        if not success:
            # List available emails for debugging
            customers = manager.list_customers()
            if customers:
                console.print(f"[yellow]Available emails:[/yellow]")
                for c in customers:
                    console.print(f"  - {c.email}")
            console.print(f"[red]✗[/red]  No user found with email: {email}")
            raise typer.Exit(1)

        if grant:
            console.print(f"[green]✓[/green]  Admin status granted to: {email}")
        else:
            console.print(f"[green]✓[/green]  Admin status revoked from: {email}")

    except Exception as e:
        console.print(f"[red]✗[/red]  Error: {e}")
        raise typer.Exit(1)


@cognito_app.command("set-password")
def set_password(
    email: str = typer.Option(..., "--email", "-e", prompt="User email", help="User email address"),
    password: str = typer.Option(..., "--password", "-p", prompt="New password", hide_input=True, help="New password"),
):
    """Set password for a Cognito user."""
    _check_aws_profile()

    try:
        import boto3

        pool_id = os.environ.get("COGNITO_USER_POOL_ID")
        if not pool_id:
            console.print("[red]✗[/red]  COGNITO_USER_POOL_ID not set")
            raise typer.Exit(1)

        cognito = boto3.client("cognito-idp")
        cognito.admin_set_user_password(
            UserPoolId=pool_id,
            Username=email,
            Password=password,
            Permanent=True,
        )

        console.print(f"[green]✓[/green]  Password set for: {email}")

    except Exception as e:
        console.print(f"[red]✗[/red]  Error: {e}")
        raise typer.Exit(1)


def _get_pool_id() -> str:
    """Get user pool ID from env or raise error."""
    pool_id = os.environ.get("COGNITO_USER_POOL_ID")
    if not pool_id:
        console.print("[red]✗[/red]  COGNITO_USER_POOL_ID not set")
        console.print("   Set it with: [cyan]export COGNITO_USER_POOL_ID=your-pool-id[/cyan]")
        raise typer.Exit(1)
    return pool_id


@cognito_app.command("list-users")
def list_users(
    limit: int = typer.Option(50, "--limit", "-l", help="Max users to list"),
):
    """List all Cognito users."""
    _check_aws_profile()
    pool_id = _get_pool_id()

    try:
        import boto3
        from daylib.config import get_settings

        settings = get_settings()
        region = settings.get_effective_region()
        cognito = boto3.client("cognito-idp", region_name=region)

        table = Table(title=f"Cognito Users ({pool_id})")
        table.add_column("Email", style="cyan")
        table.add_column("Customer ID")
        table.add_column("Status")
        table.add_column("Created")
        table.add_column("Enabled")

        paginator = cognito.get_paginator("list_users")
        user_count = 0

        for page in paginator.paginate(UserPoolId=pool_id, PaginationConfig={"MaxItems": limit}):
            for user in page.get("Users", []):
                email = ""
                customer_id = ""
                for attr in user.get("Attributes", []):
                    if attr["Name"] == "email":
                        email = attr["Value"]
                    elif attr["Name"] == "custom:customer_id":
                        customer_id = attr["Value"]

                status = user.get("UserStatus", "UNKNOWN")
                created = user.get("UserCreateDate", "")
                if created:
                    created = created.strftime("%Y-%m-%d %H:%M")
                enabled = "[green]Yes[/green]" if user.get("Enabled", False) else "[red]No[/red]"

                status_color = {
                    "CONFIRMED": "[green]CONFIRMED[/green]",
                    "UNCONFIRMED": "[yellow]UNCONFIRMED[/yellow]",
                    "FORCE_CHANGE_PASSWORD": "[yellow]FORCE_CHANGE_PASSWORD[/yellow]",
                    "COMPROMISED": "[red]COMPROMISED[/red]",
                }.get(status, status)

                table.add_row(email, customer_id, status_color, str(created), enabled)
                user_count += 1

        console.print(table)
        console.print(f"\n[dim]Total: {user_count} users[/dim]")

    except Exception as e:
        console.print(f"[red]✗[/red]  Error: {e}")
        raise typer.Exit(1)


@cognito_app.command("export")
def export_users(
    output: str = typer.Option("cognito_users.log", "--output", "-o", help="Output file path"),
):
    """Export all Cognito users to a log file."""
    _check_aws_profile()
    pool_id = _get_pool_id()

    try:
        import json
        from datetime import datetime

        import boto3
        from daylib.config import get_settings

        settings = get_settings()
        region = settings.get_effective_region()
        cognito = boto3.client("cognito-idp", region_name=region)

        console.print(f"[cyan]Exporting users from pool {pool_id}...[/cyan]")

        users = []
        paginator = cognito.get_paginator("list_users")

        for page in paginator.paginate(UserPoolId=pool_id):
            for user in page.get("Users", []):
                user_record = {
                    "username": user.get("Username"),
                    "status": user.get("UserStatus"),
                    "enabled": user.get("Enabled"),
                    "created": user.get("UserCreateDate").isoformat() if user.get("UserCreateDate") else None,
                    "modified": user.get("UserLastModifiedDate").isoformat() if user.get("UserLastModifiedDate") else None,
                    "attributes": {attr["Name"]: attr["Value"] for attr in user.get("Attributes", [])},
                }
                users.append(user_record)

        # Write to file
        export_data = {
            "exported_at": datetime.utcnow().isoformat(),
            "pool_id": pool_id,
            "region": region,
            "user_count": len(users),
            "users": users,
        }

        with open(output, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        console.print(f"[green]✓[/green]  Exported {len(users)} users to: {output}")

    except Exception as e:
        console.print(f"[red]✗[/red]  Error: {e}")
        raise typer.Exit(1)


@cognito_app.command("delete-user")
def delete_user(
    email: str = typer.Option(..., "--email", "-e", prompt="User email", help="User email to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Delete a single Cognito user."""
    _check_aws_profile()
    pool_id = _get_pool_id()

    if not force:
        console.print(f"[yellow]⚠[/yellow]  This will delete user: {email}")
        confirm = typer.confirm("Are you sure?")
        if not confirm:
            console.print("[dim]Cancelled[/dim]")
            return

    try:
        import boto3
        from daylib.config import get_settings

        settings = get_settings()
        region = settings.get_effective_region()
        cognito = boto3.client("cognito-idp", region_name=region)

        cognito.admin_delete_user(UserPoolId=pool_id, Username=email)
        console.print(f"[green]✓[/green]  Deleted user: {email}")

    except Exception as e:
        console.print(f"[red]✗[/red]  Error: {e}")
        raise typer.Exit(1)


@cognito_app.command("delete-all-users")
def delete_all_users(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Delete ALL users from the Cognito pool. Use with caution!"""
    _check_aws_profile()
    pool_id = _get_pool_id()

    if not force:
        console.print(f"[red]⚠  WARNING: This will delete ALL users from pool {pool_id}![/red]")
        confirm = typer.confirm("Are you absolutely sure?")
        if not confirm:
            console.print("[dim]Cancelled[/dim]")
            return

    try:
        import boto3
        from daylib.config import get_settings

        settings = get_settings()
        region = settings.get_effective_region()
        cognito = boto3.client("cognito-idp", region_name=region)

        console.print(f"[cyan]Deleting all users from pool {pool_id}...[/cyan]")

        deleted_count = 0
        paginator = cognito.get_paginator("list_users")

        for page in paginator.paginate(UserPoolId=pool_id):
            for user in page.get("Users", []):
                username = user.get("Username")
                try:
                    cognito.admin_delete_user(UserPoolId=pool_id, Username=username)
                    console.print(f"[dim]  Deleted: {username}[/dim]")
                    deleted_count += 1
                except Exception as e:
                    console.print(f"[yellow]  Failed to delete {username}: {e}[/yellow]")

        console.print(f"\n[green]✓[/green]  Deleted {deleted_count} users")

    except Exception as e:
        console.print(f"[red]✗[/red]  Error: {e}")
        raise typer.Exit(1)


@cognito_app.command("teardown")
def teardown(
    pool_name: str = typer.Option(None, "--name", "-n", help="Pool name to delete (if not using env var)"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Delete the Cognito User Pool and all its users."""
    _check_aws_profile()

    try:
        import boto3
        from daylib.config import get_settings

        settings = get_settings()
        region = settings.get_effective_region()
        cognito = boto3.client("cognito-idp", region_name=region)

        # Get pool ID from env or find by name
        pool_id = os.environ.get("COGNITO_USER_POOL_ID")

        if not pool_id and pool_name:
            pools = cognito.list_user_pools(MaxResults=60)
            for p in pools["UserPools"]:
                if p["Name"] == pool_name:
                    pool_id = p["Id"]
                    break

        if not pool_id:
            console.print("[red]✗[/red]  No pool ID found")
            console.print("   Set COGNITO_USER_POOL_ID or use --name")
            raise typer.Exit(1)

        # Get pool info for confirmation
        pool_info = cognito.describe_user_pool(UserPoolId=pool_id)
        pool_name_actual = pool_info["UserPool"]["Name"]

        if not force:
            console.print(f"[red]⚠  WARNING: This will delete the Cognito pool:[/red]")
            console.print(f"   Pool ID: {pool_id}")
            console.print(f"   Pool Name: {pool_name_actual}")
            console.print(f"   [red]All users will be permanently deleted![/red]")
            confirm = typer.confirm("Are you absolutely sure?")
            if not confirm:
                console.print("[dim]Cancelled[/dim]")
                return

        console.print(f"[cyan]Deleting pool {pool_id}...[/cyan]")
        cognito.delete_user_pool(UserPoolId=pool_id)
        console.print(f"[green]✓[/green]  Deleted Cognito pool: {pool_name_actual} ({pool_id})")
        console.print("\n[yellow]Remember to unset environment variables:[/yellow]")
        console.print("   unset COGNITO_USER_POOL_ID")
        console.print("   unset COGNITO_APP_CLIENT_ID")

    except Exception as e:
        console.print(f"[red]✗[/red]  Error: {e}")
        raise typer.Exit(1)

