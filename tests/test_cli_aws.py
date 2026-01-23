"""Tests for the AWS CLI commands (ursa aws setup/status/teardown)."""

import os
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from daylib.cli.aws import aws_app


runner = CliRunner()


@pytest.fixture
def mock_settings():
    """Mock settings with test table names."""
    settings = MagicMock()
    settings.get_effective_region.return_value = "us-west-2"
    settings.workset_table_name = "test-worksets"
    settings.customer_table_name = "test-customers"
    settings.daylily_linked_buckets_table = "test-linked-buckets"
    settings.daylily_manifest_table = "test-manifests"
    return settings


@pytest.fixture
def mock_dynamodb_resource():
    """Mock boto3 DynamoDB resource."""
    resource = MagicMock()
    # Create a proper exception class for ResourceNotFoundException
    resource.meta.client.exceptions.ResourceNotFoundException = type(
        "ResourceNotFoundException", (Exception,), {}
    )
    return resource


class TestAwsSetup:
    """Tests for ursa aws setup command."""

    def test_setup_requires_aws_profile(self):
        """Test that setup fails without AWS_PROFILE set."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure AWS_PROFILE is not set
            os.environ.pop("AWS_PROFILE", None)
            result = runner.invoke(aws_app, ["setup"])
            assert result.exit_code == 1
            assert "AWS_PROFILE" in result.output

    def test_setup_creates_tables(self, mock_settings, mock_dynamodb_resource):
        """Test that setup creates tables when they don't exist."""
        mock_table = MagicMock()
        mock_table.load.side_effect = mock_dynamodb_resource.meta.client.exceptions.ResourceNotFoundException(
            "Table not found"
        )
        mock_dynamodb_resource.Table.return_value = mock_table
        mock_dynamodb_resource.create_table.return_value = mock_table

        with patch.dict(os.environ, {"AWS_PROFILE": "test-profile"}):
            with patch("daylib.config.get_settings", return_value=mock_settings):
                with patch("boto3.resource", return_value=mock_dynamodb_resource):
                    result = runner.invoke(aws_app, ["setup"])
                    assert result.exit_code == 0
                    assert "Creating table" in result.output

    def test_setup_skips_existing_tables(self, mock_settings, mock_dynamodb_resource):
        """Test that setup skips tables that already exist."""
        mock_table = MagicMock()
        mock_table.load.return_value = None  # Table exists
        mock_dynamodb_resource.Table.return_value = mock_table

        with patch.dict(os.environ, {"AWS_PROFILE": "test-profile"}):
            with patch("daylib.config.get_settings", return_value=mock_settings):
                with patch("boto3.resource", return_value=mock_dynamodb_resource):
                    result = runner.invoke(aws_app, ["setup"])
                    assert result.exit_code == 0
                    assert "Table exists" in result.output


class TestAwsStatus:
    """Tests for ursa aws status command."""

    def test_status_requires_aws_profile(self):
        """Test that status fails without AWS_PROFILE set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_PROFILE", None)
            result = runner.invoke(aws_app, ["status"])
            assert result.exit_code == 1
            assert "AWS_PROFILE" in result.output

    def test_status_shows_active_tables(self, mock_settings, mock_dynamodb_resource):
        """Test that status shows active tables."""
        mock_table = MagicMock()
        mock_table.item_count = 42
        mock_dynamodb_resource.Table.return_value = mock_table

        with patch.dict(os.environ, {"AWS_PROFILE": "test-profile"}):
            with patch("daylib.config.get_settings", return_value=mock_settings):
                with patch("boto3.resource", return_value=mock_dynamodb_resource):
                    result = runner.invoke(aws_app, ["status"])
                    assert result.exit_code == 0
                    assert "test-worksets" in result.output

    def test_status_shows_missing_tables(self, mock_settings, mock_dynamodb_resource):
        """Test that status shows missing tables."""
        mock_table = MagicMock()
        mock_table.load.side_effect = mock_dynamodb_resource.meta.client.exceptions.ResourceNotFoundException(
            "Table not found"
        )
        mock_dynamodb_resource.Table.return_value = mock_table

        with patch.dict(os.environ, {"AWS_PROFILE": "test-profile"}):
            with patch("daylib.config.get_settings", return_value=mock_settings):
                with patch("boto3.resource", return_value=mock_dynamodb_resource):
                    result = runner.invoke(aws_app, ["status"])
                    assert result.exit_code == 0
                    assert "Not Found" in result.output


class TestAwsTeardown:
    """Tests for ursa aws teardown command."""

    def test_teardown_requires_aws_profile(self):
        """Test that teardown fails without AWS_PROFILE set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_PROFILE", None)
            result = runner.invoke(aws_app, ["teardown", "--force"])
            assert result.exit_code == 1
            assert "AWS_PROFILE" in result.output

    def test_teardown_prompts_without_force(self, mock_settings):
        """Test that teardown prompts for confirmation without --force."""
        with patch.dict(os.environ, {"AWS_PROFILE": "test-profile"}):
            with patch("daylib.config.get_settings", return_value=mock_settings):
                # Simulate user saying "n" to confirmation
                result = runner.invoke(aws_app, ["teardown"], input="n\n")
                assert "Cancelled" in result.output or result.exit_code == 0

    def test_teardown_deletes_tables_with_force(self, mock_settings, mock_dynamodb_resource):
        """Test that teardown deletes tables with --force flag."""
        mock_table = MagicMock()
        mock_dynamodb_resource.Table.return_value = mock_table

        with patch.dict(os.environ, {"AWS_PROFILE": "test-profile"}):
            with patch("daylib.config.get_settings", return_value=mock_settings):
                with patch("boto3.resource", return_value=mock_dynamodb_resource):
                    result = runner.invoke(aws_app, ["teardown", "--force"])
                    assert result.exit_code == 0
                    assert "Deleted table" in result.output or "teardown complete" in result.output
                    assert mock_table.delete.call_count == 3

