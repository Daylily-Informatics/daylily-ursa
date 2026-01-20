"""Tests for the Cognito CLI commands (ursa cognito *)."""

import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from daylib.cli.cognito import cognito_app


runner = CliRunner()


@pytest.fixture
def mock_settings():
    """Mock settings."""
    settings = MagicMock()
    settings.get_effective_region.return_value = "us-west-2"
    return settings


@pytest.fixture
def mock_cognito_client():
    """Mock boto3 Cognito client."""
    client = MagicMock()
    return client


class TestCognitoStatus:
    """Tests for ursa cognito status command."""

    def test_status_requires_aws_profile(self):
        """Test that status fails without AWS_PROFILE set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_PROFILE", None)
            result = runner.invoke(cognito_app, ["status"])
            assert result.exit_code == 1
            assert "AWS_PROFILE" in result.output


class TestCognitoListUsers:
    """Tests for ursa cognito list-users command."""

    def test_list_users_requires_aws_profile(self):
        """Test that list-users fails without AWS_PROFILE set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_PROFILE", None)
            result = runner.invoke(cognito_app, ["list-users"])
            assert result.exit_code == 1
            assert "AWS_PROFILE" in result.output

    def test_list_users_requires_pool_id(self):
        """Test that list-users fails without COGNITO_USER_POOL_ID."""
        with patch.dict(os.environ, {"AWS_PROFILE": "test"}, clear=True):
            result = runner.invoke(cognito_app, ["list-users"])
            assert result.exit_code == 1
            assert "COGNITO_USER_POOL_ID" in result.output

    def test_list_users_shows_users(self, mock_settings, mock_cognito_client):
        """Test that list-users displays users in a table."""
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {
                "Users": [
                    {
                        "Username": "user@example.com",
                        "UserStatus": "CONFIRMED",
                        "Enabled": True,
                        "UserCreateDate": datetime(2025, 1, 1, 12, 0),
                        "Attributes": [
                            {"Name": "email", "Value": "user@example.com"},
                            {"Name": "custom:customer_id", "Value": "CUST1"},
                        ],
                    }
                ]
            }
        ]
        mock_cognito_client.get_paginator.return_value = mock_paginator

        with patch.dict(os.environ, {"AWS_PROFILE": "test", "COGNITO_USER_POOL_ID": "us-west-2_test"}):
            with patch("daylib.config.get_settings", return_value=mock_settings):
                with patch("boto3.client", return_value=mock_cognito_client):
                    result = runner.invoke(cognito_app, ["list-users"])
                    assert result.exit_code == 0
                    assert "user@example.com" in result.output
                    assert "CUST1" in result.output


class TestCognitoExport:
    """Tests for ursa cognito export command."""

    def test_export_requires_aws_profile(self):
        """Test that export fails without AWS_PROFILE set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_PROFILE", None)
            result = runner.invoke(cognito_app, ["export"])
            assert result.exit_code == 1
            assert "AWS_PROFILE" in result.output

    def test_export_requires_pool_id(self):
        """Test that export fails without COGNITO_USER_POOL_ID."""
        with patch.dict(os.environ, {"AWS_PROFILE": "test"}, clear=True):
            result = runner.invoke(cognito_app, ["export"])
            assert result.exit_code == 1
            assert "COGNITO_USER_POOL_ID" in result.output


class TestCognitoDeleteUser:
    """Tests for ursa cognito delete-user command."""

    def test_delete_user_requires_aws_profile(self):
        """Test that delete-user fails without AWS_PROFILE set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_PROFILE", None)
            result = runner.invoke(cognito_app, ["delete-user", "--email", "test@example.com", "--force"])
            assert result.exit_code == 1
            assert "AWS_PROFILE" in result.output

    def test_delete_user_prompts_without_force(self, mock_settings, mock_cognito_client):
        """Test that delete-user prompts for confirmation."""
        with patch.dict(os.environ, {"AWS_PROFILE": "test", "COGNITO_USER_POOL_ID": "us-west-2_test"}):
            with patch("daylib.config.get_settings", return_value=mock_settings):
                with patch("boto3.client", return_value=mock_cognito_client):
                    result = runner.invoke(cognito_app, ["delete-user", "--email", "test@example.com"], input="n\n")
                    assert "Cancelled" in result.output


class TestCognitoDeleteAllUsers:
    """Tests for ursa cognito delete-all-users command."""

    def test_delete_all_users_requires_aws_profile(self):
        """Test that delete-all-users fails without AWS_PROFILE set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_PROFILE", None)
            result = runner.invoke(cognito_app, ["delete-all-users", "--force"])
            assert result.exit_code == 1
            assert "AWS_PROFILE" in result.output

    def test_delete_all_users_prompts_without_force(self, mock_settings):
        """Test that delete-all-users prompts for confirmation."""
        with patch.dict(os.environ, {"AWS_PROFILE": "test", "COGNITO_USER_POOL_ID": "us-west-2_test"}):
            result = runner.invoke(cognito_app, ["delete-all-users"], input="n\n")
            assert "Cancelled" in result.output


class TestCognitoTeardown:
    """Tests for ursa cognito teardown command."""

    def test_teardown_requires_aws_profile(self):
        """Test that teardown fails without AWS_PROFILE set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_PROFILE", None)
            result = runner.invoke(cognito_app, ["teardown", "--force"])
            assert result.exit_code == 1
            assert "AWS_PROFILE" in result.output

