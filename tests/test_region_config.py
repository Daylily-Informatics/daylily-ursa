"""Tests for AWS region configuration after AWS_DEFAULT_REGION removal.

These tests verify that:
1. Settings.get_effective_region() uses correct priority order
2. AWS_DEFAULT_REGION is NOT used even if set
3. ClusterService falls back to 'us-west-2' when no regions configured
4. _get_cognito_region() uses correct priority order
"""

import os
from unittest.mock import MagicMock, patch



class TestSettingsGetEffectiveRegion:
    """Tests for Settings.get_effective_region() method."""

    def test_returns_day_aws_region_when_set(self):
        """Test DAY_AWS_REGION takes highest priority."""
        from daylib.config import Settings

        with patch.dict(os.environ, {
            "DAY_AWS_REGION": "eu-central-1",
            "AWS_REGION": "us-east-1",
        }, clear=True):
            # Need to create new instance to pick up env vars
            settings = Settings()
            assert settings.get_effective_region() == "eu-central-1"

    def test_returns_aws_region_when_day_aws_region_not_set(self):
        """Test AWS_REGION is used when DAY_AWS_REGION not set."""
        from daylib.config import Settings

        with patch.dict(os.environ, {
            "AWS_REGION": "ap-southeast-1",
        }, clear=True):
            settings = Settings()
            assert settings.get_effective_region() == "ap-southeast-1"

    def test_returns_fallback_when_no_env_vars(self):
        """Test fallback to 'us-west-2' when no env vars set."""
        from daylib.config import Settings

        with patch.dict(os.environ, {}, clear=True):
            # Remove any region env vars
            os.environ.pop("DAY_AWS_REGION", None)
            os.environ.pop("AWS_REGION", None)
            os.environ.pop("AWS_DEFAULT_REGION", None)
            settings = Settings()
            assert settings.get_effective_region() == "us-west-2"

    def test_ignores_aws_default_region(self):
        """Test AWS_DEFAULT_REGION is intentionally ignored."""
        from daylib.config import Settings

        with patch.dict(os.environ, {
            "AWS_DEFAULT_REGION": "eu-west-1",  # This should be ignored
        }, clear=True):
            os.environ.pop("DAY_AWS_REGION", None)
            os.environ.pop("AWS_REGION", None)
            settings = Settings()
            # Should fallback to us-west-2, NOT use AWS_DEFAULT_REGION
            assert settings.get_effective_region() == "us-west-2"

    def test_aws_region_takes_priority_over_aws_default_region(self):
        """Test AWS_REGION is used even when AWS_DEFAULT_REGION is set."""
        from daylib.config import Settings

        with patch.dict(os.environ, {
            "AWS_DEFAULT_REGION": "eu-west-1",  # Should be ignored
            "AWS_REGION": "us-east-2",  # Should be used
        }, clear=True):
            os.environ.pop("DAY_AWS_REGION", None)
            settings = Settings()
            assert settings.get_effective_region() == "us-east-2"


class TestClusterServiceRegionFallback:
    """Tests for ClusterService region fallback behavior."""

    def test_fallback_to_us_west_2_when_no_regions(self):
        """Test ClusterService falls back to us-west-2 when no regions configured."""
        from daylib.cluster_service import get_cluster_service, reset_cluster_service

        # Reset any existing singleton
        reset_cluster_service()

        with patch.dict(os.environ, {}, clear=True):
            # Clear all region-related env vars
            os.environ.pop("URSA_ALLOWED_REGIONS", None)
            os.environ.pop("AWS_DEFAULT_REGION", None)
            os.environ.pop("AWS_REGION", None)

            # Mock ursa config to return no allowed regions
            mock_config = MagicMock()
            mock_config.is_configured = False
            mock_config.get_allowed_regions.return_value = []
            mock_config.aws_profile = "test-profile"

            # Patch at source module where import happens
            with patch("daylib.ursa_config.get_ursa_config", return_value=mock_config):
                with patch("daylib.cluster_service.ClusterService") as MockClusterService:
                    MockClusterService.return_value = MagicMock()
                    get_cluster_service()

                    # Verify ClusterService was created with us-west-2 fallback
                    MockClusterService.assert_called_once()
                    call_kwargs = MockClusterService.call_args.kwargs
                    assert call_kwargs["regions"] == ["us-west-2"]

        # Reset after test
        reset_cluster_service()

    def test_uses_ursa_allowed_regions_when_set(self):
        """Test URSA_ALLOWED_REGIONS is used when set."""
        from daylib.cluster_service import get_cluster_service, reset_cluster_service

        reset_cluster_service()

        with patch.dict(os.environ, {"URSA_ALLOWED_REGIONS": "eu-west-1,ap-south-1"}, clear=True):
            mock_config = MagicMock()
            mock_config.is_configured = False
            mock_config.get_allowed_regions.return_value = []
            mock_config.aws_profile = "test-profile"

            # Patch at source module
            with patch("daylib.ursa_config.get_ursa_config", return_value=mock_config):
                with patch("daylib.cluster_service.ClusterService") as MockClusterService:
                    MockClusterService.return_value = MagicMock()
                    get_cluster_service()

                    MockClusterService.assert_called_once()
                    call_kwargs = MockClusterService.call_args.kwargs
                    assert call_kwargs["regions"] == ["eu-west-1", "ap-south-1"]

        reset_cluster_service()


class TestGetCognitoRegion:
    """Tests for _get_cognito_region() in daylib/cli/cognito.py."""

    def test_returns_cognito_region_from_config(self):
        """Test cognito_region from config takes highest priority."""
        from daylib.cli.cognito import _get_cognito_region

        mock_config = MagicMock()
        mock_config.cognito_region = "eu-west-1"

        with patch.dict(os.environ, {"AWS_REGION": "us-east-1"}, clear=True):
            # Patch at source module where import happens
            with patch("daylib.ursa_config.get_ursa_config", return_value=mock_config):
                result = _get_cognito_region()
                assert result == "eu-west-1"

    def test_returns_aws_region_when_cognito_region_not_set(self):
        """Test AWS_REGION is used when cognito_region not configured."""
        from daylib.cli.cognito import _get_cognito_region

        mock_config = MagicMock()
        mock_config.cognito_region = None

        with patch.dict(os.environ, {"AWS_REGION": "ap-northeast-1"}, clear=True):
            with patch("daylib.ursa_config.get_ursa_config", return_value=mock_config):
                result = _get_cognito_region()
                assert result == "ap-northeast-1"

    def test_returns_fallback_when_nothing_configured(self):
        """Test fallback to us-west-2 when nothing configured."""
        from daylib.cli.cognito import _get_cognito_region

        mock_config = MagicMock()
        mock_config.cognito_region = None

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AWS_REGION", None)
            os.environ.pop("AWS_DEFAULT_REGION", None)
            with patch("daylib.ursa_config.get_ursa_config", return_value=mock_config):
                result = _get_cognito_region()
                assert result == "us-west-2"

    def test_ignores_aws_default_region(self):
        """Test AWS_DEFAULT_REGION is intentionally ignored."""
        from daylib.cli.cognito import _get_cognito_region

        mock_config = MagicMock()
        mock_config.cognito_region = None

        with patch.dict(os.environ, {"AWS_DEFAULT_REGION": "eu-central-1"}, clear=True):
            os.environ.pop("AWS_REGION", None)
            with patch("daylib.ursa_config.get_ursa_config", return_value=mock_config):
                result = _get_cognito_region()
                # Should fallback to us-west-2, NOT use AWS_DEFAULT_REGION
                assert result == "us-west-2"
