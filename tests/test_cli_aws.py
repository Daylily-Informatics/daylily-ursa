"""Tests for graph-native `ursa aws` infrastructure commands."""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from daylib.cli.aws import aws_app


runner = CliRunner()


class TestAwsSetup:
    def test_setup_bootstraps_components(self):
        with patch("daylib.cli.aws._effective_region", return_value="us-west-2"), patch(
            "daylib.biospecimen.BiospecimenRegistry"
        ) as mock_bio, patch("daylib.file_registry.FileRegistry") as mock_files, patch(
            "daylib.manifest_registry.ManifestRegistry"
        ) as mock_manifest, patch(
            "daylib.s3_bucket_validator.LinkedBucketManager"
        ) as mock_bucket, patch(
            "daylib.workset_customer.CustomerManager"
        ) as mock_customer, patch(
            "daylib.workset_state_db.WorksetStateDB"
        ) as mock_state:
            result = runner.invoke(aws_app, ["setup"])

        assert result.exit_code == 0
        assert "Bootstrapping Ursa TapDB resources" in result.output
        mock_state.return_value.create_table_if_not_exists.assert_called_once()
        mock_customer.return_value.create_customer_table_if_not_exists.assert_called_once()
        mock_files.return_value.create_tables_if_not_exist.assert_called_once()
        mock_manifest.return_value.create_table_if_not_exists.assert_called_once()
        mock_bio.return_value.create_tables_if_not_exist.assert_called_once()
        mock_bucket.return_value.create_table_if_not_exists.assert_called_once()

    def test_setup_returns_nonzero_on_component_failure(self):
        with patch("daylib.cli.aws._effective_region", return_value="us-west-2"), patch(
            "daylib.workset_state_db.WorksetStateDB"
        ) as mock_state:
            mock_state.return_value.create_table_if_not_exists.side_effect = RuntimeError("boom")
            result = runner.invoke(aws_app, ["setup"])

        assert result.exit_code == 1
        assert "worksets" in result.output


class TestAwsStatus:
    def test_status_lists_templates(self):
        fake_template = MagicMock(template_code="workflow/workset/analysis/1.0/")

        class _Ctx:
            def __enter__(self):
                return object()

            def __exit__(self, exc_type, exc, tb):
                return False

        fake_backend = MagicMock()
        fake_backend.session_scope.return_value = _Ctx()
        fake_backend.templates.get_template.return_value = object()

        with patch("daylib.tapdb_graph.backend.TEMPLATE_DEFINITIONS", [fake_template]), patch(
            "daylib.tapdb_graph.backend.TapDBBackend", return_value=fake_backend
        ):
            result = runner.invoke(aws_app, ["status"])

        assert result.exit_code == 0
        assert "TapDB Template Status" in result.output
        assert "workflow/workset/analysis/1.0/" in result.output


class TestAwsTeardown:
    def test_teardown_requires_force(self):
        result = runner.invoke(aws_app, ["teardown"])
        assert result.exit_code == 1
        assert "Teardown is disabled by default" in result.output

    def test_teardown_with_force_prints_instructions(self):
        result = runner.invoke(aws_app, ["teardown", "--force"])
        assert result.exit_code == 0
        assert "Manual teardown required" in result.output
        assert "Re-run `ursa aws setup`" in result.output
