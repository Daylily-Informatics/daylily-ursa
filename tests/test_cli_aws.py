"""Tests for graph-native `ursa aws` infrastructure commands."""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from daylily_ursa.cli.aws import aws_app


runner = CliRunner()


class TestAwsSetup:
    def test_setup_bootstraps_components(self):
        with (
            patch("daylily_ursa.cli.aws._effective_region", return_value="us-west-2"),
            patch("daylily_ursa.biospecimen.BiospecimenRegistry") as mock_bio,
            patch("daylily_ursa.file_registry.FileRegistry") as mock_files,
            patch("daylily_ursa.manifest_registry.ManifestRegistry") as mock_manifest,
            patch("daylily_ursa.s3_bucket_validator.LinkedBucketManager") as mock_bucket,
            patch("daylily_ursa.workset_customer.CustomerManager") as mock_customer,
            patch("daylily_ursa.workset_state_db.WorksetStateDB") as mock_state,
        ):
            result = runner.invoke(aws_app, ["setup"])

        assert result.exit_code == 0
        assert "Bootstrapping Ursa TapDB resources" in result.output
        mock_state.return_value.bootstrap.assert_called_once()
        mock_customer.return_value.bootstrap.assert_called_once()
        mock_files.return_value.bootstrap.assert_called_once()
        mock_manifest.return_value.bootstrap.assert_called_once()
        mock_bio.return_value.bootstrap.assert_called_once()
        mock_bucket.return_value.bootstrap.assert_called_once()

    def test_setup_returns_nonzero_on_component_failure(self):
        with (
            patch("daylily_ursa.cli.aws._effective_region", return_value="us-west-2"),
            patch("daylily_ursa.workset_state_db.WorksetStateDB") as mock_state,
        ):
            mock_state.return_value.bootstrap.side_effect = RuntimeError("boom")
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
        fake_backend.get_missing_instance_sequences.return_value = []
        fake_backend.list_required_instance_sequences.return_value = ["ws_instance_seq"]

        with (
            patch("daylily_ursa.tapdb_graph.backend.TEMPLATE_DEFINITIONS", [fake_template]),
            patch("daylily_ursa.tapdb_graph.backend.TapDBBackend", return_value=fake_backend),
        ):
            result = runner.invoke(aws_app, ["status"])

        assert result.exit_code == 0
        assert "TapDB Template Status" in result.output
        assert "workflow/workset/analysis/1.0/" in result.output
        assert "ws_instance_seq" in result.output

    def test_status_fails_when_sequence_missing(self):
        fake_template = MagicMock(template_code="actor/customer/account/1.0/")

        class _Ctx:
            def __enter__(self):
                return object()

            def __exit__(self, exc_type, exc, tb):
                return False

        fake_backend = MagicMock()
        fake_backend.session_scope.return_value = _Ctx()
        fake_backend.templates.get_template.return_value = object()
        fake_backend.get_missing_instance_sequences.return_value = ["ct_instance_seq"]
        fake_backend.list_required_instance_sequences.return_value = ["ct_instance_seq"]

        with (
            patch("daylily_ursa.tapdb_graph.backend.TEMPLATE_DEFINITIONS", [fake_template]),
            patch("daylily_ursa.tapdb_graph.backend.TapDBBackend", return_value=fake_backend),
        ):
            result = runner.invoke(aws_app, ["status"])

        assert result.exit_code == 1
        assert "ct_instance_seq" in result.output
        assert "Remediation:" in result.output


class TestAwsRepairSequences:
    def test_repair_sequences_dry_run(self):
        class _Ctx:
            def __enter__(self):
                return object()

            def __exit__(self, exc_type, exc, tb):
                return False

        fake_backend = MagicMock()
        fake_backend.session_scope.return_value = _Ctx()
        fake_backend.get_missing_instance_sequences.return_value = ["ct_instance_seq"]

        with patch("daylily_ursa.tapdb_graph.backend.TapDBBackend", return_value=fake_backend):
            result = runner.invoke(aws_app, ["repair-sequences", "--dry-run"])

        assert result.exit_code == 0
        assert "ct_instance_seq" in result.output
        assert "Dry run only" in result.output
        fake_backend.ensure_instance_sequences.assert_not_called()

    def test_repair_sequences_applies_changes(self):
        class _Ctx:
            def __enter__(self):
                return object()

            def __exit__(self, exc_type, exc, tb):
                return False

        fake_backend = MagicMock()
        fake_backend.session_scope.return_value = _Ctx()
        fake_backend.get_missing_instance_sequences.side_effect = [
            ["ct_instance_seq"],
            [],
        ]

        with patch("daylily_ursa.tapdb_graph.backend.TapDBBackend", return_value=fake_backend):
            result = runner.invoke(aws_app, ["repair-sequences"])

        assert result.exit_code == 0
        assert "Repaired TapDB instance sequences" in result.output
        fake_backend.ensure_instance_sequences.assert_called_once()


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
