"""Tests for workset_integration.py - State bridge between DynamoDB and S3.

Tests the integration layer that synchronizes DynamoDB state management
with S3 sentinel-based processing system.
"""

import datetime as dt
import json
from unittest.mock import MagicMock, patch, call

import pytest

from daylib.workset_integration import (
    WorksetIntegration,
    SENTINEL_FILES,
    WORK_YAML_NAME,
    INFO_YAML_NAME,
)


@pytest.fixture
def mock_state_db():
    """Create mock WorksetStateDB."""
    mock_db = MagicMock()
    mock_db.register_workset.return_value = True
    mock_db.get_workset.return_value = {
        "workset_id": "test-ws-001",
        "state": "ready",
        "priority": "normal",
        "bucket": "test-bucket",
        "prefix": "worksets/test-ws-001/",
        "created_at": "2024-01-15T10:00:00Z",
        "updated_at": "2024-01-15T10:00:00Z",
    }
    mock_db.update_state.return_value = None
    mock_db.get_ready_worksets.return_value = []
    return mock_db


@pytest.fixture
def mock_s3_client():
    """Create mock S3 client."""
    mock_s3 = MagicMock()
    mock_s3.put_object.return_value = {}
    mock_s3.get_object.return_value = {
        "Body": MagicMock(read=lambda: b"2024-01-15T10:00:00Z")
    }
    mock_s3.delete_object.return_value = {}
    mock_s3.list_objects_v2.return_value = {"Contents": [], "KeyCount": 0}
    return mock_s3


@pytest.fixture
def mock_dynamodb():
    """Mock DynamoDB resource for WorksetStateDB tests."""
    with patch("daylib.workset_state_db.boto3.Session") as mock_session:
        mock_resource = MagicMock()
        mock_table = MagicMock()
        mock_client = MagicMock()

        mock_session.return_value.resource.return_value = mock_resource
        mock_session.return_value.client.return_value = mock_client
        mock_resource.Table.return_value = mock_table

        yield {
            "session": mock_session,
            "resource": mock_resource,
            "table": mock_table,
            "client": mock_client,
        }


@pytest.fixture
def integration(mock_state_db, mock_s3_client):
    """Create WorksetIntegration instance."""
    return WorksetIntegration(
        state_db=mock_state_db,
        s3_client=mock_s3_client,
        bucket="test-bucket",
        prefix="worksets/",
        region="us-west-2",
    )


class TestWorksetIntegrationInit:
    """Test WorksetIntegration initialization."""

    def test_init_with_provided_s3_client(self, mock_state_db, mock_s3_client):
        """Test initialization with provided S3 client."""
        integration = WorksetIntegration(
            state_db=mock_state_db,
            s3_client=mock_s3_client,
            bucket="my-bucket",
            prefix="data/worksets/",
        )
        assert integration._s3 is mock_s3_client
        assert integration.bucket == "my-bucket"
        assert integration.prefix == "data/worksets/"

    def test_init_creates_s3_client_if_not_provided(self, mock_state_db):
        """Test initialization creates S3 client if not provided."""
        with patch("daylib.workset_integration.boto3") as mock_boto3:
            mock_session = MagicMock()
            mock_boto3.Session.return_value = mock_session

            integration = WorksetIntegration(
                state_db=mock_state_db,
                bucket="my-bucket",
                region="us-east-1",
            )

            mock_boto3.Session.assert_called_once_with(region_name="us-east-1")
            mock_session.client.assert_called_once_with("s3")

    def test_init_prefix_normalization(self, mock_state_db, mock_s3_client):
        """Test prefix is normalized with trailing slash."""
        integration = WorksetIntegration(
            state_db=mock_state_db,
            s3_client=mock_s3_client,
            bucket="bucket",
            prefix="worksets",  # No trailing slash
        )
        assert integration.prefix == "worksets/"


class TestRegisterWorkset:
    """Test workset registration."""

    def test_register_workset_dual_write(self, integration, mock_state_db, mock_s3_client):
        """Test registration writes to both DynamoDB and S3."""
        result = integration.register_workset(
            workset_id="new-ws-001",
            bucket="test-bucket",
            prefix="worksets/new-ws-001/",
            priority="high",
            metadata={"name": "Test Workset"},
            write_s3=True,
            write_dynamodb=True,
        )

        assert result is True
        mock_state_db.register_workset.assert_called_once()
        # S3 sentinel should be written
        assert mock_s3_client.put_object.called

    def test_register_workset_dynamodb_only(self, integration, mock_state_db, mock_s3_client):
        """Test registration to DynamoDB only."""
        result = integration.register_workset(
            workset_id="db-only-ws",
            bucket="test-bucket",
            prefix="worksets/db-only-ws/",
            write_s3=False,
            write_dynamodb=True,
        )

        assert result is True
        mock_state_db.register_workset.assert_called_once()

    def test_register_workset_s3_only(self, integration, mock_state_db, mock_s3_client):
        """Test registration to S3 only."""
        result = integration.register_workset(
            workset_id="s3-only-ws",
            bucket="test-bucket",
            prefix="worksets/s3-only-ws/",
            write_s3=True,
            write_dynamodb=False,
        )

        assert result is True
        mock_state_db.register_workset.assert_not_called()
        assert mock_s3_client.put_object.called


class TestWriteSentinel:
    """Test S3 sentinel writing."""

    def test_write_ready_sentinel(self, integration, mock_s3_client):
        """Test writing ready sentinel file."""
        timestamp = "2024-01-15T10:00:00Z"
        integration._write_sentinel(
            bucket="test-bucket",
            prefix="worksets/test/",
            state="ready",
            timestamp=timestamp,
        )

        mock_s3_client.put_object.assert_called_once()
        call_kwargs = mock_s3_client.put_object.call_args[1]
        assert call_kwargs["Bucket"] == "test-bucket"
        assert call_kwargs["Key"] == "worksets/test/daylily.ready"
        assert call_kwargs["Body"] == timestamp.encode()

    def test_write_in_progress_sentinel(self, integration, mock_s3_client):
        """Test writing in_progress sentinel file."""
        integration._write_sentinel(
            bucket="test-bucket",
            prefix="worksets/test/",
            state="in_progress",
            timestamp="2024-01-15T11:00:00Z",
        )

        call_kwargs = mock_s3_client.put_object.call_args[1]
        assert "daylily.in_progress" in call_kwargs["Key"]

    def test_write_complete_sentinel(self, integration, mock_s3_client):
        """Test writing complete sentinel file."""
        integration._write_sentinel(
            bucket="test-bucket",
            prefix="worksets/test/",
            state="complete",
            timestamp="2024-01-15T12:00:00Z",
        )

        call_kwargs = mock_s3_client.put_object.call_args[1]
        assert "daylily.complete" in call_kwargs["Key"]

    def test_write_error_sentinel(self, integration, mock_s3_client):
        """Test writing error sentinel file."""
        integration._write_sentinel(
            bucket="test-bucket",
            prefix="worksets/test/",
            state="error",
            timestamp="2024-01-15T12:00:00Z",
        )

        call_kwargs = mock_s3_client.put_object.call_args[1]
        assert "daylily.error" in call_kwargs["Key"]


class TestUpdateState:
    """Test state update synchronization."""

    def test_update_state_syncs_both(self, integration, mock_state_db, mock_s3_client):
        """Test state update syncs to both DynamoDB and S3."""
        integration.update_state(
            workset_id="test-ws-001",
            new_state="in_progress",
            reason="Started processing",
            write_s3=True,
            write_dynamodb=True,
        )

        mock_state_db.update_state.assert_called_once()
        # Should write new sentinel
        assert mock_s3_client.put_object.called

    def test_update_state_dynamodb_only(self, integration, mock_state_db, mock_s3_client):
        """Test state update to DynamoDB only."""
        mock_s3_client.reset_mock()

        integration.update_state(
            workset_id="test-ws-001",
            new_state="in_progress",
            reason="Processing started",
            write_s3=False,
            write_dynamodb=True,
        )

        mock_state_db.update_state.assert_called_once()

    def test_update_state_s3_only(self, integration, mock_state_db, mock_s3_client):
        """Test state update to S3 only."""
        mock_state_db.reset_mock()
        mock_s3_client.reset_mock()

        integration.update_state(
            workset_id="test-ws-001",
            new_state="complete",
            reason="Processing finished",
            write_s3=True,
            write_dynamodb=False,
        )

        mock_state_db.update_state.assert_not_called()
        assert mock_s3_client.put_object.called


class TestGetReadyWorksets:
    """Test getting ready worksets from DynamoDB."""

    def test_get_ready_worksets(self, integration, mock_state_db):
        """Test getting ready worksets from DynamoDB."""
        mock_state_db.get_ready_worksets_prioritized.return_value = [
            {
                "workset_id": "ws-001",
                "bucket": "test-bucket",
                "prefix": "worksets/ws-001/",
                "priority": "high",
            },
            {
                "workset_id": "ws-002",
                "bucket": "test-bucket",
                "prefix": "worksets/ws-002/",
                "priority": "normal",
            },
        ]

        worksets = integration.get_ready_worksets()

        assert len(worksets) == 2
        assert worksets[0]["workset_id"] == "ws-001"
        mock_state_db.get_ready_worksets_prioritized.assert_called_once()


class TestBuildWorkYaml:
    """Test work YAML generation from template."""

    def test_build_work_yaml_uses_template(self, integration):
        """Test that work YAML uses the template file structure."""
        metadata = {
            "samples": [{"sample_id": "S1", "fastq_r1": "s3://bucket/S1_R1.fq.gz"}],
            "reference_genome": "GRCh38",
        }

        # Returns string now, not dict
        work_yaml_content = integration._build_work_yaml("test-ws", metadata, "test-bucket", "worksets/test-ws/")

        # Should be a string (YAML content)
        assert isinstance(work_yaml_content, str)

        # Should contain template structure elements
        assert "day-clone:" in work_yaml_content
        assert "dy-r:" in work_yaml_content
        assert "export_uri:" in work_yaml_content

        # Should have {workdir_name} replaced with workset_id
        assert "test-ws" in work_yaml_content
        assert "{workdir_name}" not in work_yaml_content

    def test_build_work_yaml_with_export_uri(self, integration):
        """Test building work YAML with custom export URI."""
        metadata = {
            "samples": [{"sample_id": "S1", "fastq_r1": "s3://bucket/S1_R1.fq.gz"}],
            "reference_genome": "hg38",
            "export_uri": "s3://export-bucket/results/",
        }

        work_yaml_content = integration._build_work_yaml("test-ws", metadata, "test-bucket", "worksets/test-ws/")

        # Should contain the custom export_uri
        assert 's3://export-bucket/results/' in work_yaml_content

    def test_build_work_yaml_default_export_uri(self, integration):
        """Test that default export URI uses customer bucket."""
        metadata = {
            "samples": [{"sample_id": "S1", "fastq_r1": "s3://bucket/S1_R1.fq.gz"}],
            "reference_genome": "GRCh38",
        }

        work_yaml_content = integration._build_work_yaml("test-ws", metadata, "customer-bucket", "worksets/test-ws/")

        # Should contain the default export_uri with customer bucket
        assert 's3://customer-bucket/worksets/test-ws/results/' in work_yaml_content


class TestSyncDynamoDBToS3:
    """Test syncing DynamoDB worksets to S3."""

    def test_sync_dynamodb_to_s3(self, integration, mock_state_db, mock_s3_client):
        """Test syncing a single workset to S3."""
        # Return a dict with metadata as a dict, not a JSON string
        mock_state_db.get_workset.return_value = {
            "workset_id": "sync-ws",
            "state": "ready",
            "bucket": "test-bucket",
            "prefix": "worksets/sync-ws/",
            "metadata": {
                "samples": [{"sample_id": "S1", "fastq_r1": "s3://b/r1.fq.gz"}],
                "reference_genome": "GRCh38",
            },
        }

        result = integration.sync_dynamodb_to_s3("sync-ws")

        assert result is True
        # Should write work YAML and sentinel
        assert mock_s3_client.put_object.call_count >= 1


class TestAcquireReleaseLock:
    """Test lock acquisition and release."""

    def test_acquire_lock(self, integration, mock_state_db):
        """Test acquiring a lock."""
        mock_state_db.acquire_lock.return_value = True

        result = integration.acquire_lock("test-ws-001", "processor-1")

        assert result is True
        mock_state_db.acquire_lock.assert_called_once()

    def test_release_lock(self, integration, mock_state_db):
        """Test releasing a lock."""
        mock_state_db.release_lock.return_value = True

        result = integration.release_lock("test-ws-001", "processor-1")

        assert result is True
        mock_state_db.release_lock.assert_called_once()


class TestMonitorWorksetExistenceCheck:
    """Tests for monitor's workset existence check - monitor should NEVER create worksets."""

    @pytest.fixture
    def mock_state_db_for_monitor(self):
        """Create mock WorksetStateDB for monitor tests."""
        mock_db = MagicMock()
        return mock_db

    def test_workset_exists_returns_true_when_found(self, mock_state_db_for_monitor):
        """Test that _workset_exists_in_dynamodb returns True when workset exists."""
        from daylib.workset_monitor import WorksetMonitor

        mock_state_db_for_monitor.get_workset.return_value = {
            "workset_id": "test-ws-001",
            "state": "ready",
        }

        monitor = WorksetMonitor.__new__(WorksetMonitor)
        monitor.state_db = mock_state_db_for_monitor

        result = monitor._workset_exists_in_dynamodb("test-ws-001")

        assert result is True
        mock_state_db_for_monitor.get_workset.assert_called_once_with("test-ws-001")

    def test_workset_exists_returns_false_when_not_found(self, mock_state_db_for_monitor):
        """Test that _workset_exists_in_dynamodb returns False when workset doesn't exist."""
        from daylib.workset_monitor import WorksetMonitor

        mock_state_db_for_monitor.get_workset.return_value = None

        monitor = WorksetMonitor.__new__(WorksetMonitor)
        monitor.state_db = mock_state_db_for_monitor

        result = monitor._workset_exists_in_dynamodb("unknown-ws")

        assert result is False
        mock_state_db_for_monitor.get_workset.assert_called_once_with("unknown-ws")

    def test_workset_exists_returns_false_when_no_state_db(self):
        """Test that _workset_exists_in_dynamodb returns False when state_db is None."""
        from daylib.workset_monitor import WorksetMonitor

        monitor = WorksetMonitor.__new__(WorksetMonitor)
        monitor.state_db = None

        result = monitor._workset_exists_in_dynamodb("test-ws-001")

        assert result is False

    def test_monitor_does_not_create_worksets(self, mock_state_db_for_monitor):
        """Test that monitor does NOT call register_workset for unknown worksets."""
        from daylib.workset_monitor import WorksetMonitor

        # Workset doesn't exist in DynamoDB
        mock_state_db_for_monitor.get_workset.return_value = None

        monitor = WorksetMonitor.__new__(WorksetMonitor)
        monitor.state_db = mock_state_db_for_monitor

        # Check existence - should return False
        result = monitor._workset_exists_in_dynamodb("new-workset")

        assert result is False
        # Most importantly: register_workset should NOT be called
        mock_state_db_for_monitor.register_workset.assert_not_called()


class TestMonitorConcurrentProcessing:
    """Tests for monitor's concurrent workset processing."""

    def test_mark_workset_active_success(self):
        """Test marking a workset as active when slots available."""
        import threading
        from daylib.workset_monitor import WorksetMonitor

        monitor = WorksetMonitor.__new__(WorksetMonitor)
        monitor._active_worksets = set()
        monitor._active_worksets_lock = threading.Lock()
        monitor.config = MagicMock()
        monitor.config.monitor.max_concurrent_worksets = 3

        result = monitor._mark_workset_active("ws-001")

        assert result is True
        assert "ws-001" in monitor._active_worksets

    def test_mark_workset_active_at_capacity(self):
        """Test marking a workset as active when at capacity."""
        import threading
        from daylib.workset_monitor import WorksetMonitor

        monitor = WorksetMonitor.__new__(WorksetMonitor)
        monitor._active_worksets = {"ws-001", "ws-002", "ws-003"}
        monitor._active_worksets_lock = threading.Lock()
        monitor.config = MagicMock()
        monitor.config.monitor.max_concurrent_worksets = 3

        result = monitor._mark_workset_active("ws-004")

        assert result is False
        assert "ws-004" not in monitor._active_worksets

    def test_mark_workset_active_already_active(self):
        """Test marking a workset as active when already active."""
        import threading
        from daylib.workset_monitor import WorksetMonitor

        monitor = WorksetMonitor.__new__(WorksetMonitor)
        monitor._active_worksets = {"ws-001"}
        monitor._active_worksets_lock = threading.Lock()
        monitor.config = MagicMock()
        monitor.config.monitor.max_concurrent_worksets = 3

        result = monitor._mark_workset_active("ws-001")

        assert result is False

    def test_mark_workset_inactive(self):
        """Test marking a workset as inactive."""
        import threading
        from daylib.workset_monitor import WorksetMonitor

        monitor = WorksetMonitor.__new__(WorksetMonitor)
        monitor._active_worksets = {"ws-001", "ws-002"}
        monitor._active_worksets_lock = threading.Lock()
        monitor.config = MagicMock()
        monitor.config.monitor.max_concurrent_worksets = 3

        monitor._mark_workset_inactive("ws-001")

        assert "ws-001" not in monitor._active_worksets
        assert "ws-002" in monitor._active_worksets

    def test_get_active_count(self):
        """Test getting the count of active worksets."""
        import threading
        from daylib.workset_monitor import WorksetMonitor

        monitor = WorksetMonitor.__new__(WorksetMonitor)
        monitor._active_worksets = {"ws-001", "ws-002"}
        monitor._active_worksets_lock = threading.Lock()

        assert monitor._get_active_count() == 2

    def test_is_workset_active(self):
        """Test checking if a workset is active."""
        import threading
        from daylib.workset_monitor import WorksetMonitor

        monitor = WorksetMonitor.__new__(WorksetMonitor)
        monitor._active_worksets = {"ws-001"}
        monitor._active_worksets_lock = threading.Lock()

        assert monitor._is_workset_active("ws-001") is True
        assert monitor._is_workset_active("ws-002") is False

    def test_should_skip_workset_complete(self):
        """Test that completed worksets are skipped."""
        from daylib.workset_monitor import WorksetMonitor, SENTINEL_FILES

        monitor = WorksetMonitor.__new__(WorksetMonitor)
        monitor.attempt_restart = False

        workset = MagicMock()
        workset.sentinels = {SENTINEL_FILES["complete"]: "2024-01-01"}

        assert monitor._should_skip_workset(workset) is True

    def test_should_skip_workset_error_no_restart(self):
        """Test that errored worksets are skipped when restart not enabled."""
        from daylib.workset_monitor import WorksetMonitor, SENTINEL_FILES

        monitor = WorksetMonitor.__new__(WorksetMonitor)
        monitor.attempt_restart = False

        workset = MagicMock()
        workset.sentinels = {SENTINEL_FILES["error"]: "error message"}

        assert monitor._should_skip_workset(workset) is True

    def test_should_skip_workset_error_with_restart(self):
        """Test that errored worksets are NOT skipped when restart enabled."""
        from daylib.workset_monitor import WorksetMonitor, SENTINEL_FILES

        monitor = WorksetMonitor.__new__(WorksetMonitor)
        monitor.attempt_restart = True

        workset = MagicMock()
        workset.sentinels = {SENTINEL_FILES["error"]: "error message"}

        assert monitor._should_skip_workset(workset) is False

    def test_should_skip_workset_ignored(self):
        """Test that ignored worksets are skipped."""
        from daylib.workset_monitor import WorksetMonitor, SENTINEL_FILES

        monitor = WorksetMonitor.__new__(WorksetMonitor)
        monitor.attempt_restart = False

        workset = MagicMock()
        workset.sentinels = {SENTINEL_FILES["ignore"]: "ignored"}

        assert monitor._should_skip_workset(workset) is True

    def test_should_not_skip_ready_workset(self):
        """Test that ready worksets are NOT skipped."""
        from daylib.workset_monitor import WorksetMonitor, SENTINEL_FILES

        monitor = WorksetMonitor.__new__(WorksetMonitor)
        monitor.attempt_restart = False

        workset = MagicMock()
        workset.sentinels = {SENTINEL_FILES["ready"]: "ready"}

        assert monitor._should_skip_workset(workset) is False


class TestFormatBytes:
    """Tests for the _format_bytes helper method."""

    def test_format_bytes_bytes(self):
        """Test formatting small byte values."""
        from daylib.workset_monitor import WorksetMonitor

        monitor = WorksetMonitor.__new__(WorksetMonitor)
        assert monitor._format_bytes(0) == "0B"
        assert monitor._format_bytes(512) == "512B"
        assert monitor._format_bytes(1023) == "1023B"

    def test_format_bytes_kilobytes(self):
        """Test formatting kilobyte values."""
        from daylib.workset_monitor import WorksetMonitor

        monitor = WorksetMonitor.__new__(WorksetMonitor)
        assert monitor._format_bytes(1024) == "1.0K"
        assert monitor._format_bytes(2048) == "2.0K"
        # Note: uses integer division so 1536 -> 1K
        assert monitor._format_bytes(1536) == "1.0K"

    def test_format_bytes_megabytes(self):
        """Test formatting megabyte values."""
        from daylib.workset_monitor import WorksetMonitor

        monitor = WorksetMonitor.__new__(WorksetMonitor)
        assert monitor._format_bytes(1024 * 1024) == "1.0M"
        assert monitor._format_bytes(256 * 1024 * 1024) == "256.0M"

    def test_format_bytes_gigabytes(self):
        """Test formatting gigabyte values."""
        from daylib.workset_monitor import WorksetMonitor

        monitor = WorksetMonitor.__new__(WorksetMonitor)
        assert monitor._format_bytes(1024 * 1024 * 1024) == "1.0G"
        assert monitor._format_bytes(13 * 1024 * 1024 * 1024) == "13.0G"

    def test_format_bytes_terabytes(self):
        """Test formatting terabyte values."""
        from daylib.workset_monitor import WorksetMonitor

        monitor = WorksetMonitor.__new__(WorksetMonitor)
        assert monitor._format_bytes(1024 * 1024 * 1024 * 1024) == "1.0T"


class TestExtractSampleIdFromPath:
    """Tests for the _extract_sample_id_from_path helper method."""

    def test_extract_sample_id_cram(self):
        """Test extracting sample ID from CRAM file path."""
        from daylib.workset_monitor import WorksetMonitor

        monitor = WorksetMonitor.__new__(WorksetMonitor)

        path = "/fsx/analysis/results/day/hg38/RUN001_SAMPLE123_EXP001-L001-BARCODE-libprep-vendor-seq-platform.cram"
        assert monitor._extract_sample_id_from_path(path) == "SAMPLE123"

    def test_extract_sample_id_snv_vcf(self):
        """Test extracting sample ID from SNV VCF file path."""
        from daylib.workset_monitor import WorksetMonitor

        monitor = WorksetMonitor.__new__(WorksetMonitor)

        path = "/fsx/analysis/results/day/hg38/RUN001_SAMPLE456_EXP002.snv.sort.vcf.gz"
        assert monitor._extract_sample_id_from_path(path) == "SAMPLE456"

    def test_extract_sample_id_sv_vcf(self):
        """Test extracting sample ID from SV VCF file path."""
        from daylib.workset_monitor import WorksetMonitor

        monitor = WorksetMonitor.__new__(WorksetMonitor)

        path = "/fsx/analysis/results/day/hg38/RUN001_SAMPLE789_EXP003.sv.sort.vcf.gz"
        assert monitor._extract_sample_id_from_path(path) == "SAMPLE789"

    def test_extract_sample_id_invalid_format(self):
        """Test extracting sample ID from invalid format returns None."""
        from daylib.workset_monitor import WorksetMonitor

        monitor = WorksetMonitor.__new__(WorksetMonitor)

        # Single part filename
        path = "/fsx/analysis/results/day/hg38/singlename.cram"
        assert monitor._extract_sample_id_from_path(path) is None

    def test_extract_sample_id_simple_format(self):
        """Test extracting sample ID from simple two-part format."""
        from daylib.workset_monitor import WorksetMonitor

        monitor = WorksetMonitor.__new__(WorksetMonitor)

        path = "/fsx/analysis/RUNID_SAMPLEID.cram"
        assert monitor._extract_sample_id_from_path(path) == "SAMPLEID"


class TestCollectPostExportMetrics:
    """Tests for post-export metrics collection from S3."""

    @pytest.fixture
    def mock_monitor(self):
        """Create a mock monitor with required attributes."""
        from daylib.workset_monitor import WorksetMonitor

        monitor = WorksetMonitor.__new__(WorksetMonitor)
        monitor.state_db = MagicMock()
        monitor.dry_run = False
        monitor.debug = False
        monitor.s3_client = MagicMock()
        monitor.config = MagicMock()
        monitor.config.aws.region = "us-east-1"
        monitor.config.aws.profile = None
        return monitor

    def test_collect_directory_size_from_s3(self, mock_monitor):
        """Test collecting total directory size from S3."""
        workset = MagicMock()
        workset.name = "test-ws-001"

        # Mock S3 paginator response with files totaling ~14GB
        mock_paginator = MagicMock()
        mock_monitor.s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "prefix/workset/file1.cram", "Size": 5368709120},
                    {"Key": "prefix/workset/file2.cram", "Size": 5368709120},
                    {"Key": "prefix/workset/file3.vcf.gz", "Size": 3221225472},
                ]
            }
        ]

        mock_monitor._update_progress_step = MagicMock()
        mock_monitor.state_db.update_performance_metrics = MagicMock(return_value=True)

        # Mock PipelineStatusFetcher
        with patch("daylib.pipeline_status.PipelineStatusFetcher") as mock_fetcher_class:
            mock_fetcher = MagicMock()
            mock_fetcher_class.return_value = mock_fetcher
            mock_fetcher.fetch_performance_metrics_from_s3.return_value = {
                "alignment_stats": None,
                "benchmark_data": None,
                "cost_summary": None,
            }

            result = mock_monitor._collect_post_export_metrics(
                workset, "s3://test-bucket/prefix/workset"
            )

        assert result is not None
        assert result["analysis_directory_size_bytes"] == 13958643712
        assert result["analysis_directory_size_human"] == "13.0G"

    def test_collect_file_metrics_cram_from_s3(self, mock_monitor):
        """Test collecting CRAM file metrics from S3."""
        workset = MagicMock()
        workset.name = "test-ws-002"

        # Mock S3 paginator response with CRAM files
        mock_paginator = MagicMock()
        mock_monitor.s3_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "prefix/workset/results/day/hg38/RUN001_SAMPLE1_EXP001.cram", "Size": 5368709120},
                    {"Key": "prefix/workset/results/day/hg38/RUN001_SAMPLE2_EXP002.cram", "Size": 4294967296},
                ]
            }
        ]

        mock_monitor._update_progress_step = MagicMock()
        mock_monitor.state_db.update_performance_metrics = MagicMock(return_value=True)

        # Mock PipelineStatusFetcher
        with patch("daylib.pipeline_status.PipelineStatusFetcher") as mock_fetcher_class:
            mock_fetcher = MagicMock()
            mock_fetcher_class.return_value = mock_fetcher
            mock_fetcher.fetch_performance_metrics_from_s3.return_value = {
                "alignment_stats": None,
                "benchmark_data": None,
                "cost_summary": None,
            }

            result = mock_monitor._collect_post_export_metrics(
                workset, "s3://test-bucket/prefix/workset"
            )

        assert result is not None
        assert "per_sample_metrics" in result
        assert "SAMPLE1" in result["per_sample_metrics"]
        assert result["per_sample_metrics"]["SAMPLE1"]["cram_count"] == 1
        assert result["per_sample_metrics"]["SAMPLE1"]["cram_size_bytes"] == 5368709120
        assert "SAMPLE2" in result["per_sample_metrics"]
        assert result["per_sample_metrics"]["SAMPLE2"]["cram_count"] == 1
        assert result["per_sample_metrics"]["SAMPLE2"]["cram_size_bytes"] == 4294967296

    def test_collect_metrics_no_s3_uri(self, mock_monitor):
        """Test that metrics collection gracefully handles missing S3 URI."""
        workset = MagicMock()
        workset.name = "test-ws-003"

        mock_monitor._update_progress_step = MagicMock()

        # Should return None when no S3 URI provided
        result = mock_monitor._collect_post_export_metrics(workset, None)

        assert result is None

    def test_collect_metrics_invalid_s3_uri(self, mock_monitor):
        """Test that metrics collection handles invalid S3 URI."""
        workset = MagicMock()
        workset.name = "test-ws-004"

        mock_monitor._update_progress_step = MagicMock()

        # Should return None for invalid S3 URI
        result = mock_monitor._collect_post_export_metrics(workset, "not-an-s3-uri")

        assert result is None


class TestParseBenchmarkCosts:
    """Tests for benchmark cost parsing."""

    @pytest.fixture
    def mock_monitor(self):
        """Create a mock monitor for benchmark tests."""
        from daylib.workset_monitor import WorksetMonitor

        monitor = WorksetMonitor.__new__(WorksetMonitor)
        return monitor

    def test_parse_benchmark_costs_success(self, mock_monitor):
        """Test parsing benchmark TSV with task_cost column."""
        import subprocess

        workset = MagicMock()
        workset.name = "test-ws-001"

        benchmark_content = (
            b"rule\tsample\ttask_cost\truntime\n"
            b"align\tSAMPLE1\t0.1234\t100\n"
            b"variant_call\tSAMPLE1\t0.5678\t200\n"
            b"align\tSAMPLE2\t0.2345\t150\n"
            b"variant_call\tSAMPLE2\t0.4321\t180\n"
        )

        cat_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=benchmark_content, stderr=b""
        )

        mock_monitor._run_headnode_command = MagicMock(return_value=cat_result)

        per_sample_metrics = {}
        result = mock_monitor._parse_benchmark_costs(
            workset,
            "test-cluster",
            "/fsx/analysis/results/day/hg38/other_reports/rules_benchmark_data_mqc.tsv",
            per_sample_metrics,
        )

        assert result is not None
        expected_total = 0.1234 + 0.5678 + 0.2345 + 0.4321
        assert abs(result["total_cost"] - expected_total) < 0.0001
        assert "SAMPLE1" in result["sample_costs"]
        assert abs(result["sample_costs"]["SAMPLE1"] - (0.1234 + 0.5678)) < 0.0001
        assert "SAMPLE2" in result["sample_costs"]
        assert abs(result["sample_costs"]["SAMPLE2"] - (0.2345 + 0.4321)) < 0.0001

        # Check per_sample_metrics was updated
        assert "SAMPLE1" in per_sample_metrics
        assert "compute_cost_usd" in per_sample_metrics["SAMPLE1"]

    def test_parse_benchmark_costs_file_not_found(self, mock_monitor):
        """Test handling when benchmark file doesn't exist."""
        import subprocess

        workset = MagicMock()
        workset.name = "test-ws-002"

        cat_result = subprocess.CompletedProcess(
            args=[], returncode=1, stdout=b"", stderr=b"No such file"
        )

        mock_monitor._run_headnode_command = MagicMock(return_value=cat_result)

        result = mock_monitor._parse_benchmark_costs(
            workset, "test-cluster", "/nonexistent/path.tsv", {}
        )

        assert result is None

    def test_parse_benchmark_costs_no_task_cost_column(self, mock_monitor):
        """Test handling when task_cost column is missing."""
        import subprocess

        workset = MagicMock()
        workset.name = "test-ws-003"

        benchmark_content = (
            b"rule\tsample\truntime\n"
            b"align\tSAMPLE1\t100\n"
        )

        cat_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=benchmark_content, stderr=b""
        )

        mock_monitor._run_headnode_command = MagicMock(return_value=cat_result)

        result = mock_monitor._parse_benchmark_costs(
            workset, "test-cluster", "/fsx/path.tsv", {}
        )

        assert result is None


class TestHeadnodeAnalysisPath:
    """Tests for headnode_analysis_path storage in DynamoDB."""

    def test_record_pipeline_location_stores_path(self):
        """Test that _record_pipeline_location stores headnode_analysis_path in DynamoDB."""
        from daylib.workset_monitor import WorksetMonitor
        from pathlib import Path, PurePosixPath
        import tempfile

        monitor = WorksetMonitor.__new__(WorksetMonitor)
        monitor.state_db = MagicMock()
        monitor._pipeline_locations = {}
        monitor._workset_metrics = {}

        workset = MagicMock()
        workset.name = "test-ws-path-001"

        # Use temp directory for local state
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            monitor._local_state_dir = MagicMock(return_value=state_dir)
            monitor._save_metrics = MagicMock()

            location = PurePosixPath("/fsx/analysis_results/ubuntu/test-ws-path-001/daylily-omics-analysis")
            monitor._record_pipeline_location(workset, location)

            # Verify DynamoDB was called with headnode_analysis_path
            monitor.state_db.update_execution_environment.assert_called_once_with(
                "test-ws-path-001",
                headnode_analysis_path="/fsx/analysis_results/ubuntu/test-ws-path-001/daylily-omics-analysis",
            )

    def test_record_pipeline_location_handles_db_failure(self):
        """Test that _record_pipeline_location continues on DynamoDB failure."""
        from daylib.workset_monitor import WorksetMonitor
        from pathlib import Path, PurePosixPath
        import tempfile

        monitor = WorksetMonitor.__new__(WorksetMonitor)
        monitor.state_db = MagicMock()
        monitor.state_db.update_execution_environment.side_effect = Exception("DB error")
        monitor._pipeline_locations = {}
        monitor._workset_metrics = {}

        workset = MagicMock()
        workset.name = "test-ws-path-002"

        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            monitor._local_state_dir = MagicMock(return_value=state_dir)
            monitor._save_metrics = MagicMock()

            location = PurePosixPath("/fsx/analysis")
            # Should not raise exception
            monitor._record_pipeline_location(workset, location)

            # Local state should still be recorded
            assert workset.name in monitor._pipeline_locations


class TestProgressStepsMetrics:
    """Tests for metrics-related progress steps."""

    def test_progress_step_collecting_metrics_exists(self):
        """Test that COLLECTING_METRICS progress step exists."""
        from daylib.workset_state_db import WorksetProgressStep

        assert hasattr(WorksetProgressStep, "COLLECTING_METRICS")
        assert WorksetProgressStep.COLLECTING_METRICS.value == "collecting_metrics"

    def test_progress_step_metrics_complete_exists(self):
        """Test that METRICS_COMPLETE progress step exists."""
        from daylib.workset_state_db import WorksetProgressStep

        assert hasattr(WorksetProgressStep, "METRICS_COMPLETE")
        assert WorksetProgressStep.METRICS_COMPLETE.value == "metrics_complete"

    def test_progress_step_metrics_failed_exists(self):
        """Test that METRICS_FAILED progress step exists."""
        from daylib.workset_state_db import WorksetProgressStep

        assert hasattr(WorksetProgressStep, "METRICS_FAILED")
        assert WorksetProgressStep.METRICS_FAILED.value == "metrics_failed"


class TestUpdateExecutionEnvironmentHeadnodePath:
    """Tests for headnode_analysis_path in update_execution_environment."""

    def test_update_with_headnode_analysis_path(self, mock_dynamodb):
        """Test that headnode_analysis_path is stored in DynamoDB."""
        from daylib.workset_state_db import WorksetStateDB

        db = WorksetStateDB(
            table_name="test-worksets",
            region="us-west-2",
            profile=None,
        )

        mock_table = mock_dynamodb["table"]
        mock_table.update_item.return_value = {}

        db.update_execution_environment(
            workset_id="test-ws-001",
            headnode_analysis_path="/fsx/analysis_results/ubuntu/test-ws-001/daylily-omics-analysis",
        )

        mock_table.update_item.assert_called_once()
        call_args = mock_table.update_item.call_args

        # Check that headnode_analysis_path is in the update expression
        update_expr = call_args.kwargs["UpdateExpression"]
        expr_values = call_args.kwargs["ExpressionAttributeValues"]

        assert "headnode_analysis_path" in update_expr
        assert ":analysis_path" in expr_values
        assert expr_values[":analysis_path"] == "/fsx/analysis_results/ubuntu/test-ws-001/daylily-omics-analysis"

    def test_update_with_multiple_fields_including_path(self, mock_dynamodb):
        """Test updating multiple fields including headnode_analysis_path."""
        from daylib.workset_state_db import WorksetStateDB

        db = WorksetStateDB(
            table_name="test-worksets",
            region="us-west-2",
            profile=None,
        )

        mock_table = mock_dynamodb["table"]
        mock_table.update_item.return_value = {}

        db.update_execution_environment(
            workset_id="test-ws-002",
            cluster_name="test-cluster",
            headnode_ip="10.0.1.100",
            headnode_analysis_path="/fsx/analysis",
        )

        mock_table.update_item.assert_called_once()
        call_args = mock_table.update_item.call_args
        update_expr = call_args.kwargs["UpdateExpression"]
        expr_values = call_args.kwargs["ExpressionAttributeValues"]

        # All fields should be present
        assert "execution_cluster_name" in update_expr
        assert "execution_headnode_ip" in update_expr
        assert "headnode_analysis_path" in update_expr
        assert ":exec_cluster" in expr_values
        assert ":exec_ip" in expr_values
        assert ":analysis_path" in expr_values


class TestStageReferenceBucket:
    """Tests for _stage_reference_bucket region substitution."""

    @pytest.fixture
    def mock_monitor(self):
        """Create a mock monitor with config for testing."""
        from daylib.workset_monitor import WorksetMonitor

        monitor = WorksetMonitor.__new__(WorksetMonitor)
        monitor.config = MagicMock()
        return monitor

    def test_returns_configured_bucket_without_region(self, mock_monitor):
        """Test that configured bucket is returned as-is when no region specified."""
        mock_monitor.config.pipeline.reference_bucket = "s3://my-bucket/"

        result = mock_monitor._stage_reference_bucket()

        assert result == "s3://my-bucket/"

    def test_adds_trailing_slash(self, mock_monitor):
        """Test that trailing slash is added if missing."""
        mock_monitor.config.pipeline.reference_bucket = "s3://my-bucket"

        result = mock_monitor._stage_reference_bucket()

        assert result == "s3://my-bucket/"

    def test_substitutes_region_placeholder(self, mock_monitor):
        """Test that {region} placeholder is substituted."""
        mock_monitor.config.pipeline.reference_bucket = "s3://bucket-{region}/"

        result = mock_monitor._stage_reference_bucket(region="eu-central-1")

        assert result == "s3://bucket-eu-central-1/"

    def test_substitutes_region_suffix_us_west_2_to_eu_central_1(self, mock_monitor):
        """Test region suffix substitution from us-west-2 to eu-central-1."""
        mock_monitor.config.pipeline.reference_bucket = "s3://lsmc-dayoa-omics-analysis-us-west-2/"

        result = mock_monitor._stage_reference_bucket(region="eu-central-1")

        assert result == "s3://lsmc-dayoa-omics-analysis-eu-central-1/"

    def test_substitutes_region_suffix_eu_central_1_to_us_west_2(self, mock_monitor):
        """Test region suffix substitution from eu-central-1 to us-west-2."""
        mock_monitor.config.pipeline.reference_bucket = "s3://lsmc-dayoa-omics-analysis-eu-central-1/"

        result = mock_monitor._stage_reference_bucket(region="us-west-2")

        assert result == "s3://lsmc-dayoa-omics-analysis-us-west-2/"

    def test_substitutes_region_suffix_us_east_1(self, mock_monitor):
        """Test region suffix substitution to us-east-1."""
        mock_monitor.config.pipeline.reference_bucket = "s3://bucket-us-west-2/"

        result = mock_monitor._stage_reference_bucket(region="us-east-1")

        assert result == "s3://bucket-us-east-1/"

    def test_substitutes_region_suffix_ap_south_1(self, mock_monitor):
        """Test region suffix substitution to ap-south-1."""
        mock_monitor.config.pipeline.reference_bucket = "s3://bucket-us-west-2/"

        result = mock_monitor._stage_reference_bucket(region="ap-south-1")

        assert result == "s3://bucket-ap-south-1/"

    def test_no_substitution_when_region_matches(self, mock_monitor):
        """Test no substitution when target region matches bucket region."""
        mock_monitor.config.pipeline.reference_bucket = "s3://bucket-us-west-2/"

        result = mock_monitor._stage_reference_bucket(region="us-west-2")

        assert result == "s3://bucket-us-west-2/"

    def test_no_substitution_for_non_regional_bucket(self, mock_monitor):
        """Test no substitution for bucket without region suffix."""
        mock_monitor.config.pipeline.reference_bucket = "s3://my-generic-bucket/"

        result = mock_monitor._stage_reference_bucket(region="eu-central-1")

        assert result == "s3://my-generic-bucket/"

    def test_raises_error_when_not_configured(self, mock_monitor):
        """Test MonitorError raised when reference_bucket not configured."""
        from daylib.workset_monitor import MonitorError

        mock_monitor.config.pipeline.reference_bucket = None

        with pytest.raises(MonitorError, match="pipeline.reference_bucket must be configured"):
            mock_monitor._stage_reference_bucket()

    def test_raises_error_for_empty_string(self, mock_monitor):
        """Test MonitorError raised for empty string."""
        from daylib.workset_monitor import MonitorError

        mock_monitor.config.pipeline.reference_bucket = ""

        with pytest.raises(MonitorError, match="pipeline.reference_bucket must be configured"):
            mock_monitor._stage_reference_bucket()


class TestStageSamplesMultiRegion:
    """Tests for _stage_samples multi-region support."""

    @pytest.fixture
    def mock_monitor(self):
        """Create a mock monitor with required attributes for staging."""
        from daylib.workset_monitor import WorksetMonitor
        import tempfile
        from pathlib import Path

        monitor = WorksetMonitor.__new__(WorksetMonitor)
        monitor.config = MagicMock()
        monitor.config.aws.profile = "test-profile"
        monitor.config.aws.region = "us-west-2"
        monitor.config.pipeline.reference_bucket = "s3://bucket-us-west-2/"
        monitor.config.pipeline.stage_command = (
            "./bin/stage --profile {profile} --region {region} "
            "--reference-bucket {reference_bucket} --config-dir {output_dir} {analysis_samples}"
        )
        monitor.config.pipeline.ssh_identity_file = "~/.ssh/key.pem"
        monitor.config.pipeline.ssh_user = "ubuntu"
        monitor.config.pipeline.ssh_extra_args = []

        # Mock methods
        monitor._stage_artifacts = {}
        monitor._relative_manifest_argument = MagicMock(return_value="manifest.tsv")

        # Use temp directory for output
        tmpdir = tempfile.mkdtemp()
        monitor._local_state_dir = MagicMock(return_value=Path(tmpdir))

        return monitor

    def test_uses_cluster_region_in_command(self, mock_monitor):
        """Test that cluster region is used in staging command."""
        mock_monitor._run_monitored_command = MagicMock()
        mock_monitor._run_monitored_command.return_value = MagicMock(
            stdout="Remote FSx stage directory: /fsx/data/staged/remote_stage_123\nStaged files (1):\n  /fsx/data/staged/remote_stage_123/sample.fq.gz",
            stderr="",
        )
        mock_monitor._parse_stage_samples_output = MagicMock()

        workset = MagicMock()
        workset.name = "test-workset"
        manifest_path = MagicMock()

        mock_monitor._stage_samples(workset, manifest_path, "euc1a-jem", region="eu-central-1")

        # Verify the command was called with correct region
        call_args = mock_monitor._run_monitored_command.call_args
        cmd = call_args[0][1]  # Second positional arg is the command string

        assert "--region eu-central-1" in cmd
        assert "--reference-bucket s3://bucket-eu-central-1/" in cmd

    def test_uses_config_region_when_none_provided(self, mock_monitor):
        """Test that config region is used when no region parameter provided."""
        mock_monitor._run_monitored_command = MagicMock()
        mock_monitor._run_monitored_command.return_value = MagicMock(stdout="", stderr="")
        mock_monitor._parse_stage_samples_output = MagicMock()

        workset = MagicMock()
        workset.name = "test-workset"
        manifest_path = MagicMock()

        mock_monitor._stage_samples(workset, manifest_path, "usw2d-B", region=None)

        call_args = mock_monitor._run_monitored_command.call_args
        cmd = call_args[0][1]

        assert "--region us-west-2" in cmd
        assert "--reference-bucket s3://bucket-us-west-2/" in cmd

    def test_different_regions_produce_different_commands(self, mock_monitor):
        """Test that different regions produce different staging commands."""
        mock_monitor._run_monitored_command = MagicMock()
        mock_monitor._run_monitored_command.return_value = MagicMock(stdout="", stderr="")
        mock_monitor._parse_stage_samples_output = MagicMock()

        workset = MagicMock()
        workset.name = "test-workset"
        manifest_path = MagicMock()

        # Stage to us-west-2
        mock_monitor._stage_samples(workset, manifest_path, "usw2d-B", region="us-west-2")
        usw2_cmd = mock_monitor._run_monitored_command.call_args[0][1]

        # Stage to eu-central-1
        mock_monitor._stage_samples(workset, manifest_path, "euc1a-jem", region="eu-central-1")
        euc1_cmd = mock_monitor._run_monitored_command.call_args[0][1]

        # Stage to ap-south-1
        mock_monitor._stage_samples(workset, manifest_path, "aps1-cluster", region="ap-south-1")
        aps1_cmd = mock_monitor._run_monitored_command.call_args[0][1]

        # Verify each has correct region
        assert "--region us-west-2" in usw2_cmd
        assert "--region eu-central-1" in euc1_cmd
        assert "--region ap-south-1" in aps1_cmd

        # Verify each has correct bucket
        assert "bucket-us-west-2" in usw2_cmd
        assert "bucket-eu-central-1" in euc1_cmd
        assert "bucket-ap-south-1" in aps1_cmd
