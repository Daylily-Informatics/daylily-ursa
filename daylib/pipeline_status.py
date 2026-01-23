"""Pipeline status fetcher for Snakemake pipelines running on headnodes.

This module provides SSH-based monitoring of Snakemake pipeline progress,
parsing log files and gathering status information from remote headnodes.
It also supports fetching performance metrics from S3 when headnode is unavailable.
"""

from __future__ import annotations

import logging
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

LOGGER = logging.getLogger("daylib.pipeline_status")

# Regex patterns for parsing Snakemake logs
PROGRESS_PATTERN = re.compile(r"(\d+)\s+of\s+(\d+)\s+steps?\s*\((\d+(?:\.\d+)?)\s*%\)\s*done", re.IGNORECASE)
RULE_PATTERN = re.compile(r"^rule\s+(\w+):", re.MULTILINE)
ERROR_PATTERNS = [
    re.compile(r"error", re.IGNORECASE),
    re.compile(r"failed", re.IGNORECASE),
    re.compile(r"exception", re.IGNORECASE),
    re.compile(r"traceback", re.IGNORECASE),
]

# Default timeout for SSH commands (seconds)
DEFAULT_SSH_TIMEOUT = 5


@dataclass
class PipelineStatus:
    """Status of a Snakemake pipeline running on a headnode."""

    is_running: bool = False
    steps_completed: int = 0
    steps_total: int = 0
    percent_complete: float = 0.0
    current_rule: Optional[str] = None
    duration_seconds: int = 0
    storage_bytes: int = 0
    recent_log_lines: List[str] = field(default_factory=list)
    log_files: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    error_message: Optional[str] = None  # For fetch errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_running": self.is_running,
            "steps_completed": self.steps_completed,
            "steps_total": self.steps_total,
            "percent_complete": self.percent_complete,
            "current_rule": self.current_rule,
            "duration_seconds": self.duration_seconds,
            "storage_bytes": self.storage_bytes,
            "recent_log_lines": self.recent_log_lines,
            "log_files": self.log_files,
            "errors": self.errors,
            "error_message": self.error_message,
        }


class PipelineStatusFetcher:
    """Fetches pipeline status from a remote headnode via SSH."""

    def __init__(
        self,
        ssh_user: str = "ubuntu",
        ssh_identity_file: Optional[str] = None,
        ssh_extra_args: Optional[List[str]] = None,
        timeout: int = DEFAULT_SSH_TIMEOUT,
        clone_dest_root: str = "/fsx/analysis_results/ubuntu",
        repo_dir_name: str = "daylily-omics-analysis",
    ):
        self.ssh_user = ssh_user
        self.ssh_identity_file = ssh_identity_file
        self.ssh_extra_args = ssh_extra_args or []
        self.timeout = timeout
        self.clone_dest_root = clone_dest_root
        self.repo_dir_name = repo_dir_name

    def _ssh_options(self) -> List[str]:
        """Build SSH options list."""
        options = [
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", f"ConnectTimeout={self.timeout}",
        ]
        if self.ssh_extra_args:
            options.extend(self.ssh_extra_args)
        return options

    def _build_ssh_command(
        self, headnode_ip: str, command: str, *, shell: bool = True
    ) -> List[str]:
        """Build SSH command to execute on headnode."""
        ssh_cmd: List[str] = ["ssh"]
        if self.ssh_identity_file:
            identity = str(Path(self.ssh_identity_file).expanduser())
            ssh_cmd.extend(["-i", identity])
        ssh_cmd.extend(self._ssh_options())
        ssh_cmd.append(f"{self.ssh_user}@{headnode_ip}")
        if shell:
            ssh_cmd.append(f"bash -lc {shlex.quote(command)}")
        else:
            ssh_cmd.append(command)
        return ssh_cmd

    def _run_ssh_command(
        self, headnode_ip: str, command: str, *, shell: bool = True
    ) -> Optional[str]:
        """Run SSH command and return stdout, or None on failure."""
        ssh_cmd = self._build_ssh_command(headnode_ip, command, shell=shell)
        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
            if result.returncode == 0:
                return result.stdout.decode("utf-8", errors="ignore")
            return None
        except subprocess.TimeoutExpired:
            LOGGER.warning("SSH command timed out after %ds: %s", self.timeout, command[:100])
            return None
        except Exception as e:
            LOGGER.warning("SSH command failed: %s", str(e))
            return None

    def _get_pipeline_dir(self, workset_name: str) -> str:
        """Get the expected pipeline directory path."""
        return f"{self.clone_dest_root}/{workset_name}/{self.repo_dir_name}"

    def _get_snakemake_log_dir(self, workset_name: str) -> str:
        """Get the Snakemake log directory path."""
        return f"{self._get_pipeline_dir(workset_name)}/.snakemake/log"

    def get_headnode_ip(
        self, cluster_name: str, region: str, profile: Optional[str] = None
    ) -> Optional[str]:
        """Get the headnode IP address for a cluster using pcluster CLI.

        Args:
            cluster_name: Name of the ParallelCluster
            region: AWS region
            profile: Optional AWS profile name (passed via AWS_PROFILE env var)

        Returns:
            Headnode IP address or None if not found
        """
        import json
        import os

        cmd = ["pcluster", "describe-cluster-instances", "--region", region, "-n", cluster_name]

        # pcluster doesn't accept --profile flag; use AWS_PROFILE environment variable
        env = os.environ.copy()
        if profile:
            env["AWS_PROFILE"] = profile

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=30,
                check=False,
                env=env,
            )
            if result.returncode != 0:
                LOGGER.warning(
                    "pcluster describe-cluster-instances failed for %s: %s",
                    cluster_name,
                    result.stderr.decode("utf-8", errors="ignore")[:200],
                )
                return None

            payload = json.loads(result.stdout.decode("utf-8", errors="ignore"))
            instances = payload.get("instances", [])
            for instance in instances:
                if not isinstance(instance, dict):
                    continue
                node_type = instance.get("nodeType") or instance.get("NodeType")
                if isinstance(node_type, str) and node_type.lower() == "headnode":
                    ip = instance.get("publicIpAddress") or instance.get("PublicIpAddress")
                    if isinstance(ip, str) and ip:
                        return ip

            LOGGER.warning("No headnode found in cluster %s", cluster_name)
            return None

        except subprocess.TimeoutExpired:
            LOGGER.warning("pcluster command timed out for cluster %s", cluster_name)
            return None
        except json.JSONDecodeError as e:
            LOGGER.warning("Failed to parse pcluster output for %s: %s", cluster_name, str(e))
            return None
        except Exception as e:
            LOGGER.warning("Failed to get headnode IP for %s: %s", cluster_name, str(e))
            return None

    def check_tmux_session(self, headnode_ip: str, session_name: str) -> bool:
        """Check if a tmux session exists on the headnode."""
        result = self._run_ssh_command(
            headnode_ip, f"tmux has-session -t {shlex.quote(session_name)} 2>/dev/null"
        )
        # tmux has-session returns 0 if session exists
        return result is not None

    def get_log_files(self, headnode_ip: str, workset_name: str) -> List[str]:
        """Get list of Snakemake log files."""
        log_dir = self._get_snakemake_log_dir(workset_name)
        cmd = f"ls -1 {shlex.quote(log_dir)}/*.snakemake.log 2>/dev/null | xargs -n1 basename 2>/dev/null"
        output = self._run_ssh_command(headnode_ip, cmd)
        if not output:
            return []
        return [f.strip() for f in output.strip().split("\n") if f.strip()]

    def get_progress_lines(
        self, headnode_ip: str, workset_name: str
    ) -> Optional[str]:
        """Get all progress lines from Snakemake logs using grep.

        Uses: grep ' steps ' <log_dir>/*.log | sort
        This reliably finds all progress updates across all log files.
        """
        log_dir = self._get_snakemake_log_dir(workset_name)
        cmd = f"grep ' steps ' {shlex.quote(log_dir)}/*.log 2>/dev/null | sort"
        return self._run_ssh_command(headnode_ip, cmd)

    def get_latest_log_content(
        self, headnode_ip: str, workset_name: str, tail_lines: int = 50
    ) -> Optional[str]:
        """Get the tail of the latest Snakemake log file.

        Used for fetching recent log lines, current rule, and errors.
        For progress parsing, use get_progress_lines() instead.
        """
        log_dir = self._get_snakemake_log_dir(workset_name)
        cmd = (
            f"LOG=$(ls -t {shlex.quote(log_dir)}/*.snakemake.log 2>/dev/null | head -1) && "
            f"[ -f \"$LOG\" ] && tail -{tail_lines} \"$LOG\""
        )
        return self._run_ssh_command(headnode_ip, cmd)

    def get_full_log_content(
        self, headnode_ip: str, workset_name: str, log_filename: str
    ) -> Optional[str]:
        """Get the full content of a specific Snakemake log file."""
        log_dir = self._get_snakemake_log_dir(workset_name)
        log_path = f"{log_dir}/{log_filename}"
        cmd = f"cat {shlex.quote(log_path)}"
        return self._run_ssh_command(headnode_ip, cmd)

    def parse_progress(self, log_content: str) -> tuple[int, int, float]:
        """Parse progress from Snakemake log content.

        Returns (steps_completed, steps_total, percent_complete).
        """
        # Find all progress lines and take the last one
        matches = list(PROGRESS_PATTERN.finditer(log_content))
        if matches:
            last_match = matches[-1]
            steps_completed = int(last_match.group(1))
            steps_total = int(last_match.group(2))
            percent_complete = float(last_match.group(3))
            return steps_completed, steps_total, percent_complete
        return 0, 0, 0.0

    def parse_current_rule(self, log_content: str) -> Optional[str]:
        """Parse the most recent rule being executed from log content."""
        matches = list(RULE_PATTERN.finditer(log_content))
        if matches:
            return matches[-1].group(1)
        return None

    def parse_errors(self, log_content: str, max_errors: int = 20) -> List[str]:
        """Extract error lines from log content."""
        errors: List[str] = []
        for line in log_content.split("\n"):
            line = line.strip()
            if not line:
                continue
            for pattern in ERROR_PATTERNS:
                if pattern.search(line):
                    errors.append(line)
                    break
            if len(errors) >= max_errors:
                break
        return errors

    def get_storage_size(self, headnode_ip: str, workset_name: str) -> int:
        """Get the storage size of the analysis directory in bytes."""
        pipeline_dir = self._get_pipeline_dir(workset_name)
        cmd = f"du -sb {shlex.quote(pipeline_dir)} 2>/dev/null | cut -f1"
        output = self._run_ssh_command(headnode_ip, cmd)
        if output:
            try:
                return int(output.strip())
            except ValueError:
                pass
        return 0

    def get_duration(self, headnode_ip: str, workset_name: str) -> int:
        """Calculate duration in seconds from directory creation to completion or now.

        Uses the oldest timestamp in the analysis directory as start time,
        and DYMEMBERS file timestamp (if exists) or current time as end time.
        """
        pipeline_dir = self._get_pipeline_dir(workset_name)
        # Get start time (directory creation time)
        start_cmd = f"stat -c '%Y' {shlex.quote(pipeline_dir)} 2>/dev/null"
        start_output = self._run_ssh_command(headnode_ip, start_cmd)
        if not start_output:
            return 0
        try:
            start_time = int(start_output.strip())
        except ValueError:
            return 0

        # Check for DYMEMBERS file (completion marker)
        dymembers_path = f"{pipeline_dir}/DYMEMBERS"
        end_cmd = f"stat -c '%Y' {shlex.quote(dymembers_path)} 2>/dev/null || date +%s"
        end_output = self._run_ssh_command(headnode_ip, end_cmd)
        if not end_output:
            return 0
        try:
            end_time = int(end_output.strip())
        except ValueError:
            return 0

        return max(0, end_time - start_time)

    def fetch_status(
        self,
        headnode_ip: str,
        workset_name: str,
        tmux_session_name: Optional[str] = None,
    ) -> PipelineStatus:
        """Fetch complete pipeline status from headnode.

        Args:
            headnode_ip: IP address of the headnode
            workset_name: Name of the workset (used to derive directory paths)
            tmux_session_name: Optional tmux session name to check if running

        Returns:
            PipelineStatus with all available information
        """
        status = PipelineStatus()

        try:
            # Check if tmux session is running
            if tmux_session_name:
                status.is_running = self.check_tmux_session(headnode_ip, tmux_session_name)

            # Get log files
            status.log_files = self.get_log_files(headnode_ip, workset_name)

            # Get progress using grep (more reliable than tail)
            progress_lines = self.get_progress_lines(headnode_ip, workset_name)
            if progress_lines:
                completed, total, percent = self.parse_progress(progress_lines)
                status.steps_completed = completed
                status.steps_total = total
                status.percent_complete = percent

            # Get latest log content for other info (current rule, errors, recent lines)
            log_content = self.get_latest_log_content(headnode_ip, workset_name)
            if log_content:
                # Parse current rule
                status.current_rule = self.parse_current_rule(log_content)

                # Get recent log lines
                status.recent_log_lines = [
                    line for line in log_content.strip().split("\n")
                    if line.strip()
                ][-50:]

                # Parse errors
                status.errors = self.parse_errors(log_content)

            # Get storage size
            status.storage_bytes = self.get_storage_size(headnode_ip, workset_name)

            # Get duration
            status.duration_seconds = self.get_duration(headnode_ip, workset_name)

        except Exception as e:
            LOGGER.error("Failed to fetch pipeline status: %s", str(e))
            status.error_message = f"Failed to fetch status: {str(e)}"

        return status

    def _get_metrics_file_path(self, workset_name: str, filename: str) -> str:
        """Build full path to metrics file on headnode.

        Pattern: /fsx/analysis_results/ubuntu/{workset_name}/daylily-omics-analysis/results/day/hg38/other_reports/{filename}
        """
        return f"{self.clone_dest_root}/{workset_name}/{self.repo_dir_name}/results/day/hg38/other_reports/{filename}"

    def fetch_tsv_file(
        self,
        headnode_ip: str,
        workset_name: str,
        filename: str,
        expected_columns: Optional[List[str]] = None,
    ) -> Optional[List[Dict[str, str]]]:
        """Fetch a TSV file from headnode and parse into list of dicts.

        Args:
            headnode_ip: IP address of the headnode
            workset_name: Name of the workset
            filename: Name of the TSV file (e.g., 'alignstats_combo_mqc.tsv')
            expected_columns: Expected column names to identify the header line
                             (handles shell init noise in SSH output)

        Returns:
            List of dicts (one per row) or None if file not found/error
        """
        file_path = self._get_metrics_file_path(workset_name, filename)
        cmd = f"cat {shlex.quote(file_path)} 2>/dev/null"
        content = self._run_ssh_command(headnode_ip, cmd)

        if not content or not content.strip():
            LOGGER.debug("TSV file not found or empty: %s", file_path)
            return None

        return self._parse_tsv(content, expected_columns)

    # Expected column names for different TSV file types
    # Used to find the real header line (skipping shell init noise)
    ALIGNSTATS_EXPECTED_COLS = ["sample", "YieldReads", "WgsCoverageMean", "MappedReadsPct"]
    BENCHMARK_EXPECTED_COLS = ["sample", "task_cost", "rule", "s"]  # "s" is common time column

    def _parse_tsv(
        self, content: str, expected_columns: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """Parse TSV content into list of dicts.

        Args:
            content: Raw TSV content (may include preamble noise from shell init)
            expected_columns: If provided, find header line by looking for these columns.
                             This handles cases where SSH output includes shell init
                             messages before the actual TSV content.

        Returns:
            List of dicts (one per row), empty list if parsing fails
        """
        lines = content.strip().split("\n")
        if len(lines) < 2:  # Need at least header + 1 data row
            return []

        # Find header line - scan for first line containing expected columns
        header_idx = 0
        if expected_columns:
            for i, line in enumerate(lines):
                cols = line.split("\t")
                # Check if any expected column is present in this line
                if any(exp_col in cols for exp_col in expected_columns):
                    header_idx = i
                    break
            else:
                # No line matched expected columns - fall back to first line
                LOGGER.debug(
                    "No line matched expected columns %s, using first line as header",
                    expected_columns,
                )

        if header_idx >= len(lines) - 1:  # Need at least header + 1 data row after header
            return []

        headers = lines[header_idx].split("\t")
        rows = []
        for line in lines[header_idx + 1:]:
            if not line.strip():
                continue
            values = line.split("\t")
            # Pad values if row has fewer columns than header
            while len(values) < len(headers):
                values.append("")
            row = {h: v for h, v in zip(headers, values)}
            rows.append(row)
        return rows

    def fetch_alignment_stats(
        self, headnode_ip: str, workset_name: str
    ) -> Optional[List[Dict[str, str]]]:
        """Fetch alignment statistics from alignstats_combo_mqc.tsv.

        Returns list of per-sample alignment metrics.
        """
        return self.fetch_tsv_file(
            headnode_ip, workset_name, "alignstats_combo_mqc.tsv", self.ALIGNSTATS_EXPECTED_COLS
        )

    def fetch_benchmark_data(
        self, headnode_ip: str, workset_name: str
    ) -> Optional[List[Dict[str, str]]]:
        """Fetch rules benchmark data from rules_benchmark_data_singleton.tsv.

        Falls back to rules_benchmark_data_mqc.tsv if singleton file not found.
        Returns list of per-rule performance metrics.
        """
        # Try singleton file first
        result = self.fetch_tsv_file(
            headnode_ip, workset_name, "rules_benchmark_data_singleton.tsv", self.BENCHMARK_EXPECTED_COLS
        )
        if result:
            return result
        # Fall back to mqc file
        return self.fetch_tsv_file(
            headnode_ip, workset_name, "rules_benchmark_data_mqc.tsv", self.BENCHMARK_EXPECTED_COLS
        )

    def fetch_performance_metrics(
        self, headnode_ip: str, workset_name: str
    ) -> Dict[str, Any]:
        """Fetch all performance metrics for a workset.

        Returns dict with:
            - alignment_stats: List of per-sample alignment metrics
            - benchmark_data: List of per-rule benchmark data
            - cost_summary: Dict with per-sample and total costs
        """
        result: Dict[str, Any] = {
            "alignment_stats": None,
            "benchmark_data": None,
            "cost_summary": None,
        }

        # Fetch alignment stats
        align_stats = self.fetch_alignment_stats(headnode_ip, workset_name)
        if align_stats:
            result["alignment_stats"] = align_stats

        # Fetch benchmark data
        benchmark_data = self.fetch_benchmark_data(headnode_ip, workset_name)
        if benchmark_data:
            result["benchmark_data"] = benchmark_data
            result["cost_summary"] = self._compute_cost_summary(benchmark_data)

        return result

    def _compute_cost_summary(
        self, benchmark_data: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Compute cost summary from benchmark data.

        Returns:
            Dict with:
                - per_sample_costs: Dict[sample_name, cost]
                - total_cost: Total cost for all rules
                - rule_count: Number of rules
                - sample_count: Number of unique samples (excludes pseudo-samples like 'all.')
        """
        # Pseudo-samples that are aggregate rules, not real samples
        PSEUDO_SAMPLES = {"all.", "all", "unknown"}

        per_sample: Dict[str, float] = {}
        total_cost = 0.0

        for row in benchmark_data:
            try:
                cost = float(row.get("task_cost", "0") or "0")
                sample = row.get("sample", "unknown")

                total_cost += cost
                per_sample[sample] = per_sample.get(sample, 0.0) + cost
            except (ValueError, TypeError):
                continue

        # Filter out pseudo-samples for the per-sample cost display
        real_sample_costs = {k: v for k, v in per_sample.items() if k not in PSEUDO_SAMPLES}

        return {
            "per_sample_costs": real_sample_costs,
            "total_cost": round(total_cost, 4),
            "rule_count": len(benchmark_data),
            "sample_count": len(real_sample_costs),
        }

    # ==========================================================================
    # S3 Fallback Methods - Used when headnode is unavailable
    # ==========================================================================

    def _get_s3_metrics_path(self, results_s3_uri: str, filename: str) -> str:
        """Build S3 path to metrics file from results_s3_uri.

        Args:
            results_s3_uri: Base S3 URI where results were exported
                           (e.g., s3://bucket/prefix/workset-name)
            filename: Name of the TSV file (e.g., 'alignstats_combo_mqc.tsv')

        Returns:
            Full S3 path to the metrics file
        """
        # The results_s3_uri points to the pipeline directory root
        # Metrics files are in: {results_s3_uri}/daylily-omics-analysis/results/day/hg38/other_reports/
        base_uri = results_s3_uri.rstrip("/")
        return f"{base_uri}/{self.repo_dir_name}/results/day/hg38/other_reports/{filename}"

    def fetch_tsv_from_s3(
        self,
        results_s3_uri: str,
        filename: str,
        region: Optional[str] = None,
        expected_columns: Optional[List[str]] = None,
    ) -> Optional[List[Dict[str, str]]]:
        """Fetch a TSV file from S3 and parse into list of dicts.

        Args:
            results_s3_uri: Base S3 URI where results were exported
            filename: Name of the TSV file (e.g., 'alignstats_combo_mqc.tsv')
            region: AWS region for S3 client
            expected_columns: Expected column names to identify the header line

        Returns:
            List of dicts (one per row) or None if file not found/error
        """
        s3_path = self._get_s3_metrics_path(results_s3_uri, filename)
        parsed = urlparse(s3_path)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        LOGGER.debug("Fetching TSV from S3: s3://%s/%s", bucket, key)

        try:
            s3_client = boto3.client("s3", region_name=region) if region else boto3.client("s3")
            response = s3_client.get_object(Bucket=bucket, Key=key)
            content = response["Body"].read().decode("utf-8")

            if not content or not content.strip():
                LOGGER.debug("S3 TSV file is empty: %s", s3_path)
                return None

            return self._parse_tsv(content, expected_columns)

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("NoSuchKey", "404"):
                LOGGER.debug("S3 TSV file not found: %s", s3_path)
            else:
                LOGGER.warning("Failed to fetch TSV from S3 %s: %s", s3_path, e)
            return None
        except Exception as e:
            LOGGER.warning("Error fetching TSV from S3 %s: %s", s3_path, e)
            return None

    def fetch_alignment_stats_from_s3(
        self, results_s3_uri: str, region: Optional[str] = None
    ) -> Optional[List[Dict[str, str]]]:
        """Fetch alignment statistics from S3.

        Args:
            results_s3_uri: Base S3 URI where results were exported
            region: AWS region for S3 client

        Returns:
            List of per-sample alignment metrics or None
        """
        return self.fetch_tsv_from_s3(
            results_s3_uri, "alignstats_combo_mqc.tsv", region, self.ALIGNSTATS_EXPECTED_COLS
        )

    def fetch_benchmark_data_from_s3(
        self, results_s3_uri: str, region: Optional[str] = None
    ) -> Optional[List[Dict[str, str]]]:
        """Fetch rules benchmark data from S3.

        Tries rules_benchmark_data_singleton.tsv first, then falls back to
        rules_benchmark_data_mqc.tsv if not found.

        Args:
            results_s3_uri: Base S3 URI where results were exported
            region: AWS region for S3 client

        Returns:
            List of per-rule performance metrics or None
        """
        # Try singleton file first
        result = self.fetch_tsv_from_s3(
            results_s3_uri, "rules_benchmark_data_singleton.tsv", region, self.BENCHMARK_EXPECTED_COLS
        )
        if result:
            return result
        # Fall back to mqc file
        return self.fetch_tsv_from_s3(
            results_s3_uri, "rules_benchmark_data_mqc.tsv", region, self.BENCHMARK_EXPECTED_COLS
        )

    def fetch_duration_from_s3(
        self, results_s3_uri: str, region: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Parse pipeline duration from Snakemake logs in S3.

        Extracts start time from log filename and end time from last timestamp
        in the log content.

        Args:
            results_s3_uri: Base S3 URI where results were exported
            region: AWS region for S3 client

        Returns:
            Dict with duration_seconds, start_time, end_time, or None if unavailable
        """
        import datetime as dt

        # Build path to .snakemake/log/ directory
        parsed = urlparse(results_s3_uri)
        bucket = parsed.netloc
        base_prefix = parsed.path.lstrip("/")
        if not base_prefix.endswith("/"):
            base_prefix = base_prefix + "/"
        log_prefix = f"{base_prefix}daylily-omics-analysis/.snakemake/log/"

        try:
            s3_client = boto3.client("s3", region_name=region) if region else boto3.client("s3")

            # List log files
            resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=log_prefix, MaxKeys=100)
            log_files = [
                obj for obj in resp.get("Contents", [])
                if obj["Key"].endswith(".snakemake.log")
            ]

            if not log_files:
                LOGGER.debug("No snakemake log files found at %s", log_prefix)
                return None

            # Get the most recent log file (by LastModified)
            log_files.sort(key=lambda x: x["LastModified"], reverse=True)
            latest_log = log_files[0]
            log_key = latest_log["Key"]
            log_filename = log_key.split("/")[-1]

            # Parse start time from filename: 2026-01-21T222820.936299.snakemake.log
            # Format: YYYY-MM-DDTHHMMSS.microseconds.snakemake.log
            start_time = None
            match = re.match(r"(\d{4}-\d{2}-\d{2})T(\d{2})(\d{2})(\d{2})\.(\d+)\.snakemake\.log", log_filename)
            if match:
                date_str = match.group(1)
                hour, minute, second = match.group(2), match.group(3), match.group(4)
                try:
                    start_time = dt.datetime.strptime(
                        f"{date_str} {hour}:{minute}:{second}", "%Y-%m-%d %H:%M:%S"
                    ).replace(tzinfo=dt.timezone.utc)
                except ValueError:
                    pass

            if not start_time:
                LOGGER.debug("Could not parse start time from log filename: %s", log_filename)
                return None

            # Fetch log content to find end time
            resp = s3_client.get_object(Bucket=bucket, Key=log_key)
            content = resp["Body"].read().decode("utf-8", errors="ignore")

            # Parse timestamps from log content - format: [Wed Jan 21 23:09:06 2026]
            timestamp_pattern = re.compile(r"\[(\w{3} \w{3} \d{1,2} \d{2}:\d{2}:\d{2} \d{4})\]")
            timestamps = timestamp_pattern.findall(content)

            end_time = None
            if timestamps:
                # Get the last timestamp
                last_ts = timestamps[-1]
                try:
                    end_time = dt.datetime.strptime(last_ts, "%a %b %d %H:%M:%S %Y").replace(tzinfo=dt.timezone.utc)
                except ValueError:
                    pass

            if not end_time:
                # Fall back to S3 object LastModified
                end_time = latest_log["LastModified"]

            duration_seconds = int((end_time - start_time).total_seconds())
            if duration_seconds < 0:
                duration_seconds = 0

            return {
                "duration_seconds": duration_seconds,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "log_file": log_filename,
            }

        except ClientError as e:
            LOGGER.debug("Failed to fetch duration from S3: %s", e)
            return None
        except Exception as e:
            LOGGER.warning("Error parsing duration from S3 logs: %s", e)
            return None

    def fetch_performance_metrics_from_s3(
        self, results_s3_uri: str, region: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fetch all performance metrics from S3.

        This is the S3 fallback for fetch_performance_metrics() when
        the headnode is unavailable.

        Args:
            results_s3_uri: Base S3 URI where results were exported
            region: AWS region for S3 client

        Returns:
            Dict with alignment_stats, benchmark_data, cost_summary, duration_info
        """
        result: Dict[str, Any] = {
            "alignment_stats": None,
            "benchmark_data": None,
            "cost_summary": None,
            "duration_info": None,
        }

        # Fetch alignment stats
        align_stats = self.fetch_alignment_stats_from_s3(results_s3_uri, region)
        if align_stats:
            result["alignment_stats"] = align_stats

        # Fetch benchmark data
        benchmark_data = self.fetch_benchmark_data_from_s3(results_s3_uri, region)
        if benchmark_data:
            result["benchmark_data"] = benchmark_data
            result["cost_summary"] = self._compute_cost_summary(benchmark_data)

        # Fetch duration from Snakemake logs
        duration_info = self.fetch_duration_from_s3(results_s3_uri, region)
        if duration_info:
            result["duration_info"] = duration_info

        return result

