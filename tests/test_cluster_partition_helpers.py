from __future__ import annotations

from importlib import metadata as importlib_metadata
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from daylib_ursa.config import Settings
from daylib_ursa.ephemeral_cluster import runner
from daylib_ursa.workset_api import (
    ClusterPartitionRequest,
    ClusterPartitionPricingRequest,
    build_cluster_partition_pricing,
    build_cluster_partition_verification,
    collect_daylily_cluster_pricing_snapshot,
    load_daylily_partition_instance_types,
    resolve_cluster_partition_selection,
    resolve_daylily_cluster_config_path,
    run_cluster_partition_verification,
)


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "aws_profile": "lsmc",
        "cors_origins": "*",
        "ursa_internal_api_key": "ursa-test-key",
        "session_secret_key": "ursa-session-secret",
        "cognito_app_client_secret": "ursa-cognito-secret",
        "bloom_base_url": "https://bloom.example",
        "atlas_base_url": "https://atlas.example",
        "ursa_internal_output_bucket": "ursa-internal",
        "ursa_tapdb_mount_enabled": False,
        "deployment_name": "inflec3",
        "day_aws_region": "us-west-2",
        "ui_show_environment_chrome": True,
    }
    values.update(overrides)
    return Settings(**values)


def test_require_daylily_ec_version_accepts_required_version(monkeypatch) -> None:
    monkeypatch.setattr(runner.importlib_metadata, "version", lambda _name: "2.0.2")

    assert runner.require_daylily_ec_version() == "2.0.2"


def test_require_daylily_ec_version_rejects_missing_distribution(monkeypatch) -> None:
    def _missing(_name: str) -> str:
        raise importlib_metadata.PackageNotFoundError

    monkeypatch.setattr(runner.importlib_metadata, "version", _missing)

    with pytest.raises(RuntimeError, match="daylily-ephemeral-cluster is not installed"):
        runner.require_daylily_ec_version()


def test_require_daylily_ec_version_rejects_mismatched_distribution(monkeypatch) -> None:
    monkeypatch.setattr(runner.importlib_metadata, "version", lambda _name: "1.9.9")

    with pytest.raises(RuntimeError, match="expected 2.0.2, found 1.9.9"):
        runner.require_daylily_ec_version()


def test_write_generated_ec_config_expands_required_triplets(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "require_daylily_ec_version", lambda: "2.0.2")
    config_path = runner.write_generated_ec_config(
        dest=tmp_path / "cluster.yaml",
        cluster_name="cluster-1",
        ssh_key_name="omics-key",
        s3_bucket_name="omics-bucket",
        contact_email="ops@example.com",
    )

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = payload["ephemeral_cluster"]["config"]

    assert config["cluster_name"] == ["USESETVALUE", "", "cluster-1"]
    assert config["ssh_key_name"] == ["USESETVALUE", "", "omics-key"]
    assert config["s3_bucket_name"] == ["USESETVALUE", "", "omics-bucket"]
    assert config["budget_email"] == ["USESETVALUE", "", "ops@example.com"]
    assert config["cluster_template_yaml"] == ["PROMPTUSER", "", ""]
    assert config["public_subnet_id"] == ["PROMPTUSER", "", ""]
    assert config["private_subnet_id"] == ["PROMPTUSER", "", ""]
    assert config["spot_instance_allocation_strategy"] == ["PROMPTUSER", "", ""]


def test_cluster_partition_request_validates_region_and_az_pairing() -> None:
    request = ClusterPartitionRequest(region="us-west-2", region_az="us-west-2b")

    assert request.region == "us-west-2"
    assert request.region_az == "us-west-2b"

    with pytest.raises(ValidationError, match="region_az must identify an availability zone"):
        ClusterPartitionRequest(region="us-west-2", region_az="us-east-1a")


def test_cluster_partition_pricing_request_requires_region() -> None:
    request = ClusterPartitionPricingRequest(region=" us-west-2 ")

    assert request.region == "us-west-2"

    with pytest.raises(ValidationError, match="region is required"):
        ClusterPartitionPricingRequest(region=" ")


def test_resolve_cluster_partition_selection_prefers_explicit_region() -> None:
    selection = resolve_cluster_partition_selection(
        region="us-east-2",
        region_az="us-east-2b",
    )

    assert selection.region == "us-east-2"
    assert selection.region_az == "us-east-2b"


def test_resolve_cluster_partition_selection_infers_region_from_region_az() -> None:
    selection = resolve_cluster_partition_selection(
        region="",
        region_az="eu-central-1a",
    )

    assert selection.region == "eu-central-1"
    assert selection.region_az == "eu-central-1a"


def test_resolve_daylily_cluster_config_path_prefers_settings_override(monkeypatch) -> None:
    captured: dict[str, str | None] = {}

    def _resolve_cluster_config_path(configured: str | None) -> str:
        captured["configured"] = configured
        return configured or "/pkg/default_cluster.yaml"

    monkeypatch.setattr(
        "daylib_ursa.workset_api._load_daylily_ec_pricing_helpers",
        lambda: (None, None, _resolve_cluster_config_path),
    )

    resolved = resolve_daylily_cluster_config_path(
        _settings(ursa_cost_monitor_config_path="/tmp/custom_cluster.yaml")
    )

    assert resolved == Path("/tmp/custom_cluster.yaml")
    assert captured == {"configured": "/tmp/custom_cluster.yaml"}


def test_load_daylily_partition_instance_types_uses_effective_cluster_yaml(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cluster_config_path = tmp_path / "prod_cluster.yaml"
    cluster_config_path.write_text(
        yaml.safe_dump(
            {
                "Scheduling": {
                    "SlurmQueues": [
                        {"Name": "i8"},
                        {"QueueName": "gpu"},
                    ]
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def _load_partition_instance_types(
        *, cluster_config_path: str, partitions: list[str]
    ) -> dict[str, list[str]]:
        captured["cluster_config_path"] = cluster_config_path
        captured["partitions"] = partitions
        return {
            "i8": ["m7i.2xlarge", "c7i.2xlarge", "c7i.2xlarge"],
            "gpu": ["g6.12xlarge"],
        }

    monkeypatch.setattr(
        "daylib_ursa.workset_api.resolve_daylily_cluster_config_path",
        lambda _settings: cluster_config_path,
    )
    monkeypatch.setattr(
        "daylib_ursa.workset_api._load_daylily_ec_pricing_helpers",
        lambda: (None, _load_partition_instance_types, None),
    )

    resolved_path, partition_map = load_daylily_partition_instance_types(_settings())

    assert resolved_path == cluster_config_path
    assert captured == {
        "cluster_config_path": str(cluster_config_path),
        "partitions": ["i8", "gpu"],
    }
    assert partition_map == {
        "i8": ["c7i.2xlarge", "m7i.2xlarge"],
        "gpu": ["g6.12xlarge"],
    }


def test_collect_daylily_cluster_pricing_snapshot_uses_profile_region_and_partitions(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _Snapshot:
        def to_dict(self) -> dict[str, object]:
            return {
                "captured_at": "2026-04-15T18:00:00Z",
                "points": [],
            }

    def _collect_pricing_snapshot(
        *,
        regions: list[str],
        partitions: list[str],
        cluster_config_path: str,
        profile: str | None,
    ) -> _Snapshot:
        captured["regions"] = regions
        captured["partitions"] = partitions
        captured["cluster_config_path"] = cluster_config_path
        captured["profile"] = profile
        return _Snapshot()

    monkeypatch.setattr(
        "daylib_ursa.workset_api._load_daylily_ec_pricing_helpers",
        lambda: (_collect_pricing_snapshot, None, None),
    )
    monkeypatch.setattr(
        "daylib_ursa.workset_api.resolve_daylily_cluster_config_path",
        lambda _settings: Path("/tmp/prod_cluster.yaml"),
    )

    snapshot = collect_daylily_cluster_pricing_snapshot(
        _settings(aws_profile="lsmc"),
        region="us-west-2",
        partitions=["i8", "gpu"],
    )

    assert snapshot == {
        "captured_at": "2026-04-15T18:00:00Z",
        "points": [],
    }
    assert captured == {
        "regions": ["us-west-2"],
        "partitions": ["i8", "gpu"],
        "cluster_config_path": "/tmp/prod_cluster.yaml",
        "profile": "lsmc",
    }


def test_build_cluster_partition_verification_reports_pass_warn_and_fail() -> None:
    response = build_cluster_partition_verification(
        region="us-west-2",
        region_az="us-west-2a",
        cluster_config_path=Path("/tmp/prod_cluster.yaml"),
        partition_instances={
            "i8": ["c7i.2xlarge"],
            "i192": ["c7i.48xlarge", "m7i.48xlarge"],
            "gpu": ["g6.12xlarge"],
        },
        snapshot={
            "captured_at": "2026-04-15T18:00:00Z",
            "points": [
                {
                    "region": "us-west-2",
                    "availability_zone": "us-west-2a",
                    "partition": "i8",
                    "instance_type": "c7i.2xlarge",
                    "hourly_spot_price": 0.42,
                },
                {
                    "region": "us-west-2",
                    "availability_zone": "us-west-2a",
                    "partition": "i192",
                    "instance_type": "c7i.48xlarge",
                    "hourly_spot_price": 8.4,
                },
            ],
        },
    )

    assert response.has_failures is True
    assert [item.status for item in response.partitions] == ["PASS", "WARN", "FAIL"]
    assert response.partitions[0].summary.startswith("All 1 configured instance types")
    assert response.partitions[1].missing_instance_types == ["m7i.48xlarge"]
    assert response.partitions[2].spot_available_instance_types == []


def test_run_cluster_partition_verification_uses_effective_config_and_snapshot(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "daylib_ursa.workset_api.load_daylily_partition_instance_types",
        lambda _settings: (
            Path("/tmp/prod_cluster.yaml"),
            {"i8": ["c7i.2xlarge"]},
        ),
    )
    monkeypatch.setattr(
        "daylib_ursa.workset_api.collect_daylily_cluster_pricing_snapshot",
        lambda _settings, *, region, partitions: {
            "captured_at": "2026-04-15T18:00:00Z",
            "points": [
                {
                    "region": region,
                    "availability_zone": "us-west-2a",
                    "partition": "i8",
                    "instance_type": "c7i.2xlarge",
                    "hourly_spot_price": 0.42,
                }
            ],
        },
    )

    response = run_cluster_partition_verification(
        _settings(),
        region="us-west-2",
        region_az="us-west-2a",
    )

    assert response.region == "us-west-2"
    assert response.region_az == "us-west-2a"
    assert response.has_failures is False
    assert response.partitions[0].status == "PASS"


def test_build_cluster_partition_pricing_uses_hourly_spot_price_statistics() -> None:
    response = build_cluster_partition_pricing(
        region="us-west-2",
        cluster_config_path=Path("/tmp/prod_cluster.yaml"),
        partition_instances={
            "i8": ["c7i.2xlarge"],
            "i192": ["c7i.large", "c7i.xlarge", "c7i.2xlarge", "c7i.4xlarge"],
            "gpu": ["g6.12xlarge"],
        },
        snapshot={
            "captured_at": "2026-04-15T18:00:00Z",
            "points": [
                {
                    "region": "us-west-2",
                    "availability_zone": "us-west-2a",
                    "partition": "i8",
                    "instance_type": "c7i.2xlarge",
                    "hourly_spot_price": 0.42,
                },
                {
                    "region": "us-west-2",
                    "availability_zone": "us-west-2a",
                    "partition": "i192",
                    "instance_type": "c7i.large",
                    "hourly_spot_price": 1.0,
                },
                {
                    "region": "us-west-2",
                    "availability_zone": "us-west-2a",
                    "partition": "i192",
                    "instance_type": "c7i.xlarge",
                    "hourly_spot_price": 2.0,
                },
                {
                    "region": "us-west-2",
                    "availability_zone": "us-west-2a",
                    "partition": "i192",
                    "instance_type": "c7i.2xlarge",
                    "hourly_spot_price": 2.0,
                },
                {
                    "region": "us-west-2",
                    "availability_zone": "us-west-2b",
                    "partition": "i192",
                    "instance_type": "c7i.2xlarge",
                    "hourly_spot_price": 4.0,
                },
                {
                    "region": "us-west-2",
                    "availability_zone": "us-west-2b",
                    "partition": "i192",
                    "instance_type": "c7i.4xlarge",
                    "hourly_spot_price": 8.0,
                },
            ],
        },
    )

    assert response.captured_at == "2026-04-15T18:00:00Z"
    assert response.availability_zones == ["us-west-2a", "us-west-2b"]
    assert [item.partition for item in response.partitions] == ["i8", "i192", "gpu"]
    assert [item.availability_zone for item in response.partitions[0].availability_zones] == [
        "us-west-2a",
        "us-west-2b",
    ]
    assert response.partitions[0].availability_zones[0].mean == 0.42
    assert response.partitions[0].availability_zones[1].count == 0
    assert response.partitions[0].availability_zones[1].mean is None
    assert response.partitions[1].availability_zones[0].count == 3
    assert response.partitions[1].availability_zones[0].min == 1.0
    assert response.partitions[1].availability_zones[0].q1 == 1.5
    assert response.partitions[1].availability_zones[0].median == 2.0
    assert response.partitions[1].availability_zones[0].mean == 1.66666667
    assert response.partitions[1].availability_zones[0].q3 == 2.0
    assert response.partitions[1].availability_zones[0].max == 2.0
    assert response.partitions[1].availability_zones[1].count == 2
    assert response.partitions[1].availability_zones[1].median == 6.0
    assert response.partitions[1].availability_zones[1].mean == 6.0
    assert [
        point.instance_type for point in response.partitions[1].availability_zones[0].points
    ] == [
        "c7i.2xlarge",
        "c7i.large",
        "c7i.xlarge",
    ]
    assert [
        point.instance_type for point in response.partitions[1].availability_zones[1].points
    ] == [
        "c7i.2xlarge",
        "c7i.4xlarge",
    ]
    assert response.partitions[2].availability_zones[0].count == 0
    assert response.partitions[2].availability_zones[1].mean is None
