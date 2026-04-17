from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from daylib_ursa import config as config_module
from daylib_ursa.ursa_config import parse_regions_csv, update_config_regions


def test_parse_regions_csv_deduplicates_and_requires_values() -> None:
    assert parse_regions_csv(" us-west-2,us-east-1,us-west-2 , eu-central-1 ") == [
        "us-west-2",
        "us-east-1",
        "eu-central-1",
    ]

    with pytest.raises(ValueError, match="At least one AWS region is required"):
        parse_regions_csv(" , , ")


def test_update_config_regions_preserves_existing_region_options(tmp_path: Path) -> None:
    config_path = tmp_path / "ursa-config-test.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "aws_profile": "lsmc",
                "regions": [
                    {"us-west-2": {"ssh_pem": "~/.ssh/lsmc-us-west-2.pem"}},
                    {"us-east-1": {"ssh_pem": "~/.ssh/lsmc-us-east-1.pem"}},
                    "ap-south-1",
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    updated = update_config_regions(
        regions=["us-east-1", "eu-central-1", "us-west-2"],
        config_path=config_path,
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert payload["regions"] == [
        {"us-east-1": {"ssh_pem": "~/.ssh/lsmc-us-east-1.pem"}},
        "eu-central-1",
        {"us-west-2": {"ssh_pem": "~/.ssh/lsmc-us-west-2.pem"}},
    ]
    assert updated.get_allowed_regions() == ["us-east-1", "eu-central-1", "us-west-2"]


def test_yaml_seed_from_ursa_config_includes_allowed_regions(monkeypatch) -> None:
    config = SimpleNamespace(
        aws_profile="lsmc",
        get_allowed_regions=lambda: ["us-west-2", "us-east-1"],
        ursa_portal_default_customer_id=None,
        cognito_group_role_map=None,
        whitelist_domains=None,
        session_secret_key="ursa-session-secret",
        default_tenant_id="00000000-0000-0000-0000-000000000000",
        auto_provision_allowed_domains=["lsmc.com"],
        ursa_internal_output_bucket="ursa-internal",
        tapdb_client_id="local",
        tapdb_database_name="ursa",
        tapdb_env="dev",
        tapdb_config_path="/tmp/ursa-tapdb.yaml",
        cognito_user_pool_id=None,
        cognito_app_client_id=None,
        cognito_app_client_secret=None,
        cognito_domain=None,
        cognito_region=None,
        cognito_callback_url=None,
        cognito_logout_url=None,
        api_host="0.0.0.0",
        api_port=8913,
        bloom_base_url="https://bloom.example",
        bloom_verify_ssl=True,
        atlas_base_url="https://atlas.example",
        atlas_verify_ssl=True,
        dewey_enabled=False,
        dewey_base_url=None,
        dewey_api_token=None,
        dewey_verify_ssl=True,
        ursa_internal_api_key="ursa-internal-key",
        deployment_name="inflec3",
        deployment_color="#7521ca",
        deployment_is_production=False,
        ui_show_environment_chrome=True,
    )
    monkeypatch.setattr("daylib_ursa.ursa_config.get_ursa_config", lambda: config)

    seeded = config_module._yaml_seed_from_ursa_config()

    assert seeded["ursa_allowed_regions"] == "us-west-2,us-east-1"
    assert seeded["aws_profile"] == "lsmc"
    assert seeded["tapdb_config_path"] == "/tmp/ursa-tapdb.yaml"
    assert seeded["ursa_internal_api_key"] == "ursa-internal-key"
