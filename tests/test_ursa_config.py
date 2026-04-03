from __future__ import annotations

from pathlib import Path

from daylib_ursa import config as settings_config
from daylib_ursa import ursa_config


def test_validate_config_file_accepts_runtime_seed_fields(tmp_path: Path):
    config_path = tmp_path / "ursa-config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "regions:",
                "  - us-west-2",
                "tapdb_config_path: /tmp/tapdb-config.yaml",
                "ursa_internal_api_key: ursa-test-key",
                "",
            ]
        ),
        encoding="utf-8",
    )

    is_valid, errors, warnings = ursa_config.validate_config_file(config_path)

    assert is_valid is True
    assert errors == []
    assert warnings == []


def test_yaml_seed_from_ursa_config_preserves_runtime_seed_fields(monkeypatch):
    monkeypatch.setattr(
        ursa_config,
        "get_ursa_config",
        lambda reload=False: ursa_config.UrsaConfig(
            tapdb_config_path="/tmp/tapdb-config.yaml",
            ursa_internal_api_key="ursa-test-key",
        ),
    )

    seeded = settings_config._yaml_seed_from_ursa_config()

    assert seeded["tapdb_config_path"] == "/tmp/tapdb-config.yaml"
    assert seeded["ursa_internal_api_key"] == "ursa-test-key"
