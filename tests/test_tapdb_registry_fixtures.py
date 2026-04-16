from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from daylily_tapdb import resolve_seed_config_dirs, validate_template_configs

import daylib_ursa.tapdb_templates as tapdb_templates
from daylib_ursa.config import get_settings_for_testing
from daylib_ursa.tapdb_templates import (
    claim_ursa_template_prefixes,
    seed_ursa_templates,
    template_config_root,
)


def _fixture_root() -> Path:
    return Path(__file__).resolve().parents[1] / "etc"


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_ursa_registry_fixtures_match_template_pack() -> None:
    templates, issues = validate_template_configs(
        resolve_seed_config_dirs(template_config_root()),
        strict=True,
    )

    domain_payload = _load_json(_fixture_root() / "domain_code_registry.json")
    prefix_payload = _load_json(_fixture_root() / "prefix_ownership_registry.json")

    assert domain_payload["domains"]["Z"]["name"] == "localhost"
    template_prefixes = {
        str(template["instance_prefix"])
        for template in templates
        if str(template["instance_prefix"]) not in {"MSG", "SYS"}
    }
    claimed_prefixes = set(prefix_payload["ownership"]["Z"])
    assert claimed_prefixes == template_prefixes
    assert {
        str(claim["issuer_app_code"]) for claim in prefix_payload["ownership"]["Z"].values()
    } == {"ursa"}


def test_ursa_settings_accept_explicit_registry_paths() -> None:
    settings = get_settings_for_testing(
        ursa_internal_output_bucket="bucket",
        tapdb_domain_registry_path="/tmp/domain_code_registry.json",
        tapdb_prefix_ownership_registry_path="/tmp/prefix_ownership_registry.json",
    )

    assert settings.tapdb_domain_registry_path == "/tmp/domain_code_registry.json"
    assert settings.tapdb_prefix_ownership_registry_path == "/tmp/prefix_ownership_registry.json"


def test_ursa_claim_helper_rejects_prefix_collisions(tmp_path: Path) -> None:
    domain_registry = tmp_path / "domain_code_registry.json"
    prefix_registry = tmp_path / "prefix_ownership_registry.json"
    domain_registry.write_text(
        json.dumps({"version": "0.4.0", "domains": {"Z": {"name": "localhost"}}}) + "\n",
        encoding="utf-8",
    )
    prefix_registry.write_text(
        json.dumps(
            {
                "version": "0.4.0",
                "ownership": {"Z": {"RGX": {"issuer_app_code": "other-repo"}}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="claimed by 'other-repo'"):
        claim_ursa_template_prefixes(
            [{"instance_prefix": "RGX"}],
            domain_code="Z",
            owner_repo_name="ursa",
            domain_registry_path=domain_registry,
            prefix_registry_path=prefix_registry,
        )


def test_ursa_seed_prefers_explicit_registry_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}
    explicit_domain_registry = tmp_path / "domain_code_registry.json"
    explicit_prefix_registry = tmp_path / "prefix_ownership_registry.json"
    explicit_domain_registry.write_text(
        json.dumps({"version": "0.4.0", "domains": {"Z": {"name": "localhost"}}}) + "\n",
        encoding="utf-8",
    )
    explicit_prefix_registry.write_text(
        json.dumps({"version": "0.4.0", "ownership": {"Z": {}}}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        tapdb_templates,
        "validate_template_configs",
        lambda *_args, **_kwargs: ([{"instance_prefix": "RGX"}], []),
    )
    monkeypatch.setattr(
        tapdb_templates,
        "resolve_seed_config_dirs",
        lambda config_root: [config_root],
    )
    monkeypatch.setattr(
        tapdb_templates,
        "claim_ursa_template_prefixes",
        lambda *args, **kwargs: (
            calls.update(
                {
                    "claim_domain_path": kwargs["domain_registry_path"],
                    "claim_prefix_path": kwargs["prefix_registry_path"],
                }
            )
            or ["RGX"]
        ),
    )
    monkeypatch.setattr(
        tapdb_templates,
        "seed_templates",
        lambda *args, **kwargs: calls.update(
            {
                "seed_domain_path": kwargs["domain_registry_path"],
                "seed_prefix_path": kwargs["prefix_registry_path"],
            }
        ),
    )
    monkeypatch.setattr(
        tapdb_templates,
        "find_tapdb_core_config_dir",
        lambda: Path("/tmp/core"),
    )
    monkeypatch.setattr(
        "daylib_ursa.config.get_settings",
        lambda: SimpleNamespace(
            tapdb_domain_registry_path="/tmp/default-domain.json",
            tapdb_prefix_ownership_registry_path="/tmp/default-prefix.json",
        ),
    )

    seed_ursa_templates(
        object(),
        domain_registry_path=explicit_domain_registry,
        prefix_registry_path=explicit_prefix_registry,
    )

    assert Path(calls["claim_domain_path"]).resolve() == explicit_domain_registry.resolve()
    assert Path(calls["claim_prefix_path"]).resolve() == explicit_prefix_registry.resolve()
    assert Path(calls["seed_domain_path"]).resolve() == explicit_domain_registry.resolve()
    assert Path(calls["seed_prefix_path"]).resolve() == explicit_prefix_registry.resolve()
