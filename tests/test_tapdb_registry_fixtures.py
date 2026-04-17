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


def test_ursa_seed_prefers_settings_registry_paths_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAPDB_DOMAIN_REGISTRY_PATH", "/tmp/from-env-domain.json")
    monkeypatch.setenv("TAPDB_PREFIX_OWNERSHIP_REGISTRY_PATH", "/tmp/from-env-prefix.json")
    monkeypatch.setattr(
        "daylib_ursa.config.get_settings",
        lambda: SimpleNamespace(
            tapdb_domain_registry_path="/tmp/from-settings-domain.json",
            tapdb_prefix_ownership_registry_path="/tmp/from-settings-prefix.json",
        ),
    )

    resolved_domain, resolved_prefix = tapdb_templates._resolve_registry_paths()

    assert resolved_domain == Path("/tmp/from-settings-domain.json").resolve()
    assert resolved_prefix == Path("/tmp/from-settings-prefix.json").resolve()


def test_ursa_resolve_registry_paths_requires_explicit_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TAPDB_DOMAIN_REGISTRY_PATH", raising=False)
    monkeypatch.delenv("TAPDB_PREFIX_OWNERSHIP_REGISTRY_PATH", raising=False)
    monkeypatch.setattr(
        "daylib_ursa.config.get_settings",
        lambda: SimpleNamespace(
            tapdb_domain_registry_path="",
            tapdb_prefix_ownership_registry_path="",
        ),
    )

    with pytest.raises(RuntimeError, match="explicit registry paths"):
        tapdb_templates._resolve_registry_paths()


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

    def fake_validate_template_configs(config_dirs, strict=True):
        assert strict is True
        resolved = [Path(item) for item in config_dirs]
        if resolved == [Path("/tmp/core")]:
            return ([{"instance_prefix": "SYS"}], [])
        return ([{"instance_prefix": "RGX"}], [])

    monkeypatch.setattr(
        tapdb_templates,
        "validate_template_configs",
        fake_validate_template_configs,
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
        lambda *args, **kwargs: calls.setdefault("seed_calls", []).append(
            {
                "templates": args[1],
                "owner_repo_name": kwargs["owner_repo_name"],
                "domain_path": kwargs["domain_registry_path"],
                "prefix_path": kwargs["prefix_registry_path"],
            }
        ),
    )
    monkeypatch.setattr(
        tapdb_templates,
        "_ensure_identity_prefix_config",
        lambda session, **kwargs: calls.setdefault("identity_calls", []).append(kwargs),
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
    seed_calls = calls["seed_calls"]
    assert len(seed_calls) == 2
    assert seed_calls[0]["templates"] == [{"instance_prefix": "SYS"}]
    assert seed_calls[0]["owner_repo_name"] == "daylily-tapdb"
    assert Path(seed_calls[0]["domain_path"]).resolve() == explicit_domain_registry.resolve()
    assert Path(seed_calls[0]["prefix_path"]).resolve() == explicit_prefix_registry.resolve()
    assert seed_calls[1]["templates"] == [{"instance_prefix": "RGX"}]
    assert seed_calls[1]["owner_repo_name"] == "ursa"
    assert Path(seed_calls[1]["domain_path"]).resolve() == explicit_domain_registry.resolve()
    assert Path(seed_calls[1]["prefix_path"]).resolve() == explicit_prefix_registry.resolve()
    identity_calls = calls["identity_calls"]
    assert identity_calls == [
        {
            "entity": "generic_template",
            "domain_code": "Z",
            "owner_repo_name": "ursa",
            "prefix": "RGX",
        },
        {
            "entity": "generic_instance_lineage",
            "domain_code": "Z",
            "owner_repo_name": "ursa",
            "prefix": "EDG",
        },
        {
            "entity": "audit_log",
            "domain_code": "Z",
            "owner_repo_name": "ursa",
            "prefix": "ADT",
        },
    ]
