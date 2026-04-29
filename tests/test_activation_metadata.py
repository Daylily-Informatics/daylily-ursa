from __future__ import annotations

import json
import tomllib
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_pyproject() -> dict[str, object]:
    return tomllib.loads(_load_text(_project_root() / "pyproject.toml"))


def test_environment_yaml_is_system_only() -> None:
    environment = _load_text(_project_root() / "environment.yaml")
    pyproject = _load_pyproject()
    project = pyproject["project"]

    assert "pip:" not in environment
    assert "optional-dependencies" not in project

    for dependency in project["dependencies"]:
        assert dependency not in environment

    for expected in (
        "python=3.11",
        "awscli=2.34.31",
        "prompt-toolkit=3.0.51",
        "ruamel.yaml=0.19.1",
        "aws-parallelcluster=3.13.2",
        "pip",
        "setuptools<81",
        "bash=5.2.37",
        "jq=1.8.1",
        "yq=3.4.3",
        "rclone=1.71.0",
        "parallel",
        "perl",
        "yamllint",
        "fd-find",
        "postgresql",
        "nodejs",
    ):
        assert expected in environment


def test_pyproject_contains_the_single_python_install_set() -> None:
    pyproject = _load_pyproject()
    project = pyproject["project"]
    dependencies = project["dependencies"]

    assert "optional-dependencies" not in project
    assert project["requires-python"] == ">=3.11"
    assert pyproject["tool"]["black"]["target-version"] == ["py311", "py312"]
    assert pyproject["tool"]["ruff"]["target-version"] == "py311"
    assert pyproject["tool"]["mypy"]["python_version"] == "3.11"

    for expected in (
        "bandit[toml]>=1.8.0",
        "black>=23.0.0",
        "boto3-stubs[s3,sns,cloudwatch]>=1.28.0",
        "daylily-auth-cognito==2.1.5",
        "daylily-ephemeral-cluster==2.1.11",
        "daylily-tapdb==6.0.8",
        "fastapi>=0.104.0",
        "httpx>=0.25.0",
        "itsdangerous>=2.2.0",
        "jinja2>=3.1.0",
        "jsonschema>=4.17.0",
        "moto>=4.2.0",
        "mypy>=1.5.0",
        "passlib[bcrypt]>=1.7.4",
        "playwright>=1.42.0",
        "pre-commit>=3.8.0",
        "pyyaml>=6.0",
        "pydantic>=2.0.0",
        "pydantic[email]>=2.0.0",
        "pydantic-settings>=2.0.0",
        "python-jose[cryptography]>=3.3.0",
        "python-multipart>=0.0.6",
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.1.0",
        "pytest-playwright>=0.4.4",
        "ruff>=0.1.0",
        "rich>=13.0.0",
        "sqlalchemy>=2.0.0",
        "tabulate",
        "typer>=0.9.0",
        "uvicorn[standard]>=0.24.0",
        "cli-core-yo==2.1.1",
        "zebra_day==6.0.1",
        "boto3>=1.26.0",
    ):
        assert expected in dependencies


def test_ecosystem_versions_track_ephemeral_cluster_baseline() -> None:
    payload = json.loads(_load_text(_project_root() / "config" / "ecosystem-versions.json"))

    assert payload["last_updated"] == "2026-04-29"
    assert payload["tested_combinations"][0]["date"] == "2026-04-29"
    assert payload["components"]["daylily-ephemeral-cluster"]["current"] == "2.1.11"
    assert payload["components"]["daylily-auth-cognito"]["current"] == "2.1.5"
    assert payload["components"]["daylily-tapdb"]["current"] == "6.0.8"
    assert payload["components"]["cli-core-yo"]["current"] == "2.1.1"
    assert payload["components"]["zebra_day"]["current"] == "6.0.1"
    assert payload["tested_combinations"][0]["ephemeral_cluster"] == "2.1.11"
    assert payload["tested_combinations"][0]["cognito"] == "2.1.5"
    assert payload["tested_combinations"][0]["tapdb"] == "6.0.8"
    assert payload["tested_combinations"][0]["cli_core_yo"] == "2.1.1"
    assert payload["tested_combinations"][0]["zebra_day"] == "6.0.1"
    assert "daylily-ephemeral-cluster to 2.1.11" in payload["tested_combinations"][0]["notes"]


def test_workset_monitor_configs_use_daylily_ec_samples_stage() -> None:
    config_dir = _project_root() / "config"
    monitor_configs = sorted(config_dir.glob("*workset-monitor*.yaml"))

    assert monitor_configs
    for path in monitor_configs:
        text = _load_text(path)
        assert "daylily-stage-samples-from-local-to-headnode" not in text, path
        assert "stage_command: daylily-ec samples stage " in text, path


def test_activate_is_env_only() -> None:
    activate_script = _load_text(_project_root() / "activate")

    assert 'conda env create -n "$CONDA_ENV_NAME" -f "$ENV_FILE"' in activate_script
    assert 'conda activate "$CONDA_ENV_NAME"' in activate_script
    assert '"${CONDA_PREFIX}/bin/python" -m pip install -e "${SCRIPT_DIR}"' in activate_script
    assert 'python -m pip install -e "${SCRIPT_DIR}"' not in activate_script
    assert 'python -m pip install -e ".[dev]"' not in activate_script
    assert 'export PATH="${CONDA_PREFIX}/bin:${PATH:-}"' in activate_script
    assert "hash -r 2>/dev/null || true" in activate_script
    assert "bin/dev_setup.sh" not in activate_script
    assert "pre-commit" not in activate_script
    assert "playwright" not in activate_script
    assert "require_tool" not in activate_script
    assert "require_python_import" not in activate_script
    assert "TAPDB_CONFIG_PATH" not in activate_script
    assert "TAPDB_OWNER_REPO" not in activate_script
    assert "MERIDIAN_DOMAIN_CODE" not in activate_script
    assert "BIN_DIR" not in activate_script
    assert "prepare_tapdb_config_path" not in activate_script
    assert "distribution_is_editable_from_repo" not in activate_script
    assert "bootstrap_local_ursa_repo" not in activate_script


def test_readme_no_longer_teaches_dev_setup_or_dev_extras() -> None:
    readme = _load_text(_project_root() / "README.md")

    assert "bin/dev_setup.sh" not in readme
    assert ".[dev]" not in readme
    assert "optional-dependencies.dev" not in readme


def test_agents_document_cli_path_precedence_rule() -> None:
    agents = _load_text(_project_root() / "AGENTS.md")

    assert "minimal `${CONDA_PREFIX}/bin` prepend after `conda activate`" in agents
    assert "secondary install set" in agents
    assert "`.[dev]`" in agents
    assert "`project.optional-dependencies`" in agents


def test_env_validate_hint_points_to_config_init() -> None:
    env_cli = _load_text(_project_root() / "daylib_ursa" / "cli" / "env.py")

    assert "ursa config init" in env_cli
    assert "ursa config generate" not in env_cli


def test_user_facing_files_do_not_reference_dev_extras_or_optional_groups() -> None:
    for relative_path in ("README.md", "activate"):
        text = _load_text(_project_root() / relative_path)
        assert ".[dev]" not in text, relative_path
        assert "optional-dependencies.dev" not in text, relative_path
