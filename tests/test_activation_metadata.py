from __future__ import annotations

import tomllib
from pathlib import Path


def _load_pyproject() -> str:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    return pyproject.read_text(encoding="utf-8")


def test_pyproject_relies_on_conda_env_yaml_for_tool_union() -> None:
    pyproject_text = _load_pyproject()

    assert "[project.optional-dependencies]" in pyproject_text
    assert "\ntools = [" not in pyproject_text


def test_pyproject_uses_minimum_internal_package_versions() -> None:
    pyproject = tomllib.loads(_load_pyproject())
    dependencies = pyproject["project"]["dependencies"]
    cluster_extra = pyproject["project"]["optional-dependencies"]["cluster"]

    assert "cli-core-yo>=1.3.1" in dependencies
    assert "daylily-cognito>=1.1.7" in dependencies
    assert "daylily-tapdb>=4.0.7" in dependencies
    assert "daylily-ephemeral-cluster==0.7.614" in cluster_extra


def test_activate_bootstraps_local_ursa_repo_only() -> None:
    project_root = Path(__file__).resolve().parents[1]
    environment = (project_root / "environment.yaml").read_text(encoding="utf-8")
    assert (project_root / "environment.yaml").is_file()
    assert not (project_root / "config" / "ursa_env.yaml").exists()
    assert "-e ." not in environment
    assert "\n  - pre-commit\n" in environment

    activate_script = (Path(__file__).resolve().parents[1] / "activate").read_text(encoding="utf-8")

    assert 'CONDA_ENV_BASE="URSA"' in activate_script
    assert 'CONDA_ENV_NAME="${CONDA_ENV_BASE}-${CONDA_ENV_DEPLOYMENT_CODE}"' in activate_script
    assert 'if [ "$#" -ne 1 ]; then' in activate_script
    assert 'CONDA_ENV_DEPLOYMENT_CODE="$1"' in activate_script
    assert 'if ! validate_deploy_name "${CONDA_ENV_DEPLOYMENT_CODE}"; then' in activate_script
    assert 'export URSA_DEPLOYMENT_CODE="${CONDA_ENV_DEPLOYMENT_CODE}"' in activate_script
    assert 'export DEPLOYMENT_CODE="${CONDA_ENV_DEPLOYMENT_CODE}"' in activate_script
    assert 'export LSMC_DEPLOYMENT_CODE="${CONDA_ENV_DEPLOYMENT_CODE}"' in activate_script
    assert 'export MERIDIAN_DOMAIN_CODE="${MERIDIAN_DOMAIN_CODE:-R}"' in activate_script
    assert 'export TAPDB_APP_CODE="${TAPDB_APP_CODE:-R}"' in activate_script
    assert 'ENV_FILE="${SCRIPT_DIR}/environment.yaml"' in activate_script
    assert 'conda env create -n "$CONDA_ENV_NAME" -f "$ENV_FILE"' in activate_script
    assert 'conda activate "$CONDA_ENV_NAME"' in activate_script
    assert 'require_tool "conda env" "pre-commit"' in activate_script
    assert "pre-commit install --hook-type pre-commit --hook-type pre-push" in activate_script
    assert "URSA_REQUIRED_IMPORTS=(" in activate_script
    assert '"daylily_tapdb:daylily-tapdb"' in activate_script
    assert '"pytest:pytest"' in activate_script
    assert 'URSA_PIP_INSTALL_EXTRAS="[auth,dev]"' in activate_script
    assert 'python -m pip install --upgrade -e "$install_target" -q' in activate_script
    assert 'install_target="${SCRIPT_DIR}${URSA_PIP_INSTALL_EXTRAS}"' in activate_script
    assert (
        'if ! distribution_is_editable_from_repo "daylily-ursa" "${SCRIPT_DIR}"; then'
        in activate_script
    )
    assert "../daylily-tapdb" not in activate_script
    assert "../daylily-cognito" not in activate_script
    assert "../cli-core-yo" not in activate_script
    assert "--no-deps" not in activate_script
    assert ".venv" not in activate_script
    assert "[auth,cluster,dev,tools]" not in activate_script
    assert "MERIDIAN_DOMAIN_CODE=R" in (
        project_root / "config" / "ursa-config.example.yaml"
    ).read_text(encoding="utf-8")
    assert "TAPDB_APP_CODE=R" in (project_root / "config" / "ursa-config.example.yaml").read_text(
        encoding="utf-8"
    )
    assert "MERIDIAN_DOMAIN_CODE=R" in (
        project_root / "config" / "tapdb-config-ursa.yaml"
    ).read_text(encoding="utf-8")
    assert "TAPDB_APP_CODE=R" in (project_root / "config" / "tapdb-config-ursa.yaml").read_text(
        encoding="utf-8"
    )
