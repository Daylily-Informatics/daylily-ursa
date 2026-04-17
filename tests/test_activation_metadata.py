from __future__ import annotations

import tomllib
from pathlib import Path

from daylib_ursa.ephemeral_cluster.runner import REQUIRED_DAYLILY_EC_VERSION


def _load_pyproject() -> str:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    return pyproject.read_text(encoding="utf-8")


def _tapdb_dependency_spec() -> str:
    return _dependency_spec("daylily-tapdb")


def _dependency_spec(distribution_name: str) -> str:
    pyproject = tomllib.loads(_load_pyproject())
    dependencies = pyproject["project"]["dependencies"]
    for dependency in dependencies:
        if dependency.startswith(distribution_name):
            return dependency
    raise AssertionError(f"{distribution_name} dependency missing from pyproject.toml")


def _optional_dependency_spec(extra_name: str, distribution_name: str) -> str:
    pyproject = tomllib.loads(_load_pyproject())
    dependencies = pyproject["project"]["optional-dependencies"][extra_name]
    for dependency in dependencies:
        if dependency.startswith(distribution_name):
            return dependency
    raise AssertionError(
        f"{distribution_name} dependency missing from optional dependency '{extra_name}'"
    )


def test_pyproject_relies_on_conda_env_yaml_for_tool_union() -> None:
    pyproject_text = _load_pyproject()

    assert "[project.optional-dependencies]" in pyproject_text
    assert "\ntools = [" not in pyproject_text


def test_pyproject_uses_requested_internal_package_versions() -> None:
    pyproject = tomllib.loads(_load_pyproject())
    dependencies = pyproject["project"]["dependencies"]
    tapdb_spec = _tapdb_dependency_spec()
    cluster_spec = _optional_dependency_spec("cluster", "daylily-ephemeral-cluster")

    assert "cli-core-yo==2.1.0" in dependencies
    assert "daylily-auth-cognito==2.1.1" in dependencies
    assert tapdb_spec == "daylily-tapdb==6.0.4"
    assert not any(dep.startswith("daylily-omics-analysis") for dep in dependencies)
    assert cluster_spec == "daylily-ephemeral-cluster==2.0.3"


def test_daylily_ec_pin_is_aligned_across_runtime_and_bootstrap() -> None:
    cluster_spec = _optional_dependency_spec("cluster", "daylily-ephemeral-cluster")
    activate_script = (Path(__file__).resolve().parents[1] / "activate").read_text(encoding="utf-8")

    assert REQUIRED_DAYLILY_EC_VERSION == "2.0.3"
    assert cluster_spec == f"daylily-ephemeral-cluster=={REQUIRED_DAYLILY_EC_VERSION}"
    assert "daylily-ephemeral-cluster" not in activate_script


def test_activate_bootstraps_local_ursa_repo_only() -> None:
    project_root = Path(__file__).resolve().parents[1]
    environment = (project_root / "environment.yaml").read_text(encoding="utf-8")
    assert (project_root / "environment.yaml").is_file()
    assert not (project_root / "config" / "ursa_env.yaml").exists()
    assert "-e ." not in environment
    assert "cli-core-yo==2.1.0" in environment
    assert "daylily-auth-cognito==2.1.1" in environment
    assert "daylily-tapdb==6.0.4" in environment
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
    assert 'export MERIDIAN_DOMAIN_CODE="${MERIDIAN_DOMAIN_CODE:-Z}"' in activate_script
    assert 'export TAPDB_OWNER_REPO="${TAPDB_OWNER_REPO:-ursa}"' in activate_script
    assert 'ENV_FILE="${SCRIPT_DIR}/environment.yaml"' in activate_script
    assert "cli-core-yo==2.1.0" in activate_script
    assert "daylily-auth-cognito==2.1.1" in activate_script
    assert (
        'URSA_TAPDB_PACKAGE_SPEC="$(dependency_spec_from_pyproject "${SCRIPT_DIR}" "daylily-tapdb")"'
        in activate_script
    )
    assert 'python - "${project_root}/pyproject.toml" "${dependency_name}"' in activate_script
    assert 'conda env create -n "$CONDA_ENV_NAME" -f "$ENV_FILE"' in activate_script
    assert 'conda activate "$CONDA_ENV_NAME"' in activate_script
    assert 'require_tool "conda env" "pre-commit"' in activate_script
    assert "pre-commit install --hook-type pre-commit --hook-type pre-push" in activate_script
    assert "URSA_REQUIRED_IMPORTS=(" in activate_script
    assert '"daylily_tapdb:daylily-tapdb"' in activate_script
    assert '"daylily_ec:daylily-ephemeral-cluster"' not in activate_script
    assert '"daylily_omics_analysis:daylily-omics-analysis"' not in activate_script
    assert '"pytest:pytest"' in activate_script
    assert 'python -m pip install -e "$install_target" -q' in activate_script
    assert 'install_target="${SCRIPT_DIR}"' in activate_script
    assert "Installing local Ursa checkout..." in activate_script
    assert "pyproject.toml" in activate_script
    assert "env_created=0" in activate_script
    assert "env_created=1" in activate_script
    assert 'if [ "${env_created}" -eq 1 ]; then' in activate_script
    assert "if ! bootstrap_local_ursa_repo; then" in activate_script
    assert (
        'elif distribution_is_editable_from_repo "daylily-ursa" "${SCRIPT_DIR}"; then'
        in activate_script
    )
    assert "else" in activate_script
    assert (
        'if ! distribution_is_editable_from_repo "daylily-ursa" "${SCRIPT_DIR}"; then'
        in activate_script
    )
    assert "ensure_local_ursa_checkout()" not in activate_script
    assert "USE_LOCAL_CLI_CORE_YO" not in activate_script
    assert "TAPDB_APP_CODE" not in activate_script
    assert "daylily-tapdb==" not in activate_script
    assert "meridian-euid==0.4.2" in activate_script
    assert "../daylily-tapdb" not in activate_script
    assert "../daylily-auth-cognito" not in activate_script
    assert "../cli-core-yo" not in activate_script
    assert "from packaging.requirements import Requirement" not in activate_script
    assert "--no-deps" not in activate_script
    assert ".venv" not in activate_script
    assert "URSA_PIP_INSTALL_EXTRAS" not in activate_script
    assert "[auth,cluster,dev,tools]" not in activate_script
    assert "daylily-ephemeral-cluster" not in activate_script
    assert "MERIDIAN_DOMAIN_CODE=Z" in (
        project_root / "config" / "ursa-config.example.yaml"
    ).read_text(encoding="utf-8")
    assert "TAPDB_OWNER_REPO=ursa" in (
        project_root / "config" / "ursa-config.example.yaml"
    ).read_text(encoding="utf-8")
    assert "MERIDIAN_DOMAIN_CODE=Z" in (
        project_root / "config" / "tapdb-config-ursa.yaml"
    ).read_text(encoding="utf-8")
    assert "TAPDB_OWNER_REPO=ursa" in (
        project_root / "config" / "tapdb-config-ursa.yaml"
    ).read_text(encoding="utf-8")


def test_workset_monitor_configs_pin_analysis_repo_to_release_tag() -> None:
    project_root = Path(__file__).resolve().parents[1]

    for config_path in (
        project_root / "config" / "daylily-workset-monitor.yaml",
        project_root / "config" / "workset-monitor-config.yaml",
        project_root / "config" / "workset-monitor-config.template.yaml",
    ):
        config_text = config_path.read_text(encoding="utf-8")
        assert "repo_tag: 0.7.641" in config_text
        assert "--repo Daylily-Informatics/daylily-omics-analysis --branch 0.7.641" in config_text
