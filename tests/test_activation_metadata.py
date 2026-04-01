from __future__ import annotations

from pathlib import Path


def _load_pyproject() -> str:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    return pyproject.read_text(encoding="utf-8")


def test_pyproject_relies_on_conda_env_yaml_for_tool_union() -> None:
    pyproject_text = _load_pyproject()

    assert "[project.optional-dependencies]" in pyproject_text
    assert "\ntools = [" not in pyproject_text


def test_activate_uses_conda_only_bootstrap() -> None:
    activate_script = (Path(__file__).resolve().parents[1] / "activate").read_text(
        encoding="utf-8"
    )

    assert 'CONDA_ENV_BASE="URSA"' in activate_script
    assert 'CONDA_ENV_NAME="${CONDA_ENV_BASE}-${CONDA_ENV_DEPLOYMENT_CODE}"' in activate_script
    assert 'if [ "$#" -ne 1 ]; then' in activate_script
    assert 'CONDA_ENV_DEPLOYMENT_CODE="$1"' in activate_script
    assert 'if ! validate_deploy_name "${CONDA_ENV_DEPLOYMENT_CODE}"; then' in activate_script
    assert 'export URSA_DEPLOYMENT_CODE="${CONDA_ENV_DEPLOYMENT_CODE}"' in activate_script
    assert 'export DEPLOYMENT_CODE="${CONDA_ENV_DEPLOYMENT_CODE}"' in activate_script
    assert 'export LSMC_DEPLOYMENT_CODE="${CONDA_ENV_DEPLOYMENT_CODE}"' in activate_script
    assert 'ENV_FILE="${SCRIPT_DIR}/config/ursa_env.yaml"' in activate_script
    assert 'conda env create -n "$CONDA_ENV_NAME" -f "$ENV_FILE"' in activate_script
    assert 'conda activate "$CONDA_ENV_NAME"' in activate_script
    assert 'require_python_import "daylily_tapdb" "daylily-tapdb"' in activate_script
    assert ".venv" not in activate_script
    assert "[auth,cluster,dev,tools]" not in activate_script
