from __future__ import annotations

from pathlib import Path


def _load_pyproject() -> str:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    return pyproject.read_text(encoding="utf-8")


def test_pyproject_relies_on_conda_env_yaml_for_tool_union() -> None:
    pyproject_text = _load_pyproject()

    assert "[project.optional-dependencies]" in pyproject_text
    assert "\ntools = [" not in pyproject_text


def test_ursa_activate_uses_conda_only_bootstrap() -> None:
    activate_script = (Path(__file__).resolve().parents[1] / "ursa_activate").read_text(
        encoding="utf-8"
    )

    assert 'CONDA_ENV_NAME="URSA"' in activate_script
    assert 'ENV_FILE="${SCRIPT_DIR}/config/ursa_env.yaml"' in activate_script
    assert 'conda env create -n "$CONDA_ENV_NAME" -f "$ENV_FILE"' in activate_script
    assert 'conda activate "$CONDA_ENV_NAME"' in activate_script
    assert 'require_python_import "daylily_tapdb" "daylily-tapdb==3.0.3"' in activate_script
    assert ".venv" not in activate_script
    assert "[auth,cluster,dev,tools]" not in activate_script
