from __future__ import annotations

from pathlib import Path
import tomllib


def _load_pyproject() -> dict[str, object]:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    return tomllib.loads(pyproject.read_text(encoding="utf-8"))


def test_pyproject_relies_on_conda_env_yaml_for_tool_union() -> None:
    data = _load_pyproject()
    optional = data["project"]["optional-dependencies"]

    assert "tools" not in optional


def test_ursa_activate_uses_conda_only_bootstrap() -> None:
    activate_script = (Path(__file__).resolve().parents[1] / "ursa_activate").read_text(encoding="utf-8")

    assert 'CONDA_ENV_NAME="URSA"' in activate_script
    assert 'ENV_FILE="${SCRIPT_DIR}/config/ursa_env.yaml"' in activate_script
    assert 'conda env create -n "$CONDA_ENV_NAME" -f "$ENV_FILE"' in activate_script
    assert 'conda activate "$CONDA_ENV_NAME"' in activate_script
    assert 'git+file://${TAPDB_LOCAL_REPO}@0.2.5' in activate_script
    assert ".venv" not in activate_script
    assert "[auth,cluster,dev,tools]" not in activate_script