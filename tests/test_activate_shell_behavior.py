from __future__ import annotations

import os
import shutil
import shlex
import subprocess
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVATE = REPO_ROOT / "activate"


def _strip_ansi(text: str) -> str:
    return (
        text.replace("\x1b[0m", "")
        .replace("\x1b[0;31m", "")
        .replace("\x1b[0;32m", "")
        .replace("\x1b[1;33m", "")
        .replace("\x1b[0;34m", "")
        .replace("\x1b[0;36m", "")
        .replace("\x1b[1m", "")
    )


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _make_fake_python(bin_dir: Path, *, package_version: str = "1.2.3") -> None:
    script = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -u

        if [[ "${{1:-}}" == "--version" || "${{1:-}}" == "-V" ]]; then
            printf 'Python 3.12.0\\n'
            exit 0
        fi

        if [[ "${{1:-}}" == "-m" && "${{2:-}}" == "pip" ]]; then
            shift 2
            case "${{1:-}}" in
                show)
                    dist="${{2:-}}"
                    if [[ "$dist" == "daylily-ursa" ]]; then
                        printf 'Name: daylily-ursa\\n'
                        printf 'Editable project location: %s\\n' "${{FAKE_REPO_ROOT}}"
                    else
                        printf 'Name: %s\\n' "$dist"
                        printf 'Version: 0.0.0\\n'
                    fi
                    exit 0
                    ;;
                install)
                    if [[ -n "${{FAKE_PYTHON_LOG:-}}" ]]; then
                        printf 'pip install %s\\n' "$*" >> "${{FAKE_PYTHON_LOG}}"
                    fi
                    exit 0
                    ;;
            esac
        fi

        if [[ "${{1:-}}" == "-" ]]; then
            script="$(cat)"
            shift || true
            if printf '%s' "$script" | grep -q 'getattr(daylib_ursa, "__version__"'; then
                printf '%s\\n' "${{FAKE_PACKAGE_VERSION:-{package_version}}}"
                exit 0
            fi
            if printf '%s' "$script" | grep -q 'print(importlib.metadata.version(sys.argv[1]))'; then
                case "${{1:-}}" in
                    daylily-tapdb) printf '5.0.0\\n' ;;
                    daylily-auth-cognito) printf '2.0.1\\n' ;;
                    cli-core-yo) printf '2.0.0\\n' ;;
                    *) printf '0.0.0\\n' ;;
                esac
                exit 0
            fi
            if printf '%s' "$script" | grep -q 'importlib.import_module(sys.argv[1])'; then
                if [[ "${{FAKE_FAIL_IMPORT_MODULE:-}}" == "${{1:-}}" ]]; then
                    exit 1
                fi
                exit 0
            fi
            if printf '%s' "$script" | grep -q 'sys.exit(0 if env and '\''-'\'' in env else 1)'; then
                exit 0
            fi
            exit 0
        fi

        exit 0
        """
    )
    _write_executable(bin_dir / "python", script)


def _make_fake_conda(
    bin_dir: Path,
    *,
    prefix_root: Path,
    base_root: Path,
    preexisting_envs: tuple[str, ...] = (),
) -> Path:
    envs_file = bin_dir.parent / "conda-state" / "envs"
    log_file = bin_dir.parent / "conda-state" / "conda.log"
    envs_file.parent.mkdir(parents=True, exist_ok=True)
    envs_file.write_text("".join(f"{env}\n" for env in preexisting_envs), encoding="utf-8")
    log_file.write_text("", encoding="utf-8")

    script = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -u

        state_dir="${{FAKE_CONDA_STATE_DIR:?}}"
        envs_file="${{state_dir}}/envs"
        log_file="${{state_dir}}/conda.log"
        prefix_root="${{FAKE_CONDA_PREFIX_ROOT:?}}"
        base_root="${{FAKE_CONDA_BASE:?}}"

        mkdir -p "$state_dir" "$prefix_root"
        touch "$envs_file" "$log_file"

        if [[ "${{1:-}}" == "shell.bash" && "${{2:-}}" == "hook" ]] || [[ "${{1:-}}" == "shell.zsh" && "${{2:-}}" == "hook" ]]; then
            cat <<'EOF'
conda() {{
    local cmd="${{1:-}}"
    shift || true
    case "$cmd" in
        info)
            if [[ "${{1:-}}" == "--envs" ]]; then
                if [[ -f "{envs_file}" ]]; then
                    while IFS= read -r env_name; do
                        [[ -n "$env_name" ]] || continue
                        printf '%s %s\\n' "$env_name" "{prefix_root}/$env_name"
                    done < "{envs_file}"
                fi
            elif [[ "${{1:-}}" == "--base" ]]; then
                printf '%s\\n' "{base_root}"
            fi
            ;;
        env)
            case "${{1:-}}" in
                create)
                    printf '%s\\n' "create $*" >> "{log_file}"
                    if [[ "${{FAKE_CONDA_CREATE_FAIL:-0}}" == "1" ]]; then
                        return 1
                    fi
                    local env_name=""
                    while [[ "$#" -gt 0 ]]; do
                        case "$1" in
                            -n)
                                env_name="${{2:-}}"
                                shift 2
                                ;;
                            -f)
                                shift 2
                                ;;
                            *)
                                shift
                                ;;
                        esac
                    done
                    if [[ -n "$env_name" ]] && ! grep -qxF "$env_name" "{envs_file}"; then
                        printf '%s\\n' "$env_name" >> "{envs_file}"
                    fi
                    ;;
                remove)
                    printf '%s\\n' "remove $*" >> "{log_file}"
                    local env_name=""
                    while [[ "$#" -gt 0 ]]; do
                        case "$1" in
                            -n)
                                env_name="${{2:-}}"
                                shift 2
                                ;;
                            -y)
                                shift
                                ;;
                            *)
                                shift
                                ;;
                        esac
                    done
                    if [[ -n "$env_name" ]]; then
                        grep -vxF "$env_name" "{envs_file}" > "{envs_file}.tmp" || true
                        mv "{envs_file}.tmp" "{envs_file}"
                    fi
                    ;;
            esac
            ;;
        activate)
            local env_name="${{1:-}}"
            printf '%s\\n' "activate $env_name" >> "{log_file}"
            export FAKE_CONDA_PREV_ENV="${{CONDA_DEFAULT_ENV:-}}"
            export FAKE_CONDA_PREV_PREFIX="${{CONDA_PREFIX:-}}"
            export CONDA_DEFAULT_ENV="$env_name"
            export CONDA_PREFIX="{prefix_root}/$env_name"
            mkdir -p "$CONDA_PREFIX/bin"
            ln -sf "{bin_dir}/python" "$CONDA_PREFIX/bin/python"
            ;;
        deactivate)
            printf '%s\\n' "deactivate" >> "{log_file}"
            unset CONDA_DEFAULT_ENV CONDA_PREFIX
            ;;
        *)
            printf '%s\\n' "other $cmd $*" >> "{log_file}"
            ;;
    esac
}}
EOF
            exit 0
        fi

        printf '%s\\n' "direct $*" >> "$log_file"
        exit 0
        """
    )
    _write_executable(bin_dir / "conda", script)
    return log_file


def _run_activation(
    shell_name: str,
    args: list[str],
    *,
    package_version: str = "1.2.3",
    preexisting_envs: tuple[str, ...] = (),
    previous_env: str | None = None,
    tmp_path: Path,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    shell_path = shutil.which(shell_name)
    if shell_path is None:
        pytest.skip(f"{shell_name} is not available")

    temp_root = tmp_path / f"{shell_name}-activate"
    temp_root.mkdir(parents=True, exist_ok=True)
    bin_dir = temp_root / "bin"
    bin_dir.mkdir()
    home_dir = temp_root / "home"
    home_dir.mkdir()
    state_dir = temp_root / "conda-state"
    state_dir.mkdir()
    prefix_root = temp_root / "conda-prefixes"
    prefix_root.mkdir()

    _make_fake_python(bin_dir, package_version=package_version)
    log_file = _make_fake_conda(
        bin_dir,
        prefix_root=prefix_root,
        base_root=temp_root / "conda-base",
        preexisting_envs=preexisting_envs,
    )

    for tool_name in (
        "aws",
        "pcluster",
        "jq",
        "yq",
        "rclone",
        "parallel",
        "perl",
        "yamllint",
        "fd",
        "psql",
        "node",
    ):
        _write_executable(bin_dir / tool_name, "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        bin_dir / "pre-commit",
        '#!/usr/bin/env bash\nif [[ "${1:-}" == "install" ]]; then exit 1; fi\nexit 0\n',
    )

    shell_args = " ".join(shlex.quote(arg) for arg in args)
    script_path = temp_root / "run.sh"
    script_path.write_text(
        textwrap.dedent(
            f"""\
            set +e
            export HOME={shlex.quote(str(home_dir))}
            export PATH={shlex.quote(str(bin_dir))}:$PATH
            export FAKE_REPO_ROOT={shlex.quote(str(REPO_ROOT))}
            export FAKE_PACKAGE_VERSION={shlex.quote(package_version)}
            export FAKE_CONDA_STATE_DIR={shlex.quote(str(state_dir))}
            export FAKE_CONDA_PREFIX_ROOT={shlex.quote(str(prefix_root))}
            export FAKE_CONDA_BASE={shlex.quote(str(temp_root / "conda-base"))}
            {f"export CONDA_DEFAULT_ENV={shlex.quote(previous_env)}" if previous_env else "unset CONDA_DEFAULT_ENV"}
            {f"export CONDA_PREFIX={shlex.quote(str(prefix_root / previous_env))}" if previous_env else "unset CONDA_PREFIX"}
            source ./activate {shell_args}
            status=$?
            printf 'STATUS=%s\\n' "$status"
            printf 'CONDA_DEFAULT_ENV=%s\\n' "${{CONDA_DEFAULT_ENV:-}}"
            printf 'CONDA_PREFIX=%s\\n' "${{CONDA_PREFIX:-}}"
            exit "$status"
            """
        ),
        encoding="utf-8",
    )
    script_path.chmod(0o755)

    completed = subprocess.run(
        [shell_path, str(script_path)],
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )
    return completed, log_file


def test_activate_default_name_creates_and_cleans_failed_env_in_bash(tmp_path: Path) -> None:
    result, log_file = _run_activation(
        "bash",
        [],
        package_version="1.2.3",
        previous_env="old-env",
        tmp_path=tmp_path,
    )

    assert result.returncode != 0
    stdout = _strip_ansi(result.stdout)
    assert "Conda Environment:" in stdout
    assert "URSA-1-2-3" in stdout
    assert "Deploy Name:" in stdout
    assert "1-2-3" in stdout
    assert "Cleanup:" in stdout
    assert "deleted newly created environment" in stdout
    assert "CONDA_DEFAULT_ENV=old-env" in stdout

    log_lines = log_file.read_text(encoding="utf-8").splitlines()
    assert "create -n URSA-1-2-3 -f" in log_lines[0]
    assert "activate URSA-1-2-3" in log_lines
    assert "deactivate" in log_lines
    assert any("remove -n URSA-1-2-3 -y" in line for line in log_lines)
    assert "activate old-env" in log_lines


def test_activate_preexisting_env_is_not_deleted_in_bash(tmp_path: Path) -> None:
    result, log_file = _run_activation(
        "bash",
        [],
        package_version="1.2.3",
        preexisting_envs=("URSA-1-2-3",),
        previous_env="old-env",
        tmp_path=tmp_path,
    )

    assert result.returncode != 0
    stdout = _strip_ansi(result.stdout)
    assert "already exists" in stdout
    assert "skipped because the environment pre-existed" in stdout
    assert "CONDA_DEFAULT_ENV=old-env" in stdout

    log_lines = log_file.read_text(encoding="utf-8").splitlines()
    assert "create -n URSA-1-2-3 -f" not in log_lines
    assert "remove -n URSA-1-2-3 -y" not in log_lines
    assert "activate URSA-1-2-3" in log_lines
    assert "activate old-env" in log_lines


def test_activate_debug_skips_cleanup_in_zsh(tmp_path: Path) -> None:
    result, log_file = _run_activation(
        "zsh",
        ["--debug"],
        package_version="1.2.3",
        tmp_path=tmp_path,
    )

    assert result.returncode != 0
    stdout = _strip_ansi(result.stdout)
    assert "Debug Mode:" in stdout
    assert "yes" in stdout
    assert "skipped because --debug was set" in stdout

    log_lines = log_file.read_text(encoding="utf-8").splitlines()
    assert "create -n URSA-1-2-3 -f" in log_lines[0]
    assert "activate URSA-1-2-3" in log_lines
    assert "remove -n URSA-1-2-3 -y" not in log_lines
    assert "deactivate" in log_lines
