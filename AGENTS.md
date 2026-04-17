# WHEN INITIALIZING A NEW TERMINAL SESSION

## ALWAYS DO THIS FIRST FROM THIS REPO ROOT
source ./activate <deploy-name>

## CLI Policy

- Use `ursa ...` as the primary interface for normal Ursa work.
- Use `tapdb ...` only when Ursa explicitly delegates low-level DB/runtime lifecycle to TapDB.
- Use `daycog ...` only when Ursa explicitly delegates shared Cognito lifecycle to Daycog.

## Activation Contract

- `source ./activate <deploy-name>` must only create the conda env if missing, activate it, and run exactly one `python -m pip install -e .` on first create.
- Do not add any other pip installs, conda installs, config copying, runtime env exports, pre-commit installs, Playwright installs, or tool checks to `activate`.
- The only allowed PATH adjustment in `activate` is a minimal `${CONDA_PREFIX}/bin` prepend after `conda activate` so the repo's declared console scripts win over conflicting global installs.
- If Ursa needs Python dependencies, put them in `pyproject.toml`.
- If Ursa needs system/bootstrap packages, put them in `environment.yaml`.
- If an Ursa CLI is missing from PATH after activation, fix the packaging entrypoint or the minimal conda-env bin precedence rule, not by adding broader shell hacks.
- The declared console scripts `ursa` and `daylily-workset-api` must remain available from the activated conda env.

## Packaging Boundary

- `environment.yaml` is for Python itself, `pip`, `setuptools`, and non-Python/system packages only.
- `environment.yaml` must not contain a `pip:` block or Python library dependencies.
- `pyproject.toml` owns all Python dependencies needed by the repo in `project.dependencies`.
- `pyproject.toml` must not use `project.optional-dependencies` for repo Python installs.
- Do not add any secondary install set such as `.[dev]`, `.[test]`, or `requirements-dev.txt`.

## No Circumvention Policy

- Do not bypass `ursa`, `tapdb`, or `daycog` with raw tools just because something is missing or broken.
- Do not treat direct `python -m ...`, raw `postgres`, raw AWS CLI mutations, or direct config-file edits as automatic fallbacks.
- If the intended CLI path is broken or incomplete, stop, diagnose, and ask for permission before circumventing it.
- Prefer patience and repair of the intended CLI workflow over inventing a shortcut.

## Ursa Examples

- Start with `source ./activate <deploy-name>`
- Use `ursa server start --port 8913`
- Use `ursa config ...` and `ursa env ...` for Ursa-owned runtime operations
- Use `daycog ...` for Cognito lifecycle and `tapdb ...` for DB/runtime lifecycle where Ursa docs explicitly delegate to them

## CHANGE POLICY

- Do not default to preserving legacy behavior or adding fallback behavior.
- Treat fallback/legacy compatibility as an explicit requirement, not a safe default.
- Do not assume there is existing data to migrate unless the user directly says there is.
- Do not add migration code, compatibility shims, dual-read/dual-write paths, or legacy field support unless explicitly requested.
