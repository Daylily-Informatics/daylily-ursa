# WHEN INITIALIZING A NEW TERMINAL SESSION

## ALWAYS DO THIS FIRST
source ./activate <deploy-name>

## CLI Policy

- Use `ursa ...` as the primary interface for normal Ursa work.
- Use `tapdb ...` only when Ursa explicitly delegates low-level DB/runtime lifecycle to TapDB.
- Use `daycog ...` only when Ursa explicitly delegates shared Cognito lifecycle to Daycog.

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
