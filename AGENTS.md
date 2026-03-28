# WHEN INITIALIZING A NEW TERMINAL SESSION

## ALWAYS DO THIS FIRST
source ./ursa_activate

## CHANGE POLICY

- Do not default to preserving legacy behavior or adding fallback behavior.
- Treat fallback/legacy compatibility as an explicit requirement, not a safe default.
- Do not assume there is existing data to migrate unless the user directly says there is.
- Do not add migration code, compatibility shims, dual-read/dual-write paths, or legacy field support unless explicitly requested.
