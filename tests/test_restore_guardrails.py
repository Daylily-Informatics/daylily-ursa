from __future__ import annotations

import re
from pathlib import Path


RESTORED_MODULES = [
    Path("daylib_ursa/portal.py"),
    Path("daylib_ursa/portal_graph_state.py"),
    Path("daylib_ursa/portal_onboarding.py"),
]

BANNED_IMPORT_PATTERNS = [
    re.compile(r"^\s*import\s+sqlite3\b", re.MULTILINE),
    re.compile(r"^\s*from\s+sqlite3\s+import\b", re.MULTILINE),
    re.compile(r"^\s*import\s+sqlalchemy\b", re.MULTILINE),
    re.compile(r"^\s*from\s+sqlalchemy\s+import\b", re.MULTILINE),
    re.compile(r"^\s*import\s+duckdb\b", re.MULTILINE),
    re.compile(r"^\s*from\s+duckdb\s+import\b", re.MULTILINE),
    re.compile(r"^\s*import\s+aiosqlite\b", re.MULTILINE),
    re.compile(r"^\s*from\s+aiosqlite\s+import\b", re.MULTILINE),
    re.compile(r"^\s*import\s+psycopg\b", re.MULTILINE),
    re.compile(r"^\s*from\s+psycopg\s+import\b", re.MULTILINE),
    re.compile(r"^\s*import\s+psycopg2\b", re.MULTILINE),
    re.compile(r"^\s*from\s+psycopg2\s+import\b", re.MULTILINE),
]

RAW_SQL_PATTERN = re.compile(
    r"\.execute\(\s*(?:f?\"|f?')\s*(select|insert|update|delete|create|alter|drop)\b",
    flags=re.IGNORECASE,
)


def test_restored_modules_do_not_use_forbidden_persistence_dependencies() -> None:
    for module_path in RESTORED_MODULES:
        content = module_path.read_text(encoding="utf-8")
        for pattern in BANNED_IMPORT_PATTERNS:
            assert not pattern.search(content), f"Forbidden persistence import in {module_path}: {pattern.pattern}"


def test_restored_modules_do_not_reference_dynamodb_or_ad_hoc_sql() -> None:
    for module_path in RESTORED_MODULES:
        content = module_path.read_text(encoding="utf-8").lower()
        assert "dynamodb" not in content, f"DynamoDB reference found in restored module: {module_path}"
        assert not RAW_SQL_PATTERN.search(content), f"Ad-hoc raw SQL found in restored module: {module_path}"
