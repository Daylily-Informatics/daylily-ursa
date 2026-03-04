from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest


# Keep this list specific: we want to block Dynamo-era compatibility surfaces and
# misleading terminology, without banning legitimate SQL words like "table".
BANNED_PATTERNS: list[tuple[str, str]] = [
    (r"\bDDB\b", "DDB naming"),
    (r"\bGSI\b", "Dynamo GSI naming"),
    (r"Global Tables", "Dynamo Global Tables naming"),
    (r"ResourceNotFoundException", "Dynamo-era exception naming"),
    (r"DAYLILY_TAPDB_TABLE", "Legacy env var naming"),
    (r"DAYLILY_WORKSET_TABLE", "Legacy env var naming"),
    (r"TAPDB_WORKSET_NAMESPACE", "Legacy TapDB table-namespace env var"),
    (r"TAPDB_CUSTOMER_NAMESPACE", "Legacy TapDB table-namespace env var"),
    (r"TAPDB_MANIFEST_NAMESPACE", "Legacy TapDB table-namespace env var"),
    (r"TAPDB_BUCKET_NAMESPACE", "Legacy TapDB table-namespace env var"),
    (r"tapdb_db_region", "Legacy TapDB table-region config"),
    (r"get_effective_tapdb_db_region", "Legacy TapDB table-region helper"),
    (r"--tapdb-table", "Legacy CLI flag"),
    (r"--table-name", "Legacy CLI flag"),
    (r"create_table_if_not_exists", "Legacy Dynamo-era method name"),
    (r"create_tables_if_not_exist", "Legacy Dynamo-era method name"),
    (r"TapDB Table", "Misleading TapDB table wording"),
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_tracked_files(repo_root: Path) -> list[Path]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "-z"],
            check=True,
            capture_output=True,
        )
    except Exception as exc:
        pytest.skip(f"Unable to enumerate tracked files via git: {exc}")

    parts = [p for p in proc.stdout.decode("utf-8", errors="replace").split("\x00") if p]
    return [repo_root / p for p in parts]


def _is_probably_text(path: Path) -> bool:
    # Fast skip for common binary artifacts.
    if path.suffix.lower() in {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".ico",
        ".pdf",
        ".zip",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".woff",
        ".woff2",
        ".ttf",
        ".otf",
        ".eot",
    }:
        return False

    try:
        head = path.read_bytes()[:4096]
    except OSError:
        return False

    return b"\x00" not in head


def test_no_dynamo_era_indicators_in_tracked_files() -> None:
    repo_root = _repo_root()
    tracked_files = _git_tracked_files(repo_root)

    compiled = [(re.compile(pat, re.IGNORECASE), label) for pat, label in BANNED_PATTERNS]

    violations: list[str] = []
    for path in tracked_files:
        if not path.is_file():
            continue
        if not _is_probably_text(path):
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        for lineno, line in enumerate(text.splitlines(), start=1):
            for regex, label in compiled:
                if regex.search(line):
                    rel = path.relative_to(repo_root)
                    violations.append(f"{rel}:{lineno}: {label}: {line.strip()}")

    assert not violations, "Found Dynamo-era indicators:\n" + "\n".join(violations)

