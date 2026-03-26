from __future__ import annotations

from importlib import metadata as importlib_metadata
from typing import Any

_tapdb_import_error: Exception | None = None
try:
    from daylily_tapdb import (
        URSA_TEMPLATE_DEFINITIONS,
        UrsaTapdbRepository,
        from_json_addl,
        to_action_history_entry,
        utc_now_iso,
    )
except ImportError as exc:  # pragma: no cover - compatibility path for reduced test envs
    _tapdb_import_error = exc
    URSA_TEMPLATE_DEFINITIONS: list[Any] = []

    class UrsaTapdbRepository:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "TapDB backend is unavailable in this environment"
            ) from _tapdb_import_error

    def from_json_addl(instance) -> dict[str, Any]:
        return dict(getattr(instance, "json_addl", {}) or {})

    def to_action_history_entry(*args, **kwargs) -> dict[str, Any]:
        return {
            "args": list(args),
            "kwargs": dict(kwargs),
        }

    def utc_now_iso() -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


_MINIMUM_TAPDB_VERSION = (2, 0, 1)
_MAXIMUM_TAPDB_MAJOR = 3


def _parse_version_parts(raw_version: str) -> tuple[int, int, int]:
    cleaned = raw_version.strip()
    parts = cleaned.split(".")
    numeric: list[int] = []
    for part in parts[:3]:
        digits = []
        for char in part:
            if char.isdigit():
                digits.append(char)
            else:
                break
        numeric.append(int("".join(digits) or "0"))
    while len(numeric) < 3:
        numeric.append(0)
    return tuple(numeric[:3])


def _validate_tapdb_version() -> None:
    if _tapdb_import_error is not None:
        return
    version = importlib_metadata.version("daylily-tapdb")
    parsed = _parse_version_parts(version)
    if parsed < _MINIMUM_TAPDB_VERSION or parsed[0] >= _MAXIMUM_TAPDB_MAJOR:
        raise RuntimeError(
            "Ursa requires daylily-tapdb>=2.0.1,<3. "
            f"Installed version is {version}."
        )


_validate_tapdb_version()

TEMPLATE_DEFINITIONS = URSA_TEMPLATE_DEFINITIONS


class TapDBBackend(UrsaTapdbRepository):
    """Thin Ursa adapter over TapDB's Ursa helper surface."""


__all__ = [
    "TapDBBackend",
    "TEMPLATE_DEFINITIONS",
    "from_json_addl",
    "to_action_history_entry",
    "utc_now_iso",
]
