"""Shared user preference helpers for Ursa portal surfaces."""

from __future__ import annotations

import logging
from zoneinfo import ZoneInfo, available_timezones

from sqlalchemy import text

from daylib_ursa.tapdb_graph import TapDBBackend

LOGGER = logging.getLogger("daylily.user_preferences")

try:
    from daylily_tapdb.timezone_utils import (  # type: ignore
        DEFAULT_DISPLAY_TIMEZONE,
        normalize_display_timezone,
    )
except Exception:
    DEFAULT_DISPLAY_TIMEZONE = "UTC"

    def normalize_display_timezone(value: str | None, default: str = "UTC") -> str:
        candidate = str(value or "").strip()
        if not candidate:
            return default
        if candidate.upper() in {"UTC", "GMT", "GMT+00:00", "Z"}:
            return "UTC"
        try:
            return ZoneInfo(candidate).key
        except Exception:
            return default

try:
    from daylily_tapdb.user_store import (  # type: ignore
        get_display_timezone_by_login_or_email as _get_tz_from_user_store,
    )
except Exception:
    _get_tz_from_user_store = None

try:
    from daylily_tapdb.user_store import (  # type: ignore
        set_display_timezone_by_login_or_email as _set_tz_in_user_store,
    )
except Exception:
    _set_tz_in_user_store = None


def list_display_timezone_options() -> list[str]:
    """Return sorted IANA timezone names with UTC pinned first."""
    common = [
        "UTC",
        "America/Los_Angeles",
        "America/Denver",
        "America/Chicago",
        "America/New_York",
        "Europe/London",
        "Europe/Berlin",
        "Asia/Tokyo",
        "Asia/Singapore",
        "Australia/Sydney",
    ]
    all_timezones = sorted(
        tz
        for tz in available_timezones()
        if not tz.startswith(("Etc/", "posix/", "right/"))
    )
    ordered: list[str] = []
    for tz in common + all_timezones:
        normalized = normalize_display_timezone(tz, default=DEFAULT_DISPLAY_TIMEZONE)
        if normalized not in ordered:
            ordered.append(normalized)
    return ordered


def _normalize_identifier(email: str | None) -> str:
    return str(email or "").strip().lower()


def get_display_timezone_for_email(email: str | None) -> str:
    identifier = _normalize_identifier(email)
    if not identifier:
        return DEFAULT_DISPLAY_TIMEZONE
    try:
        backend = TapDBBackend(app_username=identifier)
        with backend.session_scope(commit=False) as session:
            if _get_tz_from_user_store is not None:
                value = _get_tz_from_user_store(session, identifier)
                return normalize_display_timezone(
                    value,
                    default=DEFAULT_DISPLAY_TIMEZONE,
                )
            row = session.execute(
                text(
                    """
                    SELECT gi.json_addl->'preferences'->>'display_timezone' AS display_timezone
                    FROM generic_instance gi
                    WHERE gi.is_deleted = FALSE
                      AND gi.polymorphic_discriminator = 'actor_instance'
                      AND gi.category = 'generic'
                      AND gi.type = 'actor'
                      AND gi.subtype = 'system_user'
                      AND (
                            lower(COALESCE(gi.json_addl->>'login_identifier', '')) = :identifier
                         OR lower(COALESCE(gi.json_addl->>'email', '')) = :identifier
                      )
                    LIMIT 1
                    """
                ),
                {"identifier": identifier},
            ).mappings().first()
            return normalize_display_timezone(
                (row or {}).get("display_timezone"),
                default=DEFAULT_DISPLAY_TIMEZONE,
            )
    except Exception as exc:
        LOGGER.debug("Unable to load display timezone for %s: %s", identifier, exc)
        return DEFAULT_DISPLAY_TIMEZONE


def set_display_timezone_for_email(email: str | None, display_timezone: str | None) -> str:
    identifier = _normalize_identifier(email)
    normalized_tz = normalize_display_timezone(display_timezone, default=DEFAULT_DISPLAY_TIMEZONE)
    if not identifier:
        return normalized_tz
    try:
        backend = TapDBBackend(app_username=identifier)
        with backend.session_scope(commit=True) as session:
            if _set_tz_in_user_store is not None:
                _set_tz_in_user_store(session, identifier, normalized_tz)
                return normalized_tz
            session.execute(
                text(
                    """
                    UPDATE generic_instance gi
                    SET json_addl = jsonb_set(
                            COALESCE(gi.json_addl, '{}'::jsonb),
                            '{preferences,display_timezone}',
                            to_jsonb(CAST(:display_timezone AS text)),
                            TRUE
                        ),
                        modified_dt = NOW()
                    WHERE gi.is_deleted = FALSE
                      AND gi.polymorphic_discriminator = 'actor_instance'
                      AND gi.category = 'generic'
                      AND gi.type = 'actor'
                      AND gi.subtype = 'system_user'
                      AND (
                            lower(COALESCE(gi.json_addl->>'login_identifier', '')) = :identifier
                         OR lower(COALESCE(gi.json_addl->>'email', '')) = :identifier
                      )
                    """
                ),
                {"identifier": identifier, "display_timezone": normalized_tz},
            )
        return normalized_tz
    except Exception as exc:
        LOGGER.debug("Unable to persist display timezone for %s: %s", identifier, exc)
        return normalized_tz
