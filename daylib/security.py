"""Security utilities for input sanitization and validation.

Provides functions to sanitize user input before logging to prevent
log injection attacks.
"""

import re
from typing import Any

# Control characters (0x00-0x1f except tab, and 0x7f-0x9f) that could be
# used for log injection or terminal escape sequences.
_LOG_UNSAFE_PATTERN = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')


def sanitize_for_log(value: Any, max_length: int = 200) -> str:
    """Sanitize a value for safe inclusion in log messages.

    Removes control characters, escapes newlines, and truncates long values
    to prevent log injection attacks where malicious input could forge
    log entries or inject terminal escape sequences.

    Args:
        value: The value to sanitize (converted to str if not already)
        max_length: Maximum length before truncation (default 200)

    Returns:
        Sanitized string safe for logging
    """
    if value is None:
        return "<none>"

    s = str(value)

    # Replace newlines with visible escape sequences
    s = s.replace('\r\n', '\\r\\n').replace('\n', '\\n').replace('\r', '\\r')

    # Replace other control characters with hex representation
    s = _LOG_UNSAFE_PATTERN.sub(lambda m: f'\\x{ord(m.group(0)):02x}', s)

    # Truncate if too long
    if len(s) > max_length:
        s = s[:max_length] + "..."

    return s

