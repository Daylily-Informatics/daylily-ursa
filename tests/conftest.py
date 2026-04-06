"""Pytest configuration and shared fixtures."""

import os

# Set WHITELIST_DOMAINS to the default base allowlist for tests.
# This must be set before importing any daylib_ursa modules
os.environ.setdefault(
    "WHITELIST_DOMAINS",
    "lsmc.com,lsmc.bio,lsmc.life,daylilyinformatics.com",
)
