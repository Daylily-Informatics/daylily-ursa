"""Pytest configuration and shared fixtures."""

import os

# Set URSA_AUTH__WHITELIST_DOMAINS=all for tests to allow test@example.com
# This must be set before importing any daylib_ursa modules
os.environ.setdefault("URSA_AUTH__WHITELIST_DOMAINS", "all")
