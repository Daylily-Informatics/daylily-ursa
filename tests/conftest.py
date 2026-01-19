"""Pytest configuration and shared fixtures."""

import os

# Set WHITELIST_DOMAINS=all for tests to allow test@example.com
# This must be set before importing any daylib modules
os.environ.setdefault("WHITELIST_DOMAINS", "all")

