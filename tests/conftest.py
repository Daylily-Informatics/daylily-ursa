"""Pytest configuration and shared fixtures."""

import os
from unittest.mock import MagicMock

import pytest

from daylib.workset_state_db import WorksetStateDB

# Set WHITELIST_DOMAINS=all for tests to allow test@example.com
# This must be set before importing any daylib modules
os.environ.setdefault("WHITELIST_DOMAINS", "all")


@pytest.fixture
def mock_state_db():
    """Shared mock WorksetStateDB for route and app tests."""
    db = MagicMock(spec=WorksetStateDB)
    db.get_workset.return_value = None
    db.list_worksets_by_state.return_value = []
    return db
