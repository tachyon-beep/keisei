"""Shared fixtures for Keisei test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from keisei.db import init_db


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Path for a temporary SQLite database (not yet initialised)."""
    return tmp_path / "test.db"


@pytest.fixture
def db(db_path: Path) -> Path:
    """An initialised temporary database."""
    init_db(str(db_path))
    return db_path
