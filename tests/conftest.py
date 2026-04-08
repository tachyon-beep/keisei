"""Shared fixtures for Keisei test suite."""

from __future__ import annotations

import contextlib
from concurrent.futures import CancelledError
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from keisei.db import init_db


# ---------------------------------------------------------------------------
# WebSocket test helper
# ---------------------------------------------------------------------------


@pytest.fixture
def ws_connect():
    """Factory fixture: returns a context manager that tolerates CancelledError.

    Starlette's sync TestClient drives the ASGI event loop in a background
    thread.  When the test exits the `with` block, the WebSocket close races
    with `asyncio.to_thread` Futures in the server's TaskGroup.  A cancelled
    Future raises `concurrent.futures.CancelledError` during ExitStack
    teardown — this is harmless and expected.
    """
    @contextlib.contextmanager
    def _connect(app, path="/ws"):
        client = TestClient(app, raise_server_exceptions=False)
        try:
            with client.websocket_connect(path) as ws:
                yield ws
        except CancelledError:
            pass

    return _connect

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
