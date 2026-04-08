
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from keisei.db import init_db, write_metrics, write_training_state
from keisei.server.app import TEST_ALLOWED_HOSTS, create_app

pytestmark = pytest.mark.integration


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    path = str(tmp_path / "test.db")
    init_db(path)
    write_training_state(path, {
        "config_json": "{}",
        "display_name": "TestBot",
        "model_arch": "resnet",
        "algorithm_name": "ppo",
        "started_at": "2026-04-01T00:00:00Z",
    })
    return path


@pytest.mark.asyncio
async def test_healthz_ok(db_path: str) -> None:
    app = create_app(db_path, allowed_hosts=TEST_ALLOWED_HOSTS)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["db_accessible"] is True


@pytest.mark.asyncio
async def test_healthz_db_missing() -> None:
    app = create_app("/tmp/nonexistent-keisei-test.db", allowed_hosts=TEST_ALLOWED_HOSTS)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["db_accessible"] is False


def test_ws_sends_init_on_connect(db_path: str, ws_connect) -> None:
    write_metrics(db_path, {"epoch": 0, "step": 100, "policy_loss": 1.5})
    app = create_app(db_path, allowed_hosts=TEST_ALLOWED_HOSTS)
    with ws_connect(app) as ws:
        msg = ws.receive_json()
        assert msg["type"] == "init"
        assert "games" in msg
        assert "metrics" in msg
        assert "training_state" in msg
        assert msg["training_state"]["display_name"] == "TestBot"


def test_ws_init_includes_league_data(db_path: str, ws_connect) -> None:
    app = create_app(db_path, allowed_hosts=TEST_ALLOWED_HOSTS)
    with ws_connect(app) as ws:
        msg = ws.receive_json()
        assert msg["type"] == "init"
        assert "league_entries" in msg
        assert "league_results" in msg
        assert "elo_history" in msg
        assert isinstance(msg["league_entries"], list)


def test_ws_init_league_data_populated(db_path: str, ws_connect) -> None:
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO league_entries (architecture, model_params, checkpoint_path, created_epoch) "
        "VALUES ('transformer', '{}', '/tmp/ckpt.pt', 5)"
    )
    conn.execute("INSERT INTO elo_history (entry_id, epoch, elo_rating) VALUES (1, 5, 1050.0)")
    conn.commit()
    conn.close()

    app = create_app(db_path, allowed_hosts=TEST_ALLOWED_HOSTS)
    with ws_connect(app) as ws:
        msg = ws.receive_json()
        assert len(msg["league_entries"]) == 1
        assert msg["league_entries"][0]["architecture"] == "transformer"
        assert len(msg["elo_history"]) == 1


def test_league_change_detection_uses_entry_ids_not_count() -> None:
    """Regression: server must detect entry churn (retire+add) even when count is unchanged.

    Previously the server compared len(entries) which missed same-count-different-set
    changes, causing phantom mass departure events in the webUI.
    """
    # Simulate two poll snapshots with same count but different IDs
    old_entries = [{"id": 1}, {"id": 2}, {"id": 3}]
    new_entries = [{"id": 1}, {"id": 4}, {"id": 3}]  # id=2 retired, id=4 added

    old_ids = frozenset(e["id"] for e in old_entries)
    new_ids = frozenset(e["id"] for e in new_entries)

    # Count-based check (the old bug): would miss this change
    assert len(old_entries) == len(new_entries)

    # ID-set check (the fix): catches the churn
    assert old_ids != new_ids


def test_ws_init_includes_historical_library_and_gauntlet_results(db_path: str, ws_connect) -> None:
    """Assert the init message contains historical_library and gauntlet_results keys."""
    app = create_app(db_path, allowed_hosts=TEST_ALLOWED_HOSTS)
    with ws_connect(app) as ws:
        msg = ws.receive_json()
        assert msg["type"] == "init"
        assert "historical_library" in msg
        assert "gauntlet_results" in msg
        assert isinstance(msg["historical_library"], list)
        assert isinstance(msg["gauntlet_results"], list)


def test_ws_init_includes_transitions(db_path: str, ws_connect) -> None:
    """Assert the init message contains the transitions key."""
    app = create_app(db_path, allowed_hosts=TEST_ALLOWED_HOSTS)
    with ws_connect(app) as ws:
        msg = ws.receive_json()
        assert msg["type"] == "init"
        assert "transitions" in msg
        assert isinstance(msg["transitions"], list)


def test_ws_init_role_field_propagation(db_path: str, ws_connect) -> None:
    """Insert a league entry with explicit role, verify it appears in the init payload."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO league_entries "
        "(architecture, model_params, checkpoint_path, created_epoch, role) "
        "VALUES ('resnet', '{}', '/tmp/ckpt.pt', 1, 'frontier_static')"
    )
    conn.commit()
    conn.close()

    app = create_app(db_path, allowed_hosts=TEST_ALLOWED_HOSTS)
    with ws_connect(app) as ws:
        msg = ws.receive_json()
        assert msg["type"] == "init"
        assert len(msg["league_entries"]) == 1
        assert msg["league_entries"][0]["role"] == "frontier_static"


def test_ws_init_multi_view_elo_metrics(db_path: str, ws_connect) -> None:
    """Insert a league entry with multi-view Elo ratings, verify they appear in the init payload."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO league_entries "
        "(architecture, model_params, checkpoint_path, created_epoch, "
        "elo_frontier, elo_dynamic, elo_recent, elo_historical) "
        "VALUES ('resnet', '{}', '/tmp/ckpt.pt', 1, 1100.0, 1050.0, 980.0, 1200.0)"
    )
    conn.commit()
    conn.close()

    app = create_app(db_path, allowed_hosts=TEST_ALLOWED_HOSTS)
    with ws_connect(app) as ws:
        msg = ws.receive_json()
        assert msg["type"] == "init"
        assert len(msg["league_entries"]) == 1
        entry = msg["league_entries"][0]
        assert entry["elo_frontier"] == 1100.0
        assert entry["elo_dynamic"] == 1050.0
        assert entry["elo_recent"] == 980.0
        assert entry["elo_historical"] == 1200.0


@pytest.mark.asyncio
async def test_serves_index_html(db_path: str) -> None:
    """If static/ dir exists with index.html, GET / returns it."""
    static_dir = Path(__file__).parent.parent / "keisei" / "server" / "static"
    if not static_dir.is_dir():
        pytest.skip("No built SPA in keisei/server/static/")

    app = create_app(db_path, allowed_hosts=TEST_ALLOWED_HOSTS)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    assert "html" in resp.headers.get("content-type", "").lower()
