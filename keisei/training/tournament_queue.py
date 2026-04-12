"""Database operations for the tournament pairing queue and worker heartbeat.

All operations use atomic SQL patterns (conditional UPDATE, BEGIN IMMEDIATE)
to guarantee correctness under concurrent workers sharing the same SQLite
database in WAL mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from keisei.db import _connect


@dataclass(frozen=True)
class ClaimedPairing:
    id: int
    round_id: int
    entry_a_id: int
    entry_b_id: int
    games_target: int
    worker_id: str
    status: str
    enqueued_epoch: int


@dataclass(frozen=True)
class WorkerHealth:
    worker_id: str
    pid: int
    device: str
    last_seen: str
    pairings_done: int


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Pairing queue ────────────────────────────────────────────────────


def enqueue_pairings(
    db_path: str,
    *,
    round_id: int,
    epoch: int,
    pairings: list[tuple[int, int, int]],
    priorities: list[float] | None = None,
) -> None:
    """Insert a batch of pairings for a round.

    Each pairing is ``(entry_a_id, entry_b_id, games_target)``.
    ``priorities`` is optional; when provided it must have the same length
    as ``pairings`` and governs claim order (highest first).
    """
    if not pairings:
        return
    if priorities is not None and len(priorities) != len(pairings):
        raise ValueError("priorities length must match pairings length")
    conn = _connect(db_path)
    try:
        with conn:
            conn.executemany(
                "INSERT INTO tournament_pairing_queue "
                "(round_id, entry_a_id, entry_b_id, games_target, "
                "priority, enqueued_epoch, status) "
                "VALUES (?, ?, ?, ?, ?, ?, 'pending')",
                [
                    (
                        round_id, a, b, gt,
                        (priorities[i] if priorities is not None else 0.0),
                        epoch,
                    )
                    for i, (a, b, gt) in enumerate(pairings)
                ],
            )
    finally:
        conn.close()


def claim_next_pairing(
    db_path: str, *, worker_id: str,
) -> ClaimedPairing | None:
    """Atomically claim the highest-priority pending pairing.

    Uses BEGIN IMMEDIATE to serialize the SELECT + UPDATE against other
    workers under WAL. Returns None if no pending pairings remain.
    """
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT id FROM tournament_pairing_queue "
                "WHERE status = 'pending' "
                "ORDER BY priority DESC, id ASC LIMIT 1"
            ).fetchone()
            if row is None:
                conn.execute("ROLLBACK")
                return None
            now = _now_iso()
            conn.execute(
                "UPDATE tournament_pairing_queue "
                "SET status = 'playing', worker_id = ?, claimed_at = ? "
                "WHERE id = ? AND status = 'pending'",
                (worker_id, now, row["id"]),
            )
            full = conn.execute(
                "SELECT * FROM tournament_pairing_queue WHERE id = ?",
                (row["id"],),
            ).fetchone()
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return ClaimedPairing(
            id=full["id"], round_id=full["round_id"],
            entry_a_id=full["entry_a_id"], entry_b_id=full["entry_b_id"],
            games_target=full["games_target"], worker_id=full["worker_id"],
            status=full["status"], enqueued_epoch=full["enqueued_epoch"],
        )
    finally:
        conn.close()


def mark_pairing_done(
    db_path: str, pairing_id: int, *, status: str = "done",
) -> None:
    """Mark a pairing as 'done', 'failed', or 'expired'. Sets completed_at."""
    if status not in ("done", "failed", "expired"):
        raise ValueError(f"invalid status: {status!r}")
    now = _now_iso()
    conn = _connect(db_path)
    try:
        with conn:
            conn.execute(
                "UPDATE tournament_pairing_queue "
                "SET status = ?, completed_at = ? WHERE id = ?",
                (status, now, pairing_id),
            )
    finally:
        conn.close()


def get_round_status(db_path: str, round_id: int) -> dict[str, int]:
    """Return {status: count} for a given round."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT status, COUNT(*) as n FROM tournament_pairing_queue "
            "WHERE round_id = ? GROUP BY status",
            (round_id,),
        ).fetchall()
        return {r["status"]: r["n"] for r in rows}
    finally:
        conn.close()


def get_active_queue_depth(db_path: str) -> int:
    """Count pending + playing pairings across all rounds."""
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) as n FROM tournament_pairing_queue "
            "WHERE status IN ('pending', 'playing')"
        ).fetchone()
        return int(row["n"])
    finally:
        conn.close()


def reset_stale_playing(
    db_path: str, *, worker_id: str | None = None,
) -> int:
    """Reset 'playing' pairings back to 'pending' for crash recovery.

    If ``worker_id`` is given, only reset pairings claimed by that worker.
    Otherwise reset ALL playing pairings (used when recovering from unknown
    worker crashes). Returns the number of rows reset.
    """
    conn = _connect(db_path)
    try:
        with conn:
            if worker_id is not None:
                cur = conn.execute(
                    "UPDATE tournament_pairing_queue "
                    "SET status = 'pending', worker_id = NULL, claimed_at = NULL "
                    "WHERE status = 'playing' AND worker_id = ?",
                    (worker_id,),
                )
            else:
                cur = conn.execute(
                    "UPDATE tournament_pairing_queue "
                    "SET status = 'pending', worker_id = NULL, claimed_at = NULL "
                    "WHERE status = 'playing'",
                )
            return cur.rowcount
    finally:
        conn.close()


# ── Worker heartbeat ─────────────────────────────────────────────────


def write_worker_heartbeat(
    db_path: str, *,
    worker_id: str, pid: int, device: str, pairings_done: int = 0,
) -> None:
    """Upsert a worker heartbeat row. Sets last_seen to now."""
    now = _now_iso()
    conn = _connect(db_path)
    try:
        with conn:
            conn.execute(
                "INSERT INTO tournament_worker_heartbeat "
                "(worker_id, pid, device, last_seen, pairings_done) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(worker_id) DO UPDATE SET "
                "pid = excluded.pid, device = excluded.device, "
                "last_seen = excluded.last_seen, "
                "pairings_done = excluded.pairings_done",
                (worker_id, pid, device, now, pairings_done),
            )
    finally:
        conn.close()


def get_worker_health(
    db_path: str, *, stale_after_seconds: int = 60,
) -> list[WorkerHealth]:
    """Return workers whose last_seen is newer than now - stale_after_seconds."""
    conn = _connect(db_path)
    try:
        cutoff = datetime.now(timezone.utc).timestamp() - stale_after_seconds
        rows = conn.execute(
            "SELECT * FROM tournament_worker_heartbeat"
        ).fetchall()
        alive = []
        for r in rows:
            try:
                last = datetime.fromisoformat(r["last_seen"]).timestamp()
            except ValueError:
                continue
            if last >= cutoff:
                alive.append(WorkerHealth(
                    worker_id=r["worker_id"], pid=r["pid"],
                    device=r["device"], last_seen=r["last_seen"],
                    pairings_done=r["pairings_done"],
                ))
        return alive
    finally:
        conn.close()


# ── DynamicTrainer single-writer claim ───────────────────────────────


def claim_dynamic_update(
    db_path: str, *, entry_id: int, worker_id: str,
) -> bool:
    """Atomically claim the Dynamic update slot for an entry.

    Returns True if this worker successfully claimed the slot, False if
    another worker already holds it.
    """
    conn = _connect(db_path)
    try:
        with conn:
            cur = conn.execute(
                "UPDATE league_entries SET dynamic_update_worker = ? "
                "WHERE id = ? AND dynamic_update_worker IS NULL",
                (worker_id, entry_id),
            )
            return cur.rowcount == 1
    finally:
        conn.close()


def release_dynamic_update(
    db_path: str, *, entry_id: int, worker_id: str,
) -> None:
    """Release the Dynamic update claim. No-op if held by a different worker."""
    conn = _connect(db_path)
    try:
        with conn:
            conn.execute(
                "UPDATE league_entries SET dynamic_update_worker = NULL "
                "WHERE id = ? AND dynamic_update_worker = ?",
                (entry_id, worker_id),
            )
    finally:
        conn.close()
