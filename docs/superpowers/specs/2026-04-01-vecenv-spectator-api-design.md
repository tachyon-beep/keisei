# VecEnv Spectator & Stats API Extensions

**Date:** 2026-04-01
**Status:** Approved
**Scope:** shogi-gym crate (shogi-engine/crates/shogi-gym)

## Summary

Four additions to the shogi-gym Python API that expose spectator data, SFEN access, and episode-level stats from the Rust engine — eliminating the need for Python-side state mirroring and enabling SQLite-based resume capability.

## Change 1: Shared Spectator Dict Builder + VecEnv.get_spectator_data()

### Problem

`SpectatorEnv.to_dict()` produces a rich dict for dashboard rendering, but getting this data for VecEnv's N parallel games requires maintaining N `SpectatorEnv` mirrors in Python — fragile sync-bug territory.

### Design

Extract the dict-building logic from `SpectatorEnv::to_dict()` into a free function in a new module `spectator_data.rs`:

```rust
// shogi-gym/src/spectator_data.rs

/// Build a spectator-format Python dict from a GameState.
/// Omits move_history (caller supplies it if available).
pub fn build_spectator_dict(
    py: Python<'_>,
    game: &GameState,
) -> PyResult<Py<PyDict>>
```

The function lives in `shogi-gym` (not `shogi-core`) because it needs PyO3 types. `shogi-core` stays pure Rust with no Python dependency.

**Refactoring:** `SpectatorEnv::to_dict()` calls `build_spectator_dict()` then appends the `move_history` key.

**New method on VecEnv:**

```rust
/// Return spectator-format dicts for all N games.
pub fn get_spectator_data(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>>
```

Calls `build_spectator_dict()` for each internal `GameState`.

### Dict Shape

```python
{
    "board": [None | {"type": str, "color": str, "promoted": bool, "row": int, "col": int}] * 81,
    "hands": {"black": {"pawn": int, ...}, "white": {"pawn": int, ...}},
    "current_player": "black" | "white",
    "ply": int,
    "is_over": bool,
    "result": "in_progress" | "checkmate" | "repetition" | "perpetual_check" | "impasse" | "max_moves",
    "sfen": str,
    "in_check": bool,
    # move_history: ABSENT from VecEnv output, PRESENT in SpectatorEnv output
}
```

Move history is tracked on the Python side — the training loop already knows which actions it chose and can maintain per-game history lists, resetting on terminated/truncated flags.

## Change 2: SFEN Import for SpectatorEnv

### Problem

If spectator state drifts from the real game (e.g., missed update, dashboard reconnect), there's no recovery path. The dashboard needs a way to re-sync from a known SFEN.

### Design

Add a static constructor to `SpectatorEnv`:

```rust
#[staticmethod]
pub fn from_sfen(sfen: &str, max_ply: Option<u32>) -> PyResult<Self>
```

Constructs a new `SpectatorEnv` with `GameState::from_sfen(sfen, max_ply.unwrap_or(500))`, empty move history, fresh mapper/obs_gen. Invalid SFEN raises Python `ValueError`.

**Why static constructor, not mutating method:** Avoids ambiguity about stale state (move history, mapper). A fresh object is unambiguous.

### Usage

```python
sfen = db.get_latest_sfen(game_id)
env = SpectatorEnv.from_sfen(sfen)
state = env.to_dict()
```

## Change 3: Per-Game SFEN from VecEnv

### Problem

The training loop needs to write SFENs to SQLite for checkpoint/resume capability. Currently there's no way to extract SFENs from VecEnv's internal GameStates.

### Design

Two new methods on `VecEnv`:

```rust
/// Get SFEN for a single game by index.
/// Raises IndexError if game_id >= num_envs.
pub fn get_sfen(&self, game_id: usize) -> PyResult<String>

/// Get SFENs for all games. Returns list[str] of length num_envs.
pub fn get_sfens(&self) -> Vec<String>
```

Both wrap `self.games[i].position.to_sfen()`. No GIL release needed — `to_sfen()` is a pure Rust string operation, fast enough that `py.allow_threads()` overhead would dominate at typical N (512).

### Usage

```python
# Batch checkpoint
sfens = vec_env.get_sfens()
db.write_sfens(step_number, sfens)

# Single game inspection
sfen = vec_env.get_sfen(42)
```

## Change 4: Episode-Level Stats from VecEnv

### Problem

VecEnv tracks `episodes_completed`, `episodes_drawn`, `episodes_truncated`, and `draw_rate()`. Missing: truncation rate and mean episode length. `ply_count` in `StepMetadata` is per-step, not aggregated across episodes.

### Design

One new atomic counter in VecEnv:

```rust
total_episode_ply: AtomicU64  // sum of ply at termination across all completed episodes
```

Accumulated in the auto-reset path (same point where `episodes_completed` is incremented):

```rust
self.total_episode_ply.fetch_add(game.ply as u64, Ordering::Relaxed);
```

Two new computed properties:

```rust
/// Mean episode length across all completed episodes since last reset_stats().
/// Returns 0.0 if no episodes completed yet.
pub fn mean_episode_length(&self) -> f64

/// Fraction of completed episodes that were truncated (hit max_ply).
/// Returns 0.0 if no episodes completed yet.
pub fn truncation_rate(&self) -> f64
```

Where:
- `mean_episode_length = total_episode_ply / episodes_completed`
- `truncation_rate = episodes_truncated / episodes_completed`

`reset_stats()` is extended to also zero out `total_episode_ply`. No separate counter for episode count — reuses `episodes_completed` to avoid shadowing and consistency risk.

### Usage

```python
logger.info(f"Mean ep length: {vec_env.mean_episode_length():.1f}")
logger.info(f"Truncation rate: {vec_env.truncation_rate():.2%}")
vec_env.reset_stats()
```

## Testing Strategy

Each change gets Rust unit tests + Python integration tests:

1. **Shared dict builder:** Assert dict shape, field types, values against known positions. Test edge cases (empty hands, promoted pieces, game-over states). Verify SpectatorEnv.to_dict() and VecEnv.get_spectator_data() produce identical output (minus move_history).
2. **from_sfen:** Round-trip test (to_sfen -> from_sfen -> to_sfen). Invalid SFEN raises ValueError. Verify constructed env is playable (legal_actions, step work).
3. **get_sfen/get_sfens:** Round-trip against known positions. Bounds checking (IndexError for invalid game_id). Verify SFEN changes after step.
4. **Episode stats:** Run games to completion, verify mean_episode_length and truncation_rate against manually tracked values. Verify reset_stats() zeroes everything.

## Files Modified

- `shogi-gym/src/spectator_data.rs` — **NEW** — shared `build_spectator_dict()` function
- `shogi-gym/src/spectator.rs` — refactor `to_dict()` to use shared builder; add `from_sfen()`
- `shogi-gym/src/vec_env.rs` — add `get_spectator_data()`, `get_sfen()`, `get_sfens()`, `total_episode_ply` atomic, `mean_episode_length()`, `truncation_rate()`
- `shogi-gym/src/lib.rs` — add `mod spectator_data`
- `shogi-engine/python/shogi_gym/__init__.py` — no changes needed (methods auto-exposed via PyO3)
- Test files for each change

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Move history in VecEnv | Omit (Python tracks) | Hot path stays clean; training loop already has the actions |
| Episode length tracking | Running stats (sum/count) | Lightweight; no allocation; dashboard only needs mean |
| Dict builder location | Free function in shogi-gym | Needs PyO3 types; keeps shogi-core pure Rust |
| SFEN import API | Static constructor | No stale state ambiguity |
| GIL release for get_sfens | No | to_sfen() is fast; GIL release overhead dominates at N=512 |
