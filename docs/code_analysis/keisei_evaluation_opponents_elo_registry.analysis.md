# Code Analysis: `keisei/evaluation/opponents/elo_registry.py`

**Analyzed:** 2026-02-07
**Lines:** 131
**Package:** Evaluation -- Opponents (Package 18)

---

## 1. Purpose & Role

This module provides a simple Elo rating system for tracking opponent strength. It persists ratings to a JSON file and supports loading, saving, querying, and batch-updating ratings based on match results. It is used directly by `OpponentPool` (which holds an optional `EloRegistry` instance) to track Elo ratings for checkpoint-based opponents.

## 2. Interface Contracts

- **Constructor (`__init__`, lines 18-33):** Requires a `file_path: Path`, optional `initial_rating` (default 1500.0), optional `k_factor` (default 32.0). Immediately calls `self.load()` on construction.
- **`load()` (lines 35-47):** Loads ratings from `self.file_path` if it exists. Silently initializes to empty dict on any exception.
- **`save()` (lines 49-64):** Writes ratings and metadata to `self.file_path`. Creates parent directories. Logs error on failure but does not raise.
- **`get_rating(player_id)` (lines 66-70):** Returns current rating, auto-creating with `initial_rating` if absent. Note: this has a side effect of inserting the player into `self.ratings`.
- **`set_rating(player_id, rating)` (lines 72-74):** Direct setter, no validation.
- **`update_ratings(player1_id, player2_id, results)` (lines 76-122):** Batch Elo update from a list of result strings. Expected values: `"agent_win"`, `"opponent_win"`, `"draw"`.
- **`get_all_ratings()` (lines 124-126):** Returns a shallow copy of the ratings dict.
- **`get_top_players(limit)` (lines 128-131):** Returns top `limit` players sorted by rating descending.

## 3. Correctness Analysis

**Elo calculation (lines 90-117):**

The implementation normalizes scores by dividing by `len(results)` (lines 109-110) before applying the K-factor update. This means the Elo update per batch is bounded to a single K-factor adjustment regardless of how many games are in the batch. For example, 100 games with 100% win rate produces the same update as 1 game with a win. This is a non-standard Elo implementation -- standard Elo applies K per game, not per batch.

Specifically:
- Lines 94-100: Raw score accumulation is correct (1.0 for win, 0.5 for draw, 0.0 for loss).
- Line 102: `score2 = len(results) - score1` is correct for the raw total.
- Lines 109-110: Normalization divides by `len(results)`, converting to a fraction.
- Lines 113-114: The update uses `k_factor * (actual - expected)` where `actual` is the normalized fraction.

This means the K-factor effectively scales inversely with batch size. A batch of 10 games produces 1/10th the update per game compared to processing each game individually. This is mathematically coherent but differs from what most Elo implementations do. It may not be a bug if the intent is to limit rating volatility per evaluation session, but it is worth noting.

**Side-effect in `get_rating` (line 69):** Calling `get_rating` for a nonexistent player silently creates an entry. This is used intentionally by `OpponentPool.add_checkpoint()` (line 53 of `opponent_pool.py`) to ensure a rating entry exists, but could cause unintended state mutation if called in read-only contexts (e.g., reporting).

**Empty results guard (line 87-88):** Correctly short-circuits on empty results list, preventing division by zero on line 109.

## 4. Robustness & Error Handling

- **Load failure (lines 45-47):** Any exception during load resets to empty dict and logs a warning. This is resilient but means corrupted JSON files are silently discarded.
- **Save failure (lines 63-64):** Logs error but does not raise. Callers have no way to know that persistence failed.
- **No file locking:** Concurrent processes writing to the same ratings file could corrupt it. The save operation is not atomic (no write-to-temp-then-rename pattern).
- **`set_rating` has no validation (line 74):** Accepts any float, including negative, NaN, or infinity values.
- **`get_top_players` (line 128):** The return type annotation uses `List[tuple[str, float]]`, mixing old-style `List` from typing (line 10) with new-style `tuple` lowercase. This works in Python 3.9+ but is stylistically inconsistent with the rest of the type hints in the file which use `Dict`, `List`, `Optional` from typing.

## 5. Performance & Scalability

- **In-memory dict:** All ratings are held in a plain dict. This is efficient for reasonable pool sizes (hundreds to low thousands of opponents).
- **JSON serialization:** Full serialize/deserialize on every load/save. Adequate for small-to-medium rating sets but would become slow with millions of entries.
- **`update_ratings` iterates `results` list:** O(n) in the number of game results per batch. No performance concern.

## 6. Security & Safety

- **File path injection:** The `file_path` parameter is a `Path` object from the caller. No user-controlled input reaches the file system directly in this module. However, if an untrusted caller could control `file_path`, arbitrary file read/write would be possible.
- **JSON deserialization:** `json.load()` on line 40 is safe against code injection (unlike pickle). The only risk is malformed JSON causing parse errors, which is caught by the broad `except Exception`.

## 7. Maintainability

- **Clear, focused module:** 131 lines, single class, single responsibility. Easy to understand.
- **Docstrings:** Every public method has a docstring.
- **Logging:** Uses the standard `logging` module consistently.
- **Duplicate Elo system:** This module coexists with `keisei/evaluation/analytics/elo_tracker.py`, which is a separate Elo tracking system used by `EvaluationResult`. Having two independent Elo implementations in the same project increases maintenance burden and risk of divergence.
- **Legacy typing imports (line 10):** `Dict`, `List`, `Optional` from `typing` are used despite the file also using lowercase `tuple` in the return type on line 128.

## 8. Verdict

**NEEDS_ATTENTION**

The module is functionally correct for its immediate use case but has a non-standard Elo batch normalization that may produce unexpected rating behavior when batch sizes vary. The lack of atomic file writes and the duplicate Elo implementation elsewhere in the codebase are additional concerns. The side-effecting `get_rating` method could cause unintended state mutation.
