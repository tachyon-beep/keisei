# Code Analysis: `keisei/evaluation/opponents/enhanced_manager.py`

**Analyzed:** 2026-02-07
**Lines:** 608
**Package:** Evaluation -- Opponents (Package 18)

---

## 1. Purpose & Role

This module implements an advanced opponent selection and performance tracking system for the evaluation subsystem. It provides five selection strategies (random, Elo-based, adaptive difficulty, curriculum learning, diversity maximizing), historical per-opponent performance tracking with trend analysis, adaptive difficulty adjustment, and curriculum progression logic. The `EnhancedOpponentManager` class is the central coordinator, while `OpponentPerformanceData` is the per-opponent data record and `SelectionStrategy` is the enumeration of available strategies.

## 2. Interface Contracts

### `SelectionStrategy` (Enum, lines 26-33)
Five named strategies: `RANDOM`, `ELO_BASED`, `ADAPTIVE_DIFFICULTY`, `CURRICULUM_LEARNING`, `DIVERSITY_MAXIMIZING`.

### `OpponentPerformanceData` (dataclass, lines 37-126)
- **Fields:** `opponent_name`, game counts, timestamps, Elo rating, difficulty, recent win rates (list), performance trend (string), tags (set).
- **`win_rate_against` property (lines 58-63):** Returns 0.0 if no games played; otherwise `wins / total_games`.
- **`games_since_last_played` property (lines 66-71):** Returns `float("inf")` if never played. Otherwise computes `days_since * 10` as a rough approximation.
- **`update_with_result(game_result)` (lines 73-101):** Updates all tracking fields from a `GameResult` object. Maintains a rolling window of the last 10 win/loss outcomes.
- **`_update_performance_trend()` (lines 103-126):** Linear regression on recent win rates to classify trend.

### `EnhancedOpponentManager` (class, lines 129-608)
- **Constructor (lines 141-169):** Takes optional file path, target win rate, adaptation rate, curriculum threshold. Loads existing data on construction.
- **`register_opponents(opponents)` (lines 171-184):** Registers a list of `OpponentInfo` objects and initializes performance data for new ones.
- **`select_opponent(...)` (lines 186-228):** Main selection entry point. Accepts optional strategy override, win rate, recency exclusion, diversity factor.
- **`update_performance(game_result)` (lines 230-251):** Updates tracking data, adjusts difficulty, checks curriculum, and probabilistically saves.
- **`get_opponent_statistics()` (lines 252-296):** Returns comprehensive statistics dictionary.
- **Private methods:** `_select_by_elo`, `_select_adaptive_difficulty`, `_select_curriculum_learning`, `_select_diversity_maximizing`, `_filter_recent_opponents`, `_adjust_difficulty_levels`, `_check_curriculum_progression`, `_estimate_initial_difficulty`, `_win_rate_to_elo`, `_weighted_random_choice`, `_group_opponents_by_difficulty`, `_load_opponent_data`, `_save_opponent_data`.

## 3. Correctness Analysis

### `games_since_last_played` type error (line 69)
The property has return type annotation `int` (inferred from the dataclass), but when `self.last_played is None`, it returns `float("inf")` (line 69). This is a **type inconsistency** -- `float("inf")` is not an `int`. Callers comparing this value with an int threshold (e.g., line 433: `data.games_since_last_played > recent_threshold` where `recent_threshold=5`) will work due to Python's numeric comparison rules, but the type contract is violated.

### Recency filter logic inversion (lines 350-352)
In `_select_adaptive_difficulty`, the recency bonus calculation on line 351 is:
```python
recency_bonus = 1.0 / (1.0 + data.games_since_last_played / 10.0)
```
This gives a *higher* bonus to *recently* played opponents (small `games_since_last_played` = larger fraction). The comment says "Bonus for less recently played opponents" but the math awards a bonus for *more* recently played opponents. Line 352 then multiplies the selection score by `(1.0 + recency_bonus * 0.3)`, boosting recently-played opponents. This is **inverted from the stated intent**.

### Difficulty adjustment direction (lines 437-456)
When the agent wins, difficulty is *increased* (line 448-450). When the agent loses, difficulty is *decreased* (line 452-455). This is correct from the perspective of estimating the opponent's relative challenge level -- if the agent beats them, they are "easier" for the agent, so the difficulty rating should go up to reflect that the agent has improved. However, this creates a semantic ambiguity: `difficulty_level` is being used both as "how hard is this opponent inherently" and "how should we rank this opponent for selection." The adaptive difficulty selector (line 347-348) then tries to match opponents to a target difficulty, which makes the whole system internally consistent even if the variable naming is confusing.

### Elo rating never updated (line 47 of dataclass, entire module)
`OpponentPerformanceData.elo_rating` defaults to 1200.0 (line 47) and is loaded/saved from persistence (lines 561, 589). However, **no code in this module ever updates `elo_rating`** on the dataclass. The `_select_by_elo` method (line 313) reads `data.elo_rating` but there is no Elo update logic anywhere in `EnhancedOpponentManager`. The `_adjust_difficulty_levels` method adjusts `difficulty_level` but not `elo_rating`. This means the Elo-based selection strategy operates on stale initial values (always 1200.0 for all opponents unless loaded from a pre-existing data file that was manually edited).

### Curriculum progression aggregation (lines 464-471)
The `_check_curriculum_progression` method aggregates `recent_win_rates` lists from *all* opponents into a single flat list (lines 465-466), then takes the last 20 entries (line 471). Because the iteration order of `self.opponent_data.values()` is insertion-order (Python 3.7+), the "last 20" entries come from the last opponents in insertion order, not the 20 most recent games chronologically. This could produce misleading curriculum progression decisions.

### Linear regression denominator (line 119)
In `_update_performance_trend`, the slope calculation is:
```python
slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
```
When `len(recent_win_rates)` is 5+ (guard at line 105), and x = [0, 1, 2, ...], the denominator `n * sum_x2 - sum_x^2` is always positive for n >= 2 with distinct x values. Division by zero is not possible here. The calculation is correct.

### `_weighted_random_choice` edge case (lines 507-529)
If all weights are zero, the method falls back to `random.choice` (line 519). If weights contain negative values, `total_weight` could be positive but the cumulative scan could skip entries unpredictably. However, looking at all callers, scores are always >= 0 (constructed from divisions and multiplications of positive values), so negative weights do not occur in practice.

### Probabilistic save (line 249)
`update_performance` saves opponent data with only 10% probability (`random.random() < 0.1`). This means up to 90% of updates may be lost on crash. There is no explicit `save()` call at shutdown or end-of-session.

## 4. Robustness & Error Handling

- **Load failure (lines 575-576):** Broad `except Exception` catches all errors during data loading. The error is logged but the manager proceeds with empty data. This is resilient but means corrupted files are silently discarded.
- **Save failure (lines 607-608):** Same pattern -- logged but not raised. No retry logic.
- **No file locking:** The JSON data file can be corrupted by concurrent writes.
- **`_save_opponent_data` (line 602):** Does not open with `encoding="utf-8"`, unlike `_load_opponent_data` which also does not specify encoding. Both rely on the system default encoding, which is usually UTF-8 on modern systems but not guaranteed.
- **Missing `opponent_info` attribute:** `update_performance` (line 232) accesses `game_result.opponent_info.name` without checking if `opponent_info` is None. The `GameResult` dataclass makes `opponent_info` a required field, so this is safe as long as callers follow the contract.
- **numpy dependency:** The module imports `numpy` unconditionally on line 19. If numpy is unavailable, the entire module fails to import, which is caught by the conditional import in `__init__.py`.

## 5. Performance & Scalability

- **O(n) selection:** All selection strategies iterate the full candidate list once. The diversity-maximizing strategy (lines 390-424) has an inner loop computing `sum(d.total_games for d in self.opponent_data.values())` and type frequency per candidate, making it O(n*m) where m is the total opponent count. For typical evaluation pools (tens of opponents), this is negligible.
- **In-memory data:** All opponent performance data is held in a dict. Suitable for pools of hundreds of opponents.
- **`np.mean()` calls (lines 266, 272, 471):** Numpy is imported for just `np.mean()`, which could be replaced with `statistics.mean()` or a simple `sum/len`. This is a heavyweight dependency for minimal usage.
- **Probabilistic save:** The 10% save rate (line 249) reduces I/O overhead but introduces data loss risk.

## 6. Security & Safety

- **File path default (line 148-149):** If no `opponent_data_file` is provided, defaults to `Path("opponent_performance_data.json")` in the current working directory. This is a relative path, meaning the data file location depends on the process's CWD, which could be unexpected.
- **JSON serialization only:** No pickle or exec-based deserialization. Safe against code injection.
- **`datetime.fromisoformat` (line 569):** Used during loading. Malformed date strings would raise `ValueError`, which is caught by the broad exception handler on line 575.

## 7. Maintainability

- **File length:** At 608 lines, this module is substantial but not excessively long. It contains one enum, one dataclass, and one main class. The class has 15 methods.
- **Separation of concerns:** The module mixes opponent selection, performance tracking, difficulty adaptation, curriculum management, and data persistence into a single class. Each of these could be a separate component.
- **Dead Elo tracking:** The `elo_rating` field on `OpponentPerformanceData` and the `_select_by_elo` method exist but the Elo rating is never actually updated, making the Elo-based selection strategy non-functional in practice.
- **Overlap with other modules:** There are three separate systems for tracking opponent strength in the evaluation package: this module's difficulty/Elo tracking, `elo_registry.py`'s `EloRegistry`, and `analytics/elo_tracker.py`. This fragmentation increases cognitive load.
- **Hardcoded constants:** Magic numbers appear throughout: 0.1 (EMA alpha, line 90), 10 (recent window size, line 97), 0.05 (trend threshold, lines 121/123), 200.0 (Elo preference range, line 316), 5 (recency threshold, line 427), 0.1 (save probability, line 249), 20 (curriculum sample size, line 468).
- **Type hints:** Uses both old-style (`Dict`, `List`, `Optional`, `Set`, `Tuple` from typing) and imports `Any`. Consistent within itself but could use modern syntax.

## 8. Verdict

**NEEDS_ATTENTION**

The module has several notable issues:
1. The recency bonus logic in `_select_adaptive_difficulty` (line 351) is inverted from its stated intent.
2. The `elo_rating` field is never updated, making Elo-based selection non-functional.
3. The `games_since_last_played` property returns `float("inf")` despite an implicit `int` return type.
4. Curriculum progression aggregates win rates in insertion order rather than chronological order.
5. The 10% probabilistic save with no shutdown hook risks significant data loss.

None of these are crash-inducing bugs in the current codebase (the module appears to be unused or lightly used based on the overall architecture), but they would produce incorrect behavior if the enhanced opponent management features were actively relied upon.
