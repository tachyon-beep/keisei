# Code Analysis: `keisei/evaluation/analytics/performance_analyzer.py`

## 1. Purpose & Role

This module provides detailed post-hoc analysis of evaluation results beyond basic win/loss/draw statistics. It computes streak analysis, game length distribution, termination reason counts, performance by color (sente/gote), and performance against different opponent types. It is instantiated by `EvaluationResult.__post_init__` and by `AdvancedAnalytics.generate_automated_report`.

## 2. Interface Contracts

- **`PerformanceAnalyzer.__init__(results: EvaluationResult)`** (lines 22-31): Takes a full `EvaluationResult` object. Extracts `results.games` and `results.summary_stats` as instance attributes.
- **`calculate_win_loss_draw_streaks() -> Dict[str, Any]`** (lines 33-97): Returns dict with keys `max_win_streak`, `max_loss_streak`, `max_draw_streak`, `current_win_streak`, `current_loss_streak`, `current_draw_streak`.
- **`analyze_game_length_distribution(bins) -> Dict[str, Any]`** (lines 99-161): Returns dict with mean, median, std_dev, min, max, and histogram data. Uses `numpy` for histogram computation.
- **`analyze_termination_reasons() -> Dict[str, int]`** (lines 163-178): Returns frequency count of termination reasons from game metadata.
- **`get_performance_by_color() -> Dict[str, Dict[str, float]]`** (lines 180-230): Returns sente and gote performance stats. Depends on `metadata["agent_color"]` being set.
- **`get_performance_vs_opponent_types() -> Dict[str, Dict[str, Any]]`** (lines 232-271): Returns performance grouped by `opponent_info.type`.
- **`run_all_analyses() -> Dict[str, Any]`** (lines 273-294): Orchestrates all analyses into a single dict.

## 3. Correctness Analysis

### Streak Calculation (`calculate_win_loss_draw_streaks`, lines 33-97)

- **Logic correctness:** The algorithm maintains a `current_streaks` dict and a `last_outcome` tracker. When the outcome changes, the completed streak is appended to the appropriate list and all current streaks are reset. This is correct for computing maximum streaks.
- **Edge case -- games with `outcome = None` (lines 61-67):** If a `GameResult` has `winner` set to a value that is neither `0`, `1`, nor `None` (e.g., `2`), all three conditions (`is_agent_win`, `is_opponent_win`, `is_draw`) would be `False`, and `outcome` remains `None`. This would cause `current_streaks[None]` at line 70 to raise a `KeyError` since `None` is not a key in the `current_streaks` dict. However, looking at the `GameResult` definition, `winner` is typed as `Optional[int]` where 0=agent, 1=opponent, None=draw, so this edge case should not occur in practice.
- **First iteration:** On the first game, `last_outcome` is `None` and `outcome` is something like `"win"`. The `if outcome == last_outcome` check (line 69) is `False`, so it goes to the `else` branch. `last_outcome` is `None` which is falsy, so the `if last_outcome:` guard at line 72 correctly skips storing a non-existent previous streak. Then `current_streaks[outcome] = 1` starts the first streak. Correct.

### Game Length Distribution (`analyze_game_length_distribution`, lines 99-161)

- **Default bins (lines 131-145):** The default bins go up to 500. Games with `moves_count > 500` would fall outside the last bin and be excluded from the histogram count. The mean/median/min/max calculations (lines 155-159) use the raw `game_lengths` list, so they correctly include all data points. Only the histogram would miss outliers. This is a minor data presentation issue.
- **Bins check (line 130):** `if not bins:` -- this also catches `bins=[]` (empty list), which would be passed to `np.histogram` and would likely raise a numpy error. The `if not bins:` check assigns defaults, which prevents this.
- **numpy types in return (lines 147-160):** `np.histogram` returns `hist` as a numpy array of `np.int64`. Line 152 explicitly converts each count to `int(hist[i])`, which avoids JSON serialization issues for the histogram. However, `np.mean`, `np.median`, `np.std` at lines 155-157 return `np.float64`, which are wrapped in `float()` calls. These conversions are correct.

### Performance by Color (`get_performance_by_color`, lines 180-230)

- **Metadata dependency (lines 193-196):** Relies on `game.metadata.get("agent_color")` being either `"sente"` or `"gote"`. If this metadata field is missing or has a different value, those games are simply excluded from both lists. This is silently lossy -- a game without color metadata is not counted in either category.
- **Unused parameter (line 199):** The inner function `calculate_color_stats` takes `color_name` as a parameter but never uses it.

### Performance vs Opponent Types (`get_performance_vs_opponent_types`, lines 232-271)

- **Attribute access (line 247):** Accesses `game.opponent_info.type` directly. If `opponent_info` is `None`, this would raise `AttributeError`. Looking at `GameResult`, `opponent_info` is a required field (not Optional), so this is safe.

### `run_all_analyses` (`lines 273-294`)

- **`summary_stats.to_dict()` (line 283):** Calls `to_dict()` on the `SummaryStats` object. This method is defined in `evaluation_result.py` and returns a plain dict. Correct.

## 4. Robustness & Error Handling

- **Empty game list handling:** Every analysis method checks `if not self.games:` and returns an appropriate empty/zero result. This is thorough and consistent (lines 46, 118, 170).
- **No try/except blocks:** The module relies on well-typed inputs. Given that `PerformanceAnalyzer` is only instantiated internally (by `EvaluationResult` or `AdvancedAnalytics`), this is acceptable.
- **No logging:** The module does not use the logging framework at all. Errors or unusual conditions (e.g., missing metadata fields) are handled silently.

## 5. Performance & Scalability

- **All analyses are O(n)** where n is the number of games, except:
  - `np.histogram` is O(n * log(bins)) but bins is small (12 default), so effectively O(n).
  - `np.mean`, `np.median`, `np.std` are each O(n).
- **`run_all_analyses` iterates over `self.games` multiple times** (once per analysis method), so the effective cost is ~5-6 passes over the game list. For typical evaluation sizes (hundreds to thousands of games), this is negligible.
- **No caching:** `run_all_analyses` recomputes everything on each call. However, `EvaluationResult.calculate_analytics` does cache the result in `analytics_data`, so the analyzer itself is typically called only once.

## 6. Security & Safety

- **No file I/O, no network access, no dynamic code execution.** Pure computation on in-memory data.
- **numpy dependency:** The module imports numpy at the module level (line 11). If numpy is unavailable, the import fails immediately, unlike the conditional scipy import in `advanced_analytics.py`. Numpy is listed as a core dependency, so this should always be available.

## 7. Maintainability

- At 391 lines, approximately half is the `__main__` example/testing block (lines 298-391). The actual analyzer logic is ~297 lines.
- The `__main__` block at lines 298-391 references `EvaluationContext` which is not imported at the top of the file. It would fail at runtime if executed directly (line 362: `mock_context = EvaluationContext(...)` -- `EvaluationContext` is not in scope). This dead code would produce a `NameError`.
- The inner function `calculate_color_stats` (lines 198-225) takes an unused `color_name` parameter.
- Return type annotations use `Dict[str, Any]` throughout, which is accurate but loses the structured nature of the return values. Dedicated dataclasses or TypedDicts would provide stronger contracts.

## 8. Verdict

**SOUND**

The analysis algorithms are correct and well-guarded against empty inputs. The main observations are: (1) games without `agent_color` metadata are silently excluded from color-based analysis; (2) the `__main__` block contains broken code (missing `EvaluationContext` import) that would fail if run directly; (3) an unused parameter in `calculate_color_stats`. None of these affect production correctness since the `__main__` block is never imported, and the metadata dependency is a documented convention.
