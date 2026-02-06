# Code Analysis: `keisei/evaluation/analytics/advanced_analytics.py`

## 1. Purpose & Role

This module implements a statistical analytics pipeline for evaluation results. It provides four main capabilities: performance comparison between result sets using hypothesis testing (two-proportion z-test, Mann-Whitney U test), trend analysis over time using linear regression, confidence interval calculation for win rate differences, and automated report generation with insights. It depends on `scipy.stats` for statistical computations.

## 2. Interface Contracts

- **`AdvancedAnalytics.__init__(significance_level, min_practical_difference, trend_window_days)`** (lines 77-93): Constructor with validation. Raises `ValueError` for out-of-range parameters.
- **`compare_performance(baseline_results, comparison_results, ...)`** (lines 95-167): Accepts two lists of `GameResult`, returns `PerformanceComparison` dataclass. Accesses `game.is_agent_win` and `game.moves_count` on each result.
- **`analyze_trends(historical_results, metric)`** (lines 169-243): Accepts a list of `(datetime, EvaluationResult)` tuples. Accesses `result.summary_stats.win_rate`, `result.summary_stats.avg_game_length`, `result.summary_stats.total_games`. Returns `TrendAnalysis` dataclass.
- **`generate_automated_report(current_results, baseline_results, historical_data, output_file)`** (lines 427-523): Orchestrates all analyses and produces a dict report, optionally saved as JSON.
- **Dataclasses exported:** `StatisticalTest` (lines 25-34), `TrendAnalysis` (lines 38-48), `PerformanceComparison` (lines 52-62).

## 3. Correctness Analysis

### Two-Proportion Z-Test (`_two_proportion_z_test`, lines 245-293)

- **Pooled proportion formula (line 263):** `p_pool = (x1 + x2) / (n1 + n2)` -- correct for the standard pooled two-proportion z-test.
- **Standard error (line 266):** `se = sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))` -- correct.
- **Z-statistic (line 272):** `z_stat = (p2 - p1) / se` -- correct (tests direction: comparison minus baseline).
- **P-value (line 275):** `2 * (1 - norm.cdf(abs(z_stat)))` -- correct two-tailed p-value.
- **Cohen's h effect size (line 281):** `2 * (asin(sqrt(p2)) - asin(sqrt(p1)))` -- correct formula for Cohen's h.
- **Potential math domain error (line 281):** If `x1 > n1` or `x2 > n2` (which should not happen with valid game data but is not validated), `p1` or `p2` could exceed 1.0, causing `math.sqrt` to receive a value > 1 which is fine, but then `math.asin` would receive a value > 1, raising `ValueError`. There is no guard against this.
- **Division by zero when `p_pool` is 0 or 1 (line 266):** If all games in both sets have the same outcome (all wins or all losses), `p_pool` is 0 or 1, making `se = 0`. The code handles this at line 269-270 by setting `z_stat = 0.0`, which is correct behavior.

### Mann-Whitney U Test (`_mann_whitney_test`, lines 295-334)

- **Minimum sample check (line 302):** Requires at least 3 samples per group; returns a non-significant result otherwise. This is reasonable.
- **Exception handling (lines 317-320):** Catches `ValueError` and `TypeError` from scipy, logs the error, and returns a safe fallback. This is correct.
- **Missing effect size:** The Mann-Whitney test does not compute or report an effect size (rank-biserial correlation), unlike the z-test. The `StatisticalTest.effect_size` field remains `None`.

### Confidence Interval (`_calculate_win_rate_difference_ci`, lines 336-367)

- **Wald interval method:** Uses `se = sqrt(p*(1-p)/n)` for each proportion separately, then combines. This is the standard Wald confidence interval for the difference of two proportions.
- **Edge case (line 348-349):** Returns `(0.0, 0.0)` when either sample is empty, which is a degenerate but non-crashing result.

### Trend Analysis (`analyze_trends`, lines 169-243)

- **Minimum data check (line 177):** Requires at least 3 data points, returning an "insufficient_data" result otherwise. Correct.
- **Linear regression (line 208):** Uses `scipy_stats.linregress` which is appropriate for simple trend detection.
- **Trend threshold (line 215):** Uses `1e-6` as the slope threshold for "stable". This is a hard-coded magic number; whether it is appropriate depends on the scale of the metric being analyzed. For win rates (0-1 scale), a slope of `1e-6` per day is extremely small and appropriate. For `total_games` (could be in the thousands), this threshold may be too tight, causing almost everything to appear as a trend.
- **Prediction gate (line 230):** Only predicts next week if `r_squared > 0.3`, which is a reasonable threshold.
- **`trend_window_days` parameter (line 82, 93):** This constructor parameter is stored but never used anywhere in the class. The `analyze_trends` method does not filter data by this window.

### Automated Report (`generate_automated_report`, lines 427-523)

- **Hardcoded version (line 441):** `"keisei_version": "1.0.0"` is hardcoded and will drift from the actual project version.
- **`PerformanceAnalyzer` instantiation (line 459):** Creates a new `PerformanceAnalyzer(current_results)` directly. This is correct usage.
- **JSON serialization (line 518):** The report dict contains native Python types that are JSON-serializable. The `streaks` and `game_lengths` dicts from `PerformanceAnalyzer` may contain numpy types (`np.int64`, `np.float64`) from `analyze_game_length_distribution()`, which would cause `json.dump` to fail with `TypeError: Object of type int64 is not JSON serializable`. This is a latent bug.

### Automated Insights (`_generate_automated_insights`, lines 525-589)

- **Emoji usage:** The insights contain Unicode emoji characters (lines 533-587). This is not a bug but may cause issues in environments with limited Unicode support (e.g., some log viewers, terminal configurations).

## 4. Robustness & Error Handling

- **Constructor validation (lines 83-89):** Validates all three parameters with clear error messages. Good.
- **Empty input handling:** `compare_performance` handles empty lists at lines 108 and 113 (returns 0 win rate). `analyze_trends` handles < 3 data points at line 177.
- **File write error handling (lines 520-521):** Catches `OSError` and `IOError` when saving the report. Logs the error but does not propagate it, meaning the caller receives the report dict but may not know the file save failed.
- **No validation on `score_a` range in Elo calls:** The `compare_performance` method does not call Elo, so this is N/A here, but the `GameResult.is_agent_win` property can only be `True`/`False`, which is correct.
- **Missing metric fallback (lines 199-201):** When an unknown metric is passed to `analyze_trends`, it logs a warning and uses 0.0. The trend analysis then runs on a flat series of zeros, which would return "stable". This is safe but may mislead users.

## 5. Performance & Scalability

- **Linear in game count:** `compare_performance` iterates over both result lists once for wins and once for move counts -- O(n) total.
- **`analyze_trends` uses `scipy_stats.linregress`:** O(n) complexity for n data points.
- **`generate_automated_report`:** Instantiates a new `PerformanceAnalyzer` and runs its analyses. The histogram computation uses `np.histogram` which is O(n).
- No memoization or caching. If `generate_automated_report` is called multiple times on the same data, all analyses are recomputed.

## 6. Security & Safety

- **File write (line 517):** Writes to a caller-specified `output_file` path. There is no path sanitization, but this is an internal analytics tool, not a web-facing API. The file is opened with `encoding="utf-8"` which is correct.
- **JSON injection:** The report is built from trusted internal data structures, so JSON injection is not a concern.

## 7. Maintainability

- At 589 lines, this is the largest file in the package but is well-structured with clear method separation.
- The four dataclasses at the top (lines 24-62) clearly define the output contracts.
- The `trend_window_days` parameter is accepted in the constructor but never used, which is dead code.
- The hardcoded version string `"1.0.0"` at line 441 is a maintenance liability.
- The `type: ignore` comments on lines 209-211 suppress mypy warnings for scipy's untyped return value, which is acceptable.
- The `type: ignore[assignment]` on line 512 suppresses a type mismatch between `List[str]` (insights) and the `Dict[str, Any]` report value type. This is correct at runtime since dict values can be any type.

## 8. Verdict

**NEEDS_ATTENTION**

Key concerns:
1. **Latent JSON serialization bug:** The `generate_automated_report` method may fail when `json.dump` encounters numpy types from `PerformanceAnalyzer.analyze_game_length_distribution()` (line 518).
2. **Dead parameter:** `trend_window_days` is stored but never used anywhere in the class.
3. **Potential `ValueError` from `math.asin`:** If proportions exceed valid range due to corrupted input data, `_two_proportion_z_test` at line 281 would raise an unhandled `ValueError`.
4. **Hardcoded version string** at line 441 will drift from the actual project version.
