# Code Analysis: keisei/training/metrics_manager.py

## 1. Purpose & Role

`metrics_manager.py` manages all training statistics, metrics tracking, and formatting for the Keisei training system. It is one of 9 specialized managers and serves as the central repository for game outcome statistics (wins/losses/draws), PPO training metrics, Elo ratings, episode history, and progress update state. It is consumed by `TrainingLoopManager`, `DisplayManager`, `WebUIManager`, and the checkpoint system.

The file is 442 lines and defines two dataclasses (`TrainingStats`), one supporting class (`MetricsHistory`), and the main `MetricsManager` class.

## 2. Interface Contracts

### Exports
- `TrainingStats` (dataclass): Container for global timestep, total episodes completed, black wins, white wins, draws.
- `MetricsHistory` (class): Tracks historical PPO metrics, win rates, episode lengths, and episode rewards with bounded history.
- `MetricsManager` (class): The main class with the following public methods:
  - `update_episode_stats(winner_color: Optional[Color]) -> Dict[str, float]`
  - `get_win_rates() -> Tuple[float, float, float]`
  - `get_win_rates_dict() -> Dict[str, float]`
  - `log_episode_metrics(moves_made, turns_count, result, episode_reward, move_history, policy_mapper)`
  - `get_moves_per_game_trend(window_size) -> Sequence[int]`
  - `get_hot_squares(top_n) -> List[str]`
  - `get_games_completion_rate(time_window_hours) -> float`
  - `get_win_loss_draw_rates(window_size) -> Dict[str, float]`
  - `get_average_turns_trend(window_size) -> Sequence[int]`
  - `format_episode_metrics(episode_length, episode_reward) -> str`
  - `format_ppo_metrics(learn_metrics) -> str`
  - `format_ppo_metrics_for_logging(learn_metrics) -> str`
  - `update_progress_metrics(key, value)`
  - `get_progress_updates() -> Dict[str, Any]`
  - `clear_progress_updates()`
  - `set_processing(value: bool)`
  - `get_final_stats() -> Dict[str, int]`
  - `restore_from_checkpoint(checkpoint_data)`
  - `increment_timestep()` / `increment_timestep_by(amount)`
  - Properties: `global_timestep`, `total_episodes_completed`, `black_wins`, `white_wins`, `draws` (all read/write)

### Key Dependencies
- `keisei.shogi.shogi_core_definitions.Color` -- player color enumeration
- `keisei.utils.PolicyOutputMapper` -- for formatting opening moves
- `keisei.utils._coords_to_square_name` -- coordinate-to-square-name conversion
- `keisei.utils.format_move_with_description` -- move description formatting
- `.elo_rating.EloRatingSystem` -- Elo rating calculations
- `json`, `time`, `collections.Counter`, `collections.deque`

### Assumptions
- `winner_color` passed to `update_episode_stats` is either `Color.BLACK`, `Color.WHITE`, or `None`. Any other value is treated as a draw (falls through to the `else` branch at line 139).
- `checkpoint_data` passed to `restore_from_checkpoint` is a dict with optional keys `global_timestep`, `total_episodes_completed`, `black_wins`, `white_wins`, `draws`.
- Move history entries are 5-tuples in the standard Shogi move format `(from_r, from_c, to_r, to_c, promotion_or_piece)`.
- `_coords_to_square_name` is imported from `keisei.utils` -- this is a private function (prefixed with `_`) being used across module boundaries.

## 3. Correctness Analysis

### `TrainingStats` (lines 21-29)
- Simple dataclass with default values of 0. No issues.

### `MetricsHistory` (lines 32-81)

**`_trim` method (lines 55-57):**
- Uses `while len(values) > self.max_history: values.pop(0)` to enforce the history limit. `list.pop(0)` is O(n) because it requires shifting all subsequent elements. When `max_history` is 1000, this is a minor concern per call, but it runs on every metric insertion. Using `collections.deque` with `maxlen` (as is done for other metrics in `MetricsManager.__init__`) would be O(1) per trim operation. The lists that use `_trim` are: `win_rates_history`, `learning_rates`, `policy_losses`, `value_losses`, `kl_divergences`, `entropies`, `clip_fractions`, `episode_lengths`, `episode_rewards`.
- The `while` loop is technically correct but will only ever execute once per call (since only one item is added at a time). A simpler `if` check would be equivalent.

**`add_episode_data` (lines 59-61):**
- Appends win rates dict and trims. Correct.

**`add_ppo_data` (lines 63-81):**
- Conditionally appends each metric type if the key exists in the input dict. This means the different metric lists can have different lengths if some metrics are missing from certain PPO updates. Code that assumes all lists have the same length would be incorrect, but no such code exists in this file.

### `MetricsManager.__init__` (lines 95-119)
- Initializes `TrainingStats`, `MetricsHistory`, `EloRatingSystem`, and various bounded collections (`deque` with `maxlen`).
- `square_usage` is a `Counter[str]` that is never trimmed or bounded. Over a very long training run, this counter will grow proportionally to the number of unique squares used. Since a Shogi board has 81 squares, this is bounded to at most 81 entries. No issue.
- `sente_opening_history` and `gote_opening_history` are deques with `maxlen=10`, which is appropriately bounded.
- `processing` boolean flag is initialized to `False`. This flag is set externally by callers to indicate PPO update is in progress.

### `update_episode_stats` (lines 123-144)
- Increments `total_episodes_completed` first (line 133), then updates the appropriate win/draw counter. This ordering means `total_episodes_completed` is always consistent with the sum of wins and draws.
- Calls `self.elo_system.update_ratings(winner_color)` which expects `Optional[Color]`. The input `winner_color` is typed as `Optional[Color]`, which matches.
- Returns win rates dict after the update. Correct.

### `get_win_rates` (lines 146-161)
- Returns percentages (multiplied by 100). Division-by-zero is guarded by checking `total == 0`.
- Returns a tuple of `(black_rate, white_rate, draw_rate)`. These should always sum to approximately 100% unless there are floating-point rounding issues. No issue in practice.

### `log_episode_metrics` (lines 174-217)
**Opening move tracking (lines 191-204):**
- Uses `format_move_with_description(move_history[0], policy_mapper, game=None)` for sente's opening and `move_history[1]` for gote's opening. Passing `game=None` may cause issues if `format_move_with_description` tries to access game state. However, the entire block is wrapped in a try/except that catches `AttributeError`, `IndexError`, `TypeError`, `ValueError`, so any such failure is silently suppressed.
- The `if policy_mapper is not None and len(move_history) >= 1` guard is correct but subtly redundant with the outer `if move_history:` check -- if `move_history` is truthy, its length is at least 1.

**Square usage tracking (lines 206-217):**
- Iterates over all moves in `move_history` and updates `square_usage` for both source and destination squares.
- Line 210: `if mv[0] is not None and mv[1] is not None` -- correctly skips drop moves (where from-square is None).
- Line 213: `if len(mv) >= 4` -- checks move tuple length before accessing `mv[2]` and `mv[3]`. This is correct and handles potential malformed moves.
- The entire block is wrapped in try/except catching multiple exception types. Correct.

### `get_moves_per_game_trend` (lines 219-221)
- Converts deque to list and takes the last `window_size` entries. Simple and correct.

### `get_hot_squares` (lines 223-225)
- Returns the `top_n` most common squares. Correct.

### `get_games_completion_rate` (lines 227-230)
- Calculates games per hour by counting timestamps within the time window.
- Line 229: `sum(ts >= cutoff for ts in self.games_completed_timestamps)` iterates through the deque. Since the deque has `maxlen=history_size` (default 1000), this is O(1000) at worst. No performance concern.
- `time_window_hours` division is guarded with `if time_window_hours > 0`. Passing `0.0` returns `0.0`, which is correct.

### `get_win_loss_draw_rates` (lines 232-240)
- `sorted(self.win_loss_draw_history, key=lambda t: t[1])[-window_size:]` -- sorts by timestamp, then takes the last `window_size` entries. This is O(n log n) where n is the deque size (up to `history_size`). Since the deque maintains insertion order (which is chronological), the sort is redundant -- the data is already in chronological order. The slice `[-window_size:]` alone would suffice.
- The rates are calculated as fractions (0.0 to 1.0), not percentages. This is inconsistent with `get_win_rates()` which returns percentages (0 to 100). Different consumers must be aware of this difference.

### `format_episode_metrics` (lines 246-263)
- Formats a display string using current win rates. Correct.

### `format_ppo_metrics` (lines 267-304)
- Builds a formatted string from PPO metric keys, using the `MetricsHistory` constants for key names.
- Also calls `self.history.add_ppo_data(learn_metrics)` at line 303, which is a side effect in what appears to be a formatting method. The method name `format_ppo_metrics` suggests it should be pure, but it also records history. This is a misleading API.

### `format_ppo_metrics_for_logging` (lines 306-317)
- Formats all metric values to 4 decimal places and returns JSON. Simple and correct.
- Note: `f"{v:.4f}"` converts float to string, so the JSON output contains string values, not numbers. Consumers parsing this JSON will get string values like `"0.0012"` rather than numeric `0.0012`.

### `restore_from_checkpoint` (lines 362-375)
- Uses `.get()` with default 0 for all fields. This is defensive and correct.
- Does NOT restore `MetricsHistory`, `EloRatingSystem`, `moves_per_game`, `turns_per_game`, `games_completed_timestamps`, `win_loss_draw_history`, `square_usage`, or opening histories. Only the basic counters are restored. This means trend data, Elo ratings, and enhanced metrics are lost on checkpoint resume. This is a known limitation rather than a bug, but it means the Elo system always starts from the initial rating (1500) on resume.

### `increment_timestep_by` (lines 381-390)
- Validates that `amount >= 0` and raises `ValueError` for negative amounts. Correct.

### Properties (lines 394-442)
- All properties delegate to `self.stats.*` fields. They provide backward-compatible attribute access.
- The setters have no validation (e.g., setting `black_wins` to a negative number is allowed). This matches the `TrainingStats` dataclass which also has no validation.

## 4. Robustness & Error Handling

**Strengths:**
- `log_episode_metrics` wraps both opening move tracking and square usage tracking in separate try/except blocks, preventing failures in metric collection from crashing training.
- `get_win_rates` and `get_games_completion_rate` guard against division by zero.
- `restore_from_checkpoint` uses `.get()` with defaults for all fields.
- `increment_timestep_by` validates non-negative input.

**Weaknesses:**
- `update_episode_stats` does not validate that `winner_color` is one of the expected values (`Color.BLACK`, `Color.WHITE`, `None`). If a string like `"black"` is passed (which is possible given the type mismatch between `StepManager.handle_episode_end` which returns strings and `MetricsManager.update_episode_stats` which expects `Color`), it would fall through to the `else` branch and be counted as a draw. The caller in `training_loop_manager.py` does convert the string to `Color` enum before calling, so this is not a live bug, but the API contract is fragile.
- `set_processing()` has no synchronization. If called from multiple threads (e.g., WebUI thread and training thread), this could cause race conditions. The `processing` flag appears to be read by the WebUI for display purposes.
- No error handling in `get_final_stats()` or `get_win_rates_dict()`.

## 5. Performance & Scalability

- `MetricsHistory._trim()` uses `list.pop(0)` which is O(n) per call. With `max_history=1000` and the method being called on each PPO update (up to 6 times per update for the 6 metric types), this is approximately 6000 element shifts per update. This is negligible in the context of a training step but is architecturally suboptimal. The `deque` collections used for `moves_per_game`, `turns_per_game`, etc. in `MetricsManager.__init__` do not have this issue.
- `get_win_loss_draw_rates()` sorts the entire `win_loss_draw_history` deque on every call, which is O(n log n). Since the data is already in chronological order, a simple slice would suffice.
- `square_usage` Counter grows to at most 81 entries (one per Shogi board square). No concern.
- `json.dumps` in `format_ppo_metrics_for_logging` is called on a small dict (~6 entries). No concern.

## 6. Security & Safety

- No file I/O, network access, or deserialization occurs in this file.
- The `json.dumps` at line 317 serializes internal data only; no external input reaches this path.
- No injection risks.
- The `_coords_to_square_name` import is a private function used across module boundaries. This is a minor encapsulation violation but has no security implications.

## 7. Maintainability

**Code Smells:**
- **Side effect in formatting method**: `format_ppo_metrics()` (line 267) both formats metrics AND records them to history via `self.history.add_ppo_data(learn_metrics)` at line 303. This violates the single-responsibility principle and makes the method name misleading. A caller who only wants formatting will inadvertently record history.
- **Inconsistent rate units**: `get_win_rates()` returns percentages (0-100), `get_win_loss_draw_rates()` returns fractions (0-1). Both represent the same conceptual metric (win/loss/draw rates).
- **Mixed trimming strategies**: `MetricsHistory` uses manual `_trim()` with `list.pop(0)`, while `MetricsManager` uses `deque(maxlen=...)`. These are two different approaches to the same bounded-history problem within the same file.
- **Large class**: `MetricsManager` has 21 public methods plus 6 properties, making it a potential "god object" for metrics. However, the methods are mostly simple and focused.
- **Property boilerplate** (lines 394-442): 48 lines of trivial property getters/setters that delegate to `self.stats.*`. These exist for backward compatibility and add maintenance burden.

**Dead Code:**
- No obvious dead code detected. All public methods appear to have callers.

**Structure:**
- The separation between `TrainingStats`, `MetricsHistory`, and `MetricsManager` is logical. `TrainingStats` holds simple counters, `MetricsHistory` holds time-series data, and `MetricsManager` orchestrates both.
- The `EloRatingSystem` integration is clean -- it is delegated to via `update_episode_stats`.

## 8. Verdict

**NEEDS_ATTENTION**

Key findings:
1. **Side effect in formatting**: `format_ppo_metrics()` records history as a side effect, violating its apparent contract as a formatting-only method.
2. **Inconsistent rate units**: `get_win_rates()` returns percentages while `get_win_loss_draw_rates()` returns fractions, creating a footgun for consumers.
3. **Suboptimal trimming**: `MetricsHistory._trim()` uses O(n) `list.pop(0)` instead of `deque(maxlen=...)` which is used elsewhere in the same file.
4. **Unnecessary sort**: `get_win_loss_draw_rates()` sorts chronologically-ordered data.
5. **Incomplete checkpoint restore**: `restore_from_checkpoint` only restores basic counters, losing trend data, Elo ratings, and enhanced metrics on resume.
6. **No thread safety for `processing` flag**: The `processing` boolean is read/written from potentially different threads without synchronization.
