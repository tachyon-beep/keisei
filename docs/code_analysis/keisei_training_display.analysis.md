# Code Analysis: keisei/training/display.py

## 1. Purpose & Role

`display.py` contains the `TrainingDisplay` class, which is the core Rich TUI rendering engine for the Shogi RL training system. It constructs either a compact (log + progress bar) or enhanced (full dashboard with board, metrics, model evolution, Elo ratings) layout depending on terminal size, and provides methods to update progress bars and refresh all dashboard panels in real-time. It sits between `DisplayManager` (the manager facade) and `display_components.py` (reusable Rich widgets).

## 2. Interface Contracts

### Exports
- **`TrainingDisplay`**: The sole public class, instantiated by `DisplayManager.setup_display()`.
  - `update_progress(trainer, speed, pending_updates)`: Updates the Rich progress bar.
  - `refresh_dashboard_panels(trainer)`: Refreshes all dashboard panels from trainer state.
  - `start()` -> `Live`: Returns a Rich `Live` context manager for real-time rendering.

### Key Dependencies
- `rich` library (Console, Layout, Live, Panel, Progress, Table, Text, Group)
- `keisei.config_schema.DisplayConfig` for configuration
- `keisei.training.adaptive_display.AdaptiveDisplayManager` for layout selection
- `keisei.training.display_components` for all widget components (ShogiBoard, Sparkline, etc.)
- `trainer` object (duck-typed): expects `metrics_manager`, `step_manager`, `game`, `agent`, `experience_buffer`, `rich_log_messages`, and various attributes

### Assumptions
- The `trainer` object has a fully initialized `metrics_manager` with `global_timestep`, `total_episodes_completed`, `black_wins`, `white_wins`, `draws`, and a `history` object with metric lists.
- The `trainer.game` has a board state compatible with `ShogiBoard.render()`.
- The `trainer.agent.model` exposes `named_parameters()` for the model evolution panel.
- `trainer.experience_buffer` has `capacity()` and `size()` methods.

## 3. Correctness Analysis

### [44-94] `__init__`
- Line 57: `game_stats_component` is initialized to `GameStatisticsPanel()` inline, while all other components are conditionally created. This means the game stats component is always created regardless of config. Minor inconsistency but not a bug.
- Lines 85-90: The tuple unpacking of `_setup_rich_progress_display()` is fragile -- any change to the return signature requires matching changes here.

### [96-106] `_create_compact_layout`
- Straightforward two-part layout (log + progress). No issues found.

### [108-155] `_create_enhanced_layout`
- Lines 127-131: The board column has fixed sizes (`size=4` for komadai, `size=8` for moves). If the terminal height is insufficient, Rich will silently truncate these panels. No validation against terminal size occurs.
- Lines 144-151: All panels initialized with placeholder `"..."` text. These are immediately overwritten on the first `refresh_dashboard_panels()` call, so no user-facing issue.

### [157-243] `_setup_rich_progress_display`
- Lines 193-210: Initial win rates are computed with a zero-division guard (`if total_episodes_completed > 0`). Correct.
- Line 183: `getattr(self.config.training, "enable_spinner", True)` uses a safe default. Correct.
- Line 211-228: The `add_task` call passes `start=` as a boolean expression (`global_timestep < total_timesteps`). This controls whether the task timer starts immediately. Correct for resumed training.

### [245-309] `_build_metric_lines`
- Line 271: `assert self.trend_component is not None` -- this is guarded by the caller context (only called when `self.trend_component` exists from line 378 check), so it is safe. However, an `assert` in production code can be removed by `-O` flag.
- Lines 275-288: Win rate extraction from `history.win_rates_history` uses `.get()` with default `0.0`, which is safe even for empty dicts.
- Lines 290-293: Edge case handling is correct: `last_val` and `prev_val` default to `None` with length checks, and `avg_val` handles empty slices.

### [311-317] `update_progress`
- Simple delegation. No issues.

### [319-620] `refresh_dashboard_panels` -- THE LONG FUNCTION (301 lines)
This is the largest method in the file, handling all panel updates. Known issue per project memory (16+ functions > 100 lines, with `display.py:301` being the worst).

- **Lines 321-327**: Log panel update. `visible_rows = max(0, self.rich_console.size.height - 6)` -- the magic number 6 assumes a fixed overhead (progress bar + borders). If layout ratios change, this estimate becomes stale.
- **Lines 336-349**: Board panel update. Catches `AttributeError, RuntimeError, TypeError, ValueError`. The error handler logs the error AND updates the panel with an error message. Correct recovery.
- **Lines 351-362**: Komadai panel update. Same error handling pattern. Correct.
- **Lines 366-376**: Moves panel update. Catches only `AttributeError, TypeError` -- narrower than the board panel handler. If `move_log` contains unexpected data types, a `ValueError` would propagate.
- **Lines 378-415**: Trends panel update.
  - Line 386: `separator.render(available_width=50)` -- hardcoded width of 50 characters. The comment acknowledges this ("Use a reasonable panel width... since we can't access the exact panel width"). This means the separator is always 47 chars wide (50 * 0.95), regardless of actual panel width.
  - Line 395: `grad_bar.add_task("", total=50.0, completed=int(grad_norm_scaled))` -- the `completed` value is cast to `int`, losing fractional precision. For gradient norms < 1.0, this will always show 0.
- **Lines 418-458**: Stats panel update.
  - Line 419: `assert self.game_stats_component is not None` -- this assert exists inside a try/except that catches `AssertionError`, so it degrades gracefully if the assertion fails. Correct recovery pattern.
  - Line 441: `group_stats: List[RenderableType] = [panel.renderable]` -- accesses the `renderable` attribute of a `Panel`, which is a valid Rich internal attribute. This is somewhat fragile as it depends on Rich's internal structure.
- **Lines 460-486**: Config panel update. Only rendered once (`config_panel_rendered` flag). No issues.
- **Lines 488-589**: Model evolution panel.
  - Lines 495-509: Iterates all named parameters, filters by keyword, and moves tensor data to CPU with `.float().cpu().numpy()`. This is a **performance concern**: for large models, iterating all parameters and copying to CPU on every dashboard refresh could be expensive. The data transfer is synchronous and blocking.
  - Line 500: `data = p.data.float().cpu().numpy()` -- forces float32 conversion and CPU transfer. For mixed-precision training, this could be called on fp16 tensors, which is fine (`.float()` upcasts). But it creates temporary copies on every refresh.
  - Line 589: `self.previous_model_stats = copy.deepcopy(current_stats)` -- deep copy of a dict of dicts of floats. The comment says "(The Bug Fix)", implying a previous aliasing bug was fixed here. Correct.
- **Lines 591-620**: Elo panel update.
  - Lines 595-600: Defensive validation of `snap` structure checks for `.get()` method, correct `top_ratings` type, and minimum length of 2. However, line 600 requires `len(snap["top_ratings"]) >= 2`, which means the panel shows "Waiting..." if only 1 model has been rated. This seems like a UI limitation rather than a bug.

### [622-628] `start`
- Returns a `Live` context manager. The caller is responsible for using it as a context manager. No issues.

## 4. Robustness & Error Handling

- **Broad exception catching**: Multiple panel update blocks catch 4-5 exception types. This is intentionally defensive for a display component -- a rendering error should never crash training. The pattern is consistent: catch, log, show error panel.
- **Missing error handling**: The model evolution panel (lines 488-589) catches exceptions for parameter iteration (line 507) but the table construction and layout update (lines 512-589) are NOT wrapped in a try/except. If `model.named_parameters()` succeeds but a subsequent operation fails (e.g., `self.layout["evolution_panel"]` KeyError if not in enhanced layout), the exception will propagate.
- **No resource cleanup**: Tensor `.cpu().numpy()` conversions create temporary allocations. No explicit cleanup, but Python GC handles this.

## 5. Performance & Scalability

- **Critical**: Lines 495-509 iterate ALL model parameters on every dashboard refresh. For a ResNet tower with millions of parameters, this involves:
  1. Iterating all named parameters
  2. String matching on parameter names
  3. For matching parameters: `.float().cpu().numpy()` (GPU->CPU transfer)
  4. Computing mean, std, min, max on CPU
  This happens at the `refresh_per_second` rate (default 4 Hz). For large models, this could introduce significant overhead during training.
- **Deep copy**: Line 589 deep copies the stats dict on every refresh. The dict contains only floats, so this is cheap.
- **Log panel**: Line 323 slices `rich_log_messages[-visible_rows:]` which is O(visible_rows). Fine.
- **Sparkline generation**: Called 10 times per refresh (once per metric). Each generates a short string. Negligible cost.

## 6. Security & Safety

- No external input processing. All data comes from the trainer object.
- No file I/O, network access, or deserialization.
- The Rich console outputs to stderr (configured in `DisplayManager`), not a web interface.
- No security concerns.

## 7. Maintainability

- **Code smell -- God method**: `refresh_dashboard_panels` at 301 lines violates single-responsibility. It handles 8 different panel types in one method. Each panel update block could be a separate method.
- **Tight coupling to trainer**: The method directly accesses `trainer.metrics_manager`, `trainer.step_manager`, `trainer.game`, `trainer.agent.model`, `trainer.experience_buffer`, and various `getattr` lookups with defaults. This creates a fragile implicit contract.
- **Debug artifacts**: Lines 343-344 override board border style to red "for debugging". Line 356 overrides komadai border to green. These appear to be leftover debug aids that are now permanent visual elements.
- **Magic numbers**: Line 321 uses `height - 6`, line 386 uses `available_width=50`, line 389 uses `min(grad_norm, 50.0)`, line 395 uses `total=50.0`. None are documented as constants.
- **Inconsistent error handling breadth**: Board catches 4 exceptions, moves catches 2, stats catches 5. The differences appear arbitrary rather than intentional.
- **No dead code detected**: All methods and branches appear reachable.

## 8. Verdict

**NEEDS_ATTENTION**

Key findings:
1. **Performance risk**: Model parameter iteration with GPU-to-CPU transfer on every refresh (lines 495-509) could meaningfully impact training throughput for large models.
2. **301-line method**: `refresh_dashboard_panels` is the single largest method in the codebase, handling 8 distinct panel updates with no decomposition.
3. **Unprotected model evolution block**: Lines 512-589 (table building and layout update) lack exception handling, unlike all other panel update blocks.
4. **Debug border colors remain**: Red and green border overrides (lines 343-344, 356) appear to be leftover development artifacts.
5. **Hardcoded separator width**: Line 386 uses a fixed 50-char width assumption rather than querying actual panel dimensions.
