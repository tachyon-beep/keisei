# Code Analysis: keisei/training/display_components.py

## 1. Purpose & Role

`display_components.py` provides a library of reusable Rich display widgets for the training TUI. It defines 8 components: `ShogiBoard` (9x9 board renderer), `RecentMovesPanel` (move history display), `PieceStandPanel` (captured pieces), `Sparkline` (trend visualization), `RollingAverageCalculator` (windowed statistics), `MultiMetricSparkline` (multi-series trends), `GameStatisticsPanel` (detailed game stats), and supporting types (`DisplayComponent` protocol, `HorizontalSeparator`). These components are consumed by `TrainingDisplay` in `display.py`.

## 2. Interface Contracts

### Exports
- **`DisplayComponent`** (Protocol): Defines the `render() -> RenderableType` interface.
- **`HorizontalSeparator`**: Renders a centered horizontal line.
- **`ShogiBoard`**: Renders a 9x9 shogi board with piece symbols, colors, and coordinate labels.
- **`RecentMovesPanel`**: Displays recent moves with optional flashing for the newest move.
- **`PieceStandPanel`**: Displays captured pieces (komadai) for both players.
- **`Sparkline`**: Generates Unicode sparkline strings from numeric data.
- **`RollingAverageCalculator`**: Computes windowed averages and trend direction.
- **`MultiMetricSparkline`**: Aggregates multiple sparklines with labels.
- **`GameStatisticsPanel`**: Renders a comprehensive game statistics panel.

### Key Dependencies
- `rich` library (Panel, Table, Text, Align, Group, Style, box)
- `keisei.shogi.shogi_core_definitions.Color` for player color enum
- `keisei.utils._coords_to_square_name` for board coordinate formatting
- Standard library: `math`, `collections` (Counter, deque), `time` (monotonic)

### Assumptions
- Game objects passed to `ShogiBoard.render()` have a `.board` attribute (2D list of piece objects) and `.hands` dict.
- Piece objects have `.type.name` (string like "PAWN", "PROMOTED_ROOK") and `.color` (Color enum).
- `metrics_manager` passed to `GameStatisticsPanel.render()` has `sente_opening_history`, `gote_opening_history`, and `get_hot_squares()`.
- Game objects for `GameStatisticsPanel.render()` have `get_king_legal_moves(color)`, `is_in_check(color)`, `current_player`, `board`, and `hands`.

## 3. Correctness Analysis

### [22-28] `DisplayComponent` Protocol
- The protocol defines `render() -> RenderableType` but raises `NotImplementedError` in the body. This is a Protocol class (structural typing) so the body is never called -- it's there for documentation. However, none of the concrete components explicitly implement this protocol (no `class ShogiBoard(DisplayComponent):`). The protocol exists but is unused as a type constraint.
- Additionally, `ShogiBoard.render()`, `RecentMovesPanel.render()`, `PieceStandPanel.render()`, and `GameStatisticsPanel.render()` all take additional parameters beyond the zero-arg protocol signature, making them incompatible with the protocol.

### [30-54] `HorizontalSeparator`
- Line 51: `max(1, int(available_width * self.width_ratio))` -- correctly prevents zero-width separators.
- `render()` does not match the `DisplayComponent` protocol (takes `available_width` parameter). Minor inconsistency.

### [57-238] `ShogiBoard`
- **[70-120] `_piece_to_symbol`**:
  - Line 76: Returns "." for falsy pieces. Correct for None/empty squares.
  - Line 80: Non-unicode fallback calls `getattr(piece, "symbol", lambda: "?")()`. The default is a lambda that returns "?", which is called as `lambda()`. If the piece has a `symbol` attribute that is NOT callable, this will raise `TypeError`. However, line 80 is only reached if `self.use_unicode` is False, which is a non-default path.
  - Line 107: `str(piece_type_name_attr.name).upper()` -- assumes `piece_type_name_attr` has a `.name` attribute. The check on line 103 only verifies the `type` attribute is truthy, not that it has `.name`. If `piece.type` is a string rather than an enum, `.name` would raise `AttributeError`, caught by line 112.
  - Line 110: `symbols.get(lookup_key, lookup_key[0])` -- fallback to first character. If `lookup_key` is empty string, `lookup_key[0]` raises `IndexError`, caught by line 112.
  - Lines 113-120: Error handler creates a new `Console(stderr=True)` on every error. This is wasteful but only occurs in error paths. The import is inside the except block, adding latency to the error path.

- **[122-128] `_colorize`**:
  - Assumes `piece.color` is a `Color` enum. If it's anything else, the comparison `== Color.BLACK` may fail silently and default to blue styling. No crash risk.

- **[130-131] `_get_shogi_notation`**:
  - Delegates to `_coords_to_square_name`. Simple passthrough.

- **[133-135] `_pad_symbol`**:
  - Returns the symbol unchanged. The method name is misleading -- it suggests padding but performs no transformation. This appears to be dead logic, retained as a no-op after a refactor.

- **[137-172] `_create_cell_panel`**:
  - Line 146: `(r_idx + c_idx) % 2 == 0` determines light/dark squares. Standard checkerboard pattern.
  - Lines 153-154: Empty squares use the opposite background color for the dot. Correct visual design.
  - Line 167: `board_col = 8 - c_idx` mirrors the column index. This is necessary because the board rendering iterates with `reversed(row)` in `_generate_board_grid` (line 185), so columns need remapping for coordinate labels.

- **[174-189] `_generate_board_grid`**:
  - Line 185: `enumerate(reversed(row))` reverses each row before rendering. In shogi, files are numbered 9-1 from left to right, so reversing the internal 0-8 array puts file 9 on the left. Correct.

- **[191-238] `render`**:
  - Line 195: `if not board_state` -- returns a placeholder panel for `None` or falsy board state. Correct.
  - Lines 213-219: File labels generated as `range(9, 0, -1)` -- produces "9 8 7 6 5 4 3 2 1". Correct for shogi.
  - Lines 226-235: Rank labels "a" through "i" created as vertically centered text blocks. Standard shogi rank notation.

### [241-292] `RecentMovesPanel`
- **[251-258] `_stylise`**:
  - Line 254: `is_newest = self.newest_on_top and move == self._last_move` -- this compares by string equality. If two identical move strings appear in the list (unlikely but possible in theory), both would be styled as newest. Minor.
  - Line 255: `monotonic() < self._flash_deadline` -- correctly uses monotonic clock for flash timing.

- **[260-292] `render`**:
  - Line 269: `moves[-1] != self._last_move` -- detects new moves by comparing the last element. Correct.
  - Lines 272-273: `__import__("time").monotonic()` -- inline import inside a method. This is an unusual pattern; `monotonic` is already imported at the top of the file (line 7: `from time import monotonic`). This redundant import is wasteful and inconsistent.
  - Line 277: `moves[-self.max_moves :]` -- correct slicing. If `max_moves` exceeds list length, Python returns the full list.
  - Line 281: `slice_.reverse()` -- in-place reversal of a local copy. Correct and safe (doesn't mutate the caller's list because `[-n:]` creates a new list).
  - Line 284: `Text("\n").join(...)` uses Rich's `Text.join()` method. Correct.
  - Lines 287-291: Title includes total move count and ply/sec when non-zero. The truthiness check `if ply_per_sec` means 0.0 shows no rate. Correct design.

### [295-340] `PieceStandPanel`
- **[298-313] `_format_hand`**:
  - Line 309: `getattr(k, 'name', k)` handles both enum keys and string keys. Correct.
  - Line 311: Filters `v > 0` to exclude empty entries. Correct.
  - Line 313: Returns empty string for empty hands. But `render()` at line 334 checks `or "None"`, so an empty hand displays "None". Correct.

- **[315-340] `render`**:
  - Line 317: `if not game` returns placeholder. Correct.
  - Line 321: `getattr(game, "hands", {}).get(Color.BLACK.value, {})` -- double-safe access: `getattr` default, then `.get()` default. Correct.

### [343-393] `Sparkline`
- **[350-393] `generate`**:
  - Line 360: Filters NaN and Inf values. Correct.
  - Line 365-367: Single-value case fills width with `"▄"`. Correct.
  - Lines 369-370: Uses caller-specified range if provided, otherwise computes from data. Correct.
  - Lines 372-374: Handles Inf in range bounds by returning blanks. Correct.
  - Lines 376-378: Clipping logic replaces NaN/Inf values with `min_v`. Correct.
  - Lines 380-388: Handles `max_v == min_v` (flat data) as normalized index 4 (middle bar). The redundant check at line 384 (`rng == 0`) is unreachable if `max_v == min_v` is already handled at line 380. However, the additional NaN/Inf checks on `rng` are defensive.
  - Line 387: `int((v - min_v) / rng * 6)` -- normalizes to 0-6 range. The `self.chars` has 7 characters (indices 0-6), so `int(...* 6)` can produce values 0 through 6. When `v == max_v`, the result is exactly 6. Correct.
  - Line 390: `self.chars[n]` -- if `n` somehow exceeds 6, this would raise `IndexError`. Given the normalization logic, `n` should be in [0,6]. Correct.

### [396-414] `RollingAverageCalculator`
- Line 401: Uses `deque(maxlen=window_size)` for automatic windowing. Correct.
- Line 405: `sum(self.values) / len(self.values)` -- `len(self.values)` is always >= 1 after `append()`. No division-by-zero risk.
- Lines 408-414: Trend compares first and last values in the window. This is a simple but potentially noisy trend indicator (a single outlier at position 0 or -1 dominates).

### [417-436] `MultiMetricSparkline`
- Line 428: `add_data_point` silently ignores unknown metric names. Correct.
- Line 434: `values[-self.width :]` -- slices before generating. Correct.
- The `data` dict grows unboundedly (line 428: `self.data[metric_name].append(value)`). Over very long training runs, these lists could consume significant memory.

### [439-610] `GameStatisticsPanel`
- **[442-458] `_calculate_material`**:
  - Line 456: `piece.type.name.replace("PROMOTED_", "")` -- promoted pieces are valued at their base piece value. This is a simplification (in shogi, promoted pieces have different strategic value) but acceptable for a display metric.

- **[460-475] `_format_hand`**:
  - This is a duplicate of `PieceStandPanel._format_hand` (lines 298-313). Identical logic, identical symbol dict. Code duplication.

- **[477-513] `_format_opening_name`**:
  - Line 479: `if not move_str or len(move_str) < 2` -- guard for empty/short strings. Correct.
  - Line 497: `destination = move_str[2:]` -- for a drop like "P*2c", this gives "2c". Correct.
  - Line 504: `move_str = move_str[:-1]` -- mutates the parameter to strip "+". This is safe since strings are immutable (creates new string).
  - Line 506: `len(move_str) == 4` check after stripping "+" handles standard moves. Correct.

- **[515-610] `render`**:
  - Line 529: `if not game or not move_history or not metrics_manager` -- returns placeholder if any input is missing. This means an empty `move_history` list (`[]`) evaluates as falsy, returning "Waiting for game to start..." even if the game has started but no moves have been made. This could be confusing during the first moments of a game.
  - Line 545: `is_in_check = game.is_in_check(game.current_player)` -- calls a game method that (per the shogi_game.py analysis) temporarily mutates `current_player`. In a display context this is read-only, but if called from a thread different from the game logic, this could cause a race condition. However, the display system appears to run in the same thread.
  - Line 553: `Counter(sente_openings).most_common(1)[0][0]` -- if `sente_openings` is non-empty, `most_common(1)` always returns at least one element. Correct.
  - Lines 566-570: "Moves since capture" searches backwards through move history for "captur" substring (case-insensitive). This depends on the move string format containing "captur" for captures. If the format changes, this heuristic silently breaks.
  - Line 573-574: `game.get_king_legal_moves(Color.BLACK)` -- per the shogi_game.py analysis, this method temporarily mutates game state. Same threading concern as line 545.
  - Line 598: `", ".join(hot_squares) or "N/A"` -- if `hot_squares` is an empty list, `", ".join([])` returns `""` which is falsy, so "N/A" is displayed. Correct.
  - Lines 605-606: `game.hands.get(Color.BLACK.value, {})` -- accesses hands directly (no `getattr` safety). If `game.hands` doesn't exist, this raises `AttributeError`. However, the caller (`refresh_dashboard_panels`) wraps this in a try/except.

## 4. Robustness & Error Handling

- **ShogiBoard** has comprehensive error handling in `_piece_to_symbol` (lines 100-120), catching AttributeError and IndexError with fallback symbols.
- **RecentMovesPanel** has no explicit error handling, but its inputs are simple (list of strings, float).
- **Sparkline** defensively handles NaN, Inf, empty inputs, and flat data. Robust.
- **GameStatisticsPanel.render** has NO try/except blocks internally. All error handling is deferred to the caller (`refresh_dashboard_panels` in display.py). If `game.get_king_legal_moves()` or `game.is_in_check()` raises an unexpected exception, it propagates up.
- **MultiMetricSparkline** data lists grow without bounds. Over very long training runs (millions of data points), this could lead to memory exhaustion.

## 5. Performance & Scalability

- **ShogiBoard**: Creates 81 Panel objects per render (9x9 grid). Each panel involves style computation, text centering, and border rendering. At 4 refreshes/second, this is 324 panels/second. Rich handles this efficiently, but it is the most object-heavy component.
- **Sparkline.generate**: O(n) where n is the number of values, with filtering and normalization. Called 10 times per refresh. Negligible.
- **GameStatisticsPanel._calculate_material**: Iterates all 81 board squares. O(81) per call. Negligible.
- **RecentMovesPanel**: Slices and potentially reverses a small list (max 20 moves by default). Negligible.
- **MultiMetricSparkline.data**: Unbounded growth. If `add_data_point` is called every episode, and episodes are short (100 steps), a 1M timestep run could accumulate 10,000+ data points per metric. Not critical but worth noting.

## 6. Security & Safety

- No external input processing. All data comes from the training system.
- No file I/O, network access, or deserialization.
- The `__import__("time")` call at line 273 is an unusual pattern but not a security risk.
- No security concerns.

## 7. Maintainability

- **Duplicated code**: `_format_hand` is duplicated between `PieceStandPanel` (line 298) and `GameStatisticsPanel` (line 460). Identical symbol dictionaries and formatting logic.
- **Unused protocol**: `DisplayComponent` protocol at line 22 is never used as a type constraint. None of the concrete classes declare conformance, and their `render()` signatures are incompatible with the protocol's zero-argument definition.
- **Dead method**: `ShogiBoard._pad_symbol` (lines 133-135) returns its input unchanged. The docstring says "Return the symbol unchanged." It appears to be a vestigial no-op.
- **Inconsistent import style**: Line 273 uses `__import__("time").monotonic()` instead of the already-imported `monotonic` from line 7.
- **Heuristic string matching**: Line 568 searches for "captur" in move strings. This creates an implicit dependency on the move formatting system's output strings.
- **Well-structured components**: Despite the above issues, each component has clear responsibilities and clean render interfaces. The sparkline implementation is particularly robust with thorough edge case handling.

## 8. Verdict

**NEEDS_ATTENTION**

Key findings:
1. **Code duplication**: `_format_hand` is duplicated verbatim across two classes (lines 298-313 and 460-475).
2. **Unused/incompatible protocol**: `DisplayComponent` protocol exists but is never applied. Concrete `render()` signatures don't match it.
3. **Unbounded data accumulation**: `MultiMetricSparkline.data` lists grow without limit (line 428).
4. **Heuristic string matching**: "Moves since capture" detection at line 568 depends on implicit move string formatting.
5. **Redundant import**: `__import__("time")` at line 273 when `monotonic` is already imported at file scope.
6. **Dead method**: `_pad_symbol` at line 133 is a no-op.
7. **No internal error handling** in `GameStatisticsPanel.render` -- relies entirely on caller exception handling.
