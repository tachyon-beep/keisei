# Code Analysis: keisei/training/adaptive_display.py

## 1. Purpose & Role

`adaptive_display.py` provides the `AdaptiveDisplayManager` class and `TerminalInfo` dataclass, which determine whether the training TUI should use a "compact" or "enhanced" layout based on terminal dimensions and configuration. It is a small helper consumed by `TrainingDisplay._setup_rich_progress_display()` during display initialization. The decision is made once at startup and not re-evaluated during training.

## 2. Interface Contracts

### Exports
- **`TerminalInfo`** (dataclass): Holds `width`, `height`, and `unicode_ok` fields.
- **`AdaptiveDisplayManager`**: Determines layout type.
  - `__init__(config: DisplayConfig)`: Takes display configuration.
  - `get_terminal_info(console: Console)` -> `TerminalInfo`: Detects terminal capabilities.
  - `choose_layout(console: Console)` -> `str`: Returns `"enhanced"` or `"compact"`.

### Key Dependencies
- `os` (for `os.get_terminal_size()` fallback)
- `rich.console.Console` (for primary terminal size detection)
- `keisei.config_schema.DisplayConfig` (for `enable_enhanced_layout` flag)

### Assumptions
- The `console` parameter is a fully initialized Rich `Console` object.
- The `DisplayConfig` has an `enable_enhanced_layout` boolean field.
- Terminal size is stable (checked once, not monitored for resize).

## 3. Correctness Analysis

### [14-18] `TerminalInfo`
- Simple dataclass with three fields. No issues.
- `unicode_ok` is detected but never consumed outside this file -- the caller (`TrainingDisplay`) does not check this field. This is dead data in the current codebase.

### [21-27] Class constants and `__init__`
- `MIN_WIDTH_ENHANCED = 120`: Requires 120 columns for enhanced layout. Reasonable threshold for a three-column dashboard.
- `MIN_HEIGHT_ENHANCED = 25`: Requires 25 rows. The enhanced layout has multiple stacked panels; 25 rows is relatively low and may still result in truncated panels.
- `__init__` stores config. No issues.

### [30-38] `_get_terminal_size`
- Line 32: Primary path uses `console.size.width, console.size.height`. The Rich `Console.size` property always returns a `ConsoleDimensions` object, even for non-terminal outputs (it falls back to default 80x25). The bare `except Exception` at line 33 is overly broad but serves as a safety net.
- Lines 34-37: Fallback to `os.get_terminal_size()`, which raises `OSError` if no terminal is available (e.g., in CI environments). Caught correctly.
- Line 38: Final fallback returns `(80, 24)`. This is a safe default. Note: 24 < `MIN_HEIGHT_ENHANCED` (25), so the fallback always results in compact layout. Correct behavior for non-terminal environments.

### [40-47] `get_terminal_info`
- Lines 43-44: Unicode detection attempts to encode `"▁"` using the console's encoding. If encoding is `None`, falls back to `"utf-8"`.
  - `console.encoding` could return `None` for non-terminal outputs. The `or "utf-8"` fallback is correct.
  - The test encodes a single sparkline character. This is a reasonable proxy for Unicode support, though it doesn't test all characters used by the display system (e.g., Japanese kanji like "歩").
- Line 45: Catches `UnicodeEncodeError`. Correct exception for encoding failures.
- The `unicode_ok` field is populated but, as noted, never used by the caller.

### [49-57] `choose_layout`
- Line 50: Calls `get_terminal_info` which internally calls `_get_terminal_size`. Correct composition.
- Lines 51-56: Three conditions must ALL be true for enhanced layout:
  1. Width >= 120
  2. Height >= 25
  3. `config.enable_enhanced_layout` is True
- Line 57: Default to "compact". Correct.
- The method returns string literals (`"enhanced"`, `"compact"`). These are matched by string comparison in the caller (`display.py` line 237). Type safety relies on consistent string usage; an enum would be safer but this is adequate for two values.

## 4. Robustness & Error Handling

- **Triple-layered fallback** for terminal size: Rich console -> OS -> hardcoded default. This is thorough.
- **Bare `except Exception`** at line 33 catches everything including `KeyboardInterrupt` and `SystemExit` via the `Exception` base. However, `KeyboardInterrupt` and `SystemExit` derive from `BaseException`, not `Exception`, so they are NOT caught. This is actually correct behavior.
- **No exceptions can escape `choose_layout`**: All internal calls are wrapped in try/except blocks, and the method returns a valid string in all paths. Robust.

## 5. Performance & Scalability

- Called once during display initialization. All operations are O(1).
- No allocations, loops, or I/O beyond the terminal size query.
- No performance concerns.

## 6. Security & Safety

- No external input, file I/O, network access, or deserialization.
- Terminal size queries use standard OS/library interfaces.
- No security concerns.

## 7. Maintainability

- **Well-structured**: At 57 lines, this is compact and focused. Clear separation between terminal detection and layout decision.
- **Constants**: Min thresholds are class-level constants. Easy to adjust.
- **Unused data**: `TerminalInfo.unicode_ok` is computed but never used by consumers. This is dead logic that adds unnecessary complexity to `get_terminal_info`.
- **String-typed return**: `choose_layout` returns string literals. A `Literal["enhanced", "compact"]` or enum type annotation would provide type safety.
- **No dead code**: All methods are called (via `choose_layout` which calls `get_terminal_info` which calls `_get_terminal_size`).
- **Clean and readable**: Straightforward logic with clear intent.

## 8. Verdict

**SOUND**

This is a clean, well-structured utility with thorough fallback handling. Minor findings:
1. **Unused `unicode_ok` field**: Computed in `get_terminal_info` (lines 43-46) but never consumed by any caller.
2. **String-typed layout names**: `"enhanced"` / `"compact"` return values lack type safety (could use `Literal` or an enum).
3. **Bare `except Exception`** at line 33 is broader than necessary (could use `(AttributeError, ValueError, OSError)`), but does not catch `BaseException` subclasses, so it is functionally correct.

No bugs, no risks, no correctness issues.
