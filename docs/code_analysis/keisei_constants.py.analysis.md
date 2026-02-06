# Analysis: keisei/constants.py

**Lines:** 183
**Role:** Centralizes application-wide constants and magic numbers used throughout the codebase. Organized by functional area: Shogi game constants, action/observation spaces, training parameters, test constants, file patterns, numerical epsilons, resource limits, and error thresholds. Also defines `GameTerminationReason` as a plain class.
**Key dependencies:**
- Imports from: nothing (zero imports -- this is a leaf module)
- Imported by: `keisei/shogi/features.py` (MOVE_COUNT_NORMALIZATION_FACTOR), `keisei/training/models/resnet_tower.py` (SHOGI_BOARD_SIZE, SHOGI_BOARD_SQUARES), and indirectly by test files via `keisei.constants`
**Analysis depth:** FULL

## Summary

This file has one clear bug (duplicate `TEST_BATCH_SIZE` definition on lines 75 and 152 with different values), significant duplication of observation constants already defined in `shogi_core_definitions.py`, and an unused `GameTerminationReason` class that shadows the real `TerminationReason` enum. The file is over 60% test constants by line count, which raises questions about whether test-specific values belong in a production constants module. No security concerns, but the duplicate constant is a data integrity risk for any test that imports it.

## Critical Findings

### [75+152] `TEST_BATCH_SIZE` is defined twice with different values

**What:** `TEST_BATCH_SIZE` is defined on line 75 as `16` and again on line 152 as `32`. Python allows this without error; the second definition silently overwrites the first. Any code that imports `TEST_BATCH_SIZE` gets `32`, not `16`.

**Why it matters:** This is a silent data corruption bug. If the first definition (16) was the intended value for some tests and the second (32) for others, one set of tests is using the wrong value. If someone later "fixes" this by removing the duplicate, tests that depend on the current effective value (32) could break, while tests that depended on the original value (16) have been silently wrong this whole time. Searching the codebase shows `tests/core/test_ppo_agent_edge_cases.py` imports and uses `TEST_BATCH_SIZE` (effective value: 32) as a batch size parameter -- this may or may not be the originally intended value.

**Evidence:**
```python
# Line 75:
TEST_BATCH_SIZE = 16

# Line 152:
TEST_BATCH_SIZE = 32  # Batch size for training tests
```

Only the second value (32) is effective at runtime. The first definition on line 75 is dead code that exists only to mislead.

## Warnings

### [20-29] Observation channel constants duplicate `shogi_core_definitions.py` with different naming conventions

**What:** Constants `OBS_CURRENT_PLAYER_UNPROMOTED_START`, `OBS_CURRENT_PLAYER_PROMOTED_START`, `OBS_OPPONENT_UNPROMOTED_START`, etc. in `constants.py` duplicate the semantically identical constants `OBS_CURR_PLAYER_UNPROMOTED_START`, `OBS_CURR_PLAYER_PROMOTED_START`, `OBS_OPP_PLAYER_UNPROMOTED_START`, etc. in `shogi/shogi_core_definitions.py`. The values are identical (both files define the same channel indices), but the names differ (`CURRENT_PLAYER` vs `CURR_PLAYER`, `OPPONENT` vs `OPP_PLAYER`).

**Why it matters:** Having two sets of identically-valued constants with different names creates a maintenance hazard. If someone changes a channel layout in `shogi_core_definitions.py` (the authoritative source, since it is actually imported by the game engine code in `shogi_game_io.py` and `features.py`), they would have no reason to update `constants.py`. The constants in `constants.py` would become stale and wrong. Currently, nothing in the production codebase imports the observation constants from `constants.py` (grep confirms only `MOVE_COUNT_NORMALIZATION_FACTOR`, `SHOGI_BOARD_SIZE`, and `SHOGI_BOARD_SQUARES` are imported from constants.py), so the duplicated OBS constants in `constants.py` are entirely dead code.

**Evidence:** Grepping the codebase for `from.*constants.*import.*OBS_` returns zero results. All actual observation constant usage imports from `shogi_core_definitions`.

### [45-51] `GameTerminationReason` is a plain class that shadows the real `TerminationReason` Enum

**What:** `GameTerminationReason` is a plain class with three string class attributes (`INVALID_MOVE`, `STALEMATE`, `POLICY_ERROR`). The real, comprehensive termination reason type is `TerminationReason` in `shogi/shogi_core_definitions.py`, which is a proper `Enum` with 10 values including `CHECKMATE`, `STALEMATE`, `REPETITION`, `MAX_MOVES_EXCEEDED`, etc.

**Why it matters:** Grepping the codebase confirms that `GameTerminationReason` is never imported or referenced anywhere outside `constants.py`. It is dead code. However, its existence is confusing because a maintainer might discover it and use it instead of the authoritative `TerminationReason` enum, leading to incorrect termination handling. The overlap on `STALEMATE` has different value strings: `GameTerminationReason.STALEMATE = "stalemate"` vs `TerminationReason.STALEMATE = "stalemate"` (matching), but `GameTerminationReason.INVALID_MOVE = "invalid_move"` vs `TerminationReason.ILLEGAL_MOVE = "illegal_move"` (different name and value for what appears to be the same concept).

**Evidence:**
```python
# constants.py
class GameTerminationReason:
    INVALID_MOVE = "invalid_move"
    STALEMATE = "stalemate"
    POLICY_ERROR = "policy_error"

# shogi_core_definitions.py
class TerminationReason(Enum):
    CHECKMATE = "Tsumi"
    STALEMATE = "stalemate"
    ...
    ILLEGAL_MOVE = "illegal_move"
```

`GameTerminationReason` is never imported by any file in the codebase.

### [14-15] `FULL_ACTION_SPACE` and `ALTERNATIVE_ACTION_SPACE` have no connection to config_schema.py's `num_actions_total`

**What:** `FULL_ACTION_SPACE = 13527` and `ALTERNATIVE_ACTION_SPACE = 6480` are defined here, while `EnvConfig.num_actions_total` defaults to `13527` in `config_schema.py`. These are not linked -- the config default is a hardcoded integer, not a reference to this constant.

**Why it matters:** If the action space calculation changes, three locations must be updated: `constants.py`, `config_schema.py`, and `default_config.yaml`. The comment on line 15 explains the derivation (`12960 + 567 = 13527`), but this derivation is not enforced programmatically. The constant `ALTERNATIVE_ACTION_SPACE = 6480` is not used anywhere in the codebase (confirmed by grep).

### [31-63] Many empty section headers with no constants defined

**What:** Lines 31-63 contain numerous section comment headers (e.g., "# Training constants", "# GAE and advantage computation", "# Model architecture defaults", "# Rendering and display defaults", etc.) followed by no actual constants. The sections are empty placeholders.

**Why it matters:** These empty sections suggest the file was designed as a comprehensive constants repository but was never fully populated. Constants that logically belong in these sections remain as magic numbers elsewhere in the codebase. The empty sections add visual noise without value.

## Observations

### [66-183] Over 60% of the file is test-specific constants

**What:** Lines 66 through 183 (117 lines out of 183 total) define constants prefixed with `TEST_`. These include test thresholds, buffer sizes, reward values, learning rates, scheduler parameters, etc.

**Why it matters:** Including test-specific constants in a production module (`keisei/constants.py`) violates separation of concerns. These constants should be in a test-specific module (e.g., `tests/test_constants.py` or `tests/conftest.py`). Their presence in the production module means:
1. They are included in the installed package.
2. Changes to test constants require editing a production file.
3. The file is harder to navigate for production constant needs.
4. Some test constants duplicate production defaults (e.g., `TEST_GAE_LAMBDA_DEFAULT = 0.95` duplicates `TrainingConfig.lambda_gae`'s default of 0.95).

### [90-92] Epsilon constants are well-organized but naming could be more specific

**What:** `EPSILON_SMALL = 1e-8`, `EPSILON_MEDIUM = 1e-6`, `EPSILON_LARGE = 1e-4` are defined. The names describe relative magnitude but not intended use case.

**Why it matters:** A developer choosing between these would not know which to use for numerical stability in softmax, advantage normalization, or floating-point comparison. Names like `NUMERICAL_STABILITY_EPSILON`, `COMPARISON_EPSILON`, and `LOOSE_COMPARISON_EPSILON` would be more self-documenting. However, as general-purpose constants, the current naming is acceptable.

### [83-85] File pattern constants are only partially used

**What:** `CHECKPOINT_FILE_PATTERN = "*.pt"`, `CONFIG_FILE_EXTENSION = ".yaml"`, `LOG_FILE_EXTENSION = ".log"` are defined. The actual log file paths in `config_schema.py` use `.txt` extension (`log_file: "logs/training_log.txt"`), not `.log`.

**Why it matters:** The `LOG_FILE_EXTENSION` constant is inconsistent with actual usage. This is minor but could confuse someone who tries to use the constant and finds that actual log files have a different extension.

### [9-10] `SHOGI_BOARD_SQUARES` is correctly computed from `SHOGI_BOARD_SIZE`

**What:** `SHOGI_BOARD_SQUARES = SHOGI_BOARD_SIZE * SHOGI_BOARD_SIZE` is clean and correct.

### [78] `SEED_OFFSET_MULTIPLIER` matches `ParallelConfig.worker_seed_offset` default

**What:** `SEED_OFFSET_MULTIPLIER = 1000` in constants.py matches `worker_seed_offset: int = Field(1000, ...)` in config_schema.py, but neither references the other.

## Verdict

**Status:** NEEDS_ATTENTION
**Recommended action:**
1. **Immediate:** Fix the duplicate `TEST_BATCH_SIZE` definition. Determine which value (16 or 32) is correct, remove the other, and verify test behavior.
2. **Short-term:** Remove the dead `GameTerminationReason` class and the dead `OBS_*` constants that duplicate `shogi_core_definitions.py`.
3. **Medium-term:** Move all `TEST_*` constants to a test-specific module to clean up the production constants file.
4. **Medium-term:** Link `FULL_ACTION_SPACE` to `EnvConfig.num_actions_total` programmatically, or at minimum add a comment cross-referencing the two locations.
5. **Cleanup:** Remove empty section comment headers that have no constants beneath them.
**Confidence:** HIGH -- The duplicate `TEST_BATCH_SIZE` is directly observable in the source. The dead code findings are confirmed by comprehensive grep searches across the codebase. The `ALTERNATIVE_ACTION_SPACE` and `GameTerminationReason` being unused is confirmed by zero import references.
