# Code Analysis: keisei/training/step_manager.py

## 1. Purpose & Role

`step_manager.py` manages the execution of individual training steps and the lifecycle of training episodes. It sits at the boundary between the PPO agent and the Shogi game environment, translating agent decisions into game moves and collecting experience for the replay buffer. It is one of 9 specialized managers orchestrated by the Trainer class and is invoked by `TrainingLoopManager` on every training timestep.

The file is 644 lines and defines two dataclasses (`EpisodeState`, `StepResult`) and one class (`StepManager`) with 9 methods.

## 2. Interface Contracts

### Exports
- `EpisodeState` (dataclass): Holds current observation, observation tensor, cumulative episode reward, and episode length.
- `StepResult` (dataclass): Holds next observation, reward, done flag, info dict, selected move details, log probability, value prediction, success flag, and optional error message.
- `StepManager` (class): The main class with the following public methods:
  - `execute_step(episode_state, global_timestep, logger_func) -> StepResult`
  - `handle_episode_end(episode_state, step_result, game_stats, total_episodes_completed, logger_func) -> Tuple[EpisodeState, Optional[str]]`
  - `reset_episode() -> EpisodeState`
  - `update_episode_state(episode_state, step_result) -> EpisodeState`

### Key Dependencies
- `keisei.config_schema.AppConfig` -- configuration object
- `keisei.core.experience_buffer.ExperienceBuffer` -- stores training experiences
- `keisei.core.ppo_agent.PPOAgent` -- action selection
- `keisei.shogi.ShogiGame` -- game environment
- `keisei.shogi.Color` -- player color enumeration
- `keisei.utils.PolicyOutputMapper` -- legal move masking and formatting
- `keisei.utils.format_move_with_description_enhanced` -- move display formatting
- `torch`, `numpy`, `time`

### Assumptions
- `ShogiGame.make_move()` returns a 4-tuple `(obs, reward, done, info)` (validated at line 230-231).
- `ShogiGame.reset()` returns an `np.ndarray` (validated at line 440).
- `ShogiGame.get_legal_moves()` returns a list of 5-tuples `(from_r, from_c, to_r, to_c, promotion_flag)`.
- The `logger_func` callable accepts `(message, also_to_wandb, wandb_data, log_level)` positionally.
- Moves are 5-element tuples where `(None, None, to_r, to_c, piece_type)` signals a drop.
- `selected_shogi_move[4]` is a boolean for promotion flag (line 227).

## 3. Correctness Analysis

### `__init__` (lines 61-96)
- Straightforward initialization. Stores all dependencies and initializes per-episode tracking counters.
- No issues detected.

### `execute_step` (lines 98-348)
This is the largest method at approximately 250 lines. It handles the full step lifecycle.

**No legal moves handling (lines 120-152):**
- When `legal_shogi_moves` is empty, the game is reset and a `StepResult` with `success=False` and `done=True` is returned. This is correct.
- However, at line 148, `success=False` is set but `done=True` is set. The caller (`training_loop_manager.py`) triggers `handle_episode_end` for any result where `done=True`. Since `success=False` results do NOT add experiences to the buffer (the `add()` call is skipped), this creates a clean terminal state with no partial data in the buffer. This is correct.

**Agent failure handling (lines 166-197):**
- When `selected_shogi_move is None`, the game is reset and a failure result is returned with `done=False`. This means the caller will NOT trigger `handle_episode_end`, and the next iteration will use the reset observation. However, this also means the episode counter is not incremented, and the `move_history`/`move_log` are not cleared. Since the game was reset, these stale counters could accumulate across failed resets within the same "logical episode." This is a minor correctness concern -- repeated agent failures without episode ends would leave growing `move_history` and `move_log` lists, along with stale capture/drop/promotion counters.

**Move tuple indexing (lines 205-212, 224-228):**
- Line 208-209: `selected_shogi_move[0]` and `selected_shogi_move[1]` are checked for `not None` before being used as coordinates. This is correct.
- Line 225: Drop detection checks `selected_shogi_move[0] is None and selected_shogi_move[1] is None`. This is correct for the move tuple format.
- Line 227: Promotion detection checks `isinstance(selected_shogi_move[4], bool) and selected_shogi_move[4]`. This is robust -- it guards against the case where index 4 is not a boolean (e.g., for drops where it might be a piece type string). However, there is a potential correctness issue: for drop moves, `selected_shogi_move[4]` is the piece type being dropped (a string or enum), not a boolean. `isinstance(str, bool)` returns `False`, so this will correctly evaluate to `False` for drops. The logic is sound.

**Move execution and capture tracking (lines 229-270):**
- `move_result` is validated to be a 4-tuple (line 230-231). If not, a `ValueError` is raised and caught by the outer try/except at line 303.
- Capture value map (lines 240-248): Defines piece values for tracking best captures. The map does not include "KING" -- capturing a king is not a normal Shogi move (checkmate ends the game before capture), so this omission is correct.
- `base_name = captured_name.replace("PROMOTED_", "")` (line 249): This strips the promotion prefix to look up value. This assumes the naming convention is `PROMOTED_<PIECE>`. If any piece uses a different prefix, the value lookup would return 0 (the default). This is acceptable.

**Experience buffer addition (lines 272-281):**
- `episode_state.current_obs_tensor.squeeze(0)` removes the batch dimension before storing. This is correct since the tensor was created with `unsqueeze(0)`.

**Error recovery (lines 303-348):**
- Only `ValueError` is caught. Other exceptions (e.g., `RuntimeError` from PyTorch, `TypeError` from incorrect move formatting) would propagate unhandled to the caller. This is intentionally narrow but may miss some failure modes.
- If the game reset in the error handler itself fails (line 334), the method returns the current (potentially corrupted) observation with `done=True` to force an episode end. This is a reasonable fallback.

### `handle_episode_end` (lines 350-481)
**Temporary stats copy (lines 378-409):**
- The method explicitly copies `game_stats` to avoid double-counting (comment "Fix B2" at line 378). The caller (`training_loop_manager.py` line 471-474) passes the current cumulative stats from `metrics_manager`, and after `handle_episode_end` returns, the caller separately calls `metrics_manager.update_episode_stats()` at line 492. This means the win/loss/draw increment happens in `handle_episode_end` only for the temporary copy used in logging, while the authoritative update happens in `metrics_manager.update_episode_stats()`. This is correct and avoids double-counting.

**WandB logging (lines 412-435):**
- The logged values (`black_wins_total`, `white_wins_total`, `draws_total`) use the temporary copy with the current game's result included. This is correct -- the logged values reflect the state AFTER this game.
- Win rates are calculated as fractions (not percentages), while `MetricsManager.get_win_rates()` returns percentages. These are separate logging paths and the inconsistency is intentional (WandB gets fractions, display gets percentages).

**Reset failure handling (lines 470-481):**
- When reset fails, the method returns the original `episode_state` unchanged. The caller will proceed with the same observation, which is problematic because the game state may be inconsistent. The caller (`_handle_step_outcome`) does not check whether the returned episode state is actually new. This could lead to an infinite loop if the game environment is in an unrecoverable state.

### `reset_episode` (lines 483-514)
- Duplicates the reset logic from `handle_episode_end` (lines 438-467). Both clear `move_history`, `move_log`, and all capture/drop/promotion counters. This is an explicit code duplication.
- Does not handle exceptions from `self.game.reset()`. If the game reset fails, the exception propagates to the caller. This is different from `handle_episode_end` which has try/except.

### `update_episode_state` (lines 516-534)
- Creates a new `EpisodeState` with accumulated reward and incremented length. This is a pure function (no side effects), which is correct.
- Does not validate that `step_result.success` is True before updating. The caller is responsible for checking this.

### `_prepare_demo_info` (lines 536-561)
- This method is dead code. It was the original demo info preparation logic that was superseded by the inline logic at lines 200-213 in `execute_step()`. It is never called from anywhere in the codebase (confirmed by grep). The inline version at lines 200-213 uses the actual selected move instead of the first legal move, which was the bug this method had.

### `_handle_demo_mode` (lines 563-608)
- `time.sleep(demo_delay)` at line 608 blocks the training loop. This is intentional for demo/display mode but will halt all training progress during the sleep.
- `getattr(self.game.current_player, "name", str(self.game.current_player))` at line 579-584: Double-defense for player name retrieval. Correct.

### `_determine_winner_and_reason` (lines 610-634)
- Handles the case where `reason_from_info` is "Tsumi" but `winner_from_info` is None by falling back to `self.game.winner`. This is a defensive fallback for incomplete game info dicts.
- The `hasattr` check for `self.game.winner` is defensive -- `ShogiGame` should always have this attribute, but the check prevents `AttributeError` if the game object is mocked or subclassed.

### `_format_game_outcome_message` (lines 636-644)
- Simple string formatting. The fallback at line 644 handles unexpected winner values (e.g., if winner is a non-None, non-"black", non-"white" string). This is correct.

## 4. Robustness & Error Handling

**Strengths:**
- The main `execute_step` method has comprehensive error handling with a try/except for `ValueError` and a nested try/except for reset failures.
- `handle_episode_end` handles game reset failures gracefully by returning the original episode state.
- Demo mode operations are wrapped in try/except blocks that silently ignore errors.

**Weaknesses:**
- `execute_step` only catches `ValueError` (line 303). A `RuntimeError` from PyTorch operations or a `TypeError` from malformed move data would propagate unhandled.
- `reset_episode()` has no error handling at all. If `self.game.reset()` raises, the caller gets an unhandled exception.
- When `handle_episode_end` fails to reset (line 470-481), it returns the old episode state. The caller does not distinguish between a successful reset and a failed one, which could cause the system to retry with stale state indefinitely.
- Agent failure at line 166 returns `done=False`, meaning episode tracking state (move history, counters) is not cleared despite the game being reset. This is a state inconsistency.

## 5. Performance & Scalability

- `time.sleep(demo_delay)` at line 608 blocks the training thread. This is only active when `config.display.display_moves` is True, which is a demo/streaming feature (default: False). In normal training, this code path is skipped.
- Tensor creation at lines 134-138, 179-183, 284-288, 315-319, 443-447 allocates new tensors on every step. This is standard for RL training and unavoidable given the API design, but the repeated `torch.tensor(...).unsqueeze(0)` pattern could be a minor hotspot.
- The `move_history` and `move_log` lists grow unbounded within an episode (cleared only on episode end). For very long games (up to `max_moves_per_game=500`), this is manageable.
- `value_map` dictionary (lines 240-248) is recreated on every step that involves a capture. This is a micro-optimization opportunity but has negligible impact.

## 6. Security & Safety

- No file I/O, network access, or deserialization occurs in this file.
- No external input validation concerns -- all inputs come from internal game engine and agent.
- No injection risks.

## 7. Maintainability

**Code Smells:**
- **`execute_step` is ~250 lines** (lines 98-348). This is the known long-function issue flagged in project notes. It handles legal move checking, agent action selection, demo mode logging, move execution, capture/drop/promotion tracking, experience buffering, and error recovery all in one method. Extracting the capture/drop/promotion tracking (lines 234-270) into a helper method would reduce complexity.
- **Duplicated reset logic**: The counter-clearing code at lines 449-460 in `handle_episode_end` is duplicated verbatim at lines 496-507 in `reset_episode`. There is no shared `_clear_episode_counters()` helper.
- **Dead code**: `_prepare_demo_info` (lines 536-561) is never called. It was replaced by inline logic at lines 200-213 but not removed.
- **`# Added import` comments** at lines 9, 13, 14, 19: These are stale comments from a previous editing session that provide no value.
- The `logger_func` type hint at line 102 (`Callable[[str, bool, Optional[Dict], str], None]`) is precise but fragile -- if the logging interface changes, this signature must be updated. The `handle_episode_end` version at line 356 uses the more relaxed `Callable[..., None]` instead, creating an inconsistency.

**Structure:**
- The dataclasses `EpisodeState` and `StepResult` are well-designed value objects that clearly communicate step outcomes.
- The class uses instance variables for per-episode tracking (capture counts, move history) which makes the state management clear but creates the reset-duplication problem.

## 8. Verdict

**NEEDS_ATTENTION**

Key findings:
1. **Dead code**: `_prepare_demo_info` method is never called, superseded by inline logic.
2. **Duplicated reset logic**: Episode counter/history clearing is duplicated between `handle_episode_end` and `reset_episode` with no shared helper.
3. **Long method**: `execute_step` at ~250 lines exceeds reasonable complexity, as flagged in project notes.
4. **Narrow exception handling**: Only `ValueError` is caught in `execute_step`; `RuntimeError` and `TypeError` propagate unhandled.
5. **State inconsistency on agent failure**: When the agent fails to select a move (line 166), the game is reset but episode tracking counters are not cleared because `done=False` is returned.
6. **Silent failure on reset**: `handle_episode_end` returns the old episode state when reset fails, with no way for the caller to detect the failure.
