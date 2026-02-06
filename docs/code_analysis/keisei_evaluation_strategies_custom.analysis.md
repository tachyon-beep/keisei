# Code Analysis: `keisei/evaluation/strategies/custom.py`

## 1. Purpose & Role

This module implements `CustomEvaluator`, a flexible evaluation strategy that allows users to define custom evaluation scenarios through configuration parameters. It supports three evaluation modes (round-robin, single-elimination, and custom sequence), multiple opponent configuration methods (direct list, opponent pool, single opponent, fallback default), and both file-based and in-memory evaluation paths. The class is registered with the `EvaluatorFactory` at module load time (line 414) under the `EvaluationStrategy.CUSTOM` key.

## 2. Interface Contracts

- **Extends**: `BaseEvaluator`, implementing the three required abstract methods: `evaluate()` (line 112), `evaluate_step()` (line 253), and `get_opponents()` (line 42).
- **`evaluate(agent_info, context)`**: Returns `EvaluationResult`. Determines evaluation mode from `strategy_params["evaluation_mode"]` (default "round_robin") and dispatches to the appropriate private method.
- **`evaluate_in_memory(agent_info, context, *, agent_weights, opponent_weights, opponent_info)`**: Overrides the base class default. Wraps weights in metadata and delegates to the same mode dispatch logic.
- **`evaluate_step(agent_info, opponent_info, context)`**: Returns a `GameResult` -- currently a placeholder with random outcomes (lines 271-286).
- **`get_opponents(context)`**: Returns `List[OpponentInfo]`, resolved from configuration using a priority cascade: direct list > opponent pool > single opponent > default random.
- **Configuration keys used**: `custom_opponents`, `opponent_pool_size`, `opponent_pool_type`, `single_opponent`, `evaluation_mode`, `games_per_opponent`, `custom_sequence`.

## 3. Correctness Analysis

### 3.1 CRITICAL: `evaluate_step()` Returns Random Results

Lines 271-286: The `evaluate_step()` method is a **placeholder implementation** that generates random outcomes:
```python
import random
winner = random.choice([0, 1, None])  # 0=agent, 1=opponent, None=draw

return GameResult(
    game_id=game_id,
    winner=winner,
    moves_count=random.randint(10, 100),
    duration_seconds=random.uniform(1.0, 10.0),
    ...
)
```
This means **all evaluation results from `CustomEvaluator` are fabricated**. Every call to `evaluate()`, `evaluate_in_memory()`, `_run_round_robin_evaluation()`, `_run_single_elimination_evaluation()`, and `_run_custom_sequence_evaluation()` produces meaningless results based on random dice rolls. The comments at lines 260-261 acknowledge this: "delegate to the base implementation which will raise NotImplementedError" and "This will need to be implemented based on the specific game engine integration." The comment is also misleading -- it says it would delegate to the base, but it actually returns random data instead.

### 3.2 `evaluate_in_memory()` Does Not Use Weights

Lines 166-251: The method wraps `agent_weights` into `agent_info.metadata["agent_weights"]` (line 189) and `opponent_weights` into `opponent_info.metadata["opponent_weights"]` (line 206). However, since `evaluate_step()` at line 271 is a random placeholder, these weights are never actually loaded or used. The metadata-based weight passing approach also differs fundamentally from `SingleOpponentEvaluator`, which stores weights on `self` and uses dedicated `_load_*_in_memory()` methods. If the placeholder were replaced with a real implementation, it would need to extract weights from metadata, creating an inconsistent pattern across evaluators.

### 3.3 Game ID Non-Uniqueness

Line 265: `game_id = f"custom_game_{context.session_id}_{opponent_info.name}"`. When multiple games are played against the same opponent (as in `_run_round_robin_evaluation` with `games_per_opponent > 1`), the game ID will be identical for all games against that opponent within the same session. This violates the expected uniqueness of `game_id` seen in `SingleOpponentEvaluator` which uses `uuid.uuid4().hex[:8]` for uniqueness.

### 3.4 Color Balancing Metadata Set but Never Used

Lines 309 and 396: Both `_run_round_robin_evaluation()` and `_run_custom_sequence_evaluation()` set `metadata["agent_plays_sente_in_eval_step"]` to alternate based on game index (`(game_num % 2) == 0`). However, `evaluate_step()` at line 253 never reads this metadata. The metadata key name (`agent_plays_sente_in_eval_step`) also differs from `SingleOpponentEvaluator`'s convention (`agent_plays_second`), indicating no shared color-handling protocol exists.

### 3.5 Single Elimination Logic

Lines 325-348: The single-elimination mode stops when `result.winner == 1` (opponent won). This is correct for the standard single-elimination semantic where the agent is eliminated on first loss. However, because `evaluate_step()` returns random winners, draws (`winner is None`) are silently treated as non-losses and the agent continues, which may not match user expectations for a "single elimination" mode.

## 4. Robustness & Error Handling

### 4.1 Exception Handling in Evaluation Modes

- `_run_round_robin_evaluation()` (lines 288-323): Per-game exceptions are caught and logged at lines 318-321, allowing remaining games to proceed. However, errors are logged but not collected into any error list that would appear in the final `EvaluationResult`. The `evaluate()` method at line 160 always sets `errors=[]`.
- `_run_single_elimination_evaluation()` (lines 325-348): On exception, the agent is eliminated (break at line 347), which may be overly aggressive -- a transient error ends the entire evaluation.
- `_run_custom_sequence_evaluation()` (lines 351-410): Per-game exceptions are caught at line 405-408. Same issue as round-robin: errors are logged but not collected.

### 4.2 No Input Validation in `evaluate()`

Unlike `SingleOpponentEvaluator.evaluate()` which calls `validate_agent()` and `validate_config()` before proceeding, `CustomEvaluator.evaluate()` (line 112) performs no validation. Invalid agent info or configuration will only surface as runtime errors during game execution.

### 4.3 Configuration Type Safety

`get_opponents()` at lines 42-110 defensively checks types: `isinstance(custom_opponents_config, list)` at line 50, `isinstance(opp_config, dict)` at line 52, `isinstance(single_opponent_config, dict)` at line 85. This is good defensive programming given that strategy_params is an untyped dictionary.

### 4.4 Fallback Cascade in `get_opponents()`

Lines 42-110 implement a four-level fallback: direct list, opponent pool, single opponent, default random. The `not opponents` checks at lines 69 and 84 ensure later methods only activate if earlier ones produced no results. The final fallback at line 96 logs a warning and provides a default. This is robust.

## 5. Performance & Scalability

### 5.1 Sequential Execution

All three evaluation modes execute games sequentially. The base class provides `run_concurrent_games()` but it is not used. For round-robin with many opponents and many games per opponent, this could be slow.

### 5.2 No Entity Caching

Similar to `SingleOpponentEvaluator`, there is no caching of loaded entities. However, since `evaluate_step()` is currently a placeholder that does not load any entities, this is not a realized concern yet.

### 5.3 Opponent List Reconstruction

In `_run_round_robin_evaluation()` (lines 301-311) and `_run_custom_sequence_evaluation()` (lines 388-398), a new `OpponentInfo` object is constructed for every game by copying the base opponent's metadata and adding game-specific fields. This involves dictionary unpacking and creation per game, which is lightweight but unnecessary for the current placeholder implementation.

## 6. Security & Safety

- **`import random` inside method** (line 271): The standard `random` module is used for placeholder game results. This is not cryptographically secure, but that is irrelevant for a placeholder. The concern is that this import suggests the placeholder might be used in production without replacement.
- **Unvalidated configuration**: Strategy params from user configuration are used without schema validation (e.g., `custom_sequence` at line 361 is expected to be a list of dicts but this is not enforced beyond type checks at line 371).
- **No file system access**: The placeholder does not read or write files.

## 7. Maintainability

### 7.1 Placeholder Implementation Shipped as Production Code

The most significant maintainability concern is that `evaluate_step()` is a non-functional placeholder. The file contains 414 lines of evaluation orchestration code (mode dispatch, opponent configuration, round-robin logic, single elimination logic, custom sequence logic, in-memory weight handling) that all ultimately delegate to a method producing random noise. There are no runtime warnings or `NotImplementedError` raises to alert users that this evaluator produces fake results.

### 7.2 Inconsistency with SingleOpponentEvaluator

- `SingleOpponentEvaluator` uses `uuid.uuid4()` for game IDs; `CustomEvaluator` uses concatenated strings.
- `SingleOpponentEvaluator` uses `agent_plays_second` metadata key; `CustomEvaluator` uses `agent_plays_sente_in_eval_step`.
- `SingleOpponentEvaluator` stores in-memory weights on `self`; `CustomEvaluator` passes them via metadata.
- `SingleOpponentEvaluator` validates agent and config before evaluation; `CustomEvaluator` does not.
- `SingleOpponentEvaluator` collects errors into the result; `CustomEvaluator` always returns `errors=[]`.

These inconsistencies make it harder to maintain a uniform evaluator contract across strategies.

### 7.3 Module-Level Registration Side Effect

Line 413-414: Factory registration at module import time, consistent with other strategy modules. The import is clean (imports `EvaluationStrategy` and `EvaluatorFactory` from `..core`).

### 7.4 Unused Import

Line 11: `import torch` is imported at the module level but is only used in the `evaluate_in_memory()` method signature's type hints (`Dict[str, torch.Tensor]`). This could be moved under `TYPE_CHECKING` for lighter import weight, though as a practical matter it is already loaded elsewhere.

### 7.5 Code Structure

The file is well-organized with clear method separation:
- Opponent resolution: `get_opponents()` (lines 42-110)
- Main entry points: `evaluate()`, `evaluate_in_memory()` (lines 112-251)
- Game execution: `evaluate_step()` (lines 253-286)
- Evaluation modes: three private methods (lines 288-410)

## 8. Verdict

**CRITICAL**

The `evaluate_step()` method at lines 253-286 is a **non-functional placeholder** that returns random game outcomes (random winner, random move count, random duration). This means the entire `CustomEvaluator` produces fabricated evaluation results with no actual game play. There is no warning, no `NotImplementedError`, and no logging to indicate that results are fake. Any code path that uses `EvaluationStrategy.CUSTOM` will silently produce meaningless metrics. The 328 lines of evaluation orchestration code (modes, opponent configuration, in-memory support) built around this placeholder create a false impression of functionality. Additionally, errors during game execution are logged but never collected into the `EvaluationResult.errors` list, so error visibility is lost.
