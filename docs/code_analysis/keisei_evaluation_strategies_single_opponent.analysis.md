# Code Analysis: `keisei/evaluation/strategies/single_opponent.py`

## 1. Purpose & Role

This module implements `SingleOpponentEvaluator`, the primary evaluation strategy for running a specified number of games between an agent and a single opponent. It is the largest strategy implementation at 894 lines and supports both file-based checkpoint loading and in-memory weight evaluation. It handles color balancing (agent as Sente or Gote), game loop execution, move validation, winner determination, and analytics calculation. The class is registered with the `EvaluatorFactory` at module load time (line 894).

## 2. Interface Contracts

- **Extends**: `BaseEvaluator` (abstract base class), implementing the three required abstract methods: `evaluate()` (line 234), `evaluate_step()` (line 585), and `get_opponents()` (line 755).
- **Constructor**: Accepts `EvaluationConfig`, initializes a `PolicyOutputMapper`, and optional weight storage attributes (lines 52-61).
- **`evaluate(agent_info, context)`**: Returns `EvaluationResult` containing all game results, summary statistics, analytics, and errors. Delegates to `evaluate_step()` per game.
- **`evaluate_in_memory(agent_info, context, *, agent_weights, opponent_weights, opponent_info)`**: Same as `evaluate()` but uses in-memory model weights, delegates to `evaluate_step_in_memory()`.
- **`evaluate_step(agent_info, opponent_info, context)`**: Runs a single game, returns `GameResult`.
- **`evaluate_step_in_memory(...)`**: Same as `evaluate_step()` but loads entities from in-memory weights.
- **`get_opponents(context)`**: Returns a single-element list of `OpponentInfo`.
- **Factory registration**: Line 894 registers with `EvaluatorFactory` under `EvaluationStrategy.SINGLE_OPPONENT`.

## 3. Correctness Analysis

### 3.1 Color Balancing Logic -- Imbalance Calculation Bug

Lines 800-814, `_calculate_game_distribution()`:
```python
imbalance = abs(half_games - (half_games + remainder)) / total_games
```
This computes `abs(-remainder) / total_games`, which simplifies to `remainder / total_games`. Since `remainder = total_games % 2`, the imbalance is always either `0` (even number of games) or `1/total_games` (odd number). For any `total_games >= 3`, this value `1/N` is always less than the default tolerance of `0.1`. The warning at lines 810-814 can therefore only trigger when `total_games` is exactly 1 (imbalance = 1.0) or when a custom tolerance < `1/total_games` is set. The calculation is technically correct but the warning is effectively dead code for typical use cases, potentially masking configuration issues.

### 3.2 Winner Mapping Logic -- Correct but Duplicated

The winner remapping logic at lines 629-637 and lines 712-720 is identical across `evaluate_step()` and `evaluate_step_in_memory()`. The mapping from Sente/Gote winner to agent/opponent winner is correct:
- When `game_outcome["winner"]` is 0 (Sente won) and agent plays Sente, agent wins (code 0).
- When `game_outcome["winner"]` is 1 (Gote won) and agent plays Gote, agent wins (code 0).
- Draw (`winner is None`) maps to `final_winner_code = None`.

### 3.3 Game Loop -- No-Legal-Moves Handling

At lines 174-180, when `legal_moves` is empty, `game.game_over` is set to True but `game.winner` is not explicitly set. The value of `game.winner` depends on whatever state `ShogiGame` leaves it in. If `ShogiGame` does not set `winner` when no legal moves exist (e.g., stalemate), the winner will be `None`, correctly treated as a draw. However, in standard Shogi rules, having no legal moves is a loss (not a draw). The game engine may handle this, but the evaluator does not verify or enforce this rule.

### 3.4 `_run_game_loop` Parameter Naming

Line 162-163: The method signature uses `agent_player` and `opponent_player`, but after color swapping in `evaluate_step()` (lines 614-619), these become `sente_player` and `gote_player` respectively. Inside `_run_game_loop`, the `player_map` at line 168 maps `{0: agent_player, 1: opponent_player}`. This is semantically correct because the parameters actually represent Sente/Gote after swapping, but the parameter names `agent_player`/`opponent_player` are misleading since they may not correspond to the logical agent/opponent.

### 3.5 Opponent Type Mapping Inconsistency

In `evaluate()` (line 273), the opponent type mapping falls back to `"random"` when the opponent name is not found in the mapping dictionary. In `get_opponents()` (line 768), the fallback is `"unknown"`, with an additional check at lines 772-773 that promotes `"unknown"` to `"ppo"` if a checkpoint path exists. These two code paths use different mapping logic for the same conceptual task, creating a potential inconsistency in how opponents are typed.

### 3.6 `agent_plays_second` Metadata Not Used in `_run_game_loop`

The `agent_plays_second` flag is set in `opponent_info.metadata` (line 325, 450) and read in `evaluate_step()` (line 599) and `evaluate_step_in_memory()` (line 682). This is correctly used to swap Sente/Gote assignment before calling `_run_game_loop()`. The `_run_game_loop` itself does not need to know about color assignment -- it just plays from Sente's perspective. This is correct.

## 4. Robustness & Error Handling

### 4.1 Exception Handling in Game Execution

- `_get_player_action()` (line 96): Propagates exceptions upward (no try/except). The caller at line 199 catches exceptions and terminates the game.
- `_validate_and_make_move()` (line 119): Handles three failure modes: (1) illegal/no move, (2) `test_move` failure, (3) `make_move` exception. All three correctly set `game.game_over = True`, assign the winner as the other player, and set a termination reason.
- `_run_game_loop()` (line 162): Catches action selection errors at line 203 and terminates the game.
- `evaluate_step()` (lines 585-666): Wrapped in a try/except that returns a "no winner, 0 moves" `GameResult` on catastrophic failure, ensuring the evaluation session can continue.
- `evaluate()` (lines 234-367): Per-game exceptions are caught individually (lines 296-298, 340-345). A top-level try/except at line 347 catches critical errors affecting the entire evaluation.

### 4.2 State Mutation on Instance

Lines 399-400 store `agent_weights` and `opponent_weights` on the instance (`self.agent_weights`, `self.opponent_weights`). These are cleaned up in a `finally` block at lines 489-492. This is correct for cleanup but makes the evaluator non-thread-safe during in-memory evaluation. Concurrent calls to `evaluate_in_memory()` would corrupt shared state.

### 4.3 Device Determination

Lines 186-194: The code attempts to determine the device of the current player entity. It first checks for a `device` attribute of type `torch.device`, then falls back to constructing a `torch.device` from a string attribute. If the entity has no `device` attribute at all, it defaults to `"cpu"`. This is defensive but relies on duck-typing with `hasattr` and `getattr`, which is fragile.

### 4.4 `time` Import Inside Method

Lines 594, 677: `import time` is performed inside `evaluate_step()` and `evaluate_step_in_memory()`. This is a function-level import of a stdlib module. While harmless (Python caches module imports), it is unconventional and incurs a minor overhead per call.

## 5. Performance & Scalability

### 5.1 Sequential Game Execution

Both `evaluate()` and `evaluate_in_memory()` run games sequentially in for-loops (lines 291-298, 302-345, 431-441, 444-467). The base class provides a `run_concurrent_games()` method with semaphore-based parallelism, but `SingleOpponentEvaluator` does not use it. For large numbers of evaluation games, this is a significant bottleneck.

### 5.2 Entity Re-Loading Per Game

In `evaluate_step()` (lines 606-611) and `evaluate_step_in_memory()` (lines 689-694), both the agent and opponent are loaded from scratch for every single game. There is no caching of loaded entities across games within an evaluation session. For checkpoint-based loading, this involves reading and deserializing model weights from disk for every game.

### 5.3 PolicyOutputMapper Created Per Evaluator

Line 56: A new `PolicyOutputMapper` is created for each evaluator instance. If the mapper is expensive to construct (e.g., building action index tables), this adds initialization cost. However, the evaluator is typically instantiated once per evaluation session, so this is likely acceptable.

### 5.4 Median Calculation

Lines 861, 868 in `_calculate_analytics()`: Median is computed via `sorted(list)[len(list) // 2]`, which is O(n log n). For large game counts, this is fine. However, for even-length lists, this returns the upper-middle element rather than the average of the two middle elements (the standard median definition).

## 6. Security & Safety

- **No file path injection**: Checkpoint paths come from configuration, not user input in a web context.
- **No network exposure**: The evaluator does not open sockets or accept external connections.
- **Error messages**: Exception messages are logged with `exc_info=True` (e.g., line 655), which dumps full stack traces. In production logging, this could expose internal paths or state. This is appropriate for a training system but should be considered for deployment contexts.
- **`torch` device strings**: Device strings from configuration are passed directly to `torch.device()` without sanitization. Malicious device strings are not a practical concern for this system.

## 7. Maintainability

### 7.1 Substantial Code Duplication

The module contains significant duplication between file-based and in-memory evaluation paths:
- `evaluate()` (lines 234-367) and `evaluate_in_memory()` (lines 369-492) share nearly identical game loop/distribution/error-handling logic.
- `evaluate_step()` (lines 585-666) and `evaluate_step_in_memory()` (lines 668-753) are structurally identical except for the entity loading method called.
- `_load_evaluation_entity()` (lines 63-94) and `_load_evaluation_entity_in_memory()` (lines 494-510) are parallel dispatchers.
- `_load_agent_in_memory()` (lines 512-543) and `_load_opponent_in_memory()` (lines 545-583) duplicate loading logic with minor variations.

This duplication means any bug fix or behavioral change must be applied in multiple places, increasing the risk of drift.

### 7.2 Comments as Design Notes

Lines 305-320 contain extensive inline comments that read more like design deliberation than code documentation. These comments discuss implementation alternatives and unresolved design questions (e.g., "This needs careful handling", "This part needs more thought"). While informative, they suggest the color-swapping feature was implemented tentatively.

### 7.3 Module-Level Registration

Line 892-894 perform a module-level import and factory registration:
```python
from ..core import EvaluationStrategy, EvaluatorFactory
EvaluatorFactory.register(EvaluationStrategy.SINGLE_OPPONENT, SingleOpponentEvaluator)
```
This is a side effect at import time. The import ordering matters: if `EvaluatorFactory` is not yet available when `single_opponent.py` is imported, this will fail. The current import chain (strategies `__init__.py` imports single_opponent, which imports from `..core`) relies on `core/__init__.py` being fully loaded first, which works because `strategies/__init__.py` is not imported from `core/__init__.py`.

### 7.4 Type Hints

- Line 68: Return type is `Any`, losing type safety for loaded entities.
- Line 97: `player_entity` is typed as `Any`, `legal_mask` as `Any`, return as `Any`.
- Line 164: Both `agent_player` and `opponent_player` are `Any`.
- Extensive use of `Any` throughout reduces the value of static type checking.

### 7.5 File Length

At 894 lines, this is a large single file. The core game loop logic (`_run_game_loop`, `_get_player_action`, `_validate_and_make_move`) could be factored out, and the in-memory variants could share more infrastructure with the file-based variants.

## 8. Verdict

**NEEDS_ATTENTION**

The module is functionally correct for its primary use case (single-opponent evaluation with color balancing), and error handling is thorough. However, there are several concerns:

1. **Significant code duplication** between file-based and in-memory paths, creating maintenance risk.
2. **Entity re-loading per game** is a performance concern for checkpoint-based evaluation.
3. **Non-thread-safe instance state** (`self.agent_weights`, `self.opponent_weights`) during in-memory evaluation.
4. **Opponent type mapping inconsistency** between `evaluate()` and `get_opponents()`.
5. **Misleading parameter names** in `_run_game_loop` after color swapping.
6. **Near-dead imbalance warning** in `_calculate_game_distribution()`.
7. **Incorrect median calculation** (integer division for even-length lists).
