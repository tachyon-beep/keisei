# Code Analysis: `keisei/evaluation/strategies/tournament.py`

## 1. Purpose & Role

`TournamentEvaluator` implements a round-robin tournament evaluation strategy where an agent is pitted against multiple opponents from a configured pool. It extends `BaseEvaluator` and manages full game lifecycle: loading agents/opponents, executing game loops, alternating sente/gote assignments, and calculating per-opponent standings. It also provides an `evaluate_in_memory` path for evaluating with pre-loaded model weights rather than loading from disk.

## 2. Interface Contracts

### Class: `TournamentEvaluator(BaseEvaluator)`

**Constructor (line 64):**
- Accepts `EvaluationConfig`, stores it and creates a `PolicyOutputMapper`.

**Abstract method implementations from `BaseEvaluator`:**
- `evaluate(agent_info, context)` -> `EvaluationResult` (line 282): Runs full tournament.
- `evaluate_step(agent_info, opponent_info, context)` -> `GameResult` (line 104): Plays one game.
- `get_opponents(context)` -> `List[OpponentInfo]` (line 69): Returns configured opponents or a default random opponent.

**Additional public methods:**
- `evaluate_in_memory(agent_info, context, *, agent_weights, opponent_weights, opponent_info)` -> `EvaluationResult` (line 350): In-memory variant of `evaluate`.
- `validate_config()` -> `bool` (line 460): Checks `opponent_pool_config`.
- `evaluate_step_in_memory(agent_info, opponent_info, context)` -> `GameResult` (line 800): Single game with in-memory weights.

**Winner convention:** `0` = agent wins, `1` = opponent wins, `None` = draw.

**Registration (line 828-830):** Registers as `EvaluationStrategy.TOURNAMENT` with `EvaluatorFactory`.

## 3. Correctness Analysis

### 3.1 In-memory evaluation is a no-op (lines 800-823)
`evaluate_step_in_memory` extracts `agent_weights` and `opponent_weights` from metadata at lines 808-815, but then discards them entirely and falls back to `self.evaluate_step()` at line 823. The debug log message at line 821 acknowledges this: "currently falls back to regular evaluation". This means the entire `evaluate_in_memory` pipeline (lines 350-458), which carefully packs weights into metadata, has no functional effect -- all in-memory evaluation calls behave identically to regular file-based evaluation. This is not a bug per se (the comment says it "could be enhanced later"), but it is dead/misleading code that gives callers a false expectation of functionality.

### 3.2 `hasattr(self, "evaluate_step_in_memory")` always True (line 433)
At line 433, `evaluate_in_memory` checks `if hasattr(self, "evaluate_step_in_memory")`. Since `evaluate_step_in_memory` is defined at line 800 of this same class, this condition is always `True`. The `else` branch (lines 437-439) is unreachable dead code.

### 3.3 Redundant `opponent_info is not None` checks (lines 414-417)
At line 414, the condition checks `opponent_weights is not None and opponent_info is not None`. Inside the `if` block at lines 416-418, there are further `if opponent_info` ternary checks, which are redundant given the outer guard already ensures `opponent_info is not None`.

### 3.4 Game loop return type mismatch (lines 176-203)
`_run_tournament_game_loop` has return type annotation `-> int` (line 225). However, `_execute_tournament_game` at lines 181-196 handles the return value as potentially a `dict` (for test mocks). This works at runtime due to dynamic typing, but the type annotation is incorrect -- the method can return either `int` or `dict` depending on internal behavior (or mocking).

### 3.5 Winner determination for draws via max moves (line 230)
When the game loop terminates because `moves_count >= max_moves` (line 230, the while condition), the `_determine_winner` call at line 202 returns `None` only if `game.game_over` is False or `game.winner` is `None`. If the game engine's internal state does not set `game.game_over` upon reaching the internal max_moves, and the tournament's game loop simply exits via its own `max_moves` check, the game state may be inconsistent (loop exits but `game.game_over` is False). The `ShogiGame` constructor accepts `max_moves_per_game`, but the tournament creates `ShogiGame()` at line 161 with the default (500), matching the hardcoded max at line 228. However, if these defaults ever diverge, there could be a dual-termination-condition problem.

### 3.6 `_validate_and_make_move` sets game state directly (lines 570-573, 587-589)
When an illegal move or execution error occurs, the method directly mutates `game.game_over`, `game.winner`, and `game.termination_reason`. This bypasses any internal game engine state management, which could leave the ShogiGame object in an inconsistent state if other code later inspects it. This is acceptable for termination scenarios but is a fragile coupling to the game engine's internal representation.

### 3.7 `select_action` return value handling (line 538)
The action extraction at line 538 handles both tuple and non-tuple returns: `move = move_tuple[0] if isinstance(move_tuple, tuple) else move_tuple`. However, if `select_action` returns a tuple where element `[0]` is an action index (integer) rather than a move object, the move would fail the `move not in legal_moves` check at line 561 later, since `legal_moves` contains move objects, not indices. This suggests there may be an implicit contract that agents using `select_action` return move objects or that an unmapped action index would be caught by the validation check.

## 4. Robustness & Error Handling

### 4.1 Exception handling in `evaluate_step` (lines 121-140)
Catches `ValueError, TypeError, RuntimeError, AttributeError`. There is an unusual secondary try/except at lines 124-127 catching `StopIteration` from `time.time()`, with a comment explaining this is for mock exhaustion in tests. This is defensive coding against a test infrastructure issue leaking into production code.

### 4.2 Broad exception pattern in game orchestration (lines 739)
`_play_games_against_opponent` catches the same four exception types. Errors are collected as strings in the `errors` list, allowing partial tournament results even when individual games fail. This is a reasonable resilience pattern.

### 4.3 No timeout enforcement
There is no per-game timeout mechanism. The hardcoded `max_moves = 500` at line 228 is the only game termination safeguard. If `game.make_move()` or `game.get_legal_moves()` hangs, the game loop will hang indefinitely. The `BaseEvaluator` has `timeout_per_game` in its config, but `TournamentEvaluator` does not read or enforce it.

### 4.4 No validation of opponent_pool_config contents
`validate_config()` (lines 460-477) checks that `opponent_pool_config` is a list but does not validate the structure of individual entries. Malformed entries (e.g., missing "type" key) would only surface at game time.

## 5. Performance & Scalability

### 5.1 Sequential game execution
All games against all opponents are executed sequentially in nested loops (lines 327-332). The base class provides `run_concurrent_games` for parallel execution, but it is unused. For large opponent pools with many games per opponent, this is a significant bottleneck.

### 5.2 Entity reloading per game
`_execute_tournament_game` loads both agent and opponent at lines 155-158 for every single game. For trained agents (PPO agents loaded from checkpoints), this involves repeated disk I/O and model deserialization. There is no caching of loaded entities across games against the same opponent.

### 5.3 Legal mask creation per turn
`policy_mapper.get_legal_mask()` is called every turn at line 248 for agent-type players. This is expected and necessary, but the mask is a full 13,527-dimensional tensor. For long games (up to 500 moves), this accumulates.

### 5.4 Duplicate configuration reads
`_load_tournament_opponents()` at line 594 calls `self.config.get_strategy_param("opponent_pool_config", [])` twice -- once for the empty check at line 594, and again in the loop at line 602. This is a minor inefficiency (dictionary lookups are fast) but is duplicated logic.

## 6. Security & Safety

### 6.1 Checkpoint path passed without validation
At lines 494-498, checkpoint paths from configuration are passed directly to `load_evaluation_agent()`. There is no path sanitization or validation that the path points to a legitimate model file. Arbitrary paths could lead to loading untrusted pickle files (PyTorch's default serialization).

### 6.2 `agent_instance` from metadata
At line 493, if `agent_info.metadata` contains an `agent_instance` key, that object is used directly without type checking. Any object placed in metadata under this key will be treated as a valid player entity.

## 7. Maintainability

### 7.1 Duplicated termination reason constants (lines 36-51)
The same constants are duplicated in `ladder.py` (with a comment acknowledging the copy at line 31 of ladder.py). These should be in a shared module.

### 7.2 Duplicated game orchestration methods
`_play_games_against_opponent` (lines 698-744) and `_play_games_against_opponent_in_memory` (lines 746-798) are nearly identical, differing only in which evaluate_step variant they call. This is a DRY violation.

### 7.3 Mock-specific logic in production code
Lines 181-196 contain special handling for when `_run_tournament_game_loop` returns a `dict` instead of `int`, explicitly documented as "This is a test mock returning a game outcome dictionary". Test-specific branching in production code increases complexity and couples the implementation to test infrastructure.

### 7.4 Method count and responsibility
The class has 14 methods (including inherited). The `evaluate_in_memory` path adds 4 methods that essentially duplicate the regular path. The class could benefit from extracting game execution into a separate component.

### 7.5 Type annotations
Union type syntax `AgentInfo | OpponentInfo` at line 481 requires Python 3.10+. The codebase targets Python 3.13 per the project memory, so this is acceptable but worth noting for compatibility.

### 7.6 File length
At 831 lines, this file is on the larger side. The in-memory evaluation duplication accounts for roughly 200 lines of near-identical code.

## 8. Verdict

**NEEDS_ATTENTION**

Primary concerns:
1. `evaluate_step_in_memory` is effectively a no-op that falls back to regular evaluation, making the entire in-memory evaluation pipeline misleading dead code.
2. No per-game timeout enforcement despite `timeout_per_game` being available in the config.
3. Agent/opponent entities are reloaded from disk for every single game with no caching.
4. Sequential-only game execution ignores available parallelism infrastructure.
5. Significant code duplication between regular and in-memory paths.
6. Test-mock-specific branching embedded in production logic.

The core game execution logic (game loop, move validation, winner determination) is sound. The tournament standings calculation is correct. The error handling is generally good with graceful degradation. However, the in-memory evaluation facade and the DRY violations represent meaningful maintainability and correctness risks.
