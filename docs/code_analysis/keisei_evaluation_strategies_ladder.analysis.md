# Code Analysis: `keisei/evaluation/strategies/ladder.py`

## 1. Purpose & Role

`LadderEvaluator` implements an ELO-based ladder evaluation strategy where an agent is matched against opponents selected from a pool based on rating proximity. It tracks ELO ratings across matches, updates them using a simplified ELO formula, and produces evaluation results that include rating snapshots and deltas. The file also contains a placeholder `EloTracker` class (lines 54-97) that provides basic ELO rating management, despite there being a real `elo_tracker` module in `evaluation/analytics/`.

## 2. Interface Contracts

### Class: `EloTracker` (placeholder, lines 54-97)
- `__init__(config)`: Initializes with default rating 1500, K-factor 32.
- `get_agent_rating(agent_id)` -> `float`: Returns stored rating or default 1500.
- `update_ratings(agent_id, opponent_id, game_results)`: Batch-updates ratings from game results.
- `get_elo_snapshot()` -> `Dict[str, float]`: Returns copy of all ratings.

### Class: `LadderEvaluator(BaseEvaluator)`

**Constructor (line 103):**
- Accepts `EvaluationConfig`, creates `EloTracker` and `PolicyOutputMapper`.
- Initializes fallback logger at line 111 if `BaseEvaluator` did not set one.

**Abstract method implementations from `BaseEvaluator`:**
- `evaluate(agent_info, context)` -> `EvaluationResult` (line 490): Runs full ladder evaluation.
- `evaluate_step(agent_info, opponent_info, context)` -> `GameResult` (line 632): Plays one game.
- `get_opponents(context)` -> `List[OpponentInfo]` (line 113): Returns pool opponents, capped by `num_opponents_per_evaluation`.

**Private methods:**
- `_game_load_evaluation_entity(entity_info, device_str, input_channels)` (line 161): Loads agents/opponents.
- `_game_get_player_action(player_entity, game, legal_mask)` (line 195): Gets actions from players.
- `_game_validate_and_make_move(game, move, legal_moves, ...)` (line 218): Validates and executes moves.
- `_handle_no_legal_moves(game)` (line 273): Handles stalemate/no-moves scenarios.
- `_game_process_one_turn(game, player_entity, context)` (line 292): Single turn orchestration.
- `_game_run_game_loop(sente_player, gote_player, context)` (line 344): Full game loop returning outcome dict.
- `_determine_final_winner(game_outcome, agent_plays_sente)` (line 390): Maps game winner to agent perspective.
- `_prepare_game_metadata(...)` (line 402): Constructs metadata for successful games.
- `_prepare_error_metadata(...)` (line 423): Constructs metadata for error cases.
- `_initialize_opponent_pool(context)` (line 446): Loads opponents from config.
- `_play_match_against_opponent(...)` (line 560): Plays multiple games against one opponent.
- `_select_ladder_opponents(agent_rating, context)` (line 700): Filters opponents by rating range.

**Registration (line 738):** Registers as `EvaluationStrategy.LADDER` with `EvaluatorFactory`.

## 3. Correctness Analysis

### 3.1 EloTracker shadows the real implementation (lines 54-97)
A placeholder `EloTracker` class is defined in this file, while a real one exists at `evaluation/analytics/elo_tracker.py` (visible in the import comment at line 50). The placeholder has a simplified ELO calculation that differs from standard implementations. The `update_ratings` method (lines 63-93) updates ratings iteratively within a loop over games, which means each game's update uses the running (already-modified) ratings rather than the pre-match rating. This creates order-dependent ELO updates where the same set of game results produces different final ratings depending on iteration order. Standard ELO implementations typically compute updates from a fixed base rating.

### 3.2 `num_games_per_match` access via `getattr` is incorrect (line 512)
At line 512: `num_games_per_match = getattr(self.config, "num_games_per_match", 2)`. The `EvaluationConfig` stores `num_games_per_match` inside `strategy_params` (via `configure_ladder_strategy`), not as a top-level attribute. The `getattr` will always return the default `2` because `EvaluationConfig` does not have a `num_games_per_match` attribute. The correct access would be `self.config.get_strategy_param("num_games_per_match", 2)`.

### 3.3 `num_opponents_to_select` access via `getattr` also incorrect (line 726)
At line 726: `num_opponents_to_select = getattr(self.config, "num_opponents_to_select", 5)`. There is no `num_opponents_to_select` attribute on `EvaluationConfig`. The ladder config helper sets `num_opponents_per_evaluation` (not `num_opponents_to_select`) in `strategy_params`. This always returns `5`, ignoring the configured value.

### 3.4 Type comparison bug in opponent filtering (line 715)
At line 715: `opp.name != agent_rating`. This compares a string (`opp.name`) against a float (`agent_rating`). The comparison is intended to exclude the agent from the opponent pool (i.e., should be something like `opp.name != agent_info.name`), but instead it compares against a floating-point rating value. Since a string never equals a float in Python, this condition is always `True` and has no filtering effect. This is a logic bug -- if the agent were ever in its own opponent pool, it would not be excluded.

### 3.5 Potential `KeyError` in `_game_load_evaluation_entity` (line 169)
At line 169: `if "agent_instance" in entity_info.metadata`. The `AgentInfo.metadata` has a default factory of `dict`, so it should always exist. However, unlike the tournament evaluator's version which guards with `hasattr(entity_info, "metadata") and entity_info.metadata`, the ladder version directly accesses `.metadata` without checking if it could be `None`. If an `AgentInfo` is constructed with `metadata=None` explicitly, this would raise a `TypeError` on the `in` check.

### 3.6 ELO update races in `_initialize_opponent_pool` (lines 476-477)
At lines 476-477, the method writes directly to `self.elo_tracker.ratings[name] = initial_rating` only if the name is not already present. If `_initialize_opponent_pool` is called multiple times (which `evaluate` does call it each time at line 500), the opponent pool is re-populated but existing ratings are preserved. However, `self.opponent_pool` is completely replaced at line 467 (`self.opponent_pool = []`), so the pool list and ratings dict could become inconsistent if opponent configs change between calls.

### 3.7 `_game_validate_and_make_move` uses `test_move` redundantly (lines 241-251)
At line 241, `test_move` is called to validate the move before `make_move`. However, the move was already verified to be in `legal_moves` at line 227. Calling `test_move` on a move that is already in the legal moves list should always return `True` if the game engine is consistent. This adds overhead per move and introduces a subtle inconsistency: if `test_move` and `get_legal_moves` disagree (a game engine bug), the move would be rejected even though the engine said it was legal. The tournament evaluator explicitly does NOT do this (see tournament.py line 575: "no need for redundant test_move check").

### 3.8 Color assignment balance in `_play_match_against_opponent` (line 579)
At line 579: `agent_plays_sente_in_this_game = i < (num_games_per_match + 1) // 2`. For even `num_games_per_match` (e.g., 2), this gives Sente assignment for game 0, Gote for game 1. For odd (e.g., 3), Sente for games 0-1, Gote for game 2. The agent gets one more Sente game when `num_games_per_match` is odd, which is a reasonable design choice but creates a slight first-player advantage bias.

## 4. Robustness & Error Handling

### 4.1 Broad exception catching (lines 256-258, 322, 530-531, 591, 680)
Several methods catch bare `Exception` (lines 256, 322, 531, 591, 680) rather than specific exception types. While this prevents crashes, it also silences unexpected errors (e.g., `KeyboardInterrupt` subclasses under some Python versions, `SystemExit`, etc.). The tournament evaluator is more selective with its exception types.

### 4.2 Fallback logger initialization (lines 110-111)
The constructor defensively checks for the existence of `self.logger` after `super().__init__()`. Since `BaseEvaluator.__init__` always sets `self.logger` at line 43, this fallback is unreachable but harmless.

### 4.3 Error metadata construction (lines 423-442)
`_prepare_error_metadata` properly copies opponent metadata before adding error fields, preventing mutation of the original opponent info. The `agent_plays_sente` value is labeled "Best guess" at line 435, acknowledging that in error scenarios the actual color assignment may not have been determined.

### 4.4 No timeout enforcement
Like the tournament evaluator, there is no per-game timeout. The `max_moves` cap at line 348 (default 500, configurable via context) is the only game termination guard. The `timeout_per_game` config parameter is not used.

### 4.5 `_handle_no_legal_moves` is more thorough than tournament (lines 273-290)
This method properly checks whether the game engine already set a winner/termination reason before defaulting to a draw. The tournament evaluator simply breaks out of the loop when no legal moves exist, which is less explicit.

## 5. Performance & Scalability

### 5.1 Sequential match execution
All opponent matches are executed sequentially in the `evaluate` loop (lines 514-535). For a ladder with many opponents, this is a bottleneck.

### 5.2 Entity reloading per game
Like the tournament evaluator, entities are loaded for every game via `_setup_game_entities_and_context` (lines 617-620). No caching of loaded agents across games in the same match.

### 5.3 `test_move` called per move
At line 241, `test_move` is called for every move in addition to the `in legal_moves` check. This doubles the per-move validation cost. For a 500-move game, this is 500 extra `test_move` calls.

### 5.4 Opponent pool re-initialization on every evaluate call
`_initialize_opponent_pool` (line 500) is called at the start of every `evaluate` call. It recreates `self.opponent_pool` from scratch each time, though it preserves existing ELO ratings. If `evaluate` is called repeatedly, this is redundant work.

### 5.5 `get_opponents` caches in instance state (lines 116-157)
Unlike `_initialize_opponent_pool`, `get_opponents` caches the opponent pool in `self.opponent_pool`. However, these two methods have separate pool-initialization logic (one uses `strategy_params.get`, the other uses `config.get_strategy_param`), and `evaluate` calls `_initialize_opponent_pool`, which always resets `self.opponent_pool`. If `get_opponents` is called before `evaluate`, the pool from `get_opponents` would be overwritten.

## 6. Security & Safety

### 6.1 Checkpoint path passed without validation
Same as tournament: checkpoint paths from config are passed directly to agent loading functions without sanitization, potentially enabling arbitrary file loading.

### 6.2 `agent_instance` from metadata
At line 169, pre-loaded agent instances from metadata are used without type validation.

### 6.3 Bare `Exception` catching
The broad `Exception` catches at lines 256, 322, 531, 591, and 680 could mask security-relevant errors.

## 7. Maintainability

### 7.1 Duplicated code from tournament.py
The comment at line 31 explicitly states: "copied from tournament.py for now". The termination reason constants (lines 32-46), entity loading (lines 161-193), action selection (lines 195-216), and move validation (lines 218-271) are all near-copies of the tournament evaluator's equivalents. This DRY violation means bug fixes or changes need to be applied in both files.

### 7.2 Placeholder EloTracker (lines 54-97)
A complete `EloTracker` exists at `evaluation/analytics/elo_tracker.py` (imported and commented out at line 50). The placeholder shadows it and may diverge in behavior. The `pragma: no cover` marker at line 54 suggests awareness that it is not production-quality.

### 7.3 Inconsistent config access patterns
Three different config access patterns are used:
- `self.config.get_strategy_param(key, default)` (line 117) -- correct pattern.
- `getattr(self.config, key, default)` (lines 512, 726) -- accesses top-level attributes, not strategy_params.
- `self.config.strategy_params.get(key, default)` (line 449) -- direct dict access, equivalent to `get_strategy_param` but bypasses the abstraction.

### 7.4 f-string logging (lines 487, 527, 584, 586, 608, 683, 729-731)
Multiple log calls use f-strings instead of `%s` format specifiers. This means string interpolation occurs even when the log level is disabled, wasting CPU. The tournament evaluator uses `%s` formatting throughout.

### 7.5 Two competing opponent pool initialization paths
`get_opponents` (lines 113-157) and `_initialize_opponent_pool` (lines 446-488) both populate `self.opponent_pool` with different logic and different default opponents. The `evaluate` method calls `_initialize_opponent_pool` but not `get_opponents`. The `evaluate_in_memory` method (inherited from `BaseEvaluator`) calls `get_opponents` via the base class. This could lead to different opponent pools depending on which entry point is used.

### 7.6 File length and method count
At 738 lines with 17 methods (plus the 4-method `EloTracker` class), the file is moderately large. The game-playing helpers (lines 159-386) are adapted from the tournament evaluator and account for about 230 lines.

### 7.7 Import at end of file (line 736)
The factory registration imports `EvaluationStrategy` and `EvaluatorFactory` at the module bottom (line 736) rather than at the top with other imports. While this avoids circular import issues, it is inconsistent with the tournament evaluator which imports these at the top.

## 8. Verdict

**NEEDS_ATTENTION**

Primary concerns:
1. **Config access bugs (lines 512, 726):** `getattr` on config returns hardcoded defaults, ignoring user-configured values for `num_games_per_match` and `num_opponents_to_select`. These are functional bugs that affect ladder behavior.
2. **Type comparison bug (line 715):** `opp.name != agent_rating` compares string to float, rendering the filter condition a no-op.
3. **Placeholder EloTracker (lines 54-97):** Shadows the real implementation with a simplified version that has order-dependent rating updates.
4. **Extensive code duplication from tournament.py** with subtle divergences (e.g., `test_move` usage).
5. **Two competing opponent pool initialization paths** that can produce different results depending on the entry point.
6. **No per-game timeout enforcement** despite config support.

The game execution core (game loop, turn processing, winner determination) is well-structured with clearer separation of concerns than the tournament evaluator. The error handling is thorough with detailed metadata capture. However, the config access bugs mean the ladder does not respect user configuration for key parameters, and the placeholder EloTracker means rating calculations are not production-grade.
