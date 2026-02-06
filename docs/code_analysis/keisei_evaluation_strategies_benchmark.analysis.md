# Code Analysis: keisei/evaluation/strategies/benchmark.py

**File:** `/home/john/keisei/keisei/evaluation/strategies/benchmark.py`
**Lines:** 753
**Package:** Evaluation -- Strategies: Benchmark

---

## 1. Purpose & Role

`BenchmarkEvaluator` implements a benchmark evaluation strategy that runs a trained agent against a configurable suite of benchmark opponents or scenarios. It extends `BaseEvaluator` and registers itself with `EvaluatorFactory` under `EvaluationStrategy.BENCHMARK`. The class contains full game-loop orchestration: loading entities, running turn-by-turn Shogi games, determining outcomes, and aggregating performance analytics across benchmark cases.

---

## 2. Interface Contracts

### Class Hierarchy
- Extends `BaseEvaluator` (abstract base class defined in `evaluation/core/base_evaluator.py`).
- Implements all four abstract methods: `evaluate()`, `evaluate_step()`, `get_opponents()`, and `validate_config()`.

### Public Interface

| Method | Signature | Contract |
|--------|-----------|----------|
| `__init__` | `(config: EvaluationConfig)` | Calls `super().__init__`, initializes `benchmark_suite`, `policy_mapper`, and conditionally sets `self.logger` (lines 50-56). |
| `get_opponents` | `(context: EvaluationContext) -> List[OpponentInfo]` | Returns opponent list from `suite_config` strategy param; defaults to random + heuristic if empty (lines 58-99). |
| `evaluate` | `(agent_info, context?) -> EvaluationResult` | Orchestrates full benchmark evaluation: loads suite, iterates cases, collects results, computes analytics (lines 143-190). |
| `evaluate_step` | `(agent_info, opponent_info, context) -> GameResult` | Plays a single game for one benchmark case; returns `GameResult` even on error (lines 556-637). |
| `validate_config` | `() -> bool` | Validates `suite_config` is a list and `num_games_per_benchmark_case > 0` (lines 688-703). |

### Registration
- Module-level registration at lines 748-753: `EvaluatorFactory.register(EvaluationStrategy.BENCHMARK, BenchmarkEvaluator)`.
- Uses import at module scope (line 749) rather than top-of-file, which is a side-effect import triggered at module load time.

### Key Data Types Consumed/Produced
- **Input:** `AgentInfo`, `OpponentInfo`, `EvaluationContext`, `EvaluationConfig`
- **Output:** `EvaluationResult` containing `List[GameResult]`, `SummaryStats`, benchmark analytics dict
- **Internal:** `ShogiGame` for game simulation, `PolicyOutputMapper` for legal-move masking

---

## 3. Correctness Analysis

### BUG: Inconsistent retrieval of `num_games_per_benchmark_case` (line 158)

At line 158, `num_games_per_case` is read via `getattr(self.config, "num_games_per_benchmark_case", 1)`. However, `EvaluationConfig` does not have a direct attribute `num_games_per_benchmark_case` -- this value is stored inside `strategy_params` dict. The `getattr` will always return the default `1`, ignoring the configured value. In contrast, `validate_config()` at line 697 correctly uses `self.config.get_strategy_param("num_games_per_benchmark_case", 1)`. This means the evaluate loop always uses 1 game per case regardless of configuration.

### BUG: Winner perspective logic in `_determine_final_winner_for_benchmark` (lines 457-467)

The method returns 0 for agent win and 1 for opponent win. When the agent plays Gote (line 466-467), if Sente won (`sente_won=True`), the method returns 1 (opponent win, correct). If Sente did not win (agent Gote won), it returns 0 (agent win, correct). The logic is actually correct but the code reads confusingly -- `return 1 if sente_won else 0` for the Gote case where "1" means opponent won. This is consistent with `GameResult.winner` semantics (0=agent, 1=opponent).

### Duplication: `get_opponents` vs `_load_benchmark_suite` (lines 58-141)

`get_opponents()` (lines 58-99) and `_load_benchmark_suite()` (lines 101-141) serve overlapping purposes -- both parse `suite_config` to build opponent lists. However, they produce **different defaults** when `suite_config` is empty:
- `get_opponents()` defaults to `random` and `heuristic` types (lines 66-82).
- `_load_benchmark_suite()` defaults to `ppo` and `scripted` types with placeholder paths (lines 107-119).

This means `evaluate()` (which calls `_load_benchmark_suite` at line 153) uses different default opponents than code that calls `get_opponents()` directly. This is a semantic inconsistency.

### Potential issue: `_load_benchmark_suite` default uses invalid path (line 112)

When no suite is configured, `_load_benchmark_suite` creates a placeholder `OpponentInfo` with `checkpoint_path="/path/to/strong_v1_model.ptk"` (line 112). If this placeholder is ever used in an actual game, loading will fail because this path does not exist. The log message at line 106 says "Using placeholders" but nothing prevents these from being used in actual game play.

### Correctness of game loop termination (lines 428-452)

The game loop at line 429 has dual termination: `game.game_over` or `move_count >= max_moves`. After the loop, `_determine_game_loop_termination_reason` is called only if `game.termination_reason` is not set (line 438). The logic is sound: if the game ended via `_game_process_one_turn` returning False, termination_reason is set by the turn processing methods. If the loop exits due to `move_count >= max_moves`, the fallback at line 383 handles it.

### Async methods that perform no async operations

Multiple methods are declared `async` but contain no `await` expressions internally: `_handle_no_legal_moves` (line 304), `_game_get_player_action` (line 228), `_game_validate_and_make_move` (line 251), `_determine_game_loop_termination_reason` (line 375). These are called with `await` from callers. This is not a bug (async functions without await are valid Python) but it indicates the async pattern may have been applied wholesale without need. The `_game_load_evaluation_entity` (line 194) is also async but contains no awaits.

---

## 4. Robustness & Error Handling

### Exception handling strategy

The code uses a layered exception handling approach:

1. **Outermost layer** (`evaluate`, line 171): Catches exceptions per benchmark case, logs them, and continues to the next case. Results include an `errors` list.
2. **Case processing layer** (`_process_benchmark_case`, line 741): Catches exceptions per individual game within a case.
3. **Game step layer** (`evaluate_step`, line 615): Broad try/except around the entire game loop, returns a `GameResult` with error metadata rather than propagating.
4. **Turn-level** (`_game_process_one_turn`, lines 351-366): Catches action selection errors and terminates the game cleanly.
5. **Move execution** (`_game_validate_and_make_move`, lines 286-302): Catches errors from `game.make_move()`.

This is thorough. No exception is silently swallowed -- all are logged and recorded.

### Null safety

- `benchmark_opponent_info.metadata` is accessed with `.get()` throughout but never checked for `None` before `.get()` at line 516. The `OpponentInfo.metadata` field has a `default_factory=dict` so it should never be `None`, but `_prepare_benchmark_game_metadata` at line 477 and `_prepare_benchmark_error_metadata` at line 500 defensively handle `None` with `(... or {}).copy()`.
- `game.winner` is safely checked for `None` at line 445 before accessing `.value`.

### Edge case: zero games played

If `benchmark_suite` is empty after `_load_benchmark_suite`, the loop at line 160 does not execute, `all_game_results` stays empty, and `SummaryStats.from_games([])` at line 176 handles this correctly (returns zeroed stats). The `_calculate_benchmark_performance` at line 670-680 has a guard `if results else 0` but this guard checks the outer `results` list, not the per-case denominators. The `if case_games else 0` at line 666 is actually unreachable because the preceding `if not case_games: continue` at line 650-656 already handles the empty case.

### Potential ZeroDivisionError (lines 670-680)

The `overall_pass_rate` calculation divides `sum(wins_or_passes)` by `sum(played)`. The guard is `if results else 0` which checks the `results` parameter (all game results). However, if `results` is non-empty but all benchmark cases in the suite have zero games (which should not happen in normal flow), the denominator `sum(p["played"])` could be zero. In practice, this is guarded by the fact that `results` being non-empty implies at least one case has `played > 0`.

---

## 5. Performance & Scalability

### Entity re-loading per game

In `evaluate_step` (line 581), `_setup_benchmark_game_entities_and_context` is called, which calls `_game_load_evaluation_entity` for both the agent and the opponent. This means for each game in a benchmark case, the agent and opponent models are loaded from disk again (unless an `agent_instance` is passed in metadata). With `num_games_per_case > 1`, this results in redundant model loading. No caching mechanism is present.

### Sequential game execution

Games within a benchmark case are played sequentially (line 717, `for i_game in range(num_games_per_case)`). The base class provides `run_concurrent_games` but `BenchmarkEvaluator.evaluate` does not use it. For large benchmark suites, this could be slow.

### Memory

Each `GameResult` is accumulated in `all_game_results` (line 169). For very large evaluations (many cases times many games per case), this list grows unboundedly. No streaming or batched processing is used.

### `PolicyOutputMapper` instantiation

A new `PolicyOutputMapper` is created in `__init__` (line 54) on every evaluator instantiation. The base class provides `set_runtime_context` which can inject a shared mapper (line 73 of `base_evaluator.py`), but if that is called after `__init__`, the `self.policy_mapper` created in `__init__` gets overwritten. If `set_runtime_context` is not called, the locally-created mapper is used. This is functional but creates a potential for two mapper instances existing simultaneously.

---

## 6. Security & Safety

### Arbitrary code loading via checkpoint paths

`_game_load_evaluation_entity` (lines 194-226) loads agents from checkpoint paths via `load_evaluation_agent` and `initialize_opponent`. If `checkpoint_path` comes from user-supplied configuration (e.g., `suite_config` from YAML), this could load arbitrary pickle files. This is an inherent risk of PyTorch checkpoint loading (`torch.load` uses pickle). This is standard in ML systems but worth noting.

### No input validation on SFEN strings

At line 416, `ShogiGame.from_sfen(initial_fen, ...)` is called with an `initial_fen` that comes from benchmark metadata. A `ValueError` is caught (lines 419-424), but other exceptions (e.g., from malformed data causing index errors in the SFEN parser) would propagate up to the `evaluate_step` exception handler at line 615.

### Logging includes user-supplied data

Throughout the file, benchmark case names, opponent names, and error messages are logged via f-strings and `%s` formatting. This is standard logging practice and not a security concern in this context (local training system).

---

## 7. Maintainability

### Code duplication

This file contains significant duplication with other evaluation strategies:

1. **`get_opponents` vs `_load_benchmark_suite`**: Two methods parsing the same `suite_config` with divergent defaults (lines 58-141). This is internal duplication within the same file.
2. **Game-loop helper methods** (lines 192-453): The comment at line 192 explicitly states these are "adapted from LadderEvaluator." Methods like `_game_load_evaluation_entity`, `_game_get_player_action`, `_game_validate_and_make_move`, `_handle_no_legal_moves`, `_game_process_one_turn`, `_determine_game_loop_termination_reason`, and `_game_run_game_loop` are duplicated across strategy implementations. This accounts for roughly 260 lines (~35% of the file).

### Method count and complexity

The class has 16 methods (excluding `__init__`):
- 5 public interface methods (`get_opponents`, `evaluate`, `evaluate_step`, `validate_config`, `_load_benchmark_suite`)
- 7 game-loop helpers (duplicated pattern)
- 4 benchmark-specific helpers

This is a high method count for a single class, but the individual methods are reasonably sized (longest is `_game_run_game_loop` at ~52 lines).

### Termination reason constants (lines 29-44)

Eight module-level constants define termination reason strings. These are well-named and consistently used within the file. However, they are likely duplicated in other strategy files (since the game-loop code is copied from `LadderEvaluator`).

### Module-level side-effect import and registration (lines 748-753)

The factory registration at the bottom of the file imports `EvaluationStrategy` and `EvaluatorFactory` at module scope and immediately calls `register()`. This is a common pattern (seen in Django, Flask) but means importing this module has side effects. If the module is imported multiple times or in unexpected order, the registration could fail or be duplicated (though `dict` assignment is idempotent).

### Type annotations

Type annotations are thorough. The `# type: ignore` comments at lines 50 and 52 suppress type checker complaints about `EvaluationConfig` compatibility, which may indicate a slight mismatch between the config type expected by `BaseEvaluator` and what `BenchmarkEvaluator` uses.

---

## 8. Verdict

**NEEDS_ATTENTION**

### Critical findings:
1. **Bug at line 158:** `getattr(self.config, "num_games_per_benchmark_case", 1)` always returns 1 because `num_games_per_benchmark_case` is stored in `strategy_params`, not as a direct attribute. The `evaluate()` method always plays exactly 1 game per benchmark case regardless of configuration. This directly defeats the purpose of configuring multiple games per case.
2. **Divergent defaults between `get_opponents` and `_load_benchmark_suite`:** These two methods represent the same concept (default benchmark suite) but produce incompatible defaults, which could cause confusion depending on which code path is invoked.
3. **Placeholder paths in `_load_benchmark_suite`:** The default benchmark opponent at line 112 has a fake checkpoint path (`/path/to/strong_v1_model.ptk`) that will cause runtime failures if no real suite is configured and the code attempts to load and play against it.

### Secondary findings:
4. ~260 lines of game-loop code duplicated from `LadderEvaluator` (acknowledged in comment at line 192).
5. Multiple async methods that never await anything, adding overhead without benefit.
6. Per-game entity re-loading without caching creates unnecessary I/O overhead for multi-game benchmark cases.
