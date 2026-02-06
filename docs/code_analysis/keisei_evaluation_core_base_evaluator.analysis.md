# Code Analysis: `keisei/evaluation/core/base_evaluator.py`

**Lines:** 459 (including blanks and docstrings)
**Last read:** 2026-02-07

---

## 1. Purpose & Role

This module defines `BaseEvaluator`, the abstract base class that all evaluation strategy implementations must inherit from. It establishes the interface contract for running evaluations (single-game, concurrent), validation, context setup, and logging. A companion `EvaluatorFactory` class provides registry-based strategy instantiation, and two convenience functions (`evaluate_agent`, `create_agent_info`) offer simplified entry points.

---

## 2. Interface Contracts

### `BaseEvaluator(ABC)`

| Method | Kind | Contract |
|---|---|---|
| `__init__(config: EvaluationConfig)` | Concrete | Stores config, initializes logger, sets log level from config. |
| `set_runtime_context(...)` | Concrete | Accepts optional `policy_mapper`, `device`, `model_dir`, `wandb_active`, plus `**kwargs`. Sets them as instance attributes. |
| `evaluate(agent_info, context)` | **Abstract** | Must return `EvaluationResult`. |
| `evaluate_in_memory(...)` | Concrete (default) | Falls back to `evaluate()`. Subclasses may override for in-memory optimization. |
| `evaluate_step(agent_info, opponent_info, context)` | **Abstract** | Must return a single `GameResult`. |
| `setup_context(agent_info, base_context)` | Concrete | Creates or reuses an `EvaluationContext`. Seeds RNG if configured. |
| `get_opponents(context)` | **Abstract** | Must return `List[OpponentInfo]`. |
| `run_game(agent_info, opponent_info, context, game_index)` | Concrete | Delegates to `evaluate_step`; wraps exceptions into a failed `GameResult`. |
| `run_concurrent_games(agent_info, opponents, context, max_concurrent)` | Concrete (async generator) | Uses `asyncio.Semaphore` to bound concurrency; yields `GameResult` as tasks complete. |
| `validate_agent(agent_info)` | Concrete | Checks name non-empty and checkpoint path existence (unless in-memory). |
| `validate_config()` | Concrete | Always returns `True`; dead exception handler at lines 346-348. |
| `log_evaluation_start(...)` / `log_evaluation_complete(...)` | Concrete | Logging helpers. |

### `EvaluatorFactory`

| Method | Kind | Contract |
|---|---|---|
| `register(strategy, evaluator_class)` | Classmethod | Adds to class-level `_evaluators` dict. |
| `create(config)` | Classmethod | Returns `BaseEvaluator` from registry; raises `ValueError` for unknown strategy. |
| `list_strategies()` | Classmethod | Returns registered strategy names. |

### Module-level functions

- `evaluate_agent(agent_info, config, context)` -- creates evaluator from factory, runs `evaluate()`.
- `create_agent_info(name, checkpoint_path, **kwargs)` -- thin wrapper over `AgentInfo(...)`.

---

## 3. Correctness Analysis

**Line 48: `getattr(logging, self.config.log_level.upper())`** -- If `config.log_level` is not a valid logging level string (e.g., a typo like `"DEBG"`), this will raise `AttributeError`. The method has no error handling for this case.

**Lines 72-82: `set_runtime_context` uses `setattr(self, key, value)`** for arbitrary `**kwargs`. This means any caller can inject arbitrary attributes onto the evaluator instance. While intentional for flexibility, it also means there is no type safety or name-clash protection. An accidental `kwargs` key of `config` or `logger` would silently overwrite critical fields.

**Lines 78: `wandb_active` is always set** regardless of whether it was passed as `None` or intentionally. The method sets `self.wandb_active = wandb_active` unconditionally (unlike `policy_mapper`, `device`, `model_dir` which have `if is not None` guards). If the caller passes `wandb_active=False` explicitly, it overwrites any previously set value. This asymmetry is intentional (it defaults to `False` in the signature) but worth noting.

**Lines 284-296: `run_concurrent_games` creates all tasks eagerly** (line 289, `asyncio.create_task` in a loop) before yielding. With a large `opponents` list, all tasks are created at once, though the semaphore bounds actual concurrency. This is correct for bounded parallelism, but the task list itself grows linearly.

**Lines 294-296: `asyncio.as_completed` yields futures**, not coroutines. The variable name `coro` on line 295 is misleading, but the `await coro` call is still correct since `as_completed` yields awaitables.

**Lines 335-348: `validate_config` is a dead method.** The `try` block always returns `True` (line 343). The `except` handler at lines 346-348 is unreachable because no exception can occur between `try` and `return True`. The comment on line 342 says "basic validation is done in config __post_init__" suggesting this is a stub, but the try/except is misleading.

**Line 381: `EvaluatorFactory._evaluators` is a class-level mutable dict.** This is shared across all instances and all evaluator types globally. Registration is permanent for the process lifetime, which is correct for a plugin registry pattern but means tests that register evaluators will leak state unless explicitly cleaned up.

---

## 4. Robustness & Error Handling

**Lines 238-255: `run_game` exception handling.** Catches all exceptions and returns a `GameResult` with `winner=None`. This means errors are silent at the caller level -- a game that fails due to a bug is indistinguishable from a draw unless the caller inspects `metadata["error"]`. The `GameResult.is_draw` property (line 45 of evaluation_result.py) returns `True` when `winner is None`, so error games would be counted as draws in `SummaryStats.from_games()`.

**Lines 313-331: `validate_agent` checkpoint path validation.** Catches `OSError` and `ValueError` when constructing `Path` or calling `exists()`, which is appropriate. The in-memory fallback check at line 318 (`agent_info.metadata.get("agent_instance")`) is a loose convention -- if a caller forgets to set this metadata key, validation will fail even though the agent is valid for in-memory evaluation.

**Line 48: Log level setup.** No fallback if `config.log_level` is invalid. An `AttributeError` from `getattr(logging, ...)` would propagate and crash evaluator construction.

**Lines 181-184: `setup_context` directory creation.** Uses `mkdir(parents=True, exist_ok=True)` which is safe. No exception handling for permission errors, but this is acceptable since such failures should propagate.

**Lines 186-194: Random seed setup.** Imports `random` and `numpy` inside the method body. If `numpy` is not installed, this will raise `ImportError` only when a random seed is configured. This is a latent dependency that could surface at runtime.

---

## 5. Performance & Scalability

**`run_concurrent_games` (lines 257-296):** Uses `asyncio.Semaphore(max_concurrent)` to bound parallelism. All tasks are created eagerly but only `max_concurrent` run at a time. This is a standard asyncio pattern and scales reasonably for moderate opponent counts (tens to hundreds). For very large opponent lists (thousands), the task creation overhead could become noticeable.

**`EvaluatorFactory._evaluators` (line 381):** Dictionary lookup is O(1); registration is not performance-sensitive.

**`validate_agent` calls `Path.exists()` (line 316):** This is a filesystem I/O call that could be slow on network filesystems. Called once per validation, so acceptable.

---

## 6. Security & Safety

**Lines 80-82: Arbitrary `setattr` from `**kwargs`.** `set_runtime_context` will set any key-value pair as an attribute on the evaluator. A malicious or misconfigured caller could overwrite `self.config`, `self.logger`, or `self.evaluate` method references. This is a defense-in-depth concern; it requires the caller to be within the same trust boundary.

**Line 182-183: Directory creation** (`mkdir(parents=True)`) in `setup_context` creates directories based on `self.config.save_path`. If `save_path` contains path traversal sequences (e.g., `../../sensitive`), directories could be created outside the expected location. This is low risk since the config is typically loaded from YAML under the user's control.

---

## 7. Maintainability

**Unused import:** `Union` (line 14) is imported but never used in this file.

**Inconsistent logger usage:** The module defines both a module-level `logger` (line 22) and an instance-level `self.logger` (line 43). `run_game` at line 234 uses the module-level `logger`, while `validate_agent` at line 309 also uses the module-level `logger`. Other methods like `set_runtime_context` (line 84) use `self.logger`. This inconsistency means log messages from the same evaluator instance appear under different logger names.

**`validate_config` is a no-op** (lines 335-348). It always returns `True` with a dead except block. Either it should be removed, made abstract, or actually validate something.

**`run_game` is a thin wrapper** over `evaluate_step` (lines 215-255). It adds error handling and logging but no additional game logic. The method comment at line 239 says "this is a placeholder that delegates to evaluate_step," suggesting incomplete implementation.

**Type hints are missing for `set_runtime_context` parameters.** `policy_mapper` has no type annotation (line 53); `device` defaults to `None` but is typed as `str` (line 54), not `Optional[str]`.

---

## 8. Verdict

**NEEDS_ATTENTION**

The file is structurally sound as an abstract base class with a factory pattern. The main concerns are:

1. **Error games counted as draws** -- `run_game` returns `GameResult(winner=None)` on exceptions, which `SummaryStats.from_games()` counts as draws, silently inflating draw statistics.
2. **Arbitrary `setattr` in `set_runtime_context`** -- allows callers to overwrite any instance attribute via `**kwargs`.
3. **Inconsistent logger usage** -- module-level vs. instance-level logger used interchangeably.
4. **Dead code** -- `validate_config` try/except block is unreachable; `run_game` is a documented placeholder.

None of these are crash-level bugs, but the error-as-draw conflation could lead to incorrect evaluation metrics in production.
