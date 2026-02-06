# Code Analysis: keisei/training/callbacks.py

## 1. Purpose & Role

`callbacks.py` defines the concrete callback classes that are executed during training steps: `CheckpointCallback` (periodic model saving), `EvaluationCallback` (sync periodic evaluation with Elo tracking), and `AsyncEvaluationCallback` (async variant). It also defines the abstract base classes `Callback` and `AsyncCallback`. These callbacks are managed by `CallbackManager` and invoked by the training loop at each step boundary. This is the "business logic" counterpart to `CallbackManager`'s orchestration role.

## 2. Interface Contracts

### Exports
- `Callback` -- abstract base class with `on_step_end(trainer)` method
- `AsyncCallback` -- abstract base class with `on_step_end_async(trainer)` method
- `CheckpointCallback(interval, model_dir)` -- saves model checkpoints at fixed intervals
- `EvaluationCallback(eval_cfg, interval)` -- runs synchronous periodic evaluation
- `AsyncEvaluationCallback(eval_cfg, interval)` -- runs async periodic evaluation

### Key Dependencies
- `keisei.evaluation.opponents.elo_registry.EloRegistry` (line 11)
- `Trainer` (TYPE_CHECKING only)
- `os`, `asyncio`, `pathlib.Path`

### Assumptions
- `trainer.metrics_manager.global_timestep` is a non-negative integer
- `trainer.agent` may be `None` (checked explicitly)
- `trainer.model_manager.save_checkpoint` is available with a specific signature
- `trainer.evaluation_manager` may or may not exist (checked via `hasattr`)
- `trainer.log_both` is either `None` or a callable

## 3. Correctness Analysis

### `Callback` / `AsyncCallback` base classes (lines 17-28)
- These are declared as `ABC` but do not use `@abstractmethod`. The `on_step_end` / `on_step_end_async` methods have concrete (no-op) implementations. This means subclasses that forget to override will silently do nothing rather than raising `TypeError` at instantiation. This is a design choice but reduces safety.

### `CheckpointCallback.on_step_end()` (lines 36-78)
- **Interval check (line 37):** Uses `(trainer.metrics_manager.global_timestep + 1) % self.interval == 0`. The `+1` is consistent with how the rest of the codebase interprets timesteps (1-indexed for modulo checks).
- **Agent null check (line 38):** Correctly returns early if agent is not initialized.
- **`save_checkpoint` call (lines 53-61):** Calls `trainer.model_manager.save_checkpoint()` with keyword-style positional arguments matching the `ModelManager.save_checkpoint` signature: `(agent, model_dir, timestep, episode_count, stats, run_name, is_wandb_active)`. This matches the 7-parameter signature confirmed in `model_manager.py` line 584. **Correct.**
- **Opponent pool update (lines 69-72):** Adds checkpoint to opponent pool only if `evaluation_manager` exists. Uses `hasattr` check which is appropriate since evaluation is optional.
- **Error logging pattern (lines 39-43, 74-78):** Checks `if trainer.log_both:` which will be falsy if `log_both` is `None`. This means errors can be silently lost.

### `EvaluationCallback.on_step_end()` (lines 86-221)
- **Critical Bug -- `save_checkpoint` signature mismatch (lines 147-152):** The call at line 147-152 passes 4 positional arguments:
  ```python
  trainer.model_manager.save_checkpoint(
      trainer.agent,                                    # agent
      trainer.metrics_manager.global_timestep + 1,      # model_dir (WRONG - this is an int, expects str)
      trainer.session_manager.run_artifact_dir,          # timestep (WRONG - this is a str, expects int)
      "initial_eval_checkpoint",                         # episode_count (WRONG - this is str, expects int)
  )
  ```
  The `ModelManager.save_checkpoint` signature is `(agent, model_dir, timestep, episode_count, stats, run_name, is_wandb_active)`. This call passes arguments in the wrong order and is missing 3 required arguments (`stats`, `run_name`, `is_wandb_active`). **This will raise a `TypeError` at runtime** when the initial evaluation path (no previous checkpoints) is triggered. Furthermore, `ckpt_path` at line 153 expects a path string, but the function returns `Tuple[bool, Optional[str]]`, so the truthiness check `if ckpt_path` would evaluate the tuple itself, which is always truthy. This is a **CRITICAL bug** that would cause a crash or incorrect behavior.

- **Redundant assert after null check (lines 99-101):** An `assert` statement immediately follows an `if not trainer.agent: return` block. The assert is redundant for correctness but serves as type narrowing for static analysis tools.

- **Model eval/train mode switching (lines 129-133, 168-172):** The model is set to `.eval()` before evaluation and `.train()` after. However, if `evaluate_current_agent` raises an exception, the model will remain in eval mode. No `try/finally` block protects this state transition.

- **Elo registry logic (lines 184-220):**
  - Creates a new `EloRegistry` instance each time (line 186), reading from disk. This is not cached.
  - The outcome determination (lines 194-205) compares `win_rate > loss_rate` for "win" and vice versa. Equal rates default to "draw". The access pattern `eval_results.summary_stats.win_rate` assumes `eval_results` has a `summary_stats` attribute with `win_rate` and `loss_rate` properties. If `eval_results` is a dict (which it can be based on the `isinstance(eval_results, dict)` check at line 141), accessing `.summary_stats` would raise `AttributeError`.
  - **Elo snapshot is set on trainer (line 213):** `trainer.evaluation_elo_snapshot = snapshot` -- this dynamically sets an attribute on the trainer object, which is brittle and not declared in the Trainer class.
  - Exception handling (line 214) catches `OSError, RuntimeError, ValueError, AttributeError` which is reasonable.

- **Nested indentation issue (lines 184-220):** The Elo registry logic block is inside `if trainer.log_both is not None:` (line 173). This means if `log_both` is `None`, the Elo registry is never updated, even if evaluation completed successfully. The Elo update should logically be independent of whether logging is available.

### `AsyncEvaluationCallback._run_evaluation_async()` (lines 241-358)
- Largely duplicates `EvaluationCallback.on_step_end()` with similar issues:
  - **Same Elo registry nesting bug (lines 303-339):** Elo update is inside `if trainer.log_both is not None:` block.
  - **Same dynamic attribute assignment (line 332):** `trainer.evaluation_elo_snapshot = snapshot`.
  - **Same model mode risk:** No `try/finally` around eval/train mode switching. However, the outer `try/except` at line 283-358 would catch exceptions, but the model would remain in eval mode. Actually, looking more closely, the async version does NOT explicitly call `current_model.eval()` / `current_model.train()`. The comment at line 221 in the sync callback says "EvaluationManager always handles model mode switching" but the sync callback does it manually. This is inconsistent behavior between the two paths.
- **No opponent checkpoint fallback (lines 269-275):** Unlike the sync version, the async version returns `None` if no opponent checkpoint exists, rather than running an initial evaluation against a random opponent. This means the async path will never bootstrap the Elo system.
- **Returns metrics dictionary (lines 342-349):** The async callback returns structured metrics that can be integrated into the training loop. The sync callback does not return metrics. This is an API inconsistency.
- **Exception handling (lines 352-358):** Catches all `Exception` types, which is appropriate at this level.

## 4. Robustness & Error Handling

- **Critical: `save_checkpoint` call will crash** at line 147-152 due to argument mismatch. This path is triggered only when no previous checkpoints exist (first evaluation), making it a "first-run" crash.
- **Model mode not restored on exception** in `EvaluationCallback`: If evaluation raises, the model stays in `.eval()` mode, potentially degrading training performance (BatchNorm, Dropout behave differently).
- **`asyncio` imported but unused (line 6):** Dead import.
- **`log_both` truthiness vs None checks inconsistent:** Lines 39, 64, 74 use `if trainer.log_both:` (truthiness), while lines 135, 162, 173, 277, 291 use `if trainer.log_both is not None:` (identity). These have different behavior if `log_both` is a falsy-but-not-None value (unlikely but possible).

## 5. Performance & Scalability

- **Elo registry disk I/O (lines 186, 305):** A new `EloRegistry` is created from the filesystem path on every evaluation cycle. For long training runs with frequent evaluation, this repeated disk I/O could be a minor concern, though evaluation intervals are typically large.
- **`eval_results` dict conversion (lines 140-143, 296-299):** `dict(eval_results)` is called on every evaluation result logging, creating a copy. Negligible overhead.
- **Sync evaluation blocks training (line 130, 169):** The sync `EvaluationCallback` runs `evaluate_current_agent` in the training thread, blocking training progress. For long evaluations (many games), this is significant. The async variant addresses this.

## 6. Security & Safety

- **`os.path.basename(str(opponent_ckpt))` (lines 190, 309):** Used as an identifier for Elo registry. If checkpoint paths contain malicious characters, `basename` is a reasonable sanitizer. No path traversal risk here since it's used as a lookup key, not for file operations.
- **EloRegistry path from config (lines 186, 305):** `self.eval_cfg.elo_registry_path` is trusted config input, not user-supplied at runtime.
- No deserialization, network calls, or external command execution in this file.

## 7. Maintainability

- **Significant code duplication:** `EvaluationCallback` (lines 81-221, 140 lines) and `AsyncEvaluationCallback` (lines 224-358, 134 lines) share approximately 70% of their logic (null checks, Elo registry handling, logging patterns). Only the core evaluation call and bootstrap behavior differ.
- **ABC without abstractmethod:** `Callback` and `AsyncCallback` use `ABC` but don't mark methods as `@abstractmethod`. This removes the safety guarantee that subclasses implement the required method.
- **Heavy coupling to Trainer internals:** Callbacks directly access `trainer.metrics_manager.global_timestep`, `trainer.agent.model`, `trainer.model_manager`, `trainer.evaluation_manager.opponent_pool`, `trainer.session_manager.run_artifact_dir`, `trainer.run_name`, `trainer.is_train_wandb_active`, and dynamically set `trainer.evaluation_elo_snapshot`. This creates tight coupling to the Trainer's internal structure.
- **`EvaluationCallback` is 140 lines:** The `on_step_end` method alone is 135 lines (lines 86-221) with deeply nested conditionals. This exceeds reasonable function length for readability.
- **Inconsistent return types:** Sync callbacks return `None` implicitly; async callback returns `Optional[Dict[str, float]]`. Callers must handle both patterns.

## 8. Verdict

**CRITICAL**

Key findings:
1. **CRITICAL: `save_checkpoint` call at lines 147-152 has wrong argument order and missing required arguments.** This will crash at runtime when the first evaluation is triggered with no prior checkpoints. The positional arguments pass `timestep` as `model_dir` (int where str expected) and `run_artifact_dir` as `timestep` (str where int expected), and omits `stats`, `run_name`, and `is_wandb_active`.
2. **Model eval/train mode not protected by try/finally** in sync `EvaluationCallback` -- exception during evaluation leaves model in eval mode.
3. **Elo registry update is gated by `log_both is not None`** -- if logging is unavailable, Elo tracking silently stops.
4. **Async callback lacks bootstrap path** that the sync callback has, creating inconsistent behavior between sync and async evaluation.
5. **ABC base classes lack `@abstractmethod`** decorators, allowing silent no-op subclasses.
6. **Heavy code duplication** between sync and async evaluation callbacks (~70% shared logic).
7. **Unused `asyncio` import** (dead code).
