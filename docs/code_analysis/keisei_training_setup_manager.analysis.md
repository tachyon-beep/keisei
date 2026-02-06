# Code Analysis: keisei/training/setup_manager.py

**Package:** Training -- Session & Setup
**Lines of Code:** 210
**Last Analyzed:** 2026-02-07

---

## 1. Purpose & Role

`SetupManager` extracts complex initialization and setup orchestration logic from the `Trainer` class. It serves as a coordinator for bootstrapping the training pipeline: setting up game components via `EnvManager`, creating training components (model, agent, experience buffer), configuring the `StepManager`, handling checkpoint resumption, and logging run information. It does not own long-lived state beyond its `config` and `device` references -- it is primarily a procedural helper.

---

## 2. Interface Contracts

### Constructor
- `__init__(self, config: AppConfig, device: torch.device)` (line 27)
- `config` must be a validated `AppConfig`.
- `device` must be a valid `torch.device`.

### Methods
| Method | Parameters | Returns | Side Effects |
|---|---|---|---|
| `setup_game_components(env_manager, rich_console)` | Untyped manager + console | `(game, policy_output_mapper, action_space_size, obs_space_shape)` | Calls `env_manager.setup_environment()` |
| `setup_training_components(model_manager)` | Untyped manager | `(model, agent, experience_buffer)` | Creates PPOAgent and ExperienceBuffer |
| `setup_step_manager(game, agent, policy_output_mapper, experience_buffer)` | Mixed types | `StepManager` | Instantiates StepManager |
| `handle_checkpoint_resume(model_manager, agent, model_dir, resume_path_override, metrics_manager, logger)` | Multiple untyped | `bool` (resumed or not) | Mutates agent and metrics_manager state |
| `log_event(message, log_file_path)` | `str, str` | `None` | Appends to file |
| `log_run_info(session_manager, model_manager, agent, metrics_manager, log_both)` | Multiple untyped | `None` | Logs to console and file |

### Missing Type Annotations
All method parameters except `config`, `device`, `message`, `log_file_path`, and `step` are untyped. The parameters `env_manager`, `rich_console`, `model_manager`, `agent`, `metrics_manager`, `logger`, `session_manager`, and `log_both` are all implicitly duck-typed. This is the module's most significant maintainability weakness.

---

## 3. Correctness Analysis

### `setup_game_components` (lines 38-68)
Calls `env_manager.setup_environment()` (line 51) and then reads `env_manager.action_space_size` and `env_manager.obs_space_shape` (lines 54-55). The null check on line 57 (`if game is None or policy_output_mapper is None`) is a post-hoc validation. If `setup_environment()` returns `(None, valid_mapper)`, the error is caught. However, the properties `action_space_size` and `obs_space_shape` are accessed before this null check (lines 54-55). If `setup_environment()` returns `None` for game but the properties are derived from the game, this could raise an `AttributeError` before reaching line 57. In practice, this depends on `EnvManager`'s implementation -- if properties are set inside `setup_environment()`, they would already be set by line 54.

The `except` clause (line 64) catches `(RuntimeError, ValueError, OSError)`. If `env_manager.setup_environment()` raises a different exception type (e.g., `ImportError` for missing Shogi library), it would propagate unhandled to the caller with no Rich console error message.

### `setup_training_components` (lines 70-104)
Creates the model via `model_manager.create_model()` (line 81), checks for `None` (line 83), then instantiates `PPOAgent` (line 89) and `ExperienceBuffer` (line 97). The `ExperienceBuffer` receives `device=self.config.env.device` (line 101), which is a string (e.g., `"cpu"` or `"cuda"`), while the `PPOAgent` receives `device=self.device` (line 92), which is a `torch.device` object. This is a type inconsistency: the buffer gets a string device while the agent gets a `torch.device`. Whether this causes a bug depends on `ExperienceBuffer`'s constructor -- if it expects a `torch.device`, this could fail at runtime when the device is something other than the string default.

### `handle_checkpoint_resume` (lines 128-170)
The method checks `if not agent` on line 151. For a `PPOAgent` object, this evaluates the truthiness of the agent, which defaults to `True` for non-`None` objects. This is correct as a null check. The method delegates to `model_manager.handle_checkpoint_resume()` (line 157) and then reads `model_manager.resumed_from_checkpoint` (line 163) and `model_manager.checkpoint_data` (line 166). This relies on `handle_checkpoint_resume()` setting these attributes as side effects, creating a temporal coupling with `ModelManager`'s internal state.

### `log_run_info` (lines 181-210)
The `log_wrapper` function (lines 194-195) wraps `log_both` but adds no functionality -- it is a pure pass-through. This is likely a remnant from a refactor where the wrapper originally added behavior.

On line 207, `log_both` is called with a keyword argument `also_to_wandb=False`. This means the `log_both` callable must accept this keyword argument. The parameter is typed as a plain positional parameter in the method signature (line 182), not as a `Callable` with keyword arguments. If the caller passes a function that does not accept `also_to_wandb`, this will raise a `TypeError` at runtime. This is an implicit contract that is not documented or enforced by types.

### `log_event` (lines 172-179)
The call to `log_error_to_stderr` on line 179 passes three positional arguments: `("SetupManager", "Failed to log event", e)`. Checking the signature of `log_error_to_stderr` (from unified_logger.py lines 148-149): `log_error_to_stderr(component: str, message: str, exception: Optional[Exception] = None)`. The third argument `e` is of type `OSError | IOError`, which is a subclass of `Exception`, so this call is correct.

---

## 4. Robustness & Error Handling

### Error Propagation Strategy
The module uses two error handling patterns:
1. **Re-raise as RuntimeError**: `setup_game_components` (line 68) catches specific exceptions and re-raises as `RuntimeError`.
2. **Log and continue**: `log_event` (line 178) catches file I/O errors and logs them to stderr without re-raising.

This is appropriate: component setup failures are fatal and should propagate, while logging failures are non-fatal.

### Rich Console Error Display (line 65-66)
The Rich console `print` call uses Rich markup (`[bold red]...[/bold red]`). If `rich_console` is not a Rich `Console` instance but some other object with a `print` method, the markup would appear as literal text. This is an implicit type dependency.

### No Exception Handling in `setup_training_components`
The method (lines 70-104) has no `try/except` block. If `PPOAgent.__init__` or `ExperienceBuffer.__init__` raises, the exception propagates directly to the caller. This is intentional -- these are fatal setup failures -- but contrasts with the defensive approach in `setup_game_components`.

### No Exception Handling in `setup_step_manager`
Similarly, `setup_step_manager` (lines 106-126) has no error handling. A failure in `StepManager.__init__` propagates directly.

---

## 5. Performance & Scalability

### No Performance Concerns
This module executes only during initialization, before the training loop begins. All operations are one-time setup steps. There are no hot paths, no loops over large data, and no repeated I/O.

### Object Construction
The module creates `PPOAgent` (line 89) and `ExperienceBuffer` (line 97) which may involve GPU memory allocation. These are correctly done once during setup.

---

## 6. Security & Safety

### File Writing in `log_event` (lines 172-179)
The method opens `log_file_path` in append mode. The path comes from `SessionManager.log_file_path` which is derived from configuration. There is no path validation or sanitization, inheriting the same concern noted in the `SessionManager` analysis regarding user-controlled paths.

### Model Loading in Checkpoint Resume
The `handle_checkpoint_resume` method (line 157) delegates to `model_manager.handle_checkpoint_resume()` which ultimately loads checkpoint files via `torch.load`. This is a known security surface (`torch.load` uses pickle by default), but the security boundary is in `ModelManager` / `training/utils.py`, not in this module.

---

## 7. Maintainability

### Unused Imports
- `sys` (line 5): Imported but never used in this module.
- `Tuple` (line 7): Imported from `typing` but never used (return types are not annotated).
- `ActorCriticProtocol` (line 12): Imported but never referenced in the module.
- `ExperienceBuffer` (line 13): Used in `setup_training_components` (line 97), so this import is valid.
- `PPOAgent` (line 14): Used in `setup_training_components` (line 89), so this import is valid.
- `TrainingLogger` (line 15): Imported but never used in this module.

Four unused imports (`sys`, `Tuple`, `ActorCriticProtocol`, `TrainingLogger`) indicate incomplete cleanup after refactoring.

### Weak Type Contracts
The module has 6 public methods, and the majority of their parameters are untyped. The method `log_run_info` takes 5 parameters, none of which have type annotations beyond the implicit `self`. This makes the module difficult to understand in isolation and fragile to changes in the manager interfaces it depends on.

### Pass-Through Wrapper (lines 194-195)
The `log_wrapper` function is a no-op wrapper around `log_both`. It adds an unnecessary level of indirection. This likely remains from a refactoring where the wrapper originally transformed or filtered log messages.

### Orchestration vs. Logic
The module is almost purely an orchestrator -- it calls methods on other managers and assembles their outputs. The only domain logic is the null check on line 57 and the agent truthiness check on line 151. This is a clean separation of concerns, consistent with the manager-based architecture.

### Cohesion
The class bundles two conceptually different activities: (1) component initialization (`setup_game_components`, `setup_training_components`, `setup_step_manager`, `handle_checkpoint_resume`) and (2) logging (`log_event`, `log_run_info`). The logging methods could arguably belong to `SessionManager` or a dedicated logging helper, but the current placement is pragmatically acceptable.

---

## 8. Verdict

**NEEDS_ATTENTION**

The module functions correctly as an orchestration layer for training initialization. However, the following items warrant attention:

1. **Device type inconsistency** (line 101 vs line 92): `ExperienceBuffer` receives `self.config.env.device` (a string) while `PPOAgent` receives `self.device` (a `torch.device` object). This could cause a type mismatch depending on `ExperienceBuffer`'s expectations.
2. **Four unused imports** (`sys`, `Tuple`, `ActorCriticProtocol`, `TrainingLogger`): Lines 5, 7, 12, 15 indicate incomplete post-refactor cleanup.
3. **Implicit `also_to_wandb` keyword contract** (line 207): The `log_both` callable is expected to accept an `also_to_wandb` keyword argument, but this is not expressed in any type annotation or documented.
4. **Property access before null check** (lines 54-55 vs line 57): `env_manager.action_space_size` and `obs_space_shape` are read before the null check on `game` and `policy_output_mapper`.
5. **No-op wrapper function** (lines 194-195): `log_wrapper` adds indirection without behavior, suggesting incomplete refactoring.
6. **Pervasive absence of type annotations**: Nearly all method parameters beyond `config` and `device` are untyped, reducing static analysis effectiveness and IDE support.
