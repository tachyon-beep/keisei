# Analysis: keisei/config_schema.py

**Lines:** 694
**Role:** Central configuration contract for the entire system. Defines all Pydantic-validated configuration models (`EnvConfig`, `TrainingConfig`, `EvaluationConfig`, `LoggingConfig`, `WandBConfig`, `ParallelConfig`, `DemoConfig`, `DisplayConfig`, `WebUIConfig`, and root `AppConfig`). Every subsystem in Keisei depends on these models for its runtime behavior. Also defines `EvaluationStrategy` constants and `VALID_EVALUATION_STRATEGIES`.
**Key dependencies:**
- Imports from: `pydantic` (BaseModel, Field, field_validator), `typing` (List, Literal, Optional)
- Imported by: 60+ files across the entire codebase (training managers, evaluation strategies, utilities, tests, CLI tools)
**Analysis depth:** FULL

## Summary

This is the most critical file in the system -- every subsystem depends on its contracts. The Pydantic validation is generally well-structured, but there are several latent bugs and design issues that could cause production incidents. The most serious are: (1) a duplicated `evaluation_interval_timesteps` field across `TrainingConfig` and `EvaluationConfig` that creates ambiguity in which value controls behavior, (2) the `lr_schedule_kwargs` field accepts an untyped `dict` that is passed directly to scheduler constructors without validation, (3) missing cross-field validation for numerous interdependent parameters, and (4) the `WebUIConfig` host default of `"0.0.0.0"` binds to all interfaces by default, which is a security concern for non-demo deployments.

## Critical Findings

### [91+210] Duplicate `evaluation_interval_timesteps` field in both TrainingConfig and EvaluationConfig

**What:** `evaluation_interval_timesteps` is defined in two places:
- `TrainingConfig` line 91: default 50,000
- `EvaluationConfig` line 210: default 50,000

Both fields have the same name, same default, and same description. Downstream consumers (`callback_manager.py`) contain complex fallback logic to determine which one to use:

```python
eval_interval = (
    eval_cfg.evaluation_interval_timesteps
    if eval_cfg and hasattr(eval_cfg, "evaluation_interval_timesteps")
    else getattr(self.config.training, "evaluation_interval_timesteps", 1000)
)
```

**Why it matters:** If a user sets `training.evaluation_interval_timesteps` in their config but not `evaluation.evaluation_interval_timesteps` (or vice versa), they get different behavior depending on which code path is exercised. The `callback_manager.py` prefers the evaluation config, but the `TrainingConfig` field suggests it should be in the training section. This is a source-of-truth ambiguity that leads to confusing behavior and mismatched intervals between checkpoint and evaluation callbacks. The fallback default of 1000 in the callback manager differs from both schema defaults of 50,000, creating a third possible value.

**Evidence:** See `keisei/training/callback_manager.py` lines 55-58 and 85-88, which implement a priority cascade. Also `keisei/training/session_manager.py` line 166, which logs `eval_config.evaluation_interval_timesteps` to W&B, so the training config's copy is silently ignored for logging even if it was the one the user intended to set.

### [114] `lr_schedule_kwargs` is an untyped `Optional[dict]` passed directly to scheduler constructors

**What:** The field `lr_schedule_kwargs: Optional[dict]` accepts any dictionary. This dictionary is passed through to `SchedulerFactory.create_scheduler()` (via `ppo_agent.py` line 110) and ultimately to PyTorch scheduler constructors. There is no validation that the keys match the expected parameters for the chosen `lr_schedule_type`.

**Why it matters:** If a user provides `lr_schedule_kwargs: {gamma: 0.5}` with `lr_schedule_type: "linear"` (which does not accept a `gamma` parameter), the error will surface deep inside PyTorch's scheduler factory with an unhelpful traceback pointing at PyTorch internals rather than the configuration. Worse, if a user provides a key that happens to match a different parameter name (e.g., `step_size` when using cosine scheduler), it would be silently ignored or cause unexpected behavior depending on the `**kwargs` forwarding pattern.

**Evidence:**
```python
# config_schema.py line 114
lr_schedule_kwargs: Optional[dict] = Field(
    None, description="Additional keyword arguments for the learning rate scheduler"
)

# ppo_agent.py line 110
self.scheduler = SchedulerFactory.create_scheduler(
    optimizer=self.optimizer,
    schedule_type=self.lr_schedule_type,
    total_steps=total_steps,
    schedule_kwargs=config.training.lr_schedule_kwargs,
)
```

No cross-validation exists between `lr_schedule_type` and the contents of `lr_schedule_kwargs`.

## Warnings

### [668] WebUIConfig defaults to `host: "0.0.0.0"` -- binds to all network interfaces

**What:** The `WebUIConfig.host` field defaults to `"0.0.0.0"`, meaning the WebSocket server will listen on all network interfaces when enabled.

**Why it matters:** While this is intentional for the Twitch streaming use case (remote access needed), it means that if a user enables WebUI on a server with a public IP, the WebSocket and HTTP dashboard ports (8765 and 8766) are immediately exposed to the internet without authentication. The WebUI system itself has no authentication mechanism. An attacker could connect to the WebSocket to observe training data in real-time, or potentially inject malicious messages if the WebSocket handler does not properly validate incoming data.

**Evidence:**
```python
host: str = Field("0.0.0.0", description="WebUI server host (0.0.0.0 for all interfaces)")
```

### [30-36] `EnvConfig` lacks validation on critical fields

**What:** `EnvConfig` has no field validators at all. The `device` field accepts any string (not constrained to "cpu" or "cuda" variants), `input_channels` and `num_actions_total` have no positive-value constraints, and `seed` has no range checking.

**Why it matters:** Setting `input_channels: 0` or `num_actions_total: -1` would pass config validation but cause cryptic crashes during model construction. Setting `device: "gpu"` (incorrect -- should be "cuda") would pass validation but fail at runtime when PyTorch cannot find the device. These are not exotic edge cases; they are common user mistakes when hand-editing YAML configs.

**Evidence:**
```python
class EnvConfig(BaseModel):
    device: str = Field("cpu", description="Device to use: 'cpu' or 'cuda'.")
    input_channels: int = Field(46, ...)
    num_actions_total: int = Field(13527, ...)
    seed: int = Field(42, ...)
    max_moves_per_game: int = Field(500, ...)
```
No `@field_validator` decorators exist for this class.

### [38-39] `max_moves_per_game` is defined in both EnvConfig and EvaluationConfig with no cross-validation

**What:** `EnvConfig.max_moves_per_game` (line 38, default 500) and `EvaluationConfig.max_moves_per_game` (line 239, default 500) are independent fields. The `default_config.yaml` comments note: "env.max_moves_per_game should align with evaluation.max_moves_per_game" -- but there is no code enforcing this.

**Why it matters:** If a user changes only `env.max_moves_per_game` to 1000, evaluation games would still use 500 (or vice versa). Training games and evaluation games would have different termination conditions, making evaluation results non-comparable to training conditions. This is a subtle semantic error that would not produce any warning.

**Evidence:** The YAML config comments on line 16 acknowledge the dependency: "Move limits: env.max_moves_per_game should align with evaluation.max_moves_per_game" -- but the schema enforces nothing.

### [47-48] `steps_per_epoch` has no relationship validation with `minibatch_size`

**What:** `steps_per_epoch` (default 2048) and `minibatch_size` (default 64) are independent. If `steps_per_epoch` is not divisible by `minibatch_size`, some experiences in the buffer will be silently dropped during minibatch iteration, or the last minibatch will be smaller than expected.

**Why it matters:** PPO training correctness depends on all experiences being used for gradient computation. If `steps_per_epoch=100` and `minibatch_size=64`, only 64 of 100 experiences are used per pass (or behavior depends on the buffer iteration implementation). This is a common PPO implementation bug.

**Evidence:**
```python
steps_per_epoch: int = Field(2048, description="Steps per PPO buffer/epoch.")
minibatch_size: int = Field(64, gt=1, ...)
```
No cross-field validator checks divisibility.

### [79] `model_type` is an unvalidated string with no enumeration

**What:** `model_type: str = Field("resnet", ...)` accepts any string value. The model factory (`keisei/training/models/__init__.py` line 9) only handles `"resnet"` and would presumably raise an error for anything else.

**Why it matters:** Users could specify `model_type: "cnn"` or `model_type: "ResNet"` (case mismatch) and get a confusing error from the model factory rather than a clear validation error from the config system.

**Evidence:** The model factory in `training/models/__init__.py` checks `if model_type == "resnet"` -- a simple string equality. No validation at the config layer catches invalid values early.

### [230-233] `strategy_params: dict` is completely untyped and unvalidated

**What:** The `strategy_params` field is a bare `dict` that can contain anything. Each evaluation strategy expects specific keys within this dict (e.g., `opponent_pool_config`, `num_games_per_opponent`, `elo_config`), but there is no schema for what these keys should be per strategy.

**Why it matters:** A typo in a strategy parameter key (e.g., `opponet_pool_config` instead of `opponent_pool_config`) would silently do nothing -- the strategy would use its default or get `None`, and the misspelled key would sit in the dict unconsumed. This is the anti-pattern that Pydantic's `extra = "forbid"` was designed to catch, but it cannot work on a bare `dict` field.

**Evidence:** The `configure_for_ladder` method (line 476) sets keys like `opponent_pool_config`, `elo_config`, `num_games_per_match`, etc. The `ladder.py` strategy reads these back via `self.config.strategy_params.get("opponent_pool_config", [])`. Any key mismatch between writer and reader is invisible.

### [184-190] Redundant validator for `torch_compile_mode` which is already a `Literal` type

**What:** `torch_compile_mode` is typed as `Literal["default", "reduce-overhead", "max-autotune"]` (line 127), which Pydantic already validates. The `validate_torch_compile_mode` validator on line 184-190 duplicates this validation.

**Why it matters:** This is not a bug, but it creates maintenance burden. If the set of valid modes changes, two locations must be updated. More importantly, it obscures which validators are actually doing meaningful work vs. which are redundant with the type system.

### [659-661] `_create_display_config` factory function contains suspicious `type: ignore` comment

**What:** The function `_create_display_config()` returns `DisplayConfig()` with a `# type: ignore[call-arg]` comment. This suppresses a mypy error about the call.

**Why it matters:** The `type: ignore` suggests that the type checker sees something wrong with calling `DisplayConfig()` with no arguments. Since all fields have defaults, this should be valid. The suppression may be masking a real issue or may be vestigial from a previous state of the code. Either way, it should be investigated rather than silently suppressed.

**Evidence:**
```python
def _create_display_config() -> DisplayConfig:
    """Factory function for DisplayConfig to avoid lambda in default_factory."""
    return DisplayConfig()  # type: ignore[call-arg]
```

### [690] `DemoConfig` is `Optional` while all other subsection configs are required

**What:** `AppConfig` defines `demo: Optional[DemoConfig] = None`. All consumers must check for `None` before accessing demo config fields. Other optional-by-nature sections like `WebUIConfig` are given non-None defaults via `default_factory`.

**Why it matters:** This asymmetry means every consumer of `config.demo` must include `if config.demo is not None` guards, while consumers of `config.webui` can access fields directly. Any code that forgets the None check for demo config will get `AttributeError` at runtime. This is inconsistent.

**Evidence:**
```python
class AppConfig(BaseModel):
    ...
    demo: Optional[DemoConfig] = None        # Can be None
    display: DisplayConfig = Field(...)       # Always present
    webui: WebUIConfig = Field(...)           # Always present
```

### [515-525] `LoggingConfig` paths are not validated for safety

**What:** `LoggingConfig.log_file` and `LoggingConfig.model_dir` accept arbitrary strings as file paths. There is no validation to prevent path traversal (e.g., `../../etc/passwd`), absolute paths that could overwrite system files, or paths with special characters.

**Why it matters:** In a multi-user or containerized deployment, a malicious or misconfigured config file could write log files or model checkpoints to unexpected locations. The `model_dir` path is used to create directories and write potentially large checkpoint files.

**Evidence:**
```python
class LoggingConfig(BaseModel):
    log_file: str = Field("logs/training_log.txt", ...)
    model_dir: str = Field("models/", ...)
```

## Observations

### [21-26] `EvaluationStrategy` is a plain class with string attributes, not a proper Enum

**What:** `EvaluationStrategy` is defined as a plain class with class-level string attributes (e.g., `SINGLE_OPPONENT = "single_opponent"`), not as a Python `Enum`. Meanwhile, `VALID_EVALUATION_STRATEGIES` (line 11) is a separate list that must be kept in sync manually.

**Why it matters:** A proper `Enum` would provide built-in membership checking, iteration, and would be harder to accidentally misuse. The current design requires manual synchronization between `EvaluationStrategy`, `VALID_EVALUATION_STRATEGIES`, and the `Literal` type on `EvaluationConfig.strategy` (line 215).

### [596-656] `DisplayConfig` has 30+ fields with minimal validation

**What:** `DisplayConfig` has no field validators. Fields like `sparkline_width`, `trend_history_length`, `metrics_window_size`, etc., accept any integer including zero or negative values.

**Why it matters:** Setting `sparkline_width: 0` or `trend_history_length: -5` would cause rendering errors in the Rich console UI. These are unlikely production issues but represent incomplete defensive validation.

### [163-202] Extensive field validators follow best practices but use inconsistent patterns

**What:** Most validators use `@field_validator` with the `# pylint: disable=no-self-argument` comment pattern. The validation logic is consistent (check bounds, raise ValueError). However, some fields that would benefit from validation (e.g., `gamma`, `clip_epsilon`, `se_ratio`, `tower_depth`, `tower_width`) have none.

**Why it matters:** `gamma` should be in [0, 1], `clip_epsilon` should be positive and typically in (0, 1), `se_ratio` should be non-negative, and `tower_depth`/`tower_width` should be positive. A `gamma: 1.5` or `tower_depth: -3` would pass config validation but cause incorrect training or model construction failures.

### [694] `model_config = {"extra": "forbid"}` on AppConfig is good defensive practice

**What:** The `extra = "forbid"` setting on `AppConfig` means unknown top-level config keys will raise validation errors. This is excellent for catching typos in config files.

**Why it matters:** This is a positive finding. However, note that individual sub-models (e.g., `EnvConfig`, `TrainingConfig`) do NOT set `extra = "forbid"`, so a typo within a section (e.g., `training.learnign_rate`) would be silently accepted by the individual model but caught at the `AppConfig` level only if the key does not accidentally match any existing field.

Actually, re-examining: each sub-model inherits `BaseModel` default behavior which is `extra = "ignore"`. So `training.learnign_rate` (typo) would be silently ignored within `TrainingConfig`, and the user's actual learning rate would be the default 3e-4. Only unrecognized top-level keys are caught. This is a significant gap.

### [528-546] WandBConfig has sensible defaults and good field documentation

**What:** The W&B configuration section is well-structured with appropriate defaults and types.

## Verdict

**Status:** NEEDS_ATTENTION
**Recommended action:** Address the following in priority order:
1. Remove the duplicate `evaluation_interval_timesteps` from `TrainingConfig` (keep only in `EvaluationConfig`) to eliminate source-of-truth ambiguity.
2. Add `extra = "forbid"` to all sub-model classes (not just `AppConfig`) to catch typos within config sections.
3. Add cross-field validation for `steps_per_epoch` vs `minibatch_size` divisibility, and `env.max_moves_per_game` vs `evaluation.max_moves_per_game` consistency.
4. Type or validate `lr_schedule_kwargs` per `lr_schedule_type`.
5. Add basic validation to `EnvConfig` fields (device, input_channels, num_actions_total must be positive).
6. Convert `model_type` from bare `str` to `Literal` or validated string.
7. Consider adding `extra = "forbid"` or a discriminated union pattern to `strategy_params`.
**Confidence:** HIGH -- All findings are directly evidenced by reading the source and its consumers. The duplicate field issue is confirmed across config_schema.py and callback_manager.py.
