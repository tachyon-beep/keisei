# Code Analysis: keisei/training/utils.py

## 1. Purpose & Role

`training/utils.py` provides helper functions for training setup and configuration. It handles checkpoint discovery and validation, configuration serialization, directory setup, random seed initialization, Weights & Biases (WandB) integration, sweep configuration application, and CLI argument override building. It is consumed by the training entry points (`train.py`, `train_wandb_sweep.py`) and the `SessionManager`.

The file is 232 lines and defines 7 module-level functions with no classes.

## 2. Interface Contracts

### Exports
- `_validate_checkpoint(checkpoint_path: str) -> bool` -- private, validates checkpoint file integrity
- `find_latest_checkpoint(model_dir_path: str) -> Optional[str]` -- finds newest valid checkpoint
- `serialize_config(config: AppConfig) -> str` -- serializes config to JSON
- `setup_directories(config, run_name) -> dict` -- creates run artifact directories
- `setup_seeding(config)` -- sets random seeds for reproducibility
- `setup_wandb(config, run_name, run_artifact_dir) -> bool` -- initializes WandB
- `apply_wandb_sweep_config() -> dict` -- extracts sweep parameter overrides
- `build_cli_overrides(args) -> dict` -- converts CLI arguments to config overrides

### Key Dependencies
- `torch` -- for checkpoint loading and seed setting
- `numpy`, `random` -- for seed setting
- `wandb` -- imported at module level (line 16) and locally (line 113)
- `keisei.config_schema.AppConfig` -- for config serialization
- `keisei.utils.unified_logger.log_error_to_stderr`, `log_info_to_stderr` -- for logging
- `glob`, `json`, `os`, `pickle`, `sys`

### Assumptions
- Checkpoint files use `.pth` or `.pt` extensions.
- `config` parameters in `setup_directories` and `setup_seeding` have the expected attribute structure (no type hints on these functions).
- `args` parameter in `build_cli_overrides` is an argparse `Namespace` with optional attributes.
- WandB is available as an importable module (imported at module level).

## 3. Correctness Analysis

### `_validate_checkpoint` (lines 21-38)
- Loads the entire checkpoint file into memory to validate it (`torch.load(checkpoint_path, map_location="cpu")`). The loaded data is not used and is immediately discarded.
- **Critical issue**: `torch.load` without `weights_only=True` is unsafe. With PyTorch 2.7.0 (the project's version), the default behavior changed to `weights_only=True`, which will cause this call to fail for checkpoints containing non-tensor metadata (optimizer state dicts, training stats, etc.). This function will incorrectly report all such checkpoints as corrupted.
- The exception list (`OSError, RuntimeError, EOFError, pickle.UnpicklingError`) is comprehensive for file corruption scenarios.
- This function does NOT catch `torch.serialization.StorageNotFoundError` or other PyTorch 2.7.0-specific exceptions that may be raised with `weights_only=True`.

### `find_latest_checkpoint` (lines 41-66)
- Searches for `.pth` files first, then `.pt` files if none found. This means if a directory contains both `.pth` and `.pt` files, only the `.pth` files are considered.
- Sorts by modification time (newest first) and validates each until a valid one is found. This is correct for finding the latest non-corrupted checkpoint.
- The `OSError` catch at line 64 handles permission errors and path issues.
- **Performance concern**: `_validate_checkpoint` loads the entire checkpoint file for each candidate. For large checkpoints (hundreds of MB), iterating through corrupted files could be very slow and memory-intensive. In the worst case, if all checkpoints are corrupted, every single file will be fully loaded and discarded.

### `serialize_config` (lines 69-78)
- Delegates to Pydantic's `model_dump_json(indent=4)`. Simple and correct.
- The return type is `str`. No error handling -- if `model_dump_json` fails, the exception propagates.

### `setup_directories` (lines 81-94)
- **Missing type hints**: `config` and `run_name` parameters have no type annotations. This is inconsistent with the rest of the codebase.
- Creates `run_artifact_dir` using `os.makedirs(run_artifact_dir, exist_ok=True)`. Correct.
- `log_file` is taken from `config.logging.log_file` and only the basename is used (line 86: `os.path.basename(log_file)`). If `config.logging.log_file` is already a full path, the basename extraction is correct. If it's just a filename, `os.path.basename` returns it unchanged. Either way, correct.
- Returns a dict with 4 path strings. No error handling for `os.makedirs` failure.

### `setup_seeding` (lines 97-104)
- **Missing type hints**: `config` parameter has no type annotation.
- Sets seeds for `numpy`, `torch`, `random`, and `torch.cuda` (if available).
- The seed is only set if `config.env.seed is not None`. The `EnvConfig` model has `seed: int = Field(42, ...)`, so the seed is always an int (never `None`) unless explicitly overridden. This `is not None` check suggests the seed field was previously `Optional[int]` or may become so in the future.
- Does not set `torch.backends.cudnn.deterministic = True` or `torch.backends.cudnn.benchmark = False`, which are typically needed for full reproducibility. This is a known limitation in many RL codebases but is worth noting.

### `setup_wandb` (lines 107-154)
- **Double import**: `wandb` is imported at module level (line 16) AND locally at line 113 (`import wandb as local_wandb`). The local import is used for the actual `init()` call, while the module-level import is used in `apply_wandb_sweep_config()`. This is unusual and appears to be defensive debugging code (the `log_info_to_stderr` call at line 115-118 inspects the module type).
- **Double serialization** at line 120-122: `serialize_config(config)` is called twice -- once to check if it's truthy (in the `if` condition) and once to parse the result. Since `serialize_config` returns a JSON string, it will always be truthy (even for an empty config, `model_dump_json` returns `"{}"`). The `if serialize_config(config) else {}` condition is effectively dead code -- the `else {}` branch can never execute.
- **WandB resume with run_name as ID** (lines 130-131): `resume="allow"` and `id=run_name` means that if a run with the same name already exists, WandB will attempt to resume it. This is correct for checkpoint-based training resumption but could cause issues if `run_name` is reused for a completely different training run.
- **Error handling** (lines 133-148): Catches `TypeError`, `ValueError`, `OSError`, `AttributeError`. The `AttributeError` branch includes a full traceback dump to stderr (lines 136-142). This is more verbose than the generic error path at lines 143-147. Both branches set `is_active = False` to gracefully disable WandB.
- **Redundant mode check** at line 128: `mode="online" if wandb_cfg.enabled else "disabled"`. Since this code is inside the `if is_active:` block (line 110), and `is_active = wandb_cfg.enabled` (line 109), the condition `wandb_cfg.enabled` is always True at this point. The mode will always be `"online"`.

### `apply_wandb_sweep_config` (lines 157-198)
- Checks `wandb.run is None` to detect if a sweep is active (line 165). Correct.
- The exception handling at line 168 catches `AttributeError, ImportError`, which handles cases where WandB is not properly installed or initialized.
- The `sweep_param_mapping` dict (lines 176-190) maps WandB sweep parameter names to config paths. This mapping uses dot-notation strings (e.g., `"training.learning_rate"`).
- Uses `hasattr(sweep_config, sweep_key)` at line 195 to check for parameter presence. This is correct for WandB's config object which uses attribute access.
- Always includes `"wandb.enabled": True` in sweep overrides (line 193). This ensures WandB stays enabled during sweeps.

### `build_cli_overrides` (lines 201-232)
- Uses `hasattr(args, ...)` and `args.xxx is not None` guards for each possible argument. This is defensive and handles both the main training script and the sweep script (which may have different argument sets).
- Returns a dict of dot-notation overrides.
- Line 229: `if hasattr(args, "wandb_enabled") and args.wandb_enabled:` -- only adds the override if `wandb_enabled` is truthy (True). There is no way to explicitly disable WandB via CLI if it's enabled in config, since `wandb_enabled=False` would not add the override. This is likely intentional (you wouldn't pass `--wandb_enabled` to disable it).
- No validation of override values. If `args.seed` is a string instead of an int, it would be passed through unchanged. Input validation is presumably handled downstream by Pydantic.

## 4. Robustness & Error Handling

**Strengths:**
- `find_latest_checkpoint` iterates through candidates and gracefully handles all-corrupted scenarios.
- `setup_wandb` catches a broad set of exceptions and falls back to disabled state.
- `apply_wandb_sweep_config` handles missing WandB gracefully.
- `build_cli_overrides` uses defensive `hasattr` checks for all arguments.

**Weaknesses:**
- `_validate_checkpoint` loads entire checkpoint files just to validate them. A failed load due to `weights_only=True` default in PyTorch 2.7.0 would incorrectly mark valid checkpoints as corrupted.
- `setup_directories` has no error handling for `os.makedirs` failure.
- `setup_seeding` has no error handling.
- No functions validate their input types (no type hints on `setup_directories`, `setup_seeding`, `setup_wandb`).

## 5. Performance & Scalability

- **`_validate_checkpoint` is expensive**: It fully loads each checkpoint file into CPU memory. For checkpoints that include model weights, optimizer state, and metadata, this could be hundreds of megabytes per file. `find_latest_checkpoint` calls this for each candidate until a valid one is found, meaning in the worst case (all corrupted), every checkpoint file is fully loaded.
- `serialize_config` is called twice in `setup_wandb` (line 120-122) for no benefit. The first call's result is only used for a truthiness check that always passes.
- `glob.glob` at lines 43-45 scans the entire directory. For directories with thousands of files, this could be slow, but checkpoint directories typically have fewer than 100 files.

## 6. Security & Safety

- **`torch.load` without `weights_only=True`** (line 32): This is the most significant security concern. `torch.load` uses `pickle.loads` internally, which can execute arbitrary code during deserialization. If a malicious checkpoint file is placed in the model directory, `_validate_checkpoint` would execute its payload. The comment in the existing analysis of `ppo_agent.py` notes this same issue. With PyTorch 2.7.0, the default has changed to `weights_only=True`, which mitigates this risk but will likely cause the validation to fail for legitimate checkpoint files containing non-tensor data.
- **`import pickle`** (line 8): Imported but never directly used. It appears in the exception handling of `_validate_checkpoint` (`pickle.UnpicklingError`). The import is justified for the exception type reference.
- **`import sys`** (line 10): Imported but never used anywhere in the file. Dead import.
- **Path traversal**: `find_latest_checkpoint` takes a `model_dir_path` string and uses it with `glob.glob` and `os.path.join`. No validation is performed on the path. If an attacker controls the `model_dir_path` value, they could potentially point to arbitrary directories. However, this path comes from the configuration system which is trusted.

## 7. Maintainability

**Code Smells:**
- **Missing type hints**: `setup_directories`, `setup_seeding`, and `setup_wandb` lack type annotations on their parameters. This is inconsistent with the rest of the codebase which uses type hints extensively.
- **Dead import**: `import sys` (line 10) is unused.
- **Dead import**: `import pickle` (line 8) is only used for the exception type `pickle.UnpicklingError`. While technically used, the import of the full `pickle` module for a single exception type is unusual. The exception could be imported directly: `from pickle import UnpicklingError`.
- **Double serialization**: `serialize_config(config)` is called twice in `setup_wandb` (line 120-122). The truthiness check is redundant.
- **Debug logging in production code**: Lines 112-118 in `setup_wandb` include debug-level logging (`wandb module type`, `has init`) that appears to be leftover from debugging a WandB integration issue. These log messages are written to stderr on every training run where WandB is enabled.
- **Redundant `mode` conditional** at line 128 in `setup_wandb` -- always evaluates to `"online"`.
- **Inconsistent error verbosity**: The `AttributeError` branch in `setup_wandb` (lines 135-142) includes a full traceback, while other exception types at lines 143-147 do not. This asymmetry suggests the `AttributeError` path was added to debug a specific issue and the extra logging was not removed afterward.

**Structure:**
- The file is a collection of loosely related utility functions. They share no state and have minimal coupling, which makes them easy to test individually.
- The module-level `import wandb` (line 16) means that `import keisei.training.utils` will fail if `wandb` is not installed. This is different from the optional-integration pattern described in `CLAUDE.md` ("Optional integrations must disable cleanly"). A conditional import or lazy import would be more robust.

## 8. Verdict

**NEEDS_ATTENTION**

Key findings:
1. **`torch.load` without `weights_only=True`** in `_validate_checkpoint` (line 32): This is both a security risk (arbitrary code execution from untrusted checkpoints) and a compatibility issue (PyTorch 2.7.0 defaults to `weights_only=True`, which will cause legitimate checkpoint validation to fail for checkpoints containing non-tensor metadata).
2. **Expensive checkpoint validation**: `_validate_checkpoint` fully loads checkpoint files into memory just to check integrity. For large checkpoints, this is wasteful and slow.
3. **Module-level `import wandb`**: Makes the entire module fail to import if `wandb` is not installed, violating the project's optional-integration pattern.
4. **Dead import**: `import sys` is unused.
5. **Missing type hints**: Three functions (`setup_directories`, `setup_seeding`, `setup_wandb`) lack parameter type annotations.
6. **Debug logging in production**: `setup_wandb` contains debug-level logging (WandB module inspection) that runs on every training initialization.
7. **Double serialization**: `serialize_config` is called twice unnecessarily in `setup_wandb`.
