# Code Analysis: keisei/evaluation/core/model_manager.py

**Lines:** 541 (540 non-blank)
**Git Status:** Staged as deleted (`D`) in git index, but file still exists on disk. This means a `git rm` was executed but the working tree copy was not removed, or the deletion was staged but the file was subsequently restored to disk without being re-added to the index. Any code depending on this file at import time will succeed at runtime but CI/build from the committed state would fail.

---

## 1. Purpose & Role

`ModelWeightManager` provides in-memory model weight extraction, caching, and agent reconstruction for the evaluation subsystem. It avoids file I/O during repeated evaluations by maintaining an LRU cache of opponent weight dictionaries. It also includes a `DynamicActorCritic` inner class that can reconstruct a minimal neural network model from weight tensors alone, enabling evaluation of checkpoints without the original model configuration.

This file is imported through `keisei/evaluation/core/__init__.py` (line 22) and consumed by `core_manager.py` (line 20, 46) and `strategies/single_opponent.py` (lines 526, 555).

---

## 2. Interface Contracts

### `ModelWeightManager.__init__(device, max_cache_size)`
- **device**: String device name (e.g., "cpu", "cuda:0"), converted to `torch.device` at line 43.
- **max_cache_size**: Integer, maximum opponents in cache. Default 5.

### `extract_agent_weights(agent: PPOAgent) -> Dict[str, torch.Tensor]`
- **Precondition:** Agent must have a non-None `model` attribute.
- **Postcondition:** Returns cloned, detached, CPU-resident weight dict. Original model weights are unaffected.

### `cache_opponent_weights(opponent_id: str, checkpoint_path: Path) -> Dict[str, torch.Tensor]`
- **Precondition:** `checkpoint_path` must point to a valid PyTorch checkpoint file.
- **Postcondition:** Returns weight dict, caches it, and performs LRU eviction if needed.
- **Side effect:** Modifies `_weight_cache`, `_cache_order`, `_cache_hits`, `_cache_misses`.

### `create_agent_from_weights(weights, agent_class, config, device) -> PPOAgent`
- **Precondition:** Weights dict must contain recognizable layer names for architecture inference.
- **Postcondition:** Returns a `PPOAgent` with loaded weights in eval mode.

### `_create_minimal_config(input_channels, total_actions, device) -> AppConfig`
- Returns a fully populated `AppConfig` with hardcoded minimal values.

### `DynamicActorCritic` (inner class, lines 373-432)
- Implements `get_action_and_value` and `evaluate_actions` matching `ActorCriticProtocol`.

---

## 3. Correctness Analysis

### Bug: Overly broad `except RuntimeError` in `cache_opponent_weights` swallows `FileNotFoundError`
At line 109, a `FileNotFoundError` is raised if the checkpoint does not exist. However, the `except Exception` at line 145 catches this and re-raises as `RuntimeError`. This means the `FileNotFoundError` documented in the docstring (line 88) is never actually propagated to callers -- they always see `RuntimeError`. The caller in `core_manager.py` line 412 catches `(ValueError, FileNotFoundError, RuntimeError)`, so functionally this works, but the interface contract is misleading.

### Concern: Architecture inference heuristics (lines 307-359)
The methods `_infer_input_channels_from_weights`, `_infer_conv_out_channels_from_weights`, `_infer_flattened_size_from_weights`, and `_infer_total_actions_from_weights` rely on specific layer naming conventions (`conv.weight`, `stem.weight`, `conv1.weight`, `policy_head.weight`, etc.). The project's main model is `ResNetTower` which uses different layer naming (e.g., `initial_conv.weight`). This means the inference will fall through to fallback defaults for ResNet models, and the `DynamicActorCritic` created will have a fundamentally different architecture than the original model. `load_state_dict(..., strict=True)` at line 210 would then fail with a key mismatch error.

### Concern: `DynamicActorCritic` is a single-conv-layer model
The dynamically created model (lines 373-432) is a minimal single-convolution-layer network regardless of the original model's depth. For any model more complex than this (which the ResNet tower certainly is), the `strict=True` weight loading at line 210 will fail because the state dict keys will not match. This makes `create_agent_from_weights` effectively unusable with the project's primary model architecture.

### `_create_minimal_config` hardcodes many values (lines 434-540)
The minimal config includes hardcoded paths like `/tmp/eval.log`, `/tmp/eval_games`, and specific hyperparameter values. These values are only used for in-memory agent reconstruction and should not affect training, but the hardcoded temp paths could conflict in multi-process scenarios.

### Security: `weights_only=False` on checkpoint loading (line 112)
The `torch.load` call uses `weights_only=False`, which allows arbitrary code execution from pickle-based checkpoint files. The comment says "trusted source" but there is no validation of the source. This is a known PyTorch security concern.

---

## 4. Robustness & Error Handling

- **Weight extraction** (line 63): Validates `model` attribute existence before proceeding.
- **Cache operations** (lines 91-147): Proper LRU eviction with `while` loop handles edge cases where cache is at capacity. Exception handling wraps the load and re-raises as `RuntimeError`.
- **Cache stats** (line 256-271): Returns copies of internal state (`_cache_order.copy()`) preventing external mutation.
- **Memory usage** (line 281-305): Properly checks `isinstance(tensor, torch.Tensor)` before accessing tensor-specific methods.

**Gap:** The `clear_cache` method (lines 273-279) resets hit/miss counters. This means cache statistics cannot survive a cache clear, which could be unexpected for long-running monitoring.

---

## 5. Performance & Scalability

- **LRU cache** uses a list for ordering (lines 93-94), where `remove()` is O(n). For small cache sizes (default 5), this is acceptable. For larger caches this would degrade.
- **Weight cloning** (line 69): `param.clone().detach().cpu()` creates a full copy. For large models (ResNet tower with 9 blocks at 256 channels), each extraction allocates significant memory.
- **No concurrent access protection:** The cache (`_weight_cache`, `_cache_order`) is not thread-safe. If `cache_opponent_weights` is called from multiple threads concurrently, the list operations could corrupt state. The evaluation system uses asyncio (single-threaded) so this is unlikely but not guaranteed.

---

## 6. Security & Safety

- **`torch.load` with `weights_only=False`** (line 112): Allows arbitrary pickle deserialization. If an attacker can place a malicious checkpoint file in the expected path, this would execute arbitrary code. This is flagged in PyTorch documentation as a security risk.
- **Hardcoded filesystem paths** in `_create_minimal_config` (lines 503, 516, 527): `/tmp/eval.log` and `/tmp/` are world-writable directories. In a multi-user environment, this creates potential symlink attack vectors, though the config is only used for in-memory evaluation.

---

## 7. Maintainability

- **Heavy coupling to config schema:** `_create_minimal_config` (lines 434-540) explicitly constructs an `AppConfig` with every required field. Any addition to `AppConfig`, `TrainingConfig`, `EvaluationConfig`, or other sub-configs requires updating this method. This is a significant maintenance burden -- the method is 107 lines of hardcoded configuration.
- **Tight coupling to concrete classes:** Imports `PPOAgent`, `ActorCritic`, and `PolicyOutputMapper` directly (lines 26-28), rather than using protocols or interfaces. Changes to `PPOAgent.__init__` signature would break `create_agent_from_weights`.
- **Inner class `DynamicActorCritic`** (lines 373-432) duplicates logic from the main `ActorCritic` class but with a simpler architecture. This creates a maintenance risk where protocol changes in `ActorCriticProtocol` must be mirrored here.
- **Git status anomaly:** The file is staged as deleted but exists on disk. This creates confusion about whether the file is considered part of the project.

---

## 8. Verdict

**NEEDS_ATTENTION**

Primary concerns:
1. The `DynamicActorCritic` single-layer architecture will not match weights from the project's actual ResNet models, making `create_agent_from_weights` fail with `strict=True` for any non-trivial model. This is a functional correctness issue for the in-memory evaluation path.
2. `torch.load` with `weights_only=False` is a security risk.
3. The 107-line `_create_minimal_config` method is tightly coupled to every config model and will break on any schema change.
4. The file's git status (staged as deleted) suggests it may be in the process of being removed or refactored, adding uncertainty about its lifecycle.
