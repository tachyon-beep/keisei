# Analysis: keisei/core/experience_buffer.py

**Lines:** 301
**Role:** Pre-allocated tensor-based experience buffer for storing RL training transitions (observations, actions, rewards, values, done flags, legal masks). Computes Generalized Advantage Estimation (GAE) and returns. Used to collect rollout data during self-play, then consumed by PPOAgent.learn() for policy updates.
**Key dependencies:** Imports `torch`, `log_warning_to_stderr` from unified_logger. Imported by `keisei/training/trainer.py`, `keisei/training/step_manager.py`, `keisei/core/ppo_agent.py`, `keisei/training/parallel/parallel_manager.py`.
**Analysis depth:** FULL

## Summary

The experience buffer is structurally sound with a well-designed pre-allocated tensor approach. The GAE computation is numerically correct. However, there are two meaningful concerns: (1) silent data loss when the buffer is full -- experiences are silently dropped with only a warning log, and the buffer-full condition in `add()` silently refuses writes without raising, which could cause an incomplete rollout to be used for training; (2) hard-coded observation and action dimensions create a fragile coupling to a specific Shogi configuration. Confidence in the analysis is HIGH.

## Critical Findings

### [82-97] Silent experience loss when buffer is full

**What:** When `self.ptr >= self.buffer_size`, the `add()` method logs a warning but silently drops the experience. The caller receives no indication that data was lost.

**Why it matters:** If the training loop ever has an off-by-one error or miscalculates when the buffer is full, experiences are silently discarded. This would produce a buffer with fewer samples than expected, potentially missing end-of-episode transitions. GAE computation on a not-truly-full buffer is still valid (it uses `self.ptr` as the effective size), but the training loop might assume the buffer contains `buffer_size` samples when it actually contains fewer. More critically, if the last experiences of an episode are dropped, the reward signal at terminal states is lost, which can silently degrade training quality -- a notoriously hard-to-diagnose problem.

**Evidence:**
```python
if self.ptr < self.buffer_size:
    # ... store data ...
    self.ptr += 1
else:
    log_warning_to_stderr(
        "ExperienceBuffer", "Buffer is full. Cannot add new experience."
    )
```
The `add_batch()` method (line 219) and `add_from_worker_batch()` (line 242) handle fullness with a `break` -- also silent. The `merge_from_parallel_buffers()` method (line 290) silently `return`s. None of these notify the caller that data was lost.

### [170-188] get_batch returns tensor slices (views), not copies

**What:** `get_batch()` returns slices of the pre-allocated tensors (e.g., `self.obs[:num_samples]`). These are tensor views that share memory with the buffer's internal storage.

**Why it matters:** If `clear()` is called after `get_batch()` but while the returned tensors are still in use by `PPOAgent.learn()`, the data is not corrupted because `clear()` only resets the pointer -- it does not zero the tensors. However, if a future optimization were to zero tensors on `clear()` for security/determinism, or if `add()` begins overwriting indices while `learn()` is still iterating, this would silently corrupt training data. Currently safe given the single-threaded usage pattern (trainer.py calls `learn()` then `clear()`), but fragile under concurrent usage. This is a latent bug that would activate if the architecture ever moves to overlapping collection and training.

**Evidence:**
```python
obs_tensor = self.obs[:num_samples]  # This is a VIEW, not a copy
```
The trainer at line 275 calls `self.agent.learn(self.experience_buffer)` and then line 286 calls `self.experience_buffer.clear()`. The `learn()` method calls `batch_data["obs"].to(self.device)`, which *may* create a copy if the device differs, but if the buffer is already on the correct device, `.to()` is a no-op and the data remains a view.

## Warnings

### [39-57] Hard-coded tensor dimensions create fragile coupling

**What:** The observation tensor is hard-coded to shape `(buffer_size, 46, 9, 9)` and the legal mask to `(buffer_size, 13527)`. These magic numbers are the specific Shogi board representation (46 channels, 9x9 board, 13527 possible actions).

**Why it matters:** If the observation representation changes (e.g., a different feature set specified by `config.training.input_features`), or if the action space changes, the buffer will allocate the wrong size and indexing operations will fail at runtime or silently store truncated/padded data. The `config_schema.py` already has `input_channels: int = 46` and `num_actions_total: int = 13527` as configurable fields, but the buffer ignores them. This means the buffer's shape is not guaranteed to match the model's shape.

**Evidence:**
```python
self.obs = torch.zeros((buffer_size, 46, 9, 9), ...)
self.legal_masks = torch.zeros((buffer_size, 13527), ...)
```
These should be parameterized from the config or passed as constructor arguments.

### [99-145] GAE computation assumes single-environment sequential collection

**What:** The GAE computation iterates backwards through the buffer treating it as a single contiguous trajectory, using `dones` to reset the bootstrap value. However, in parallel collection scenarios (using `add_batch`, `add_from_worker_batch`, or `merge_from_parallel_buffers`), experiences from multiple environments may be interleaved.

**Why it matters:** If experiences from multiple parallel environments are mixed in the buffer without proper ordering, the GAE computation will compute advantages incorrectly -- it will bootstrap values across environment boundaries where transitions are not temporally adjacent. The `masks_tensor[t]` mechanism only handles episode termination (done=True), not environment boundaries where episodes from different workers are concatenated. This can introduce bias into advantage estimates and silently degrade training.

**Evidence:**
```python
for t in reversed(range(self.ptr)):
    if t == self.ptr - 1:
        current_next_value = next_value_tensor
    else:
        current_next_value = values_tensor[t + 1]
    delta = rewards_tensor[t] + self.gamma * current_next_value * masks_tensor[t] - values_tensor[t]
    gae = delta + self.gamma * self.lambda_gae * masks_tensor[t] * gae
```
When experiences from different environments are interleaved, `values_tensor[t + 1]` may belong to a completely different environment/episode.

### [107-112] Empty buffer returns silently from compute_advantages_and_returns

**What:** When `self.ptr == 0`, the method logs a warning and returns without setting `_advantages_computed = True`. This means a subsequent `get_batch()` call will raise a `RuntimeError`.

**Why it matters:** The error chain is: empty buffer -> `compute_advantages_and_returns` warns and returns -> `get_batch()` raises RuntimeError. While this *does* eventually surface an error, the warning in `compute_advantages_and_returns` is misleading -- it suggests the operation completed (just on empty data), when in fact the caller's invariant that advantages are computed is not satisfied. This could confuse debugging. However, `PPOAgent.learn()` line 257 checks `if not batch_data or batch_data["obs"].shape[0] == 0` and returns early, so `get_batch()` on an empty buffer returns `{}` before the `_advantages_computed` check. The two code paths are redundant and inconsistent in their error handling.

## Observations

### [191-197] clear() does not zero tensor memory

**What:** `clear()` resets `self.ptr = 0` but leaves old data in the pre-allocated tensors. This is by design for performance.

**Why it matters:** Old data from previous epochs remains in memory after clearing. While this is not a correctness issue (the `ptr` ensures only valid data is accessed), it means tensor memory cannot be garbage-collected until the buffer itself is freed. For very large buffers on GPU, this is the expected tradeoff. However, for debugging purposes, stale data in the buffer after `clear()` could be misleading if inspected.

### [232-253] add_from_worker_batch performs item-by-item insertion

**What:** `add_from_worker_batch` and `merge_from_parallel_buffers` iterate through items and call `add()` individually, converting each tensor element to Python scalars via `.item()`.

**Why it matters:** For large batches, this is significantly slower than a vectorized copy operation (e.g., using tensor slicing to copy a block of data). The repeated `.item()` calls force GPU-to-CPU synchronization for each element. This is a performance bottleneck for parallel collection.

### [199-205] Duplicate size accessors

**What:** Both `__len__()` and `size()` return `self.ptr`. The `capacity()` method returns `self.buffer_size`.

**Why it matters:** Minor API redundancy. Having both `__len__` and `size()` doing the same thing is not harmful but slightly confusing.

## Verdict

**Status:** NEEDS_ATTENTION
**Recommended action:** (1) Parameterize observation shape and action count from config or constructor arguments instead of hard-coding `46, 9, 9` and `13527`. (2) Consider raising an exception (or returning a status) from `add()` when the buffer is full instead of silently dropping data. (3) Document the single-environment assumption for GAE computation, or handle multi-environment interleaving explicitly. (4) Consider vectorized batch insertion for parallel collection performance.
**Confidence:** HIGH -- The code is straightforward, the GAE algorithm is standard and correctly implemented for the single-environment case, and the concerns are verifiable by reading the code and its callers.
