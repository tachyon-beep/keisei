# torch.compile Evaluation for Training Hot Path

**Issue:** keisei-d1fdc4d92b
**Date:** 2026-04-03
**Approach:** A — Compile model + move GAE to GPU

## Goal

Find free throughput in the training hot path by applying `torch.compile` to the
SE-ResNet forward/backward passes and rewriting the GAE loop to run on GPU.
Target hardware: NVIDIA CUDA (4060s / H200). Measure everything, ship what helps.

## Candidates

Three hot-path regions, in order of expected impact:

1. **Mini-batch forward+backward** (`update()` inner loop) — 40-block SE-ResNet,
   called ~4 epochs x N mini-batches per update. Highest kernel time budget.
2. **Rollout forward pass** (`select_actions()`) — same model in eval mode under
   `no_grad`, called once per step during collection.
3. **GAE computation** (`compute_gae()`) — Python `for t in reversed(range(T))`
   on CPU, ~128 iterations. Leaves GPU idle.

Candidates 1 and 2 are served by compiling the model object once. Candidate 3
requires a GAE rewrite.

## Design

### 1. Model Compilation

#### 1.1 Two compiled models, not one

**Problem:** `select_actions()` (line 264) toggles the model to `eval()` mode
for BatchNorm running-stats behaviour, then restores `train()` on exit.
`update()` runs in `train()` mode (line 302), and has a *third* mode switch at
line 528 for value-prediction metrics. When `torch.compile` traces the forward
graph, it bakes the `training` flag at trace time. If the first call is in eval
mode, the compiled graph's BN kernels will use frozen running stats even during
training — a **silent correctness bug**. Under `reduce-overhead` (CUDA graphs),
mode-switching between calls can trigger expensive graph recompilation or replay
with mismatched BN state.

**Solution:** Create two separately compiled model objects — one for inference
(eval mode) and one for training (train mode). Each is compiled once and never
mode-switched:

```python
# In KataGoPPOAlgorithm.__init__():
if self.params.compile_mode is not None:
    # Train-mode compiled model — used in update() mini-batch loop
    self.forward_model.train()
    self.compiled_train = torch.compile(
        self.forward_model,
        mode=self.params.compile_mode,
        dynamic=self.params.compile_dynamic,
    )
    # Eval-mode compiled model — used in select_actions() and value metrics
    self.forward_model.eval()
    self.compiled_eval = torch.compile(
        self.forward_model,
        mode=self.params.compile_mode,
        dynamic=self.params.compile_dynamic,
    )
    self.forward_model.train()  # restore default
else:
    self.compiled_train = None
    self.compiled_eval = None
```

Call sites change to use the appropriate compiled model:

```python
# select_actions() — use compiled_eval (already in eval mode, no mode toggle)
def select_actions(self, obs, legal_masks):
    model = self.compiled_eval or self.forward_model
    if self.compiled_eval is None:
        model.eval()  # only toggle if not compiled
    try:
        output = model(obs)
        ...
    finally:
        if self.compiled_eval is None:
            model.train()

# update() mini-batch loop — use compiled_train (already in train mode)
output = (self.compiled_train or self.forward_model)(batch_obs)

# update() value metrics block (line 528) — use compiled_eval
model = self.compiled_eval or self.forward_model
if self.compiled_eval is None:
    model.eval()
try:
    sample_output = model(sample_obs)
finally:
    if self.compiled_eval is None:
        model.train()
```

#### 1.2 Parameter identity invariant

Both `compiled_train` and `compiled_eval` wrap `self.forward_model`, which shares
parameters with `self.model`. This is load-bearing: gradient clipping at line 493
operates on `self.model.parameters()`, and the optimizer at line 225 is
constructed from `model.parameters()`. If `forward_model` is ever a copy rather
than a view (e.g., a diverged DDP wrapper), gradients would be clipped on the
wrong parameter set.

**Invariant (must hold):** `self.forward_model` and `self.model` must share the
same parameter tensors. The `__init__` docstring already documents this convention
(line 215: "Pass the DataParallel wrapper here if using multi-GPU"). Add a runtime
assertion:

```python
# In __init__, after forward_model assignment:
fm_base = self.forward_model.module if hasattr(self.forward_model, 'module') else self.forward_model
m_base = self.model.module if hasattr(self.model, 'module') else self.model
assert fm_base is m_base, (
    "forward_model and model must share parameters — "
    "compile + grad clipping requires this"
)
```

#### 1.3 Dynamic shapes for split-merge mode

In league/split-merge mode, `select_actions()` is called with a variable batch
size (`learner_indices.numel()` varies per step — see `katago_loop.py` lines
85–105). With `reduce-overhead` (CUDA graphs), variable batch sizes are
fundamentally incompatible unless `dynamic=True` is set. Even with `default`
mode, each new batch size triggers a recompilation until the cache is exhausted.

**Solution:** Add `compile_dynamic: bool = True` to `KataGoPPOParams`. Default
`True` avoids recompilation storms in league mode. For fixed-batch-size runs
(non-league), the user can set `False` to allow more aggressive optimization.

```python
compile_dynamic: bool = True  # True = safe for variable batch sizes (league mode)
```

#### 1.4 Config surface

Add to `KataGoPPOParams` (frozen dataclass at `katago_ppo.py` line 52):

```python
compile_mode: str | None = None    # None, "default", "reduce-overhead", "max-autotune"
compile_dynamic: bool = True       # dynamic shapes — True for league mode safety
```

`None` = eager mode (current behavior, zero change for existing users).
`algorithm_registry.py` uses `params_cls(**params)` (line 39) with no strict key
validation, so the new fields propagate without registry changes.

### 2. GAE GPU Rewrite

The current CPU implementation loops in Python over T timesteps. The rewrite
keeps the same sequential recurrence but executes entirely on GPU tensors,
eliminating Python interpreter overhead between steps.

#### 2.1 Scope: 2D path only

The GPU GAE function **only supports the structured 2D `(T, N)` input path**
where each column is a single environment's unbroken trajectory. This corresponds
to the vectorized path in `update()` (line 312: `if total_samples == T * N`).

**Why not the 1D flat fallback?** The flat fallback (line 374) flattens all
environments into a single 1D tensor where consecutive elements may be from
different envs. The vectorized `values[1:]` trick used to compute `next_values`
would conflate transitions across environment boundaries, producing incorrect
deltas. The flat fallback is already the slowest path (rarely used, no env_ids),
so keeping it on CPU is fine.

**Why not the padded split-merge path?** The padded path (line 322) reconstructs
a `(T_max, N_env)` grid from variable-length per-env data. A GPU variant is
feasible but adds complexity for a path that's already vectorized. Revisit if
profiling shows it's a bottleneck.

**Routing in `update()` (replaces current line 312–321):**

```python
if total_samples == T * N:
    rewards_2d = data["rewards"].reshape(T, N)
    values_2d = data["values"].reshape(T, N)
    dones_2d = data["dones"].reshape(T, N)

    if device.type == "cuda":
        # GPU path: move inputs to GPU, compute GAE there
        advantages = compute_gae_gpu(
            rewards_2d.to(device), values_2d.to(device), dones_2d.to(device),
            next_values,  # already on GPU (from bootstrap forward pass)
            gamma=self.params.gamma, lam=self.params.gae_lambda,
        ).reshape(-1).cpu()  # back to CPU for advantage normalization + mini-batch indexing
    else:
        advantages = compute_gae(
            rewards_2d, values_2d, dones_2d,
            next_values_cpu, gamma=self.params.gamma, lam=self.params.gae_lambda,
        ).reshape(-1)
```

GPU GAE is independent of `compile_mode` — if CUDA is available, use the GPU
path. The two optimizations are orthogonal.

#### 2.2 New function: `compute_gae_gpu()`

Added to `gae.py`. Takes GPU tensors shaped `(T, N)`, returns `(T, N)`.

```python
def compute_gae_gpu(
    rewards: torch.Tensor,    # (T, N) on CUDA
    values: torch.Tensor,     # (T, N) on CUDA
    dones: torch.Tensor,      # (T, N) on CUDA
    next_value: torch.Tensor, # (N,) on CUDA
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """GPU GAE for structured (T, N) rollouts.

    Same recurrence as compute_gae(), but avoids CPU round-trips.
    Only valid when each column is a single environment's trajectory.
    """
    T, N = rewards.shape

    # Step 1: vectorized delta and decay (no loop)
    # Shift values forward: next_values[t] = values[t+1] for t < T-1, else bootstrap
    next_values = torch.cat([values[1:], next_value.unsqueeze(0)], dim=0)  # (T, N)
    not_done = 1.0 - dones.float()
    delta = rewards + gamma * next_values * not_done - values  # (T, N)
    decay = gamma * lam * not_done  # (T, N)

    # Step 2: sequential backward scan on GPU
    # Each iteration is a single fused kernel over N envs — no Python tensor ops
    advantages = torch.empty_like(rewards)
    last_gae = torch.zeros(N, device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(T)):
        last_gae = delta[t] + decay[t] * last_gae
        advantages[t] = last_gae

    return advantages
```

**Performance note:** The Python loop still has T=128 iterations, each launching
~2 CUDA kernels (fused multiply-add + assignment). At ~5us dispatch overhead
each, that's ~1.3ms of launch overhead. The real win is eliminating CPU→GPU data
transfer (the CPU path requires all buffer data to stay on CPU, then transfers
mini-batches piecemeal during updates). If profiling shows the GPU scan is only
marginally faster than CPU, `torch.associative_scan` with combine operator
`(a1, x1) + (a2, x2) = (a1*a2, x1*a2 + x2)` achieves O(log T) depth and
should be investigated.

#### 2.3 CPU path preserved

`compute_gae()` and `compute_gae_padded()` remain unchanged. All existing tests
continue to pass against the CPU path. The flat fallback and padded split-merge
paths continue to use CPU GAE.

### 3. Benchmarking

#### 3.1 CUDA event timer placement

Instrument three regions using `torch.cuda.Event(enable_timing=True)`:

1. **Rollout forward pass** — bracket **only** the `model(obs)` call inside
   `select_actions()`, NOT the full method. The method includes a CPU-syncing
   `(legal_counts == 0).any()` guard (line 270–276) that performs a `.nonzero()`
   call; including that in the timing window would add CPU dispatch latency to
   what should be a pure GPU measurement.

2. **Update forward+backward** — bracket the `forward_model(batch_obs)` call
   through `scaler.update()` in the mini-batch inner loop (lines 413–497).

3. **GAE computation** — bracket the `compute_gae` / `compute_gae_gpu` call.

#### 3.2 Variants to compare

| Variant | Model | GAE | Notes |
|---------|-------|-----|-------|
| Baseline | Eager | CPU | Current code path |
| V1 | `compile(mode="default")` | CPU | Safe baseline compile |
| V2 | `compile(mode="reduce-overhead")` | CPU | CUDA graphs — best for fixed shapes |
| V3 | `compile(mode="max-autotune")` | CPU | Triton auto-tune — slow warmup, best steady-state |
| V4 | Best compile mode from V1-V3 | GPU | Full optimization |

**Between variants:** Call `torch._dynamo.reset()` to clear the compilation cache.
Without this, compiled kernels from a previous variant may be reused, making
comparison unreliable.

#### 3.3 Warmup strategy

`max-autotune` triggers Triton kernel auto-tuning on first use, which can take
minutes for a 40-block model with multiple unique kernel shapes. A single epoch
is insufficient because:

- The mini-batch loop may produce a final smaller mini-batch when
  `total_samples % batch_size != 0`, triggering auto-tuning for that shape too
- The eval-mode compiled model (`compiled_eval`) has a different trace than
  train-mode and needs its own warmup

**Warmup protocol:**
1. Run one complete epoch (all mini-batch sizes, both rollout and update paths)
   as explicit warmup before starting the timer
2. First *measured* epoch is excluded from timing as additional safety margin
3. Report timing from epochs 2–5

#### 3.4 Metrics collected

- Median and p95 forward pass time (rollout)
- Median and p95 forward+backward time (update)
- Median GAE computation time
- Total epoch wall-clock time
- Samples/sec throughput

### 4. Correctness Verification

#### 4.1 Fast gate (tolerance)

- **Forward pass:** Capture one output from compiled model and eager model on
  the same input, assert `allclose(rtol=1e-5, atol=1e-5)` on policy logits,
  value logits, and score. **Important:** Freeze BN running stats before
  comparison (or compare before any training steps), since divergent running
  stats will cause outputs to differ independently of compile correctness.
- **GAE:** Compare GPU path output against CPU path on same inputs, same
  tolerance. Use the 2D `(T, N)` path only (GPU GAE doesn't support 1D).
- Runs as unit tests, takes seconds.

#### 4.2 Statistical gate

- Run 5 training epochs with eager vs best compiled variant.
- Compare: mean policy loss, mean value loss, advantage mean/std, entropy.
- Manual inspection for systematic drift — no formal statistical test needed.

## Files Modified

| File | Change |
|------|--------|
| `keisei/training/katago_ppo.py` | Add `compile_mode` and `compile_dynamic` to `KataGoPPOParams`. Create `compiled_train` and `compiled_eval` in `__init__`. Add parameter identity assertion. Update `select_actions()` and `update()` to use appropriate compiled model without mode-switching. Route 2D GAE to GPU path when CUDA available. Add CUDA event timers around forward pass and GAE calls. |
| `keisei/training/gae.py` | Add `compute_gae_gpu()` — 2D `(T, N)` only, GPU tensors. |
| `tests/test_gae.py` | Add tolerance tests: CPU `compute_gae` vs GPU `compute_gae_gpu` on same 2D inputs, `allclose(rtol=1e-5, atol=1e-5)`. |
| `tests/test_compile.py` | New file. Tests: (1) model compiles without error in both train and eval modes, (2) compiled forward pass matches eager within tolerance with frozen BN stats, (3) compiled model parameter identity with base model. |

## Files NOT Modified

- `se_resnet.py` — model unchanged, compile wraps externally
- `katago_loop.py` — loop orchestration unchanged, compilation is internal to PPO
- `algorithm_registry.py` — no strict key validation, new params propagate automatically
- No new dependencies — `torch.compile` is built into PyTorch 2.x

## Hazards and Mitigations

| # | Hazard | Severity | Mitigation |
|---|--------|----------|------------|
| H1 | BN mode-switch bakes wrong `training` flag in compiled graph | **Critical** — silent training correctness bug | Two compiled models: `compiled_train` (always train mode) and `compiled_eval` (always eval mode). Neither is ever mode-switched. |
| H2 | Variable batch size in league/split-merge triggers recompilation storms | **High** — performance regression, possible CUDA graph failure | `compile_dynamic=True` default. Users can set `False` for fixed-batch runs. |
| H3 | GPU GAE 1D flat path produces wrong results (`values[1:]` conflates env boundaries) | **High** — numerical incorrectness | GPU GAE only supports 2D `(T, N)` path. Flat fallback stays on CPU. |
| H4 | `forward_model` and `model` parameter divergence breaks grad clipping | **Medium** — silent gradient clipping on wrong params | Runtime assertion in `__init__` that base modules are the same object. |
| H5 | Third mode-switch in `update()` value metrics block (line 528) | **Medium** — same as H1 for that call site | Uses `compiled_eval`, no mode toggle needed. |
| H6 | GPU GAE scan has 128 sequential kernel launches (~1.3ms overhead) | **Low** — may not outperform CPU for small N | Benchmark will reveal; `torch.associative_scan` is the fallback if marginal. |
| H7 | `max-autotune` warmup insufficient with single-epoch exclusion | **Low** — timing includes tuning overhead | Explicit warmup epoch + exclude first measured epoch (two-epoch buffer). |
| H8 | CUDA event timer includes CPU-syncing legal-mask guard | **Low** — inflated rollout timing | Timer brackets only `model(obs)`, not the full `select_actions()` method. |
| H9 | `autocast` outside compiled call prevents dtype-cast fusion | **Info** — minor missed optimization | Not addressed in this spec. Future cleanup: move `autocast` inside model `forward()`. |

## Out of Scope

- Compiling full `update()` method (Approach C — future experiment, strictly
  additive on top of this work)
- `torch.associative_scan` for O(log T) GAE — revisit if GPU scan is marginal
- Multi-GPU / DDP interaction with compile
- Persistent compilation cache tuning
- Moving `autocast` inside model `forward()` for fusion (future cleanup)
