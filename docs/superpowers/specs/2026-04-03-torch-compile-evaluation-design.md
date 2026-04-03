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

`torch.compile` wraps the model in `KataGoPPOAlgorithm.__init__()` after
`self.forward_model` is assigned. The compiled model transparently serves both
`select_actions()` (eval, `no_grad`) and `update()` (train, AMP + backward).

```python
# In KataGoPPOAlgorithm.__init__():
if self.params.compile_mode is not None:
    self.forward_model = torch.compile(
        self.forward_model, mode=self.params.compile_mode
    )
```

No changes to `select_actions()` or `update()` call sites — they already use
`self.forward_model(obs)`.

**Config:**

Add to `KataGoPPOParams`:

```python
compile_mode: str | None = None  # None, "default", "reduce-overhead", "max-autotune"
```

`None` = eager mode (current behavior, zero change). All three compile modes
will be benchmarked.

### 2. GAE GPU Rewrite

The current CPU implementation loops in Python over T timesteps. The rewrite
keeps the same sequential recurrence but executes entirely on GPU tensors,
eliminating Python interpreter overhead between steps.

**New functions in `gae.py`:**

- `compute_gae_gpu(rewards, values, dones, next_value, gamma, lam)` — expects
  GPU tensors shaped `(T,)` or `(T, N)`. Computes all deltas and decay factors
  in one vectorized pass, then runs the backward scan on GPU.
- `compute_gae_padded_gpu(...)` — matching GPU variant for variable-length
  episodes.

**The computation:**

```python
# Step 1: vectorized delta and decay (no loop)
next_values = torch.cat([values[1:], next_value.unsqueeze(0)], dim=0)
not_done = 1.0 - dones.float()
delta = rewards + gamma * next_values * not_done - values
decay = gamma * lam * not_done

# Step 2: sequential scan on GPU (loop over T, but each step is a GPU op on N envs)
advantages = torch.zeros_like(rewards)
last_gae = torch.zeros_like(next_value)
for t in reversed(range(T)):
    last_gae = delta[t] + decay[t] * last_gae
    advantages[t] = last_gae
```

The Python loop remains O(T) sequential, but each iteration is a single fused
GPU kernel over the full environment batch (N=512), not a CPU tensor op with
kernel launch overhead. For T=128 this eliminates ~128 Python round-trips.

**Why not `torch.associative_scan`:**

GAE's recurrence includes a per-timestep multiplicative decay
(`gamma * lam * not_done[t]`), so it's not a simple prefix sum. An associative
scan with custom combine `(a1,x1) + (a2,x2) = (a1*a2, x1*a2+x2)` would achieve
O(log T) depth, but for T=128 the practical gain over sequential GPU scan is
marginal, and the sequential version is far easier to validate. Revisit if T
grows significantly.

**Routing in `update()`:**

The `update()` method selects the GAE path based on device availability:

```python
if device.type == "cuda":
    advantages = compute_gae_gpu(...)  # GPU path (works with or without compile)
else:
    advantages = compute_gae(...)  # existing CPU path
```

GPU GAE is independent of `compile_mode` — if a CUDA device is available, use
the GPU path regardless. The two optimizations are orthogonal.

**CPU path preserved:** `compute_gae()` and `compute_gae_padded()` remain
unchanged. All existing tests continue to pass against the CPU path.

### 3. Benchmarking

Instrument the training loop with CUDA event timers around three regions:

1. Forward pass in `select_actions()`
2. Forward+backward in `update()` mini-batch loop
3. GAE computation

Use `torch.cuda.Event(enable_timing=True)` with `synchronize()` for accurate
GPU kernel timing.

**Variants to compare:**

| Variant | Model | GAE |
|---------|-------|-----|
| Baseline | Eager | CPU |
| V1 | `compile(mode="default")` | CPU |
| V2 | `compile(mode="reduce-overhead")` | CPU |
| V3 | `compile(mode="max-autotune")` | CPU |
| V4 | Best compile mode from V1-V3 | GPU |

Each variant runs ~5 epochs on the same config. First epoch excluded from
timing (warmup / compilation overhead).

**Metrics collected:**
- Median and p95 forward pass time (rollout)
- Median and p95 forward+backward time (update)
- Median GAE computation time
- Total epoch wall-clock time
- Samples/sec throughput

### 4. Correctness Verification

**Fast gate (tolerance):**
- Forward pass: capture one output from compiled model and eager model on same
  input, assert `allclose(rtol=1e-5, atol=1e-5)` on policy logits, value
  logits, and score.
- GAE: compare GPU path output against CPU path on same inputs, same tolerance.
- Runs as unit tests, takes seconds.

**Statistical gate:**
- Run 5 training epochs with eager vs best compiled variant.
- Compare: mean policy loss, mean value loss, advantage mean/std, entropy.
- Manual inspection for systematic drift — no formal statistical test needed.

## Files Modified

| File | Change |
|------|--------|
| `keisei/training/katago_ppo.py` | Add `compile_mode` to params, compile model in `__init__`, add CUDA event timers, route GAE to GPU path |
| `keisei/training/gae.py` | Add `compute_gae_gpu()` and `compute_gae_padded_gpu()` |
| `tests/test_gae.py` | Add tolerance tests: CPU vs GPU GAE |
| `tests/test_compile.py` | New — model compiles without error, forward pass matches eager |

## Files NOT Modified

- `se_resnet.py` — model unchanged, compile wraps externally
- `katago_loop.py` — loop orchestration unchanged, compilation is internal to PPO
- No new dependencies — `torch.compile` is built into PyTorch 2.x

## Out of Scope

- Compiling full `update()` method (Approach C — future experiment)
- `torch.associative_scan` for O(log T) GAE
- Multi-GPU / DDP interaction with compile
- Persistent compilation cache tuning
