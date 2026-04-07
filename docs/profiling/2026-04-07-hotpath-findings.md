# Hot-Path Profiling Findings — 2026-04-07

## Executive Summary

**AMP (mixed precision) is the single most impactful optimization**, delivering 2.1x speedup on forward passes at production scale. `torch.compile` adds a further 1.5x on top of AMP. Combined, **compile(default)+AMP delivers a 3.2x forward-pass speedup** (164ms → 51ms) and **2.5x full PPO update speedup** (73s → 29s). The residual tower is 100% of forward-pass cost; GAE and loss components are negligible. The `torch.associative_scan` investigation (keisei-b7b5820caf) is not worth pursuing — GAE is ~1.6ms out of a 29s update cycle.

## Hardware

- **GPU:** NVIDIA GeForce RTX 4060 Ti (16 GB VRAM)
- **PyTorch:** 2.11.0+cu130
- **CUDA:** 13.0

## Component Timing Breakdown

### Forward Pass

| Component | Small (10b/128ch) | Production (40b/256ch) |
|-----------|-------------------|----------------------|
| Input stem | 0.52 ms | 0.74 ms |
| Single block | 1.23 ms | 4.17 ms |
| Residual tower | 11.70 ms (95%) | 164.24 ms (100%) |
| Policy head | 0.11 ms (<1%) | 0.16 ms (<1%) |
| Value+Score heads | 0.13 ms (1%) | 0.23 ms (<1%) |
| **Total forward (eval, bs=256)** | **12.35 ms** | **164.67 ms** |
| **Forward+backward+step (bs=256)** | **45.12 ms** | **564.01 ms** |

The residual tower completely dominates. At production scale, heads are <0.3% of forward time. The backward pass is ~3.4x the forward pass (164ms forward → 564ms fwd+bwd+step).

### Policy Sampling

| Batch size | Time |
|------------|------|
| 16 | 0.16 ms |
| 64 | 0.16 ms |
| 256 | 0.36 ms |

Negligible. The masked softmax over 11,259 actions is fast even at bs=256 (0.36ms vs 164ms forward pass). Not worth optimizing.

### GAE

| Config | CPU | GPU | Speedup |
|--------|-----|-----|---------|
| T=128, N=32 | 1.84 ms | 1.58 ms | 1.2x |
| T=128, N=64 | 1.83 ms | 1.59 ms | 1.2x |
| T=128, N=128 | 1.85 ms | 1.59 ms | 1.2x |
| T=128, N=256 | 1.86 ms | 1.59 ms | 1.2x |

GPU GAE provides minimal benefit (1.2x). The backward scan loop (T=128 iterations of ~2 CUDA kernels each) is kernel-launch-overhead-dominated, not compute-dominated. Increasing N beyond 32 doesn't help because each kernel is already trivially small.

**Verdict on associative_scan (keisei-b7b5820caf):** Not worth pursuing. GAE is 1.6ms out of a 29s update cycle (0.005%). Even a 10x improvement would save 1.4ms — unmeasurable.

### Buffer Operations

| Operation | Time | Notes |
|-----------|------|-------|
| buffer_fill+flatten T=128 N=64 | 120 ms | CPU-bound: Python list append + torch.cat |
| GPU transfer T=128 N=64 | 17 ms | 225 MB over PCIe |

Buffer fill+flatten is ~120ms — dominated by CPU tensor creation in `add()`, not the `torch.cat` in `flatten()`. The GPU transfer of 225 MB at 17ms implies ~13 GB/s PCIe throughput (reasonable for PCIe 4.0 x16). Not a bottleneck relative to the 564ms fwd+bwd+step.

### Loss Components

| Component | Time (bs=256) |
|-----------|--------------|
| Policy log_softmax+gather | 0.15 ms |
| PPO clip loss | 0.04 ms |
| Entropy | 0.34 ms |
| WDL cross-entropy | 0.01 ms |
| Score MSE | 0.02 ms |
| **Total** | **0.56 ms** |

All loss computations combined are 0.56ms — completely negligible vs the 564ms fwd+bwd+step. Entropy is the most expensive loss component due to the `(B, 11259)` tensor operations, but still trivial.

## Memory Analysis

| Metric | Small | Production |
|--------|-------|------------|
| Parameters | 13.9 MB | 213.7 MB |
| Gradients | 13.9 MB | 213.7 MB |
| Optimizer (Adam) | 27.9 MB | 427.4 MB |
| Total static | 55.8 MB | 854.8 MB |
| Inference peak (bs=256) | 78.5 MB | 123.3 MB |
| Training peak (bs=256) | 742.6 MB | 5,330.1 MB |
| VRAM headroom | 16.0 GB | 11.4 GB |

Production model uses 5.3 GB VRAM during training at bs=256 — well within the 16 GB RTX 4060 Ti. Headroom is 11.4 GB, enough for larger batch sizes (up to ~bs=768 estimated before OOM).

## torch.compile Assessment

### Compile Diagnostics

- **Graph breaks:** 0 (all configurations: dynamic, static, AMP)
- **Recompilation:** None across batch sizes [4, 16, 64, 256]
- **Single graph:** The entire SE-ResNet compiles into one graph with no breaks

This is ideal — the model is fully compilable. No graph breaks means inductor can fuse the entire forward pass.

### Forward Pass Speedup

| Mode | Small (10b) | Production (40b) |
|------|-------------|-------------------|
| Eager | 12.56 ms | 164.06 ms |
| Eager + AMP | 7.78 ms (1.6x) | 76.74 ms (2.1x) |
| compile(default) | 8.97 ms (1.4x) | 123.78 ms (1.3x) |
| compile(default) + AMP | 4.37 ms (2.9x) | 50.65 ms (3.2x) |
| compile(reduce-overhead) | 9.10 ms (1.4x) | 121.88 ms (1.3x) |
| compile(max-autotune) | 8.82 ms (1.4x) | 123.37 ms (1.3x) |
| compile(max-autotune) + AMP | 4.24 ms (3.0x) | 49.49 ms (3.3x) |

Key observations:
1. **AMP alone is more impactful than compile alone** (2.1x vs 1.3x at production scale)
2. **Compile modes are equivalent** — default, reduce-overhead, and max-autotune give the same speedup (1.3x). The SE-ResNet uses cuDNN convolutions which are already well-optimized; compile mainly fuses the smaller ops (BN, ReLU, SE attention, global pool bias).
3. **AMP + compile stack multiplicatively** — the compound effect gives 3.2-3.3x
4. **max-autotune adds ~1ms advantage** over default at production scale (49.49 vs 50.65), which is within noise. The "Not enough SMs" warning suggests max-autotune can't use its GEMM autotuning on the 4060 Ti.

### Full PPO Update Speedup

| Configuration | Update time | Mini-batch fwd+bwd | Speedup vs baseline |
|---------------|-------------|--------------------|--------------------|
| Eager | 73,191 ms | 570.52 ms | 1.0x |
| Eager + AMP | 38,796 ms | 301.91 ms | 1.9x |
| compile(default) | 59,172 ms | 460.72 ms | 1.2x |
| **compile(default) + AMP** | **29,154 ms** | **225.54 ms** | **2.5x** |

The full update cycle includes buffer fill/flatten (~120ms) and GPU transfer (~17ms) which are unaffected by compile/AMP. The 2.5x end-to-end speedup (73s → 29s) is excellent.

### Recommendation

**Enable `compile_mode="default"` with `use_amp=True` in all training configs.**

- `default` is recommended over `max-autotune` because it produces the same speedup with faster compilation startup and no "Not enough SMs" warning.
- `compile_dynamic=True` is fine for league mode; zero graph breaks with dynamic shapes.
- `reduce-overhead` is not recommended because `compile_dynamic=True` disables CUDA graph capture (the main benefit of that mode).

## Optimization Opportunities (Ranked by Impact)

### 1. Enable AMP — Expected: 1.9x update speedup, Effort: config change

**What:** Set `use_amp = true` in training configs.

**Why:** The 40-block SE-ResNet is memory-bandwidth-bound on the 4060 Ti. The 9x9 convolutions are small enough that bandwidth (not compute) is the bottleneck. bf16/fp16 halves the data moved per operation.

**Already supported:** `KataGoPPOParams.use_amp` and `KataGoBaseModel.configure_amp()` are implemented. Just a config change.

### 2. Enable torch.compile — Expected: additional 1.3x on top of AMP, Effort: config change

**What:** Set `compile_mode = "default"` in training configs.

**Why:** Zero graph breaks, fuses BN+ReLU+SE attention ops in the residual blocks. Combined with AMP gives 2.5x total update speedup.

**Already supported:** `compile_mode` config field, `compiled_train`/`compiled_eval` infrastructure in `KataGoPPOAlgorithm.__init__`.

### 3. Set TF32 matmul precision — Expected: ~5-15% for fp32 paths, Effort: one line

**What:** Add `torch.set_float32_matmul_precision('high')` at training startup.

**Why:** PyTorch warned "TensorFloat32 tensor cores available but not enabled". TF32 uses tensor cores for fp32 matrix multiplication with minimal precision loss. Benefits fp32 code paths (when AMP is off or for loss computation).

### 4. Pre-allocate rollout buffer tensors — Expected: reduce buffer fill from ~120ms, Effort: medium

**What:** Replace Python list append + `torch.cat` in `KataGoRolloutBuffer` with pre-allocated tensors indexed by step.

**Why:** `buffer_fill+flatten` is 120ms — the CPU tensor creation and list management dominate. Pre-allocation would avoid repeated Python object creation and the final `torch.cat` concatenation. This would also enable `pin_memory` for faster GPU transfer (filigree issue keisei-32065dc74b).

### 5. Increase batch size — Expected: better GPU utilization, Effort: config change

**What:** Try `batch_size=512` or `batch_size=1024` if memory allows.

**Why:** The 4060 Ti has 11.4 GB headroom at bs=256. Larger batches amortize kernel-launch overhead and improve GPU occupancy, particularly for the small SE/global-pool linear layers.

## What NOT to Optimize

- **GAE (CPU or GPU):** 1.6ms — 0.005% of update cycle. Leave as-is.
- **Policy sampling:** 0.36ms — negligible.
- **Loss computation:** 0.56ms total — negligible.
- **Policy/value/score heads:** <0.5ms combined — negligible.
- **torch.associative_scan for GAE:** The sequential scan is not the bottleneck.

## Appendix: Raw Data

Profile outputs saved in `profiles/`:
- `hotpath-small.txt` — small-scale component timing
- `hotpath-small-all.txt` — small-scale full benchmark with compile matrix
- `hotpath-production.txt` — production-scale component timing
- `hotpath-production-all.txt` — production-scale full benchmark with compile matrix
- `compile-diagnostics-small.txt` — torch._dynamo analysis
