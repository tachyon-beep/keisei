#!/usr/bin/env python3
"""Component-level hot-path profiling for Keisei training.

Profiles each component of the PPO training loop in isolation:
  1. Model forward pass (inference + training mode)
  2. Policy head: masked softmax + sampling
  3. GAE computation (CPU vs GPU)
  4. Buffer flatten + GPU transfer
  5. PPO mini-batch update (forward + backward + optimizer step)
  6. torch.compile diagnostic (graph breaks, recompilation)

Each component is measured with proper warmup, CUDA sync, and statistics.

Usage:
    uv run python scripts/profile_hotpath.py [--scale small|production] [--device cuda:0]
    uv run python scripts/profile_hotpath.py --compile-diagnostics
    uv run python scripts/profile_hotpath.py --all
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from keisei.training.gae import compute_gae, compute_gae_gpu
from keisei.training.katago_ppo import (
    KataGoPPOAlgorithm,
    KataGoPPOParams,
    KataGoRolloutBuffer,
)
from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams


# ---------------------------------------------------------------------------
# Model scales
# ---------------------------------------------------------------------------

SCALES: dict[str, SEResNetParams] = {
    "small": SEResNetParams(
        num_blocks=10, channels=128, se_reduction=8,
        global_pool_channels=64, policy_channels=16,
        value_fc_size=128, score_fc_size=64, obs_channels=50,
    ),
    "production": SEResNetParams(
        num_blocks=40, channels=256, se_reduction=16,
        global_pool_channels=128, policy_channels=32,
        value_fc_size=256, score_fc_size=128, obs_channels=50,
    ),
}


def create_model_at_scale(
    scale: str, device: str | torch.device,
) -> tuple[SEResNetModel, SEResNetParams]:
    """Create an SEResNetModel at the given scale."""
    params = SCALES[scale]
    model = SEResNetModel(params).to(device)
    return model, params


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------

@dataclass
class TimingResult:
    """Statistical summary of a timed operation."""
    name: str
    times_ms: list[float]

    @property
    def median_ms(self) -> float:
        return statistics.median(self.times_ms) if self.times_ms else 0.0

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.times_ms) if self.times_ms else 0.0

    @property
    def std_ms(self) -> float:
        return statistics.stdev(self.times_ms) if len(self.times_ms) > 1 else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.times_ms:
            return 0.0
        sorted_t = sorted(self.times_ms)
        idx = max(0, int(len(sorted_t) * 0.95) - 1)
        return sorted_t[idx]

    def __str__(self) -> str:
        return (
            f"{self.name:<45} "
            f"median={self.median_ms:8.2f}ms  "
            f"mean={self.mean_ms:8.2f}ms  "
            f"std={self.std_ms:7.2f}ms  "
            f"p95={self.p95_ms:8.2f}ms  "
            f"(n={len(self.times_ms)})"
        )


def time_cuda_op(
    fn, device: torch.device, warmup: int = 5, repeats: int = 20,
) -> TimingResult:
    """Time a CUDA operation with proper warmup and synchronization.

    Uses CUDA events for accurate GPU timing (not wall-clock).
    """
    # Warmup
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize(device)
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(device)
        times.append(start.elapsed_time(end))

    return TimingResult(name="", times_ms=times)


def time_cpu_op(
    fn, warmup: int = 3, repeats: int = 10,
) -> TimingResult:
    """Time a CPU operation with warmup."""
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)

    return TimingResult(name="", times_ms=times)


# ---------------------------------------------------------------------------
# Component benchmarks
# ---------------------------------------------------------------------------

def profile_model_forward(
    scale: str, device: torch.device, batch_sizes: list[int],
) -> list[TimingResult]:
    """Profile model forward pass at various batch sizes."""
    model, params = create_model_at_scale(scale, device)
    model.eval()
    results = []

    for bs in batch_sizes:
        obs = torch.randn(bs, 50, 9, 9, device=device)

        def forward_pass(obs=obs):
            with torch.no_grad():
                model(obs)

        r = time_cuda_op(forward_pass, device)
        r.name = f"forward(eval) bs={bs} [{scale}]"
        results.append(r)

    # Training mode (includes BN update)
    model.train()
    for bs in batch_sizes:
        obs = torch.randn(bs, 50, 9, 9, device=device)

        def forward_train(obs=obs):
            model(obs)

        r = time_cuda_op(forward_train, device)
        r.name = f"forward(train) bs={bs} [{scale}]"
        results.append(r)

    return results


def profile_model_forward_backward(
    scale: str, device: torch.device, batch_size: int = 256,
) -> list[TimingResult]:
    """Profile forward + backward pass together."""
    model, params = create_model_at_scale(scale, device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    results = []

    obs = torch.randn(batch_size, 50, 9, 9, device=device)

    def fwd_bwd():
        optimizer.zero_grad(set_to_none=True)
        out = model(obs)
        # Simulate combined loss
        loss = out.policy_logits.sum() + out.value_logits.sum() + out.score_lead.sum()
        loss.backward()
        optimizer.step()

    r = time_cuda_op(fwd_bwd, device)
    r.name = f"forward+backward+step bs={batch_size} [{scale}]"
    results.append(r)
    return results


def profile_policy_sampling(
    device: torch.device, batch_sizes: list[int],
) -> list[TimingResult]:
    """Profile the policy head hot path: masked_fill + softmax + Categorical."""
    action_space = 11259
    results = []

    for bs in batch_sizes:
        logits = torch.randn(bs, action_space, device=device)
        legal_mask = torch.ones(bs, action_space, dtype=torch.bool, device=device)
        # Make ~30% of actions illegal (realistic)
        legal_mask[:, action_space // 3:] = False

        def sample():
            masked = logits.masked_fill(~legal_mask, float("-inf"))
            probs = F.softmax(masked, dim=-1)
            dist = torch.distributions.Categorical(probs, validate_args=False)
            actions = dist.sample()
            _ = dist.log_prob(actions)

        r = time_cuda_op(sample, device)
        r.name = f"policy_sample bs={bs}"
        results.append(r)

    return results


def profile_gae(
    device: torch.device, T: int = 128, N_values: list[int] | None = None,
) -> list[TimingResult]:
    """Profile GAE computation: CPU vs GPU at various environment counts."""
    if N_values is None:
        N_values = [32, 64, 128, 256]
    results = []

    for N in N_values:
        rewards = torch.randn(T, N)
        values = torch.randn(T, N)
        terminated = torch.zeros(T, N)
        next_value = torch.randn(N)

        # CPU GAE
        def gae_cpu():
            compute_gae(rewards, values, terminated, next_value, gamma=0.99, lam=0.95)

        r = time_cpu_op(gae_cpu)
        r.name = f"GAE CPU T={T} N={N}"
        results.append(r)

        # GPU GAE
        rewards_gpu = rewards.to(device)
        values_gpu = values.to(device)
        terminated_gpu = terminated.to(device)
        next_value_gpu = next_value.to(device)

        def gae_gpu():
            compute_gae_gpu(
                rewards_gpu, values_gpu, terminated_gpu, next_value_gpu,
                gamma=0.99, lam=0.95,
            )

        r = time_cuda_op(gae_gpu, device)
        r.name = f"GAE GPU T={T} N={N}"
        results.append(r)

    return results


def profile_buffer_ops(
    device: torch.device, T: int = 128, N: int = 64,
) -> list[TimingResult]:
    """Profile buffer flatten + GPU transfer."""
    action_space = 11259
    results = []

    def make_buffer() -> KataGoRolloutBuffer:
        buf = KataGoRolloutBuffer(
            num_envs=N, obs_shape=(50, 9, 9), action_space=action_space,
        )
        for t in range(T):
            is_last = t == T - 1
            buf.add(
                obs=torch.randn(N, 50, 9, 9),
                actions=torch.randint(0, action_space, (N,)),
                log_probs=torch.randn(N),
                values=torch.randn(N),
                rewards=torch.randn(N) * 0.1,
                dones=torch.full((N,), is_last, dtype=torch.float32),
                terminated=torch.full((N,), is_last, dtype=torch.float32),
                legal_masks=torch.ones(N, action_space, dtype=torch.bool),
                value_categories=torch.where(
                    torch.tensor([is_last] * N),
                    torch.randint(0, 3, (N,)),
                    torch.full((N,), -1),
                ),
                score_targets=torch.randn(N).clamp(-1.5, 1.5),
            )
        return buf

    # Profile flatten
    buf = make_buffer()

    def flatten_op():
        # Re-fill buffer since flatten doesn't clear it, but we need fresh data
        nonlocal buf
        buf = make_buffer()
        buf.flatten()

    r = time_cpu_op(flatten_op, warmup=1, repeats=5)
    r.name = f"buffer_fill+flatten T={T} N={N}"
    results.append(r)

    # Profile GPU transfer (the bulk transfer in update())
    data = make_buffer().flatten()

    def gpu_transfer():
        data["observations"].to(device, non_blocking=True)
        data["actions"].to(device, non_blocking=True)
        data["log_probs"].to(device, non_blocking=True)
        data["legal_masks"].to(device, non_blocking=True)
        data["value_categories"].to(device, non_blocking=True)
        data["score_targets"].to(device, non_blocking=True)
        torch.cuda.synchronize(device)

    r = time_cuda_op(gpu_transfer, device, warmup=2, repeats=10)
    r.name = f"gpu_transfer T={T} N={N}"
    results.append(r)

    # Report transfer sizes
    total_bytes = sum(
        data[k].nelement() * data[k].element_size()
        for k in ["observations", "actions", "log_probs", "legal_masks",
                   "value_categories", "score_targets"]
    )
    print(f"  GPU transfer size: {total_bytes / 1e6:.1f} MB")

    return results


def profile_ppo_update(
    scale: str, device: torch.device,
    T: int = 128, N: int = 64,
    compile_mode: str | None = None,
    use_amp: bool = False,
) -> list[TimingResult]:
    """Profile a full PPO update cycle."""
    model, params = create_model_at_scale(scale, device)
    action_space = 11259

    ppo_params = KataGoPPOParams(
        compile_mode=compile_mode,
        compile_dynamic=False,
        batch_size=256,
        use_amp=use_amp,
    )
    ppo = KataGoPPOAlgorithm(ppo_params, model)

    results = []
    label = f"[{scale}, compile={compile_mode}, amp={use_amp}]"

    def run_update():
        buf = KataGoRolloutBuffer(
            num_envs=N, obs_shape=(50, 9, 9), action_space=action_space,
        )
        for t in range(T):
            is_last = t == T - 1
            buf.add(
                obs=torch.randn(N, 50, 9, 9),
                actions=torch.randint(0, action_space, (N,)),
                log_probs=torch.randn(N),
                values=torch.randn(N),
                rewards=torch.randn(N) * 0.1,
                dones=torch.full((N,), is_last, dtype=torch.float32),
                terminated=torch.full((N,), is_last, dtype=torch.float32),
                legal_masks=torch.ones(N, action_space, dtype=torch.bool),
                value_categories=torch.where(
                    torch.tensor([is_last] * N),
                    torch.randint(0, 3, (N,)),
                    torch.full((N,), -1),
                ),
                score_targets=torch.randn(N).clamp(-1.5, 1.5),
            )
        next_values = torch.randn(N, device=device)
        ppo.update(buf, next_values)

    # Warmup iterations (discard internal timings)
    for _ in range(2):
        run_update()
    ppo.flush_timings()  # discard warmup timings

    # Measured iterations
    torch.cuda.synchronize(device)
    times = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_update()
        end.record()
        torch.cuda.synchronize(device)
        times.append(start.elapsed_time(end))

    r = TimingResult(name=f"ppo_update T={T} N={N} {label}", times_ms=times)
    results.append(r)

    # Internal timings — only from measured iterations
    ppo.flush_timings()
    for key, vals in ppo.timings.items():
        if vals:
            r2 = TimingResult(name=f"  {key} {label}", times_ms=vals)
            results.append(r2)

    return results


# ---------------------------------------------------------------------------
# torch.compile diagnostics
# ---------------------------------------------------------------------------

def run_compile_diagnostics(scale: str, device: torch.device) -> None:
    """Analyze torch.compile behavior: graph breaks, recompilations."""
    import torch._dynamo

    model, params = create_model_at_scale(scale, device)
    model.eval()
    obs = torch.randn(4, 50, 9, 9, device=device)

    print(f"\n{'='*80}")
    print(f"TORCH.COMPILE DIAGNOSTICS [{scale}]")
    print(f"{'='*80}")

    # Explain what dynamo can/can't compile
    explanation = torch._dynamo.explain(model)(obs)
    print(f"\nGraph breaks: {explanation.break_count}")
    print(f"Ops in graph: {explanation.ops_per_graph}")
    if explanation.break_reasons:
        print("\nBreak reasons:")
        for reason in explanation.break_reasons:
            print(f"  - {reason}")

    # Check for recompilations with different batch sizes
    print(f"\n--- Recompilation test (varying batch sizes) ---")
    torch._dynamo.reset()
    compiled = torch.compile(model, mode="default", dynamic=True)

    for bs in [4, 16, 64, 256]:
        obs = torch.randn(bs, 50, 9, 9, device=device)
        with torch.no_grad():
            compiled(obs)
        counters = torch._dynamo.utils.counters
        guards = counters.get("graph_break", {})
        print(f"  bs={bs:>4}: graph_breaks={sum(guards.values()) if guards else 0}")

    # Check with compile_dynamic=False (fixed shapes)
    print(f"\n--- Static shapes (compile_dynamic=False) ---")
    torch._dynamo.reset()
    compiled_static = torch.compile(model, mode="default", dynamic=False)
    obs = torch.randn(256, 50, 9, 9, device=device)
    with torch.no_grad():
        compiled_static(obs)
    explanation_static = torch._dynamo.explain(model)(obs)
    print(f"  Graph breaks (static): {explanation_static.break_count}")

    # AMP interaction check
    print(f"\n--- AMP + compile interaction ---")
    torch._dynamo.reset()
    model_amp = SEResNetModel(params).to(device)
    model_amp.configure_amp(enabled=True, dtype=torch.bfloat16, device_type="cuda")
    model_amp.eval()
    explanation_amp = torch._dynamo.explain(model_amp)(obs)
    print(f"  Graph breaks (AMP): {explanation_amp.break_count}")
    if explanation_amp.break_reasons:
        for reason in explanation_amp.break_reasons:
            print(f"    - {reason}")


# ---------------------------------------------------------------------------
# Compile speedup matrix
# ---------------------------------------------------------------------------

def profile_compile_matrix(
    scale: str, device: torch.device, batch_size: int = 256,
) -> list[TimingResult]:
    """Compare eager vs compile modes x AMP on/off for forward pass."""
    results = []
    obs = torch.randn(batch_size, 50, 9, 9, device=device)

    configs = [
        ("eager", None, False),
        ("eager+AMP", None, True),
        ("compile(default)", "default", False),
        ("compile(default)+AMP", "default", True),
        ("compile(reduce-overhead)", "reduce-overhead", False),
        ("compile(max-autotune)", "max-autotune", False),
        ("compile(max-autotune)+AMP", "max-autotune", True),
    ]

    for label, compile_mode, use_amp in configs:
        if compile_mode:
            torch._dynamo.reset()
        torch.cuda.empty_cache()

        model, params = create_model_at_scale(scale, device)

        if use_amp:
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model.configure_amp(enabled=True, dtype=amp_dtype, device_type="cuda")

        model.eval()

        if compile_mode:
            compiled = torch.compile(model, mode=compile_mode, dynamic=False)
        else:
            compiled = model

        def forward_fn(m=compiled, x=obs):
            with torch.no_grad():
                m(x)

        r = time_cuda_op(forward_fn, device, warmup=5, repeats=20)
        r.name = f"forward {label} bs={batch_size} [{scale}]"
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# Per-block profiling
# ---------------------------------------------------------------------------

def profile_per_block(
    scale: str, device: torch.device, batch_size: int = 256,
) -> list[TimingResult]:
    """Profile time spent in each component of the model."""
    model, params = create_model_at_scale(scale, device)
    model.eval()
    results = []

    obs = torch.randn(batch_size, 50, 9, 9, device=device)

    # Input conv + BN + ReLU
    def input_stem():
        with torch.no_grad():
            x = F.relu(model.input_bn(model.input_conv(obs)))
        return x

    r = time_cuda_op(input_stem, device)
    r.name = f"input_stem [{scale}]"
    results.append(r)

    # Get intermediate tensor for subsequent profiling
    with torch.no_grad():
        x = F.relu(model.input_bn(model.input_conv(obs)))

    # Single residual block
    block = model.blocks[0]

    def one_block():
        with torch.no_grad():
            block(x)

    r = time_cuda_op(one_block, device)
    r.name = f"single_block [{scale}]"
    results.append(r)

    # Full residual tower
    def full_tower():
        with torch.no_grad():
            model.blocks(x)

    r = time_cuda_op(full_tower, device)
    r.name = f"residual_tower ({params.num_blocks} blocks) [{scale}]"
    results.append(r)

    # Policy head
    with torch.no_grad():
        trunk_out = model.blocks(x)

    def policy_head():
        with torch.no_grad():
            p = F.relu(model.policy_bn1(model.policy_conv1(trunk_out)))
            model.policy_conv2(p)

    r = time_cuda_op(policy_head, device)
    r.name = f"policy_head [{scale}]"
    results.append(r)

    # Value + Score heads (shared global pool)
    from keisei.training.models.se_resnet import _global_pool

    def value_score_heads():
        with torch.no_grad():
            pool = _global_pool(trunk_out)
            F.relu(model.value_fc1(pool))
            F.relu(model.score_fc1(pool))

    r = time_cuda_op(value_score_heads, device)
    r.name = f"value+score_heads [{scale}]"
    results.append(r)

    return results


def profile_loss_components(
    device: str | torch.device, batch_size: int = 256,
) -> list[TimingResult]:
    """Profile individual loss computation components."""
    device = torch.device(device)
    action_space = 11259
    results = []
    is_cuda = device.type == "cuda"
    timer = (lambda fn: time_cuda_op(fn, device)) if is_cuda else time_cpu_op

    # Synthetic data
    logits = torch.randn(batch_size, action_space, device=device)
    legal_mask = torch.ones(batch_size, action_space, dtype=torch.bool, device=device)
    legal_mask[:, action_space // 3:] = False
    old_log_probs = torch.randn(batch_size, device=device)
    advantages = torch.randn(batch_size, device=device)
    value_logits = torch.randn(batch_size, 3, device=device)
    value_cats = torch.randint(0, 3, (batch_size,), device=device)
    score_pred = torch.randn(batch_size, 1, device=device)
    score_targets = torch.randn(batch_size, device=device)

    # 1. Masked log_softmax + gather (policy preprocessing)
    def policy_preprocess():
        masked = logits.masked_fill(~legal_mask, float("-inf"))
        log_p = F.log_softmax(masked, dim=-1)
        actions = torch.randint(0, action_space, (batch_size,), device=device)
        log_p.gather(1, actions.unsqueeze(1)).squeeze(1)

    r = timer(policy_preprocess)
    r.name = f"policy_log_softmax+gather bs={batch_size}"
    results.append(r)

    # 2. PPO clip loss
    new_log_probs = torch.randn(batch_size, device=device)

    def clip_loss():
        from keisei.training.katago_ppo import ppo_clip_loss
        ppo_clip_loss(new_log_probs, old_log_probs, advantages, 0.2)

    r = timer(clip_loss)
    r.name = f"ppo_clip_loss bs={batch_size}"
    results.append(r)

    # 3. Entropy computation
    def entropy_op():
        masked = logits.masked_fill(~legal_mask, float("-inf"))
        log_p = F.log_softmax(masked, dim=-1)
        probs = log_p.exp()
        safe_log = log_p.masked_fill(~legal_mask, 0.0)
        -(probs * safe_log).sum(dim=-1).mean()

    r = timer(entropy_op)
    r.name = f"entropy bs={batch_size}"
    results.append(r)

    # 4. WDL cross-entropy
    def wdl_loss():
        F.cross_entropy(value_logits, value_cats, ignore_index=-1)

    r = timer(wdl_loss)
    r.name = f"wdl_cross_entropy bs={batch_size}"
    results.append(r)

    # 5. Score MSE
    def score_loss():
        F.mse_loss(score_pred.squeeze(-1), score_targets)

    r = timer(score_loss)
    r.name = f"score_mse bs={batch_size}"
    results.append(r)

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_section(title: str, results: list[TimingResult]) -> None:
    print(f"\n{'='*80}")
    print(title)
    print(f"{'='*80}")
    for r in results:
        print(f"  {r}")


def print_recommendations(all_results: dict[str, list[TimingResult]]) -> None:
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")

    # Compare compile vs eager from matrix
    matrix = all_results.get("compile_matrix", [])
    eager_time = None
    best_compile_time = None
    best_compile_label = ""
    for r in matrix:
        if "eager" in r.name and "AMP" not in r.name:
            eager_time = r.median_ms
        if "compile" in r.name and (best_compile_time is None or r.median_ms < best_compile_time):
            best_compile_time = r.median_ms
            best_compile_label = r.name

    if eager_time and best_compile_time:
        speedup = eager_time / best_compile_time
        print(f"\n  torch.compile forward-pass speedup: {speedup:.2f}x")
        print(f"    Eager: {eager_time:.2f}ms -> Best compile ({best_compile_label}): {best_compile_time:.2f}ms")
        if speedup < 1.1:
            print("    -> Compile provides minimal benefit for forward pass alone.")
            print("    -> Focus on AMP, batch size tuning, and data loading instead.")
        elif speedup < 1.3:
            print("    -> Moderate compile benefit. Worth enabling for training.")
        else:
            print("    -> Significant compile benefit. Enable with the best mode above.")

    # GAE analysis
    gae_results = all_results.get("gae", [])
    gpu_gae = None
    cpu_gae = None
    for r in gae_results:
        if "GPU" in r.name and "N=64" in r.name:
            gpu_gae = r.median_ms
        if "CPU" in r.name and "N=64" in r.name:
            cpu_gae = r.median_ms
    if gpu_gae and cpu_gae:
        speedup = cpu_gae / gpu_gae if gpu_gae > 0 else 0
        print(f"\n  GAE GPU vs CPU (N=64): {cpu_gae:.2f}ms -> {gpu_gae:.2f}ms ({speedup:.1f}x)")
        if speedup < 1.2:
            print("    -> GPU GAE provides minimal benefit at this scale.")
        else:
            print("    -> GPU GAE worthwhile. Already enabled when device=cuda.")

    # Per-block analysis
    blocks = all_results.get("per_block", [])
    tower_time = None
    total_fwd = None
    for r in blocks:
        if "residual_tower" in r.name:
            tower_time = r.median_ms
    fwd_results = all_results.get("forward", [])
    for r in fwd_results:
        if "eval" in r.name and "bs=256" in r.name:
            total_fwd = r.median_ms
            break

    if tower_time and total_fwd:
        pct = tower_time / total_fwd * 100
        print(f"\n  Residual tower: {tower_time:.2f}ms = {pct:.0f}% of forward pass")
        if pct > 85:
            print("    -> Tower dominates. Optimization effort should focus here.")
            print("    -> Consider: fewer blocks, smaller channels, or compile mode")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Component-level hot-path profiling for Keisei",
    )
    parser.add_argument(
        "--scale", choices=list(SCALES.keys()), default="small",
        help="Model scale: small (10 blocks/128ch) or production (40 blocks/256ch)",
    )
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    parser.add_argument(
        "--compile-diagnostics", action="store_true",
        help="Run torch._dynamo diagnostic analysis",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all benchmarks including compile matrix (slow)",
    )
    parser.add_argument("--T", type=int, default=128, help="Rollout length")
    parser.add_argument("--N", type=int, default=64, help="Number of environments")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size")
    args = parser.parse_args()

    device = torch.device(args.device)
    if not torch.cuda.is_available() or device.type != "cuda":
        sys.exit("Error: profile_hotpath.py requires a CUDA device. Pass --device cuda:N.")
    scale = args.scale

    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    print(f"Scale: {scale} ({SCALES[scale].num_blocks} blocks, {SCALES[scale].channels} channels)")
    print(f"Rollout: T={args.T}, N={args.N}, batch_size={args.batch_size}")
    print(f"PyTorch: {torch.__version__}")

    all_results: dict[str, list[TimingResult]] = {}

    # 1. Model forward pass
    batch_sizes = [16, 64, 256]
    results = profile_model_forward(scale, device, batch_sizes)
    all_results["forward"] = results
    print_section("MODEL FORWARD PASS", results)

    # 2. Forward + backward + optimizer step
    results = profile_model_forward_backward(scale, device, args.batch_size)
    all_results["forward_backward"] = results
    print_section("FORWARD + BACKWARD + OPTIMIZER STEP", results)

    # 3. Policy sampling
    results = profile_policy_sampling(device, batch_sizes)
    all_results["policy_sampling"] = results
    print_section("POLICY SAMPLING (masked_fill + softmax + Categorical)", results)

    # 4. GAE
    results = profile_gae(device, T=args.T)
    all_results["gae"] = results
    print_section("GAE COMPUTATION (CPU vs GPU)", results)

    # 5. Buffer ops
    results = profile_buffer_ops(device, T=args.T, N=args.N)
    all_results["buffer"] = results
    print_section("BUFFER OPERATIONS", results)

    # 6. Per-block breakdown
    results = profile_per_block(scale, device, args.batch_size)
    all_results["per_block"] = results
    print_section("PER-COMPONENT BREAKDOWN", results)

    # 6b. Loss component breakdown
    results = profile_loss_components(device, args.batch_size)
    all_results["loss_components"] = results
    print_section("LOSS COMPONENT BREAKDOWN", results)

    # 7. Compile diagnostics (optional)
    if args.compile_diagnostics or args.all:
        run_compile_diagnostics(scale, device)

    # 8. Compile x AMP matrix (optional, slow)
    if args.all:
        results = profile_compile_matrix(scale, device, args.batch_size)
        all_results["compile_matrix"] = results
        print_section("COMPILE x AMP MATRIX (forward pass only)", results)

        # Full PPO update comparison
        update_results = []
        for compile_mode, use_amp, label in [
            (None, False, "eager"),
            (None, True, "eager+AMP"),
            ("default", False, "compile(default)"),
            ("default", True, "compile(default)+AMP"),
        ]:
            r = profile_ppo_update(
                scale, device, T=args.T, N=args.N,
                compile_mode=compile_mode, use_amp=use_amp,
            )
            update_results.extend(r)
        all_results["ppo_update"] = update_results
        print_section("PPO UPDATE (full cycle)", update_results)

    # 9. Recommendations
    print_recommendations(all_results)


if __name__ == "__main__":
    main()
