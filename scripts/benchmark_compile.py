#!/usr/bin/env python3
"""Benchmark torch.compile modes and GPU GAE for the training hot path.

Runs 5 variants (baseline + 3 compile modes + best compile + GPU GAE),
each with warmup + measured epochs. Reports median/p95 timings.

Usage:
    uv run python scripts/benchmark_compile.py [--num-envs 64] [--steps 32] [--epochs 4]
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch._dynamo

from keisei.training.katago_ppo import (
    KataGoPPOAlgorithm,
    KataGoPPOParams,
    KataGoRolloutBuffer,
)
from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams


def create_model(device: torch.device) -> SEResNetModel:
    model = SEResNetModel(SEResNetParams(
        num_blocks=10, channels=128, se_reduction=8,
        global_pool_channels=64, policy_channels=16,
        value_fc_size=128, score_fc_size=64, obs_channels=50,
    ))
    return model.to(device)


def fill_buffer(
    num_envs: int, steps: int, action_space: int = 11259,
) -> KataGoRolloutBuffer:
    buf = KataGoRolloutBuffer(
        num_envs=num_envs, obs_shape=(50, 9, 9), action_space=action_space,
    )
    for t in range(steps):
        is_last = t == steps - 1
        buf.add(
            obs=torch.randn(num_envs, 50, 9, 9),
            actions=torch.randint(0, action_space, (num_envs,)),
            log_probs=torch.randn(num_envs),
            values=torch.randn(num_envs),
            rewards=torch.randn(num_envs) * 0.1,
            dones=torch.full((num_envs,), is_last, dtype=torch.float32),
            terminated=torch.full((num_envs,), is_last, dtype=torch.float32),
            legal_masks=torch.ones(num_envs, action_space, dtype=torch.bool),
            value_categories=torch.where(
                torch.tensor([is_last] * num_envs),
                torch.randint(0, 3, (num_envs,)),
                torch.full((num_envs,), -1),
            ),
            score_targets=torch.randn(num_envs).clamp(-1.5, 1.5),
        )
    return buf


def run_epoch(
    ppo: KataGoPPOAlgorithm,
    device: torch.device,
    num_envs: int,
    steps: int,
) -> dict[str, float]:
    """Run one epoch: rollout collection + PPO update. Return wall-clock timings."""
    action_space = 11259

    # --- Rollout collection ---
    rollout_start = time.perf_counter()
    buf = KataGoRolloutBuffer(
        num_envs=num_envs, obs_shape=(50, 9, 9), action_space=action_space,
    )
    for _ in range(steps):
        obs = torch.randn(num_envs, 50, 9, 9, device=device)
        legal_masks = torch.ones(num_envs, action_space, dtype=torch.bool, device=device)
        actions, log_probs, values = ppo.select_actions(obs, legal_masks)
        buf.add(
            obs=obs,
            actions=actions,
            log_probs=log_probs,
            values=values,
            rewards=torch.randn(num_envs) * 0.1,
            dones=torch.zeros(num_envs),
            terminated=torch.zeros(num_envs),
            legal_masks=legal_masks,
            value_categories=torch.full((num_envs,), -1),
            score_targets=torch.randn(num_envs).clamp(-1.5, 1.5),
        )
    rollout_time = time.perf_counter() - rollout_start

    # --- Bootstrap value ---
    with torch.no_grad():
        boot_obs = torch.randn(num_envs, 50, 9, 9, device=device)
        if ppo.compiled_eval is not None:
            boot_output = ppo.compiled_eval(boot_obs)
        else:
            ppo.forward_model.eval()
            boot_output = ppo.forward_model(boot_obs)
            ppo.forward_model.train()
        next_values = KataGoPPOAlgorithm.scalar_value(boot_output.value_logits)

    # --- PPO update ---
    update_start = time.perf_counter()
    metrics = ppo.update(buf, next_values)
    update_time = time.perf_counter() - update_start

    # --- Flush CUDA timers ---
    ppo.flush_timings()

    return {
        "rollout_wall_s": rollout_time,
        "update_wall_s": update_time,
        "epoch_wall_s": rollout_time + update_time,
        **{k: statistics.median(v) if v else 0.0 for k, v in ppo.timings.items()},
        **metrics,
    }


def run_variant(
    name: str,
    compile_mode: str | None,
    device: torch.device,
    num_envs: int,
    steps: int,
    warmup_epochs: int,
    measure_epochs: int,
) -> dict[str, list[float]]:
    """Run a single benchmark variant with warmup + measurement."""
    print(f"\n{'='*60}")
    print(f"Variant: {name}")
    print(f"  compile_mode={compile_mode}, num_envs={num_envs}, steps={steps}")
    print(f"{'='*60}")

    # Reset dynamo cache between variants
    torch._dynamo.reset()
    torch.cuda.empty_cache()

    model = create_model(device)
    params = KataGoPPOParams(
        compile_mode=compile_mode,
        compile_dynamic=False,  # fixed batch size for benchmarking
        batch_size=min(256, num_envs * steps),
        use_amp=True,
    )
    ppo = KataGoPPOAlgorithm(params, model)

    # Warmup
    for i in range(warmup_epochs):
        t0 = time.perf_counter()
        run_epoch(ppo, device, num_envs, steps)
        dt = time.perf_counter() - t0
        print(f"  warmup {i+1}/{warmup_epochs}: {dt:.1f}s")

    # Measurement
    results: dict[str, list[float]] = {}
    for i in range(measure_epochs):
        t0 = time.perf_counter()
        epoch_metrics = run_epoch(ppo, device, num_envs, steps)
        dt = time.perf_counter() - t0
        print(f"  measure {i+1}/{measure_epochs}: {dt:.1f}s")
        for k, v in epoch_metrics.items():
            results.setdefault(k, []).append(v)

    return results


def print_summary(all_results: dict[str, dict[str, list[float]]]) -> None:
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")

    key_metrics = [
        "epoch_wall_s",
        "rollout_wall_s",
        "update_wall_s",
        "select_actions_forward_ms",
        "update_forward_backward_ms",
        "gae_ms",
    ]

    # Header
    variants = list(all_results.keys())
    header = f"{'Metric':<35}" + "".join(f"{v:>15}" for v in variants)
    print(header)
    print("-" * len(header))

    for metric in key_metrics:
        row = f"{metric:<35}"
        for variant in variants:
            values = all_results[variant].get(metric, [])
            if values:
                med = statistics.median(values)
                if "wall_s" in metric:
                    row += f"{med:>14.3f}s"
                else:
                    row += f"{med:>13.2f}ms"
            else:
                row += f"{'N/A':>15}"
        print(row)

    # Speedup vs baseline
    baseline = variants[0]
    baseline_epoch = statistics.median(all_results[baseline].get("epoch_wall_s", [1.0]))
    print(f"\n{'Speedup vs baseline':<35}", end="")
    for variant in variants:
        epoch = statistics.median(all_results[variant].get("epoch_wall_s", [1.0]))
        speedup = baseline_epoch / epoch if epoch > 0 else 0
        print(f"{speedup:>14.2f}x", end="")
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark torch.compile for training hot path")
    parser.add_argument("--num-envs", type=int, default=64, help="Number of environments")
    parser.add_argument("--steps", type=int, default=32, help="Rollout steps per epoch")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup epochs per variant")
    parser.add_argument("--epochs", type=int, default=4, help="Measurement epochs per variant")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Config: num_envs={args.num_envs}, steps={args.steps}, "
          f"warmup={args.warmup}, measure={args.epochs}")

    variants = [
        ("Baseline (eager)", None),
        ("compile(default)", "default"),
        ("compile(reduce-overhead)", "reduce-overhead"),
        ("compile(max-autotune)", "max-autotune"),
    ]

    all_results = {}
    for name, mode in variants:
        results = run_variant(
            name, mode, device,
            args.num_envs, args.steps,
            args.warmup, args.epochs,
        )
        all_results[name] = results

    print_summary(all_results)


if __name__ == "__main__":
    main()
