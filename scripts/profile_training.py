#!/usr/bin/env python3
"""Full training loop profiler for Keisei.

Runs a short training session (configurable epochs/steps) under the PyTorch
profiler, producing:
  1. Console summary (CPU + CUDA time, top-10 ops)
  2. Chrome trace (viewable at chrome://tracing or https://ui.perfetto.dev/)
  3. Phase-level wall-clock breakdown (rollout vs GAE vs PPO update vs bookkeeping)

Usage:
    uv run python scripts/profile_training.py keisei-katago.toml \
        --epochs 2 --steps-per-epoch 50 --output-dir profiles/

The script wraps the real KataGoTrainingLoop but instruments the epoch body
so you get real data flow — not a synthetic microbenchmark.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure keisei is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from keisei.config import load_config  # noqa: E402
from keisei.training.katago_loop import (  # noqa: E402
    KataGoTrainingLoop,
    _compute_value_cats,
    get_distributed_context,
    seed_all_ranks,
    setup_distributed,
    cleanup_distributed,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("profile")


# ---------------------------------------------------------------------------
# Shared rollout logic
# ---------------------------------------------------------------------------

def run_one_epoch_rollout(
    loop: KataGoTrainingLoop,
    obs: torch.Tensor,
    legal_masks: torch.Tensor,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one epoch of rollout, returning (obs, legal_masks) at end."""
    device = loop.device
    loop.buffer.clear()

    for step_i in range(steps):
        loop.global_step += 1

        # Simple self-play path (no league/opponent) — mirrors the
        # else branch in _run_training_body when _current_opponent is None.
        actions, log_probs, values = loop.ppo.select_actions(
            obs, legal_masks, value_adapter=loop.value_adapter,
        )

        action_list = actions.tolist()
        step_result = loop.vecenv.step(action_list)

        new_obs = torch.from_numpy(np.asarray(step_result.observations)).to(device)
        new_legal_masks = torch.from_numpy(np.asarray(step_result.legal_masks)).to(device)
        rewards = torch.from_numpy(np.asarray(step_result.rewards)).to(device)
        terminated = torch.from_numpy(np.asarray(step_result.terminated)).to(device)
        truncated = torch.from_numpy(np.asarray(step_result.truncated)).to(device)
        dones = terminated | truncated

        terminal_mask = terminated.bool()
        value_cats = _compute_value_cats(rewards, terminal_mask, device)

        material = torch.from_numpy(
            np.asarray(step_result.step_metadata.material_balance, dtype=np.float32),
        ).to(device)
        score_targets = material / loop.score_norm

        loop.buffer.add(
            obs, actions, log_probs, values, rewards, dones,
            terminated, legal_masks, value_cats, score_targets,
        )

        obs = new_obs
        legal_masks = new_legal_masks

    return obs, legal_masks


def run_ppo_update(loop: KataGoTrainingLoop, obs: torch.Tensor) -> dict[str, float]:
    """Bootstrap + PPO update, returns losses dict."""
    loop.ppo.forward_model.eval()
    with torch.no_grad():
        output = loop.ppo.forward_model(obs)
        next_values = loop.value_adapter.scalar_value_blended(
            output.value_logits, output.score_lead,
        )
    loop.ppo.forward_model.train()

    losses = loop.ppo.update(
        loop.buffer, next_values,
        value_adapter=loop.value_adapter,
    )
    return losses


# ---------------------------------------------------------------------------
# Phase 1: Baseline timing (no profiler overhead)
# ---------------------------------------------------------------------------

def run_baseline(loop: KataGoTrainingLoop, steps: int, num_warmup: int = 1) -> dict:
    """Run a few epochs to get wall-clock baseline per phase.

    Returns timing dict with per-phase mean/std in seconds.
    """
    device = loop.device

    phase_times: dict[str, list[float]] = {
        "epoch_total": [],
        "rollout_total": [],
        "ppo_update": [],
        "bookkeeping": [],
    }

    total_epochs = num_warmup + 2  # 2 measured epochs
    logger.info("=== Phase 1: Baseline (%d warmup + 2 measured epochs, %d steps each) ===",
                num_warmup, steps)

    # Reset environment
    reset_result = loop.vecenv.reset()
    obs = torch.from_numpy(np.asarray(reset_result.observations)).to(device)
    legal_masks = torch.from_numpy(np.asarray(reset_result.legal_masks)).to(device)

    for epoch_i in range(total_epochs):
        # Re-reset env each epoch to avoid degenerate game states
        reset_result = loop.vecenv.reset()
        obs = torch.from_numpy(np.asarray(reset_result.observations)).to(device)
        legal_masks = torch.from_numpy(np.asarray(reset_result.legal_masks)).to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()

        t_epoch_start = time.perf_counter()

        # --- Rollout ---
        t_rollout_start = time.perf_counter()
        obs, legal_masks = run_one_epoch_rollout(loop, obs, legal_masks, steps)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_rollout_end = time.perf_counter()

        # --- Bootstrap + PPO Update ---
        t_update_start = time.perf_counter()
        losses = run_ppo_update(loop, obs)
        loop.ppo.flush_timings()

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_update_end = time.perf_counter()

        t_epoch_end = time.perf_counter()

        if epoch_i >= num_warmup:
            phase_times["epoch_total"].append(t_epoch_end - t_epoch_start)
            phase_times["rollout_total"].append(t_rollout_end - t_rollout_start)
            phase_times["ppo_update"].append(t_update_end - t_update_start)
            phase_times["bookkeeping"].append(
                (t_epoch_end - t_epoch_start) -
                (t_rollout_end - t_rollout_start) -
                (t_update_end - t_update_start)
            )

            # CUDA event timings from PPO
            for key in ["select_actions_forward_ms", "gae_ms", "update_forward_backward_ms"]:
                vals = loop.ppo.timings.get(key, [])
                if vals:
                    mean_ms = sum(vals) / len(vals)
                    logger.info("  [CUDA] %s: %.2f ms mean (%d samples)",
                                key, mean_ms, len(vals))

    # Compute stats
    results = {}
    for phase, times in phase_times.items():
        if times:
            mean = sum(times) / len(times)
            std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
            results[phase] = {"mean_s": mean, "std_s": std, "samples": times}

    return results


# ---------------------------------------------------------------------------
# Phase 2+3: torch.profiler run
# ---------------------------------------------------------------------------

def run_profiler(
    loop: KataGoTrainingLoop,
    steps: int,
    output_dir: Path,
) -> None:
    """Run one epoch under torch.profiler for op-level detail."""
    from torch.profiler import profile, ProfilerActivity, schedule

    device = loop.device
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "training_trace.json"

    logger.info("=== Phase 2+3: torch.profiler (1 warmup + 1 measured epoch, %d steps) ===", steps)

    # Reset environment
    reset_result = loop.vecenv.reset()
    obs = torch.from_numpy(np.asarray(reset_result.observations)).to(device)
    legal_masks = torch.from_numpy(np.asarray(reset_result.legal_masks)).to(device)

    # We profile 2 epochs: 1 warmup (skip=1) + 1 active
    with profile(
        activities=activities,
        schedule=schedule(wait=0, warmup=1, active=1, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=lambda p: p.export_chrome_trace(str(trace_path)),
    ) as prof:
        for epoch_pass in range(2):
            obs, legal_masks = run_one_epoch_rollout(loop, obs, legal_masks, steps)
            losses = run_ppo_update(loop, obs)
            prof.step()

    logger.info("Chrome trace saved to: %s", trace_path)
    logger.info("Open in chrome://tracing or https://ui.perfetto.dev/")

    # Print summary tables
    print("\n" + "=" * 80)
    print("TOP OPERATIONS BY CPU TIME")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

    if device.type == "cuda":
        print("\n" + "=" * 80)
        print("TOP OPERATIONS BY CUDA TIME")
        print("=" * 80)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

        print("\n" + "=" * 80)
        print("MEMORY ALLOCATION")
        print("=" * 80)
        print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Profile Keisei training loop")
    parser.add_argument("config", type=Path, help="Path to TOML config file")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Epochs for baseline (default: 2 warmup + 3 measured)")
    parser.add_argument("--steps-per-epoch", type=int, default=50,
                        help="Steps per epoch (default: 50, lower = faster profiling)")
    parser.add_argument("--output-dir", type=Path, default=Path("profiles"),
                        help="Directory for trace output")
    parser.add_argument("--skip-profiler", action="store_true",
                        help="Only run baseline timing, skip torch.profiler")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dist_ctx = get_distributed_context()
    setup_distributed(dist_ctx)

    try:
        seed_all_ranks(args.seed + dist_ctx.rank)
        config = load_config(args.config)
        loop = KataGoTrainingLoop(config, dist_ctx=dist_ctx)

        device = loop.device
        logger.info("Device: %s", device)
        if device.type == "cuda":
            logger.info("GPU: %s", torch.cuda.get_device_name(device))
            logger.info("VRAM: %.1f GB", torch.cuda.get_device_properties(device).total_memory / 1e9)

        # Phase 1: Baseline
        baseline = run_baseline(loop, steps=args.steps_per_epoch, num_warmup=1)

        print("\n" + "=" * 80)
        print("BASELINE TIMING (wall-clock, per epoch)")
        print("=" * 80)
        for phase, stats in baseline.items():
            print(f"  {phase:25s}: {stats['mean_s']*1000:8.1f} ms  (std: {stats['std_s']*1000:.1f} ms)")

        # Breakdown as percentages
        total = baseline.get("epoch_total", {}).get("mean_s", 1.0)
        print(f"\n  {'BREAKDOWN':25s}:")
        for phase in ["rollout_total", "ppo_update", "bookkeeping"]:
            if phase in baseline:
                pct = baseline[phase]["mean_s"] / total * 100
                print(f"    {phase:23s}: {pct:5.1f}%")

        # Phase 2+3: Profiler
        if not args.skip_profiler:
            run_profiler(loop, steps=args.steps_per_epoch, output_dir=args.output_dir)

        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)

        if baseline.get("rollout_total", {}).get("mean_s", 0) > 0.6 * total:
            print("  [!] Rollout dominates (>60%). Likely CPU-bound on env stepping.")
            print("      -> Profile env.step() separately")
            print("      -> Consider async env stepping or more vectorization")
        elif baseline.get("ppo_update", {}).get("mean_s", 0) > 0.6 * total:
            print("  [!] PPO update dominates (>60%). Likely GPU-bound.")
            print("      -> Check CUDA trace for kernel efficiency")
            print("      -> Consider torch.compile, larger batch_size, or AMP")
        else:
            print("  Time is distributed across phases — check Chrome trace for details.")

    finally:
        cleanup_distributed(dist_ctx)


if __name__ == "__main__":
    main()
