# AMP Mixed Precision & GAE Vectorization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in AMP (Automatic Mixed Precision) support across the PPO and SL training paths, with GradScaler state persisted in checkpoints; separately, vectorize the per-env GAE loop for split-merge mode.

**Architecture:** AMP is controlled by a `use_amp: bool` flag passed through `KataGoPPOParams` and `SLConfig`. When enabled, forward+loss runs inside `torch.cuda.amp.autocast()`, backward uses `GradScaler`, and gradient clipping calls `scaler.unscale_()` first. GradScaler state is saved/loaded alongside optimizer state in checkpoints. The GAE vectorization pads per-env sequences into a (T_max, N_env) tensor for a single `compute_gae` call.

**Tech Stack:** PyTorch `torch.cuda.amp` (autocast + GradScaler), existing checkpoint module

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `keisei/training/katago_ppo.py` | Add `use_amp` to params, wrap forward+loss in autocast, use GradScaler for backward+step |
| Modify | `keisei/sl/trainer.py` | Add `use_amp` to SLConfig, wrap train_epoch loop with autocast+GradScaler |
| Modify | `keisei/training/checkpoint.py` | Save/load `grad_scaler_state_dict` when present |
| Modify | `keisei/training/katago_loop.py` | Pass `use_amp` through to PPO, pass scaler to checkpoint calls |
| Modify | `keisei/training/gae.py` | Add `compute_gae_padded()` for batched per-env GAE |
| Create | `tests/test_amp.py` | All AMP tests: PPO update, SL epoch, checkpoint round-trip, CPU fallback |
| Create | `tests/test_gae_padded.py` | Tests for padded GAE vectorization |

---

## Task 1: GradScaler state in checkpoints

**Files:**
- Modify: `keisei/training/checkpoint.py:42-69` (save_checkpoint)
- Modify: `keisei/training/checkpoint.py:72-122` (load_checkpoint)
- Create: `tests/test_amp.py`

- [ ] **Step 1: Write failing test for scaler save/load round-trip**

```python
# tests/test_amp.py
"""Tests for AMP mixed precision support."""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path

from keisei.training.checkpoint import save_checkpoint, load_checkpoint


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TestGradScalerCheckpoint:
    def test_scaler_state_round_trip(self, tmp_path: Path) -> None:
        """GradScaler state survives save → load cycle."""
        model = _TinyModel()
        optimizer = torch.optim.Adam(model.parameters())
        scaler = torch.cuda.amp.GradScaler()

        # Simulate a few steps so scaler has non-default state
        scaler._scale = torch.tensor(32768.0)
        scaler._growth_tracker = torch.tensor(5)

        save_checkpoint(
            tmp_path / "ckpt.pt", model, optimizer,
            epoch=3, step=100, grad_scaler=scaler,
        )

        model2 = _TinyModel()
        optimizer2 = torch.optim.Adam(model2.parameters())
        scaler2 = torch.cuda.amp.GradScaler()

        load_checkpoint(
            tmp_path / "ckpt.pt", model2, optimizer2, grad_scaler=scaler2,
        )

        assert scaler2.get_scale() == 32768.0
        assert scaler2._growth_tracker.item() == 5

    def test_load_checkpoint_without_scaler_state(self, tmp_path: Path) -> None:
        """Old checkpoints without scaler state load without error."""
        model = _TinyModel()
        optimizer = torch.optim.Adam(model.parameters())

        # Save without scaler
        save_checkpoint(
            tmp_path / "ckpt.pt", model, optimizer, epoch=1, step=0,
        )

        model2 = _TinyModel()
        optimizer2 = torch.optim.Adam(model2.parameters())
        scaler2 = torch.cuda.amp.GradScaler()
        original_scale = scaler2.get_scale()

        # Load with scaler — should not crash, scaler keeps defaults
        load_checkpoint(
            tmp_path / "ckpt.pt", model2, optimizer2, grad_scaler=scaler2,
        )

        assert scaler2.get_scale() == original_scale
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_amp.py::TestGradScalerCheckpoint -v`
Expected: FAIL — `save_checkpoint() got an unexpected keyword argument 'grad_scaler'`

- [ ] **Step 3: Add grad_scaler parameter to save_checkpoint**

In `keisei/training/checkpoint.py`, modify `save_checkpoint` signature and body:

```python
def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    architecture: str | None = None,
    scheduler: Any | None = None,
    grad_scaler: Any | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "rng_states": {
            "python": random.getstate(),
            "numpy": _numpy_rng_to_safe(np.random.get_state()),
            "torch_cpu": torch.random.get_rng_state(),
        },
    }
    if torch.cuda.is_available():
        data["rng_states"]["torch_cuda"] = torch.cuda.get_rng_state_all()
    if architecture is not None:
        data["architecture"] = architecture
    if scheduler is not None:
        data["scheduler_state_dict"] = scheduler.state_dict()
    if grad_scaler is not None:
        data["grad_scaler_state_dict"] = grad_scaler.state_dict()
    torch.save(data, path)
```

- [ ] **Step 4: Add grad_scaler parameter to load_checkpoint**

In `keisei/training/checkpoint.py`, modify `load_checkpoint` signature and add after the scheduler restore block:

```python
def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_architecture: str | None = None,
    scheduler: Any | None = None,
    grad_scaler: Any | None = None,
) -> dict[str, Any]:
```

Add after the scheduler restore block (after line 107):

```python
    # Restore GradScaler state if present in checkpoint and caller provided one.
    if grad_scaler is not None and "grad_scaler_state_dict" in checkpoint:
        grad_scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_amp.py::TestGradScalerCheckpoint -v`
Expected: PASS (both tests)

- [ ] **Step 6: Commit**

```bash
git add keisei/training/checkpoint.py tests/test_amp.py
git commit -m "feat: persist GradScaler state in checkpoints for AMP support"
```

---

## Task 2: AMP in PPO update loop

**Files:**
- Modify: `keisei/training/katago_ppo.py:37-49` (KataGoPPOParams)
- Modify: `keisei/training/katago_ppo.py:273-438` (update method)
- Modify: `tests/test_amp.py`

- [ ] **Step 1: Write failing test for PPO AMP update**

Append to `tests/test_amp.py`:

```python
import pytest
from unittest.mock import MagicMock
from keisei.training.katago_ppo import KataGoPPOParams, KataGoPPOAlgorithm, KataGoRolloutBuffer
from keisei.training.models.mlp import MLPModel, MLPParams


def _make_ppo(use_amp: bool = False) -> KataGoPPOAlgorithm:
    """Create a minimal PPO algorithm with a tiny MLP model."""
    params = KataGoPPOParams(use_amp=use_amp, batch_size=4, epochs_per_batch=1)
    model = MLPModel(MLPParams(hidden_size=32))
    return KataGoPPOAlgorithm(model, params)


def _fill_buffer(ppo: KataGoPPOAlgorithm, num_envs: int = 2, steps: int = 4) -> KataGoRolloutBuffer:
    """Fill a rollout buffer with random data."""
    obs_shape = (50, 9, 9)
    action_space = 81 * 139
    buf = KataGoRolloutBuffer(num_envs, obs_shape, action_space)

    for _ in range(steps):
        obs = torch.randn(num_envs, *obs_shape)
        actions = torch.randint(0, action_space, (num_envs,))
        log_probs = torch.randn(num_envs)
        values = torch.randn(num_envs)
        rewards = torch.randn(num_envs)
        dones = torch.zeros(num_envs)
        legal_masks = torch.ones(num_envs, action_space, dtype=torch.bool)
        value_cats = torch.randint(0, 3, (num_envs,))
        score_targets = torch.randn(num_envs).clamp(-1.5, 1.5)
        buf.store(obs, actions, log_probs, values, rewards, dones, legal_masks,
                  value_categories=value_cats, score_targets=score_targets)

    return buf


class TestPPOAmp:
    def test_update_with_amp_produces_finite_loss(self) -> None:
        """PPO update with use_amp=True runs without error and produces finite metrics."""
        ppo = _make_ppo(use_amp=True)
        buf = _fill_buffer(ppo)
        next_values = torch.randn(2)

        metrics = ppo.update(buf, next_values)

        assert all(
            torch.isfinite(torch.tensor(v)) for v in metrics.values() if isinstance(v, float)
        ), f"Non-finite metrics: {metrics}"

    def test_update_without_amp_still_works(self) -> None:
        """use_amp=False (default) doesn't break anything."""
        ppo = _make_ppo(use_amp=False)
        buf = _fill_buffer(ppo)
        next_values = torch.randn(2)

        metrics = ppo.update(buf, next_values)
        assert "policy_loss" in metrics

    def test_amp_on_cpu_uses_no_op_autocast(self) -> None:
        """AMP on CPU should not crash — autocast('cpu') is a valid no-op."""
        ppo = _make_ppo(use_amp=True)
        # Model is on CPU by default
        buf = _fill_buffer(ppo)
        next_values = torch.randn(2)

        metrics = ppo.update(buf, next_values)
        assert "policy_loss" in metrics
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_amp.py::TestPPOAmp -v`
Expected: FAIL — `KataGoPPOParams.__init__() got an unexpected keyword argument 'use_amp'`

- [ ] **Step 3: Add use_amp to KataGoPPOParams**

In `keisei/training/katago_ppo.py`, add field to `KataGoPPOParams`:

```python
@dataclass(frozen=True)
class KataGoPPOParams:
    learning_rate: float = 2e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    epochs_per_batch: int = 4
    batch_size: int = 256
    lambda_policy: float = 1.0
    lambda_value: float = 1.5
    lambda_score: float = 0.02
    lambda_entropy: float = 0.01
    score_normalization: float = SCORE_NORMALIZATION
    grad_clip: float = 1.0
    use_amp: bool = False
```

- [ ] **Step 4: Initialize GradScaler in KataGoPPOAlgorithm.__init__**

In `keisei/training/katago_ppo.py`, find the `__init__` of `KataGoPPOAlgorithm` and add scaler initialization. Add `from torch.cuda.amp import autocast, GradScaler` to imports at top of file.

In `__init__`, after `self.optimizer = ...` add:

```python
        self.scaler = GradScaler(enabled=params.use_amp)
```

- [ ] **Step 5: Wrap update() forward+loss in autocast, use scaler for backward+step**

In `keisei/training/katago_ppo.py`, modify the mini-batch loop inside `update()`. Replace lines 351-431 (from `output = self.forward_model(batch_obs)` through `self.optimizer.step()`) with:

```python
                device = next(self.model.parameters()).device
                amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                autocast_device = "cuda" if device.type == "cuda" else "cpu"

                with autocast(device_type=autocast_device, dtype=amp_dtype, enabled=self.params.use_amp):
                    output = self.forward_model(batch_obs)

                    # Policy loss (clipped surrogate)
                    flat_logits = output.policy_logits.reshape(batch_obs.shape[0], -1)

                    if flat_logits.isnan().any():
                        raise RuntimeError("NaN in raw policy logits from model forward pass")

                    if (batch_legal_masks.sum(dim=-1) == 0).any():
                        raise RuntimeError(
                            "Batch contains samples with zero legal actions in update(). "
                            "Check that terminal-state masks are not stored in the buffer."
                        )

                    masked_logits = flat_logits.masked_fill(~batch_legal_masks, float("-inf"))
                    log_probs_all = F.log_softmax(masked_logits, dim=-1)
                    new_log_probs = log_probs_all.gather(
                        1, batch_actions.unsqueeze(1)
                    ).squeeze(1)

                    ratio = (new_log_probs - batch_old_log_probs).exp()
                    clip = self.params.clip_epsilon
                    surr1 = ratio * batch_advantages
                    surr2 = ratio.clamp(1 - clip, 1 + clip) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    probs = F.softmax(masked_logits, dim=-1)
                    safe_log_probs = log_probs_all.masked_fill(~batch_legal_masks, 0.0)
                    entropy = -(probs * safe_log_probs).sum(dim=-1).mean()

                    if value_adapter is not None:
                        value_score_loss = value_adapter.compute_value_loss(
                            output.value_logits,
                            returns=None,
                            value_cats=batch_value_cats,
                            score_targets=batch_score_targets,
                            score_pred=output.score_lead,
                        )
                        value_loss = value_score_loss
                        score_loss = torch.tensor(0.0, device=batch_obs.device)
                    else:
                        has_valid_value_targets = (batch_value_cats >= 0).any()
                        if has_valid_value_targets:
                            value_loss = F.cross_entropy(
                                output.value_logits, batch_value_cats, ignore_index=-1
                            )
                        else:
                            value_loss = output.value_logits.sum() * 0.0

                        score_loss = F.mse_loss(
                            output.score_lead.squeeze(-1), batch_score_targets,
                        )

                        value_score_loss = (
                            self.params.lambda_value * value_loss
                            + self.params.lambda_score * score_loss
                        )

                    loss = (
                        self.params.lambda_policy * policy_loss
                        + value_score_loss
                        - self.current_entropy_coeff * entropy
                    )

                # Backward + step with scaler (outside autocast)
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.params.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
```

Note: the `device = next(...)` line already exists at line 286 — move the `amp_dtype` and `autocast_device` computation to just before the mini-batch loop (after line 335) to avoid recomputing each iteration:

```python
        amp_dtype = torch.bfloat16 if (self.params.use_amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        autocast_device = "cuda" if device.type == "cuda" else "cpu"
```

Then inside the mini-batch loop, the `with autocast(...)` uses those pre-computed values.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_amp.py::TestPPOAmp -v`
Expected: PASS (all 3 tests)

- [ ] **Step 7: Run existing PPO tests to check for regressions**

Run: `uv run pytest tests/test_katago_ppo.py -v`
Expected: All existing tests PASS (default `use_amp=False`)

- [ ] **Step 8: Commit**

```bash
git add keisei/training/katago_ppo.py tests/test_amp.py
git commit -m "feat: add AMP mixed precision to PPO update loop"
```

---

## Task 3: AMP in SL trainer

**Files:**
- Modify: `keisei/sl/trainer.py:18-28` (SLConfig)
- Modify: `keisei/sl/trainer.py:33-122` (SLTrainer)
- Modify: `tests/test_amp.py`

- [ ] **Step 1: Write failing test for SL AMP training**

Append to `tests/test_amp.py`:

```python
from keisei.sl.trainer import SLTrainer, SLConfig
from keisei.training.models.mlp import MLPModel, MLPParams


class TestSLAmp:
    def test_sl_epoch_with_amp(self, tmp_path: Path) -> None:
        """SL train_epoch with use_amp=True completes without error."""
        # Create minimal shard data
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        obs = torch.randn(8, 50, 9, 9)
        policy = torch.randint(0, 81 * 139, (8,))
        value = torch.randint(0, 3, (8,))
        score = torch.randn(8).clamp(-1.5, 1.5)
        torch.save({"observation": obs, "policy_target": policy,
                     "value_target": value, "score_target": score},
                    shard_dir / "shard_000.pt")

        model = MLPModel(MLPParams(hidden_size=32))
        config = SLConfig(data_dir=str(shard_dir), batch_size=4, use_amp=True)
        trainer = SLTrainer(model, config)
        metrics = trainer.train_epoch()

        assert all(
            torch.isfinite(torch.tensor(v)) for v in metrics.values() if isinstance(v, float)
        )

    def test_sl_epoch_without_amp(self, tmp_path: Path) -> None:
        """Default use_amp=False still works."""
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        obs = torch.randn(8, 50, 9, 9)
        policy = torch.randint(0, 81 * 139, (8,))
        value = torch.randint(0, 3, (8,))
        score = torch.randn(8).clamp(-1.5, 1.5)
        torch.save({"observation": obs, "policy_target": policy,
                     "value_target": value, "score_target": score},
                    shard_dir / "shard_000.pt")

        model = MLPModel(MLPParams(hidden_size=32))
        config = SLConfig(data_dir=str(shard_dir), batch_size=4)
        trainer = SLTrainer(model, config)
        metrics = trainer.train_epoch()

        assert "policy_loss" in metrics
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_amp.py::TestSLAmp -v`
Expected: FAIL — `SLConfig.__init__() got an unexpected keyword argument 'use_amp'`

- [ ] **Step 3: Add use_amp to SLConfig and wrap SLTrainer with autocast+GradScaler**

In `keisei/sl/trainer.py`, add import and field:

```python
from torch.cuda.amp import autocast, GradScaler
```

Add `use_amp: bool = False` field to `SLConfig`:

```python
@dataclass
class SLConfig:
    data_dir: str
    batch_size: int = 4096
    learning_rate: float = 1e-3
    total_epochs: int = 30
    num_workers: int = 0
    lambda_policy: float = 1.0
    lambda_value: float = 1.5
    lambda_score: float = 0.02
    grad_clip: float = 0.5
    use_amp: bool = False
```

In `SLTrainer.__init__`, after `self.optimizer = ...` add:

```python
        self.scaler = GradScaler(enabled=config.use_amp)
```

Replace the training loop body in `train_epoch()` (lines 78-97) with:

```python
            output = self.model(obs)
```

becomes:

```python
            amp_dtype = torch.bfloat16 if (self.config.use_amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
            autocast_device = "cuda" if self.device.type == "cuda" else "cpu"

            with autocast(device_type=autocast_device, dtype=amp_dtype, enabled=self.config.use_amp):
                output = self.model(obs)

                policy_loss = F.cross_entropy(
                    output.policy_logits.reshape(obs.shape[0], -1), policy_targets
                )
                value_loss = F.cross_entropy(output.value_logits, value_targets)
                score_loss = F.mse_loss(output.score_lead.squeeze(-1), score_targets)

                loss = (
                    self.config.lambda_policy * policy_loss
                    + self.config.lambda_value * value_loss
                    + self.config.lambda_score * score_loss
                )

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
```

Remove the old TODO comment at lines 39-42 (it's now implemented).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_amp.py::TestSLAmp -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add keisei/sl/trainer.py tests/test_amp.py
git commit -m "feat: add AMP mixed precision to SL trainer"
```

---

## Task 4: Wire AMP through the training loop and checkpoints

**Files:**
- Modify: `keisei/training/katago_loop.py:327-336` (load_checkpoint call)
- Modify: `keisei/training/katago_loop.py:590-595` (save_checkpoint call)
- Modify: `keisei/config.py:17-25` (TrainingConfig)
- Modify: `keisei/config.py:108-117` (load_config)

- [ ] **Step 1: Add use_amp to TrainingConfig and config loading**

In `keisei/config.py`, add field to `TrainingConfig`:

```python
@dataclass(frozen=True)
class TrainingConfig:
    num_games: int
    max_ply: int
    algorithm: str
    checkpoint_interval: int
    checkpoint_dir: str
    algorithm_params: dict[str, object]
    use_amp: bool = False
```

In `load_config()`, after `algorithm_params = ...` (line 108), add:

```python
    use_amp = bool(t.get("use_amp", False))
```

And add it to the `TrainingConfig(...)` constructor:

```python
    training = TrainingConfig(
        num_games=num_games,
        max_ply=max_ply,
        algorithm=algorithm,
        checkpoint_interval=checkpoint_interval,
        checkpoint_dir=checkpoint_dir,
        algorithm_params=algorithm_params,
        use_amp=use_amp,
    )
```

- [ ] **Step 2: Pass use_amp from config to PPO params in KataGoTrainingLoop**

Find where `KataGoPPOParams` is constructed in `katago_loop.py` and add `use_amp=self.config.training.use_amp`. This is done via `algorithm_params` dict — add `use_amp` to the dict before passing to the params constructor, or wire it directly. Check the actual construction site first.

Run: `uv run python -c "from keisei.training.katago_loop import KataGoTrainingLoop; help(KataGoTrainingLoop.__init__)"` to trace how params are built.

The `algorithm_params` dict from config is unpacked into `KataGoPPOParams(**algorithm_params)`. Since we added `use_amp` to `KataGoPPOParams`, the user can set `use_amp = true` in `[training.algorithm_params]` in their TOML config and it flows through automatically. Alternatively, also merge `config.training.use_amp` into the params dict as a top-level convenience. Check the actual construction in `katago_loop.py`.

- [ ] **Step 3: Pass scaler to save_checkpoint and load_checkpoint calls**

In `keisei/training/katago_loop.py`, at the `save_checkpoint` call (around line 590), add `grad_scaler=self.ppo.scaler`:

```python
                    save_checkpoint(
                        ckpt_path, self._base_model, self.ppo.optimizer,
                        epoch_i + 1, self.global_step,
                        architecture=self.config.model.architecture,
                        scheduler=self.lr_scheduler,
                        grad_scaler=self.ppo.scaler,
                    )
```

At the `load_checkpoint` call (around line 327), add `grad_scaler=self.ppo.scaler`:

```python
                meta = load_checkpoint(
                    checkpoint_path,
                    self._base_model,
                    self.ppo.optimizer,
                    expected_architecture=self.config.model.architecture,
                    scheduler=self.lr_scheduler,
                    grad_scaler=self.ppo.scaler,
                )
```

- [ ] **Step 4: Run full test suite to verify no regressions**

Run: `uv run pytest tests/ -v --timeout=60`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/config.py keisei/training/katago_loop.py
git commit -m "feat: wire AMP config through training loop and checkpoints"
```

---

## Task 5: Padded GAE vectorization for split-merge mode

**Files:**
- Modify: `keisei/training/gae.py`
- Create: `tests/test_gae_padded.py`
- Modify: `keisei/training/katago_ppo.py:301-315` (per-env GAE path)

- [ ] **Step 1: Write failing test for compute_gae_padded**

```python
# tests/test_gae_padded.py
"""Tests for padded/batched GAE computation."""

from __future__ import annotations

import torch
from keisei.training.gae import compute_gae, compute_gae_padded


class TestComputeGaePadded:
    def test_padded_matches_per_env(self) -> None:
        """Padded batched GAE must match per-env sequential GAE."""
        torch.manual_seed(42)
        gamma, lam = 0.99, 0.95
        # 3 envs with different lengths
        lengths = [5, 3, 7]
        max_T = max(lengths)
        N = len(lengths)

        rewards_list = [torch.randn(L) for L in lengths]
        values_list = [torch.randn(L) for L in lengths]
        dones_list = [torch.zeros(L) for L in lengths]
        next_values = torch.randn(N)

        # Per-env reference
        ref_advantages = []
        for i, L in enumerate(lengths):
            adv = compute_gae(rewards_list[i], values_list[i], dones_list[i],
                              next_values[i], gamma, lam)
            ref_advantages.append(adv)

        # Padded batched
        rewards_pad = torch.zeros(max_T, N)
        values_pad = torch.zeros(max_T, N)
        dones_pad = torch.ones(max_T, N)  # pad with done=1 to zero out GAE
        length_tensor = torch.tensor(lengths)

        for i, L in enumerate(lengths):
            rewards_pad[:L, i] = rewards_list[i]
            values_pad[:L, i] = values_list[i]
            dones_pad[:L, i] = dones_list[i]

        padded_adv = compute_gae_padded(
            rewards_pad, values_pad, dones_pad, next_values, length_tensor,
            gamma, lam,
        )

        for i, L in enumerate(lengths):
            torch.testing.assert_close(
                padded_adv[:L, i], ref_advantages[i],
                atol=1e-6, rtol=1e-5,
            )

    def test_equal_lengths_matches_standard(self) -> None:
        """When all envs have equal length, padded == standard 2D GAE."""
        torch.manual_seed(123)
        T, N = 8, 4
        gamma, lam = 0.99, 0.95
        rewards = torch.randn(T, N)
        values = torch.randn(T, N)
        dones = torch.zeros(T, N)
        next_values = torch.randn(N)
        lengths = torch.full((N,), T)

        ref = compute_gae(rewards, values, dones, next_values, gamma, lam)
        padded = compute_gae_padded(rewards, values, dones, next_values, lengths, gamma, lam)

        torch.testing.assert_close(padded, ref, atol=1e-6, rtol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_gae_padded.py -v`
Expected: FAIL — `cannot import name 'compute_gae_padded' from 'keisei.training.gae'`

- [ ] **Step 3: Implement compute_gae_padded**

In `keisei/training/gae.py`, add after the existing `compute_gae` function:

```python
def compute_gae_padded(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_values: torch.Tensor,
    lengths: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """Compute GAE for multiple envs with variable-length episodes via padding.

    All inputs are padded to (T_max, N). Positions beyond each env's length
    must have dones=1 so that GAE propagation is zeroed out for padding.

    Args:
        rewards: (T_max, N) padded rewards
        values: (T_max, N) padded value estimates
        dones: (T_max, N) termination flags — padding positions MUST be 1.0
        next_values: (N,) bootstrap values per env
        lengths: (N,) actual sequence length per env
        gamma: discount factor
        lam: GAE lambda

    Returns:
        (T_max, N) advantages — only [:lengths[i], i] are meaningful per env
    """
    T_max, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(next_values)

    for t in reversed(range(T_max)):
        if t == T_max - 1:
            next_val = next_values
        else:
            next_val = values[t + 1]

        not_done = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_val * not_done - values[t]
        last_gae = delta + gamma * lam * not_done * last_gae
        advantages[t] = last_gae

    return advantages
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_gae_padded.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Wire padded GAE into PPO per-env path**

In `keisei/training/katago_ppo.py`, replace the per-env GAE loop (lines 301-315) with the padded version:

```python
        elif "env_ids" in data:
            # Padded batched GAE for split-merge mode: pad per-env sequences
            # into a (T_max, N_env) tensor for a single vectorized call.
            from keisei.training.gae import compute_gae_padded

            env_ids = data["env_ids"]
            unique_envs = env_ids.unique()
            advantages = torch.zeros(total_samples)

            # Collect per-env data
            env_rewards = []
            env_values = []
            env_dones = []
            env_lengths = []
            env_masks = []

            for env_id in unique_envs:
                mask = env_ids == env_id
                env_rewards.append(data["rewards"][mask])
                env_values.append(data["values"][mask])
                env_dones.append(data["dones"][mask])
                env_lengths.append(mask.sum().item())
                env_masks.append(mask)

            max_T = max(env_lengths)
            N_env = len(unique_envs)

            # Pad into (T_max, N_env) tensors
            rewards_pad = torch.zeros(max_T, N_env)
            values_pad = torch.zeros(max_T, N_env)
            dones_pad = torch.ones(max_T, N_env)  # padding = done to zero GAE
            nv = torch.zeros(N_env)

            for i, L in enumerate(env_lengths):
                rewards_pad[:L, i] = env_rewards[i]
                values_pad[:L, i] = env_values[i]
                dones_pad[:L, i] = env_dones[i]
                nv[i] = next_values_cpu[unique_envs[i]]

            lengths_t = torch.tensor(env_lengths)
            padded_adv = compute_gae_padded(
                rewards_pad, values_pad, dones_pad, nv, lengths_t,
                gamma=self.params.gamma, lam=self.params.gae_lambda,
            )

            for i, L in enumerate(env_lengths):
                advantages[env_masks[i]] = padded_adv[:L, i]
```

- [ ] **Step 6: Run per-env GAE tests to verify no regressions**

Run: `uv run pytest tests/test_katago_ppo.py tests/test_gae_padded.py tests/test_pytorch_audit_gaps.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add keisei/training/gae.py keisei/training/katago_ppo.py tests/test_gae_padded.py
git commit -m "feat: vectorize per-env GAE with padded batching for split-merge mode"
```

---

## Task 6: AMP select_actions inference path

**Files:**
- Modify: `keisei/training/katago_ppo.py:234-271` (select_actions method)
- Modify: `tests/test_amp.py`

- [ ] **Step 1: Write failing test for AMP in select_actions**

Append to `tests/test_amp.py`:

```python
class TestSelectActionsAmp:
    def test_select_actions_with_amp(self) -> None:
        """select_actions with use_amp=True produces valid actions."""
        ppo = _make_ppo(use_amp=True)
        obs = torch.randn(2, 50, 9, 9)
        legal_masks = torch.ones(2, 81 * 139, dtype=torch.bool)
        dones = torch.zeros(2)

        actions, log_probs, values, value_cats, score_targets = ppo.select_actions(
            obs, legal_masks, dones,
        )

        assert actions.shape == (2,)
        assert log_probs.shape == (2,)
        assert values.shape == (2,)
        assert torch.isfinite(log_probs).all()
        assert torch.isfinite(values).all()
```

- [ ] **Step 2: Run test to verify current behavior**

Run: `uv run pytest tests/test_amp.py::TestSelectActionsAmp -v`
Expected: This may PASS already since select_actions runs under `@torch.no_grad()`. If it passes, we still want to add autocast for speed benefit during rollout collection.

- [ ] **Step 3: Add autocast to select_actions**

In `keisei/training/katago_ppo.py`, in the `select_actions` method, wrap the forward pass with autocast. Find the line `output = self.forward_model(obs)` (line 246) and wrap:

```python
        device = next(self.model.parameters()).device
        amp_dtype = torch.bfloat16 if (self.params.use_amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        autocast_device = "cuda" if device.type == "cuda" else "cpu"

        with autocast(device_type=autocast_device, dtype=amp_dtype, enabled=self.params.use_amp):
            output = self.forward_model(obs)
```

No GradScaler needed here — this is inference under `@torch.no_grad()`.

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_amp.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/katago_ppo.py tests/test_amp.py
git commit -m "feat: add autocast to PPO select_actions for faster rollout inference"
```

---

## Task 7: Final integration test and cleanup

**Files:**
- Modify: `tests/test_amp.py`

- [ ] **Step 1: Write integration test for AMP checkpoint round-trip with PPO**

Append to `tests/test_amp.py`:

```python
class TestAmpIntegration:
    def test_ppo_checkpoint_round_trip_with_amp(self, tmp_path: Path) -> None:
        """Full cycle: PPO update with AMP → save → load → update again."""
        ppo = _make_ppo(use_amp=True)
        buf = _fill_buffer(ppo)
        next_values = torch.randn(2)

        metrics1 = ppo.update(buf, next_values)
        assert "policy_loss" in metrics1

        # Save checkpoint
        save_checkpoint(
            tmp_path / "ckpt.pt", ppo.model, ppo.optimizer,
            epoch=1, step=10, grad_scaler=ppo.scaler,
        )

        # Create fresh PPO and load
        ppo2 = _make_ppo(use_amp=True)
        load_checkpoint(
            tmp_path / "ckpt.pt", ppo2.model, ppo2.optimizer,
            grad_scaler=ppo2.scaler,
        )

        # Verify scaler state transferred
        assert ppo2.scaler.get_scale() == ppo.scaler.get_scale()

        # Second update should work
        buf2 = _fill_buffer(ppo2)
        metrics2 = ppo2.update(buf2, next_values)
        assert "policy_loss" in metrics2
```

- [ ] **Step 2: Run all AMP tests**

Run: `uv run pytest tests/test_amp.py tests/test_gae_padded.py -v`
Expected: All PASS

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v --timeout=120`
Expected: All PASS, no regressions

- [ ] **Step 4: Commit**

```bash
git add tests/test_amp.py
git commit -m "test: add AMP integration test for checkpoint round-trip"
```
