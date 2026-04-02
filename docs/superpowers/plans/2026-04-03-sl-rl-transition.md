# SL→RL Transition: skip_optimizer + Orchestrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the SL→RL checkpoint handoff correct (skip stale optimizer state) and automated (orchestrator function that runs the full pipeline).

**Architecture:** Two changes that compose naturally. Task 1-3 add a `skip_optimizer` parameter to `load_checkpoint()` and wire it into `_check_resume()` with a `resume_mode` discriminator. Task 4-5 build a thin `sl_to_rl()` orchestrator function that calls `SLTrainer`, saves a checkpoint, then launches `KataGoTrainingLoop` — the first caller that actually wires the two halves together. The orchestrator uses `skip_optimizer=True` from Task 1-3.

**Tech Stack:** Python 3.13, PyTorch, pytest, uv

**Filigree issues:** `keisei-1a23105f48` (skip_optimizer), `keisei-0cd7dc8c70` (orchestrator)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `keisei/training/checkpoint.py` | Modify | Add `skip_optimizer` param to `load_checkpoint()` |
| `keisei/training/katago_loop.py` | Modify | Add `resume_mode` to `_check_resume()`, pass `skip_optimizer=True` for SL→RL |
| `keisei/training/transition.py` | Create | `sl_to_rl()` orchestrator function |
| `tests/test_checkpoint.py` | Modify | Test `skip_optimizer` behavior |
| `tests/test_transition.py` | Create | End-to-end SL→RL transition tests |

---

### Task 1: Add `skip_optimizer` parameter to `load_checkpoint()`

**Files:**
- Modify: `keisei/training/checkpoint.py:75-130`
- Test: `tests/test_checkpoint.py`

**Context:** Currently `load_checkpoint()` unconditionally loads optimizer state (line 97). When transitioning from SL to RL, the SL Adam momentum buffers fight RL gradients. We need a way to skip restoring optimizer state while still loading model weights, scheduler, scaler, and RNG state.

- [ ] **Step 1: Write failing test — skip_optimizer skips optimizer state**

Add to `tests/test_checkpoint.py`:

```python
def test_skip_optimizer_leaves_optimizer_fresh(tmp_path: Path, model: ResNetModel) -> None:
    """load_checkpoint(skip_optimizer=True) should NOT load optimizer state."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Run a training step to populate optimizer momentum buffers
    obs = torch.randn(1, 46, 9, 9)
    policy, value = model(obs)
    loss = policy.sum() + value.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Save checkpoint (now contains non-empty optimizer state)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, epoch=5, step=500)

    # Create fresh model and optimizer
    fresh_model = ResNetModel(ResNetParams(hidden_size=16, num_layers=1))
    fresh_optimizer = torch.optim.Adam(fresh_model.parameters(), lr=1e-3)

    # Verify fresh optimizer has no state
    assert len(fresh_optimizer.state) == 0

    # Load with skip_optimizer=True
    meta = load_checkpoint(path, fresh_model, fresh_optimizer, skip_optimizer=True)

    # Optimizer state should still be empty
    assert len(fresh_optimizer.state) == 0
    assert meta["epoch"] == 5
    assert meta["step"] == 500
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_checkpoint.py::test_skip_optimizer_leaves_optimizer_fresh -v`
Expected: FAIL — `load_checkpoint()` does not accept `skip_optimizer` keyword.

- [ ] **Step 3: Implement skip_optimizer in load_checkpoint**

In `keisei/training/checkpoint.py`, modify the `load_checkpoint` function signature and body:

```python
def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_architecture: str | None = None,
    scheduler: Any | None = None,
    grad_scaler: Any | None = None,
    skip_optimizer: bool = False,
) -> dict[str, Any]:
```

Replace lines 96-107 (the unconditional optimizer load + device migration) with:

```python
    model.load_state_dict(checkpoint["model_state_dict"])

    if not skip_optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Move optimizer state (Adam moment buffers) to match model device.
        # torch.load(map_location="cpu") puts all tensors on CPU, but the model
        # may already be on CUDA. Without this, optimizer.step() crashes with a
        # device mismatch error.
        device = next(model.parameters()).device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_checkpoint.py::test_skip_optimizer_leaves_optimizer_fresh -v`
Expected: PASS

- [ ] **Step 5: Write test — skip_optimizer still loads model weights**

Add to `tests/test_checkpoint.py`:

```python
def test_skip_optimizer_still_loads_model_weights(tmp_path: Path, model: ResNetModel) -> None:
    """skip_optimizer=True should still restore model weights correctly."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    obs = torch.randn(1, 46, 9, 9)
    with torch.no_grad():
        original_policy, original_value = model(obs)

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, epoch=1, step=100)

    # Corrupt model weights
    for p in model.parameters():
        p.data.add_(torch.randn_like(p))

    fresh_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    load_checkpoint(path, model, fresh_optimizer, skip_optimizer=True)

    with torch.no_grad():
        restored_policy, restored_value = model(obs)

    assert torch.allclose(original_policy, restored_policy, atol=1e-6)
    assert torch.allclose(original_value, restored_value, atol=1e-6)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_checkpoint.py::test_skip_optimizer_still_loads_model_weights -v`
Expected: PASS

- [ ] **Step 7: Run all existing checkpoint tests to confirm no regressions**

Run: `uv run pytest tests/test_checkpoint.py -v`
Expected: All tests PASS (the existing tests don't pass `skip_optimizer`, so they use the default `False` and behave exactly as before).

- [ ] **Step 8: Commit**

```bash
git add keisei/training/checkpoint.py tests/test_checkpoint.py
git commit -m "feat(checkpoint): add skip_optimizer parameter to load_checkpoint

When transitioning from SL to RL, Adam momentum buffers from supervised
gradients fight RL gradient signal. skip_optimizer=True loads model weights
but leaves the optimizer in its fresh state.

Refs: keisei-1a23105f48"
```

---

### Task 2: Wire `skip_optimizer` into `_check_resume()` with resume_mode

**Files:**
- Modify: `keisei/training/katago_loop.py:311-337`
- Modify: `keisei/training/katago_loop.py:154-155` (constructor signature)
- Test: `tests/test_katago_loop.py`

**Context:** `_check_resume()` currently calls `load_checkpoint()` unconditionally. We need it to pass `skip_optimizer=True` when resuming from an SL checkpoint (SL→RL transition) but `skip_optimizer=False` when resuming RL→RL (normal resume). We add a `resume_mode` parameter — `"rl"` (default, normal resume) or `"sl"` (SL→RL transition, skip optimizer).

- [ ] **Step 1: Read current test file to understand test patterns**

Read: `tests/test_katago_loop.py` lines 1-50 and 394-450 to understand the existing test fixtures and mocking patterns.

- [ ] **Step 2: Write failing test — resume_mode="sl" skips optimizer**

Add to `tests/test_katago_loop.py`. The exact fixture setup depends on what you find in Step 1 — the test needs a `KataGoTrainingLoop` instance with a mock VecEnv. The key assertion:

```python
def test_check_resume_sl_mode_skips_optimizer(self, tmp_path, ...):
    """When resume_mode='sl', _check_resume should call load_checkpoint with skip_optimizer=True."""
    from unittest.mock import patch

    # Set up a training state record pointing to a valid checkpoint
    # (use the existing fixture pattern from the test file)
    
    with patch("keisei.training.katago_loop.load_checkpoint") as mock_load:
        mock_load.return_value = {"epoch": 5, "step": 500}
        loop = KataGoTrainingLoop(config, vecenv=mock_vecenv, resume_mode="sl")
        
        # Verify load_checkpoint was called with skip_optimizer=True
        mock_load.assert_called_once()
        _, kwargs = mock_load.call_args
        assert kwargs.get("skip_optimizer") is True
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_katago_loop.py::test_check_resume_sl_mode_skips_optimizer -v`
Expected: FAIL — `KataGoTrainingLoop` does not accept `resume_mode`.

- [ ] **Step 4: Add resume_mode parameter to KataGoTrainingLoop**

In `keisei/training/katago_loop.py`, modify the constructor:

```python
class KataGoTrainingLoop:
    def __init__(self, config: AppConfig, vecenv: Any = None, resume_mode: str = "rl") -> None:
        if resume_mode not in ("rl", "sl"):
            raise ValueError(f"resume_mode must be 'rl' or 'sl', got '{resume_mode}'")
        self._resume_mode = resume_mode
        self.config = config
```

Then modify `_check_resume()` to pass the flag:

```python
    def _check_resume(self) -> None:
        # NOTE: When resuming from an SL checkpoint into RL training, the SL
        # optimizer state is intentionally discarded. KataGoTrainingLoop creates
        # a fresh Adam optimizer. The SL optimizer has momentum from supervised
        # gradients that would fight the RL gradient signal. The RL warmup
        # elevated entropy (Plan D Task 3) compensates for the overconfident
        # SL policy by encouraging exploration in early RL epochs.
        state = read_training_state(self.db_path)
        if state is not None and state.get("checkpoint_path"):
            checkpoint_path = Path(state["checkpoint_path"])
            if checkpoint_path.exists():
                skip_opt = self._resume_mode == "sl"
                logger.warning(
                    "Resuming from checkpoint: %s (epoch %d, skip_optimizer=%s)",
                    checkpoint_path,
                    state["current_epoch"],
                    skip_opt,
                )
                meta = load_checkpoint(
                    checkpoint_path,
                    self._base_model,
                    self.ppo.optimizer,
                    expected_architecture=self.config.model.architecture,
                    scheduler=self.lr_scheduler,
                    grad_scaler=self.ppo.scaler,
                    skip_optimizer=skip_opt,
                )
                self.epoch = meta["epoch"]
                self.global_step = meta["step"]
                return
        # ... rest of method unchanged
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_katago_loop.py::test_check_resume_sl_mode_skips_optimizer -v`
Expected: PASS

- [ ] **Step 6: Run full katago_loop test suite for regressions**

Run: `uv run pytest tests/test_katago_loop.py -v`
Expected: All PASS — existing tests don't pass `resume_mode`, so they use default `"rl"` which preserves existing behavior.

- [ ] **Step 7: Commit**

```bash
git add keisei/training/katago_loop.py tests/test_katago_loop.py
git commit -m "feat(katago_loop): add resume_mode parameter for SL→RL transition

resume_mode='sl' passes skip_optimizer=True to load_checkpoint, preventing
SL Adam momentum from contaminating RL training. Default 'rl' preserves
existing behavior for RL→RL resume.

Refs: keisei-1a23105f48"
```

---

### Task 3: End-to-end test — SL checkpoint loaded into RL has fresh optimizer

**Files:**
- Test: `tests/test_checkpoint.py`

**Context:** This is the "real" integration test from the issue: save an actual SL checkpoint with populated optimizer state, load it with `skip_optimizer=True` into a KataGo model, and verify the RL optimizer is clean. Uses the SE-ResNet (production architecture).

- [ ] **Step 1: Write the end-to-end test**

Add a new test class to `tests/test_checkpoint.py`:

```python
class TestSLtoRLCheckpointTransition:
    """End-to-end: SL checkpoint loaded into RL training has fresh optimizer."""

    @pytest.fixture
    def se_model(self):
        from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams

        params = SEResNetParams(
            num_blocks=2, channels=32, se_reduction=8,
            global_pool_channels=16, policy_channels=8,
            value_fc_size=32, score_fc_size=16, obs_channels=50,
        )
        return SEResNetModel(params)

    def test_sl_checkpoint_loaded_for_rl_has_fresh_optimizer(
        self, tmp_path: Path, se_model
    ) -> None:
        """Save SL checkpoint with warm optimizer, load for RL, verify optimizer is fresh."""
        from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams

        sl_optimizer = torch.optim.Adam(se_model.parameters(), lr=1e-3)

        # Train a few steps to populate Adam momentum buffers
        for _ in range(3):
            obs = torch.randn(2, 50, 9, 9)
            output = se_model(obs)
            loss = output.policy_logits.sum() + output.value_logits.sum()
            loss.backward()
            sl_optimizer.step()
            sl_optimizer.zero_grad()

        # Confirm SL optimizer has populated state
        assert len(sl_optimizer.state) > 0

        # Save SL checkpoint (includes optimizer state)
        ckpt_path = tmp_path / "sl_checkpoint.pt"
        save_checkpoint(
            ckpt_path, se_model, sl_optimizer, epoch=30, step=9000,
            architecture="se_resnet",
        )

        # Create fresh RL model and optimizer (simulating what KataGoTrainingLoop does)
        rl_params = SEResNetParams(
            num_blocks=2, channels=32, se_reduction=8,
            global_pool_channels=16, policy_channels=8,
            value_fc_size=32, score_fc_size=16, obs_channels=50,
        )
        rl_model = SEResNetModel(rl_params)
        rl_optimizer = torch.optim.Adam(rl_model.parameters(), lr=3e-4)

        # Load with skip_optimizer=True (SL→RL transition)
        meta = load_checkpoint(
            ckpt_path, rl_model, rl_optimizer,
            expected_architecture="se_resnet",
            skip_optimizer=True,
        )

        # Model weights should be loaded
        se_model.eval()
        rl_model.eval()
        test_obs = torch.randn(1, 50, 9, 9)
        with torch.no_grad():
            sl_out = se_model(test_obs)
            rl_out = rl_model(test_obs)
        assert torch.allclose(sl_out.policy_logits, rl_out.policy_logits, atol=1e-6)

        # Optimizer should be EMPTY (fresh) — no momentum from SL training
        assert len(rl_optimizer.state) == 0

        # Training metadata should be carried over
        assert meta["epoch"] == 30
        assert meta["step"] == 9000
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_checkpoint.py::TestSLtoRLCheckpointTransition -v`
Expected: PASS (implementation from Task 1 already supports this).

- [ ] **Step 3: Commit**

```bash
git add tests/test_checkpoint.py
git commit -m "test(checkpoint): add SL→RL transition end-to-end test

Verifies that loading an SL checkpoint with skip_optimizer=True produces
a model with correct weights but a fresh optimizer — no SL momentum
contamination.

Refs: keisei-1a23105f48"
```

---

### Task 4: Build the SL→RL transition orchestrator

**Files:**
- Create: `keisei/training/transition.py`
- Test: `tests/test_transition.py`

**Context:** There is no code that wires `SLTrainer` to `KataGoTrainingLoop`. The orchestrator is a single function `sl_to_rl()` that: (1) runs SL training to completion, (2) saves the SL checkpoint, (3) writes training state to the DB so `KataGoTrainingLoop._check_resume()` finds it, (4) creates and returns a `KataGoTrainingLoop` with `resume_mode="sl"`. The caller then calls `loop.run()`.

The orchestrator is intentionally thin — it's glue code, not a framework. It coordinates existing components.

- [ ] **Step 1: Write failing test — sl_to_rl produces a training loop with SL-trained weights**

Create `tests/test_transition.py`:

```python
"""Tests for the SL→RL transition orchestrator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from keisei.training.checkpoint import load_checkpoint, save_checkpoint


class TestSLToRL:
    """Verify sl_to_rl() orchestrates the SL→RL handoff correctly."""

    @pytest.fixture
    def sl_data_dir(self, tmp_path: Path) -> Path:
        """Create a minimal SL data shard."""
        from keisei.sl.dataset import RECORD_SIZE
        from keisei.sl.prepare import write_shard

        data_dir = tmp_path / "sl_data"
        data_dir.mkdir()
        n = 16
        rng = np.random.default_rng(42)
        write_shard(
            data_dir / "shard_000.bin",
            rng.standard_normal((n, 50 * 81)).astype(np.float32),
            rng.integers(0, 11259, size=n).astype(np.int64),
            rng.integers(0, 3, size=n).astype(np.int64),
            rng.standard_normal(n).astype(np.float32),
        )
        return data_dir

    def test_sl_to_rl_returns_loop_with_trained_weights(
        self, tmp_path: Path, sl_data_dir: Path
    ) -> None:
        """sl_to_rl should run SL training, save checkpoint, and return a configured loop."""
        from keisei.training.transition import sl_to_rl

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        mock_vecenv = MagicMock()
        mock_vecenv.observation_channels = 50
        mock_vecenv.action_space_size = 11259

        loop = sl_to_rl(
            sl_data_dir=sl_data_dir,
            sl_epochs=2,
            sl_batch_size=8,
            checkpoint_dir=checkpoint_dir,
            rl_config_path=None,  # We'll build a minimal config inline
            architecture="se_resnet",
            model_params={
                "num_blocks": 2, "channels": 32, "se_reduction": 8,
                "global_pool_channels": 16, "policy_channels": 8,
                "value_fc_size": 32, "score_fc_size": 16, "obs_channels": 50,
            },
            vecenv=mock_vecenv,
            db_path=str(tmp_path / "test.db"),
        )

        # A checkpoint should have been saved
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        assert len(checkpoints) == 1

        # The loop should exist and have resume_mode="sl"
        assert loop._resume_mode == "sl"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_transition.py::TestSLToRL::test_sl_to_rl_returns_loop_with_trained_weights -v`
Expected: FAIL — `keisei.training.transition` does not exist.

- [ ] **Step 3: Implement sl_to_rl()**

Create `keisei/training/transition.py`:

```python
"""SL→RL transition orchestrator.

Wires SLTrainer output into KataGoTrainingLoop input — the glue code
that automates the most important seam in the training pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from keisei.config import (
    AppConfig,
    DisplayConfig,
    ModelConfig,
    TrainingConfig,
    load_config,
)
from keisei.db import init_db, write_training_state
from keisei.sl.trainer import SLConfig, SLTrainer
from keisei.training.checkpoint import save_checkpoint
from keisei.training.katago_loop import KataGoTrainingLoop
from keisei.training.model_registry import build_model

logger = logging.getLogger(__name__)


def sl_to_rl(
    *,
    sl_data_dir: Path,
    sl_epochs: int,
    sl_batch_size: int = 4096,
    checkpoint_dir: Path,
    rl_config_path: Path | None = None,
    architecture: str = "se_resnet",
    model_params: dict[str, Any] | None = None,
    vecenv: Any = None,
    db_path: str = "keisei.db",
    sl_learning_rate: float = 1e-3,
    sl_use_amp: bool = False,
) -> KataGoTrainingLoop:
    """Run SL training, save checkpoint, and return a configured RL training loop.

    This is the orchestrator that bridges supervised pre-training and
    reinforcement learning fine-tuning. It:

    1. Builds a model and runs SL training for ``sl_epochs`` epochs.
    2. Saves the SL checkpoint (model weights + metadata, optimizer state
       included in the file but will be skipped on RL load).
    3. Writes training state to the DB so ``KataGoTrainingLoop._check_resume()``
       finds the checkpoint.
    4. Returns a ``KataGoTrainingLoop`` with ``resume_mode="sl"`` — when the
       caller invokes ``loop.run()``, it loads model weights but discards the
       SL optimizer state.

    Returns:
        A configured ``KataGoTrainingLoop`` ready for ``loop.run()``.
    """
    # --- Phase 1: SL Training ---
    model = build_model(architecture, model_params or {})
    sl_config = SLConfig(
        data_dir=str(sl_data_dir),
        batch_size=sl_batch_size,
        learning_rate=sl_learning_rate,
        total_epochs=sl_epochs,
        use_amp=sl_use_amp,
    )
    trainer = SLTrainer(model, sl_config)

    logger.info("Starting SL training: %d epochs, batch_size=%d", sl_epochs, sl_batch_size)
    for epoch in range(sl_epochs):
        metrics = trainer.train_epoch()
        logger.info("SL epoch %d/%d complete: %s", epoch + 1, sl_epochs, metrics)

    # --- Phase 2: Save SL Checkpoint ---
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / "sl_final.pt"
    save_checkpoint(
        ckpt_path,
        model,
        trainer.optimizer,
        epoch=sl_epochs,
        step=0,
        architecture=architecture,
        scheduler=trainer.scheduler,
        grad_scaler=trainer.scaler,
    )
    logger.info("SL checkpoint saved: %s", ckpt_path)

    # --- Phase 3: Write DB state for _check_resume() ---
    init_db(db_path)
    write_training_state(
        db_path,
        {
            "config_json": "{}",
            "display_name": "SL→RL",
            "model_arch": architecture,
            "algorithm_name": "katago_ppo",
            "started_at": "",
            "current_epoch": sl_epochs,
            "current_step": 0,
            "checkpoint_path": str(ckpt_path),
        },
    )

    # --- Phase 4: Build RL Training Loop ---
    if rl_config_path is not None:
        rl_config = load_config(rl_config_path)
    else:
        rl_config = AppConfig(
            training=TrainingConfig(
                num_games=8,
                max_ply=500,
                algorithm="katago_ppo",
                checkpoint_interval=50,
                checkpoint_dir=str(checkpoint_dir),
                algorithm_params={},
                use_amp=sl_use_amp,
            ),
            display=DisplayConfig(moves_per_minute=30, db_path=db_path),
            model=ModelConfig(
                display_name="SL→RL",
                architecture=architecture,
                params=model_params or {},
            ),
        )

    loop = KataGoTrainingLoop(rl_config, vecenv=vecenv, resume_mode="sl")
    logger.info("RL training loop ready (resume_mode=sl)")
    return loop
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_transition.py::TestSLToRL::test_sl_to_rl_returns_loop_with_trained_weights -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/transition.py tests/test_transition.py
git commit -m "feat(transition): add sl_to_rl() orchestrator

Wires SLTrainer → checkpoint → KataGoTrainingLoop with resume_mode='sl'.
The first code path that automates the SL→RL handoff end-to-end.

Refs: keisei-0cd7dc8c70"
```

---

### Task 5: Integration test — sl_to_rl() produces working RL loop with correct state

**Files:**
- Test: `tests/test_transition.py`

**Context:** Task 4 tested that the orchestrator returns a loop. This task verifies the *state* is correct: the RL model has SL-trained weights (not random), and the RL optimizer is fresh (no SL momentum). This is the critical correctness property.

- [ ] **Step 1: Write the integration test**

Add to `tests/test_transition.py`:

```python
    def test_sl_to_rl_model_has_trained_weights_and_fresh_optimizer(
        self, tmp_path: Path, sl_data_dir: Path
    ) -> None:
        """After sl_to_rl(), RL model should have SL weights but fresh optimizer."""
        from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams
        from keisei.training.transition import sl_to_rl

        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        mock_vecenv = MagicMock()
        mock_vecenv.observation_channels = 50
        mock_vecenv.action_space_size = 11259

        model_params = {
            "num_blocks": 2, "channels": 32, "se_reduction": 8,
            "global_pool_channels": 16, "policy_channels": 8,
            "value_fc_size": 32, "score_fc_size": 16, "obs_channels": 50,
        }

        loop = sl_to_rl(
            sl_data_dir=sl_data_dir,
            sl_epochs=1,
            sl_batch_size=8,
            checkpoint_dir=checkpoint_dir,
            architecture="se_resnet",
            model_params=model_params,
            vecenv=mock_vecenv,
            db_path=str(tmp_path / "test.db"),
        )

        # The RL model should NOT have random weights — it should have SL-trained weights.
        # Verify by checking that a random model produces different outputs.
        random_model = SEResNetModel(SEResNetParams(**model_params))
        test_obs = torch.randn(1, 50, 9, 9)
        with torch.no_grad():
            rl_out = loop._base_model(test_obs)
            random_out = random_model(test_obs)

        # SL-trained weights should differ from random init
        assert not torch.allclose(
            rl_out.policy_logits, random_out.policy_logits, atol=1e-3
        ), "RL model appears to have random weights — SL training didn't transfer"

        # The RL optimizer should be fresh (no SL momentum buffers)
        assert len(loop.ppo.optimizer.state) == 0, (
            "RL optimizer has state — SL momentum was not skipped"
        )
```

- [ ] **Step 2: Run test**

Run: `uv run pytest tests/test_transition.py::TestSLToRL::test_sl_to_rl_model_has_trained_weights_and_fresh_optimizer -v`
Expected: PASS

- [ ] **Step 3: Run full test suite to confirm no regressions**

Run: `uv run pytest tests/test_checkpoint.py tests/test_transition.py tests/test_katago_loop.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_transition.py
git commit -m "test(transition): integration test for SL→RL state correctness

Verifies the critical property: after sl_to_rl(), the RL model has
SL-trained weights (not random) but the optimizer is fresh (no SL momentum).

Refs: keisei-0cd7dc8c70"
```

---

### Task 6: Close filigree issues

- [ ] **Step 1: Close both issues**

```bash
filigree close keisei-1a23105f48 --reason="skip_optimizer added to load_checkpoint, wired into _check_resume via resume_mode"
filigree close keisei-0cd7dc8c70 --reason="sl_to_rl() orchestrator implemented in keisei/training/transition.py"
```
