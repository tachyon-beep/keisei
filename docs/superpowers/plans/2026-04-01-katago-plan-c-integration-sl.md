# KataGo Plan C: Integration & Supervised Learning Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the Rust engine extensions (Plan A) with the Python model/PPO (Plan B) into a working end-to-end training loop, then add the supervised learning warmup pipeline.

**Architecture:** New `KataGoTrainingLoop` class, enhanced checkpoint system with architecture metadata, TOML config extensions, game parser interface with CSA implementation, memory-mapped SL dataset, and `SLTrainer`.

**Tech Stack:** Python 3.13, PyTorch, TOML, numpy. Tests via `uv run pytest`.

**Dependencies:** Requires Plan A (Rust engine extensions) and Plan B (Python model/PPO) to be complete.

**Spec reference:** `docs/superpowers/specs/2026-04-01-katago-se-resnet-design.md` — Slices 5 & 6.

**Deferred to Plan D:** LR plateau scheduler, RL warmup elevated entropy, `keisei-prepare-sl` CLI, and optimized `write_shard`. See `plans/2026-04-01-katago-plan-d-deferred.md`.

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `keisei/config.py` | Add `"se_resnet"` and `"katago_ppo"` to valid sets, add SL config |
| Modify | `keisei/training/checkpoint.py` | Add architecture metadata to checkpoints |
| Create | `keisei/training/katago_loop.py` | `KataGoTrainingLoop` — end-to-end RL loop |
| Create | `keisei/sl/__init__.py` | SL package |
| Create | `keisei/sl/parsers.py` | `GameParser` ABC, `GameRecord`, `SFENParser`, `CSAParser` |
| Create | `keisei/sl/dataset.py` | `SLDataset` — memory-mapped position shards |
| Create | `keisei/sl/trainer.py` | `SLTrainer` — supervised training loop |
| Create | `keisei/sl/prepare.py` | `keisei-prepare-sl` CLI entrypoint |
| Create | `keisei-katago.toml` | TOML config for KataGo SE-ResNet |
| Create | `tests/test_katago_loop.py` | Integration tests for KataGoTrainingLoop |
| Create | `tests/test_sl_pipeline.py` | Tests for SL parsers, dataset, trainer |

---

### Task 1: Config Extensions

**Files:**
- Modify: `keisei/config.py`
- Modify: `tests/test_config.py` (if exists) or create a targeted test

- [ ] **Step 1: Write the failing test**

Create `tests/test_katago_config.py`:

```python
# tests/test_katago_config.py
"""Tests for KataGo config extensions."""

import tempfile
from pathlib import Path

import pytest

from keisei.config import load_config


KATAGO_TOML = """
[model]
display_name = "KataGo-SE-b2c32"
architecture = "se_resnet"

[model.params]
num_blocks = 2
channels = 32
se_reduction = 8
global_pool_channels = 16
policy_channels = 8
value_fc_size = 32
score_fc_size = 16
obs_channels = 50

[training]
algorithm = "katago_ppo"
num_games = 2
max_ply = 50
checkpoint_interval = 10
checkpoint_dir = "checkpoints/"

[training.algorithm_params]
learning_rate = 0.0002
gamma = 0.99
lambda_policy = 1.0
lambda_value = 1.5
lambda_score = 0.02
lambda_entropy = 0.01
score_normalization = 76.0
grad_clip = 1.0

[display]
moves_per_minute = 0
db_path = "test.db"
"""


def test_load_katago_config(tmp_path):
    toml_file = tmp_path / "katago.toml"
    toml_file.write_text(KATAGO_TOML)
    config = load_config(toml_file)
    assert config.model.architecture == "se_resnet"
    assert config.training.algorithm == "katago_ppo"
    assert config.model.params["num_blocks"] == 2
    assert config.model.params["channels"] == 32


def test_invalid_architecture_rejected(tmp_path):
    toml = KATAGO_TOML.replace('architecture = "se_resnet"', 'architecture = "invalid"')
    toml_file = tmp_path / "bad.toml"
    toml_file.write_text(toml)
    with pytest.raises(ValueError, match="Unknown architecture"):
        load_config(toml_file)


def test_load_real_katago_toml():
    """Verify the shipped keisei-katago.toml parses without error."""
    toml_path = Path(__file__).parent.parent / "keisei-katago.toml"
    if toml_path.exists():
        config = load_config(toml_path)
        assert config.model.architecture == "se_resnet"
        assert config.training.algorithm == "katago_ppo"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_katago_config.py -v`
Expected: FAIL — `"se_resnet"` not in `VALID_ARCHITECTURES` inside config.py

- [ ] **Step 3: Update config.py**

In `keisei/config.py`, update the valid sets:

```python
VALID_ARCHITECTURES = {"resnet", "mlp", "transformer", "se_resnet"}
VALID_ALGORITHMS = {"ppo", "katago_ppo"}
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_katago_config.py -v`
Expected: PASS

- [ ] **Step 5: Verify existing config tests still pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add keisei/config.py tests/test_katago_config.py
git commit -m "feat: add se_resnet and katago_ppo to valid config options"
```

---

### Task 2: Checkpoint Metadata

**Files:**
- Modify: `keisei/training/checkpoint.py`
- Create: `tests/test_katago_checkpoint.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_katago_checkpoint.py
"""Tests for enhanced checkpoint with architecture metadata."""

import tempfile
from pathlib import Path

import torch
import pytest

from keisei.training.checkpoint import save_checkpoint, load_checkpoint
from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams


@pytest.fixture
def model():
    params = SEResNetParams(
        num_blocks=2, channels=32, se_reduction=8,
        global_pool_channels=16, policy_channels=8,
        value_fc_size=32, score_fc_size=16, obs_channels=50,
    )
    return SEResNetModel(params)


def test_save_with_architecture_metadata(model):
    optimizer = torch.optim.Adam(model.parameters())
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.pt"
        save_checkpoint(path, model, optimizer, 10, 100, architecture="se_resnet")
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        assert ckpt["architecture"] == "se_resnet"


def test_load_with_architecture_check(model):
    optimizer = torch.optim.Adam(model.parameters())
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.pt"
        save_checkpoint(path, model, optimizer, 10, 100, architecture="se_resnet")
        meta = load_checkpoint(path, model, optimizer, expected_architecture="se_resnet")
        assert meta["epoch"] == 10


def test_load_architecture_mismatch_raises(model):
    optimizer = torch.optim.Adam(model.parameters())
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.pt"
        save_checkpoint(path, model, optimizer, 10, 100, architecture="se_resnet")
        with pytest.raises(ValueError, match="architecture mismatch"):
            load_checkpoint(path, model, optimizer, expected_architecture="resnet")


def test_load_legacy_checkpoint_no_architecture(model):
    """Old checkpoints without architecture field should load when no check requested."""
    optimizer = torch.optim.Adam(model.parameters())
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.pt"
        # Save without architecture (legacy format)
        save_checkpoint(path, model, optimizer, 5, 50)
        meta = load_checkpoint(path, model, optimizer)
        assert meta["epoch"] == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_katago_checkpoint.py -v`
Expected: FAIL — `save_checkpoint` doesn't accept `architecture` kwarg

- [ ] **Step 3: Update checkpoint.py**

```python
# keisei/training/checkpoint.py
"""Model checkpointing: save and load model + optimizer + training state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    architecture: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
    }
    if architecture is not None:
        data["architecture"] = architecture
    torch.save(data, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_architecture: str | None = None,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    if expected_architecture is not None:
        ckpt_arch = checkpoint.get("architecture")
        if ckpt_arch is not None and ckpt_arch != expected_architecture:
            raise ValueError(
                f"Checkpoint architecture mismatch: "
                f"expected '{expected_architecture}', got '{ckpt_arch}'"
            )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {"epoch": checkpoint["epoch"], "step": checkpoint["step"]}
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_katago_checkpoint.py -v`
Expected: PASS

- [ ] **Step 5: Verify existing checkpoint tests still pass**

Run: `uv run pytest tests/test_checkpoint.py -v`
Expected: PASS (existing callers don't pass `architecture`, which defaults to None)

- [ ] **Step 6: Commit**

```bash
git add keisei/training/checkpoint.py tests/test_katago_checkpoint.py
git commit -m "feat: add architecture metadata to checkpoint save/load"
```

---

### Task 3: KataGoTrainingLoop

**Files:**
- Create: `keisei/training/katago_loop.py`
- Create: `tests/test_katago_loop.py`

**Prerequisites:** Plan A (VecEnv mode params) and Plan B (SE-ResNet + KataGoPPO) must be complete. Verify before starting:

```bash
# Verify Plan B registries are in place
uv run python -c "from keisei.training.model_registry import VALID_ARCHITECTURES; assert 'se_resnet' in VALID_ARCHITECTURES"
uv run python -c "from keisei.training.algorithm_registry import VALID_ALGORITHMS; assert 'katago_ppo' in VALID_ALGORITHMS"
# Verify Plan A VecEnv modes work
uv run python -c "from shogi_gym import VecEnv; e = VecEnv(num_envs=1, max_ply=50, observation_mode='katago', action_mode='spatial'); print(f'obs_ch={e.observation_channels}, act={e.action_space_size}')"
```

If any of these fail, complete the prerequisite plan first.

This is the most complex task — it wires VecEnv (spatial + katago modes), SE-ResNet model, and KataGoPPO together.

- [ ] **Step 1: Write the failing test**

Tests use a mock VecEnv to avoid requiring Plan A's Rust build for unit testing:

```python
# tests/test_katago_loop.py
"""Integration tests for KataGoTrainingLoop."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import numpy as np

from keisei.config import AppConfig, TrainingConfig, DisplayConfig, ModelConfig
from keisei.training.katago_loop import KataGoTrainingLoop


def _make_mock_katago_vecenv(num_envs: int = 2) -> MagicMock:
    """Create a mock VecEnv that returns correct shapes for KataGo mode."""
    mock = MagicMock()
    mock.observation_channels = 50
    mock.action_space_size = 11259
    mock.episodes_completed = 0
    mock.mean_episode_length = 0.0
    mock.truncation_rate = 0.0

    def make_reset_result():
        result = MagicMock()
        result.observations = np.random.randn(num_envs, 50, 9, 9).astype(np.float32)
        result.legal_masks = np.ones((num_envs, 11259), dtype=bool)
        return result

    def make_step_result(actions):
        result = MagicMock()
        result.observations = np.random.randn(num_envs, 50, 9, 9).astype(np.float32)
        result.legal_masks = np.ones((num_envs, 11259), dtype=bool)
        result.rewards = np.zeros(num_envs, dtype=np.float32)
        result.terminated = np.zeros(num_envs, dtype=bool)
        result.truncated = np.zeros(num_envs, dtype=bool)
        result.current_players = np.zeros(num_envs, dtype=np.uint8)
        return result

    mock.reset.side_effect = lambda: make_reset_result()
    mock.step.side_effect = make_step_result
    mock.reset_stats = MagicMock()
    return mock


@pytest.fixture
def katago_config(tmp_path):
    return AppConfig(
        training=TrainingConfig(
            num_games=2,
            max_ply=50,
            algorithm="katago_ppo",
            checkpoint_interval=5,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            algorithm_params={
                "learning_rate": 2e-4,
                "gamma": 0.99,
                "lambda_policy": 1.0,
                "lambda_value": 1.5,
                "lambda_score": 0.02,
                "lambda_entropy": 0.01,
                "score_normalization": 76.0,
                "grad_clip": 1.0,
            },
        ),
        display=DisplayConfig(
            moves_per_minute=0,
            db_path=str(tmp_path / "test.db"),
        ),
        model=ModelConfig(
            display_name="Test-KataGo",
            architecture="se_resnet",
            params={
                "num_blocks": 2,
                "channels": 32,
                "se_reduction": 8,
                "global_pool_channels": 16,
                "policy_channels": 8,
                "value_fc_size": 32,
                "score_fc_size": 16,
                "obs_channels": 50,
            },
        ),
    )


class TestKataGoTrainingLoopInit:
    def test_initialization(self, katago_config):
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)
        assert loop.num_envs == 2

    def test_model_is_se_resnet(self, katago_config):
        from keisei.training.models.se_resnet import SEResNetModel
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)
        base = loop.model.module if hasattr(loop.model, "module") else loop.model
        assert isinstance(base, SEResNetModel)


class TestKataGoTrainingLoopRun:
    def test_run_one_epoch(self, katago_config):
        """Run one epoch of 4 steps — should complete without error."""
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)
        loop.run(num_epochs=1, steps_per_epoch=4)
        assert loop.epoch == 0
        assert loop.global_step == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_katago_loop.py -v`
Expected: FAIL — `katago_loop` module not found

- [ ] **Step 3: Write the implementation**

```python
# keisei/training/katago_loop.py
"""KataGo training loop orchestrator."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from keisei.config import AppConfig
from keisei.db import (
    init_db,
    read_training_state,
    update_heartbeat,
    update_training_progress,
    write_game_snapshots,
    write_metrics,
    write_training_state,
)
from keisei.training.algorithm_registry import validate_algorithm_params
from keisei.training.checkpoint import load_checkpoint, save_checkpoint
from keisei.training.katago_ppo import (
    KataGoPPOAlgorithm,
    KataGoPPOParams,
    KataGoRolloutBuffer,
    compute_value_metrics,
)
from keisei.training.model_registry import build_model

logger = logging.getLogger(__name__)


class KataGoTrainingLoop:
    def __init__(self, config: AppConfig, vecenv: Any = None) -> None:
        self.config = config
        self.db_path = config.display.db_path

        init_db(self.db_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = build_model(config.model.architecture, config.model.params)
        self.model = self.model.to(self.device)

        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            logger.info("Using %d GPUs via DataParallel", gpu_count)
            self.model = torch.nn.DataParallel(self.model)

        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "Model: %s (%s), params: %d, device: %s, gpus: %d",
            config.model.display_name,
            config.model.architecture,
            param_count,
            self.device,
            gpu_count,
        )

        ppo_params = validate_algorithm_params(
            config.training.algorithm, config.training.algorithm_params
        )
        assert isinstance(ppo_params, KataGoPPOParams)

        base_model = self.model.module if hasattr(self.model, "module") else self.model
        self.ppo = KataGoPPOAlgorithm(ppo_params, base_model, forward_model=self.model)

        if vecenv is not None:
            self.vecenv = vecenv
        else:
            from shogi_gym import VecEnv

            self.vecenv = VecEnv(
                num_envs=config.training.num_games,
                max_ply=config.training.max_ply,
                observation_mode="katago",
                action_mode="spatial",
            )

        # Startup assertions — fail fast on config mismatch
        assert self.vecenv.observation_channels == config.model.params.get("obs_channels", 50), (
            f"VecEnv produces {self.vecenv.observation_channels} channels "
            f"but model expects {config.model.params.get('obs_channels', 50)}"
        )
        assert self.vecenv.action_space_size == 11259, (
            f"Expected spatial action space 11259, got {self.vecenv.action_space_size}"
        )

        self.num_envs = config.training.num_games
        obs_channels = self.vecenv.observation_channels
        action_space = self.vecenv.action_space_size
        self.buffer = KataGoRolloutBuffer(
            num_envs=self.num_envs,
            obs_shape=(obs_channels, 9, 9),
            action_space=action_space,
        )

        self.score_norm = ppo_params.score_normalization
        self.moves_per_minute = config.display.moves_per_minute
        self._last_snapshot_time = 0.0
        self.epoch = 0
        self.global_step = 0
        self._last_heartbeat = time.monotonic()

        self._check_resume()

    def _check_resume(self) -> None:
        state = read_training_state(self.db_path)
        if state is not None and state.get("checkpoint_path"):
            checkpoint_path = Path(state["checkpoint_path"])
            if checkpoint_path.exists():
                logger.warning(
                    "Resuming from checkpoint: %s (epoch %d)",
                    checkpoint_path,
                    state["current_epoch"],
                )
                base_model = (
                    self.model.module if hasattr(self.model, "module") else self.model
                )
                meta = load_checkpoint(
                    checkpoint_path,
                    base_model,
                    self.ppo.optimizer,
                    expected_architecture=self.config.model.architecture,
                )
                self.epoch = meta["epoch"]
                self.global_step = meta["step"]
                return

        write_training_state(
            self.db_path,
            {
                "config_json": json.dumps(
                    {
                        "training": {
                            "num_games": self.config.training.num_games,
                            "algorithm": self.config.training.algorithm,
                        },
                        "model": {"architecture": self.config.model.architecture},
                    }
                ),
                "display_name": self.config.model.display_name,
                "model_arch": self.config.model.architecture,
                "algorithm_name": self.config.training.algorithm,
                "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        )

    def run(self, num_epochs: int, steps_per_epoch: int) -> None:
        reset_result = self.vecenv.reset()
        obs = torch.from_numpy(np.asarray(reset_result.observations)).to(self.device)
        legal_masks = torch.from_numpy(np.asarray(reset_result.legal_masks)).to(self.device)

        start_epoch = self.epoch
        for epoch_i in range(start_epoch, start_epoch + num_epochs):
            self.epoch = epoch_i
            win_count = 0
            black_wins = 0
            white_wins = 0
            draw_count = 0

            for step_i in range(steps_per_epoch):
                self.global_step += 1

                actions, log_probs, values = self.ppo.select_actions(obs, legal_masks)
                action_list = actions.tolist()
                step_result = self.vecenv.step(action_list)

                rewards = torch.from_numpy(np.asarray(step_result.rewards)).to(self.device)
                terminated = torch.from_numpy(np.asarray(step_result.terminated)).to(self.device)
                truncated = torch.from_numpy(np.asarray(step_result.truncated)).to(self.device)
                dones = terminated | truncated

                # Value categories: -1 (ignore) for non-terminal steps.
                # Only terminal steps get a real label (0=W, 1=D, 2=L).
                # F.cross_entropy(ignore_index=-1) skips these in the loss.
                # This avoids the bias of labeling all non-terminal steps as
                # any particular outcome.
                value_cats = torch.full(
                    (self.num_envs,), -1, dtype=torch.long, device=self.device
                )
                score_targets = torch.zeros(self.num_envs, device=self.device)

                for env_i in range(self.num_envs):
                    r = rewards[env_i].item()
                    if dones[env_i]:
                        if r > 0:
                            value_cats[env_i] = 0  # Win
                            win_count += 1
                        elif r < 0:
                            value_cats[env_i] = 2  # Loss
                        else:
                            value_cats[env_i] = 1  # Draw
                            draw_count += 1
                        # Score: use reward as simplified score signal, normalized.
                        # NOTE: This is outcome-conditioned (+1/-1/0), not material
                        # advantage. A real material score would require the engine
                        # to report piece counts at game end. Documented simplification.
                        score_targets[env_i] = r / self.score_norm

                self.buffer.add(
                    obs, actions, log_probs, values, rewards, dones, legal_masks,
                    value_cats, score_targets,
                )

                obs = torch.from_numpy(np.asarray(step_result.observations)).to(self.device)
                legal_masks = torch.from_numpy(np.asarray(step_result.legal_masks)).to(self.device)

                self._maybe_update_heartbeat()

            # Bootstrap value for GAE — uses shared scalar_value() method
            self.model.eval()
            with torch.no_grad():
                output = self.model(obs)
                next_values = KataGoPPOAlgorithm.scalar_value(output.value_logits)

            losses = self.ppo.update(self.buffer, next_values)

            ep_completed = getattr(self.vecenv, "episodes_completed", 0)
            # NOTE: write_metrics schema has no score_loss column.
            # score_loss is logged below but not persisted to DB.
            # To persist: add an ALTER TABLE migration or a JSON extras column.
            metrics = {
                "epoch": epoch_i,
                "step": self.global_step,
                "policy_loss": losses["policy_loss"],
                "value_loss": losses["value_loss"],
                "entropy": losses["entropy"],
                "gradient_norm": losses["gradient_norm"],
                "episodes_completed": ep_completed,
            }
            write_metrics(self.db_path, metrics)

            if hasattr(self.vecenv, "reset_stats"):
                self.vecenv.reset_stats()

            update_training_progress(self.db_path, epoch_i, self.global_step)

            logger.info(
                "Epoch %d | step %d | policy=%.4f value=%.4f score=%.4f entropy=%.4f",
                epoch_i,
                self.global_step,
                losses["policy_loss"],
                losses["value_loss"],
                losses["score_loss"],
                losses["entropy"],
            )

            if (epoch_i + 1) % self.config.training.checkpoint_interval == 0:
                ckpt_path = (
                    Path(self.config.training.checkpoint_dir)
                    / f"epoch_{epoch_i:05d}.pt"
                )
                base_model = (
                    self.model.module if hasattr(self.model, "module") else self.model
                )
                save_checkpoint(
                    ckpt_path,
                    base_model,
                    self.ppo.optimizer,
                    epoch_i + 1,
                    self.global_step,
                    architecture=self.config.model.architecture,
                )
                update_training_progress(
                    self.db_path, epoch_i + 1, self.global_step, str(ckpt_path)
                )
                logger.info("Checkpoint saved: %s", ckpt_path)

    def _maybe_update_heartbeat(self) -> None:
        now = time.monotonic()
        if now - self._last_heartbeat >= 10.0:
            self._last_heartbeat = now
            update_training_progress(self.db_path, self.epoch, self.global_step)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_katago_loop.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/katago_loop.py tests/test_katago_loop.py
git commit -m "feat: add KataGoTrainingLoop wiring SE-ResNet + KataGoPPO + VecEnv"
```

---

### Task 4: TOML Config File

**Files:**
- Create: `keisei-katago.toml`

- [ ] **Step 1: Create the config file**

```toml
# keisei-katago.toml — KataGo SE-ResNet b40c256 training config

[model]
display_name = "KataGo-SE-b40c256"
architecture = "se_resnet"

[model.params]
num_blocks = 40
channels = 256
se_reduction = 16
global_pool_channels = 128
policy_channels = 32
value_fc_size = 256
score_fc_size = 128
obs_channels = 50

[training]
num_games = 128
max_ply = 512
algorithm = "katago_ppo"
checkpoint_interval = 50
checkpoint_dir = "checkpoints/katago/"

[training.algorithm_params]
learning_rate = 0.0002
gamma = 0.99
clip_epsilon = 0.2
epochs_per_batch = 4
batch_size = 256
lambda_policy = 1.0
lambda_value = 1.5
lambda_score = 0.02
lambda_entropy = 0.01
score_normalization = 76.0
grad_clip = 1.0

[display]
moves_per_minute = 30
db_path = "keisei-katago.db"
```

- [ ] **Step 2: Verify config loads**

Run: `uv run python -c "from keisei.config import load_config; from pathlib import Path; c = load_config(Path('keisei-katago.toml')); print(f'arch={c.model.architecture} algo={c.training.algorithm}')"`
Expected: `arch=se_resnet algo=katago_ppo`

- [ ] **Step 3: Commit**

```bash
git add keisei-katago.toml
git commit -m "feat: add KataGo SE-ResNet b40c256 training config"
```

---

### Task 5: Game Parser Interface and SFENParser

**Files:**
- Create: `keisei/sl/__init__.py`
- Create: `keisei/sl/parsers.py`
- Create: `tests/test_sl_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sl_pipeline.py
"""Tests for the supervised learning pipeline."""

import tempfile
from pathlib import Path

import pytest

from keisei.sl.parsers import (
    GameOutcome,
    GameRecord,
    ParsedMove,
    SFENParser,
)


class TestSFENParser:
    def test_supported_extensions(self):
        parser = SFENParser()
        assert ".sfen" in parser.supported_extensions()

    def test_parse_single_game(self, tmp_path):
        # SFEN format: one game per block, first line = metadata, rest = moves
        sfen_content = (
            "result:win_black\n"
            "startpos\n"
            "7g7f\n"
            "3c3d\n"
        )
        sfen_file = tmp_path / "test.sfen"
        sfen_file.write_text(sfen_content)

        parser = SFENParser()
        games = list(parser.parse(sfen_file))
        assert len(games) == 1
        assert games[0].outcome == GameOutcome.WIN_BLACK
        assert len(games[0].moves) == 2
        assert games[0].moves[0].move_usi == "7g7f"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sl_pipeline.py::TestSFENParser -v`
Expected: FAIL

- [ ] **Step 3: Write the implementation**

```python
# keisei/sl/__init__.py
```

```python
# keisei/sl/parsers.py
"""Game record parsers for supervised learning data."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


class GameOutcome(Enum):
    WIN_BLACK = "win_black"
    WIN_WHITE = "win_white"
    DRAW = "draw"


@dataclass
class ParsedMove:
    move_usi: str
    sfen_before: str = ""


@dataclass
class GameRecord:
    moves: list[ParsedMove]
    outcome: GameOutcome
    metadata: dict[str, str] = field(default_factory=dict)


class GameParser(ABC):
    @abstractmethod
    def parse(self, path: Path) -> Iterator[GameRecord]: ...

    @abstractmethod
    def supported_extensions(self) -> set[str]: ...


class SFENParser(GameParser):
    """Simple SFEN-based game record parser.

    Format: blocks separated by blank lines.
    First line: key:value metadata (at minimum: result:win_black|win_white|draw)
    Second line: starting position ("startpos" or SFEN string)
    Remaining lines: one USI move per line.
    """

    def supported_extensions(self) -> set[str]:
        return {".sfen"}

    def parse(self, path: Path) -> Iterator[GameRecord]:
        text = path.read_text()
        blocks = text.strip().split("\n\n")

        for block in blocks:
            lines = [line.strip() for line in block.strip().split("\n") if line.strip()]
            if len(lines) < 2:
                continue

            # Parse metadata from first line(s)
            metadata: dict[str, str] = {}
            move_start = 0
            for i, line in enumerate(lines):
                if ":" in line and not any(c.isdigit() for c in line.split(":")[0]):
                    key, _, val = line.partition(":")
                    metadata[key.strip()] = val.strip()
                    move_start = i + 1
                else:
                    break

            # Parse outcome
            result_str = metadata.get("result", "")
            try:
                outcome = GameOutcome(result_str)
            except ValueError:
                continue  # skip games with unknown outcome

            # Skip position line (startpos or SFEN)
            if move_start < len(lines):
                position_line = lines[move_start]
                move_start += 1

            # Parse moves
            moves = []
            for line in lines[move_start:]:
                moves.append(ParsedMove(move_usi=line))

            if moves:
                yield GameRecord(moves=moves, outcome=outcome, metadata=metadata)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_sl_pipeline.py::TestSFENParser -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/sl/__init__.py keisei/sl/parsers.py tests/test_sl_pipeline.py
git commit -m "feat: add GameParser interface and SFENParser for SL pipeline"
```

---

### Task 6: CSAParser (Required for Slice 6 Completion)

**Files:**
- Modify: `keisei/sl/parsers.py`
- Modify: `tests/test_sl_pipeline.py`

CSA is the primary format for Floodgate data. This parser must handle the standard CSA format including move notation, game result, and player metadata.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_sl_pipeline.py`:

```python
from keisei.sl.parsers import CSAParser


CSA_GAME = """V2.2
N+Player1
N-Player2
P1-KY-KE-GI-KI-OU-KI-GI-KE-KY
P2 * -HI *  *  *  *  * -KA * 
P3-FU-FU-FU-FU-FU-FU-FU-FU-FU
P4 *  *  *  *  *  *  *  *  * 
P5 *  *  *  *  *  *  *  *  * 
P6 *  *  *  *  *  *  *  *  * 
P7+FU+FU+FU+FU+FU+FU+FU+FU+FU
P8 * +KA *  *  *  *  * +HI * 
P9+KY+KE+GI+KI+OU+KI+GI+KE+KY
+
+7776FU
-3334FU
%TORYO
"""


class TestCSAParser:
    def test_supported_extensions(self):
        parser = CSAParser()
        assert ".csa" in parser.supported_extensions()

    def test_parse_single_game(self, tmp_path):
        csa_file = tmp_path / "test.csa"
        csa_file.write_text(CSA_GAME)

        parser = CSAParser()
        games = list(parser.parse(csa_file))
        assert len(games) == 1
        # After -3334FU (White moves), it's Black's turn. %TORYO = side-to-move resigns.
        # Black resigns → White wins. last_mover="-" → WIN_WHITE.
        assert games[0].outcome == GameOutcome.WIN_WHITE
        assert len(games[0].moves) == 2
        assert games[0].metadata.get("player_black") == "Player1"
        assert games[0].metadata.get("player_white") == "Player2"

    def test_parse_csa_move_format(self, tmp_path):
        csa_file = tmp_path / "test.csa"
        csa_file.write_text(CSA_GAME)

        parser = CSAParser()
        games = list(parser.parse(csa_file))
        # CSA move "+7776FU" should be converted to USI "7g7f"
        assert games[0].moves[0].move_usi == "7g7f"
        assert games[0].moves[1].move_usi == "3c3d"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sl_pipeline.py::TestCSAParser -v`
Expected: FAIL — `CSAParser` not defined

- [ ] **Step 3: Write the CSAParser implementation**

Add to `keisei/sl/parsers.py`:

```python
class CSAParser(GameParser):
    """Parser for Computer Shogi Association (CSA) game record format.

    Handles V2.2 format used by Floodgate and other servers.
    Converts CSA move notation to USI notation.
    """

    # CSA column (1-9) to USI file letter
    _COL_TO_FILE = {1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9"}
    # CSA row (1-9) to USI rank letter
    _ROW_TO_RANK = {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f", 7: "g", 8: "h", 9: "i"}
    # CSA piece names to USI piece names (for drops)
    _PIECE_TO_USI = {
        "FU": "P", "KY": "L", "KE": "N", "GI": "S",
        "KI": "G", "KA": "B", "HI": "R",
        "TO": "P", "NY": "L", "NK": "N", "NG": "S",
        "UM": "B", "RY": "R", "OU": "K",
    }
    # Promoted CSA pieces
    _PROMOTED = {"TO", "NY", "NK", "NG", "UM", "RY"}

    def supported_extensions(self) -> set[str]:
        return {".csa"}

    def _csa_move_to_usi(self, csa_move: str, board: dict[tuple[int,int], str]) -> str:
        """Convert CSA move like '+7776FU' to USI move like '7g7f'.

        CSA format: [+-]<from_col><from_row><to_col><to_row><piece>
        If from is "00", it's a drop.

        `board` tracks piece names at each (col, row) position for promotion
        detection. Updated in-place by the caller after each move.
        """
        # Strip the +/- prefix
        side = csa_move[0]
        body = csa_move[1:]

        from_col = int(body[0])
        from_row = int(body[1])
        to_col = int(body[2])
        to_row = int(body[3])
        piece = body[4:]  # piece name at DESTINATION (post-move)

        if from_col == 0 and from_row == 0:
            # Drop move: "0055FU" -> "P*5e"
            usi_piece = self._PIECE_TO_USI.get(piece, piece)
            to_file = str(to_col)
            to_rank = self._ROW_TO_RANK[to_row]
            return f"{usi_piece}*{to_file}{to_rank}"

        # Board move
        from_file = str(from_col)
        from_rank = self._ROW_TO_RANK[from_row]
        to_file = str(to_col)
        to_rank = self._ROW_TO_RANK[to_row]

        usi = f"{from_file}{from_rank}{to_file}{to_rank}"

        # Promotion detection: compare piece at source (before move) with
        # piece at destination (after move). If the destination piece is a
        # promoted type but the source piece was not, promotion happened.
        source_piece = board.get((from_col, from_row), "")
        if piece in self._PROMOTED and source_piece not in self._PROMOTED:
            usi += "+"

        return usi

    @staticmethod
    def _parse_board_from_p_lines(p_lines: list[str]) -> dict[tuple[int,int], str]:
        """Parse CSA P1-P9 position lines into a (col, row) -> piece_name dict."""
        board: dict[tuple[int,int], str] = {}
        for line in p_lines:
            if not line.startswith("P") or len(line) < 3:
                continue
            row_char = line[1]
            if not row_char.isdigit():
                continue
            row = int(row_char)
            # Each position is 3 chars: " * " (empty) or "+FU" / "-FU"
            content = line[2:]
            for col_idx in range(9):
                start = col_idx * 3
                if start + 3 > len(content):
                    break
                cell = content[start:start+3]
                if cell.strip() == "*" or cell.strip() == "":
                    continue
                # cell is like "+FU" or "-KY"
                piece_name = cell[1:3] if len(cell) >= 3 else cell
                actual_col = 9 - col_idx  # CSA columns are 9..1 left-to-right
                board[(actual_col, row)] = piece_name
        return board

    def parse(self, path: Path) -> Iterator[GameRecord]:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.split("\n")

        metadata: dict[str, str] = {}
        moves: list[ParsedMove] = []
        last_mover: str = "+"
        result_line: str = ""
        p_lines: list[str] = []

        # First pass: collect P-lines for board state initialization
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith("P") and len(line_stripped) > 2 and line_stripped[1].isdigit():
                p_lines.append(line_stripped)

        # Initialize board state from position definition
        board = self._parse_board_from_p_lines(p_lines) if p_lines else {}

        for line in lines:
            line = line.strip()
            if not line or line.startswith("'"):
                continue  # skip empty lines and comments
            if line.startswith("V"):
                continue  # version
            if line.startswith("N+"):
                metadata["player_black"] = line[2:]
            elif line.startswith("N-"):
                metadata["player_white"] = line[2:]
            elif line.startswith("$"):
                key, _, val = line[1:].partition(":")
                metadata[key.lower()] = val.strip()
            elif line.startswith("P"):
                continue  # position definition lines (already parsed)
            elif line == "+" or line == "-":
                continue  # side to move indicator
            elif line.startswith("+") or line.startswith("-"):
                if "%" in line:
                    result_line = line[1:]
                else:
                    last_mover = line[0]
                    usi_move = self._csa_move_to_usi(line, board)
                    moves.append(ParsedMove(move_usi=usi_move))

                    # Update board state
                    body = line[1:]
                    from_col, from_row = int(body[0]), int(body[1])
                    to_col, to_row = int(body[2]), int(body[3])
                    piece = body[4:]
                    if from_col != 0 or from_row != 0:
                        board.pop((from_col, from_row), None)
                    board[(to_col, to_row)] = piece
            elif line.startswith("%"):
                result_line = line

        if not moves:
            return

        # Determine outcome from result line
        if result_line == "%TORYO":
            # %TORYO = side-to-move resigns. last_mover is the player who
            # made the final actual move, so the OPPONENT of last_mover is the
            # side-to-move who resigned. last_mover's side wins.
            outcome = GameOutcome.WIN_BLACK if last_mover == "+" else GameOutcome.WIN_WHITE
        elif result_line == "%SENNICHITE":
            outcome = GameOutcome.DRAW
        elif result_line in ("%JISHOGI", "%KACHI"):
            # Impasse/declaration: last mover wins
            outcome = GameOutcome.WIN_BLACK if last_mover == "+" else GameOutcome.WIN_WHITE
        elif result_line == "%HIKIWAKE":
            outcome = GameOutcome.DRAW
        else:
            outcome = GameOutcome.DRAW  # unknown result, default to draw

        yield GameRecord(moves=moves, outcome=outcome, metadata=metadata)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_sl_pipeline.py::TestCSAParser -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/sl/parsers.py tests/test_sl_pipeline.py
git commit -m "feat: add CSAParser for Floodgate game record parsing"
```

---

### Task 7: SL Dataset (Memory-Mapped Shards)

**Files:**
- Create: `keisei/sl/dataset.py`
- Modify: `tests/test_sl_pipeline.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_sl_pipeline.py`:

```python
import numpy as np
import torch

from keisei.sl.dataset import SLDataset, write_shard


class TestSLDataset:
    def test_write_and_read_shard(self, tmp_path):
        """Write a shard with 10 positions and read them back."""
        observations = np.random.randn(10, 50 * 81).astype(np.float32)
        policy_targets = np.random.randint(0, 11259, size=10).astype(np.int64)
        value_targets = np.random.randint(0, 3, size=10).astype(np.int64)
        score_targets = np.random.randn(10).astype(np.float32)

        shard_path = tmp_path / "shard_000.bin"
        write_shard(shard_path, observations, policy_targets, value_targets, score_targets)

        dataset = SLDataset(tmp_path)
        assert len(dataset) == 10

        item = dataset[0]
        assert item["observation"].shape == (50, 9, 9)
        assert item["policy_target"].shape == ()  # scalar
        assert item["value_target"].shape == ()
        assert item["score_target"].shape == ()

    def test_multiple_shards(self, tmp_path):
        for i in range(3):
            n = 5
            write_shard(
                tmp_path / f"shard_{i:03d}.bin",
                np.random.randn(n, 50 * 81).astype(np.float32),
                np.random.randint(0, 11259, size=n).astype(np.int64),
                np.random.randint(0, 3, size=n).astype(np.int64),
                np.random.randn(n).astype(np.float32),
            )
        dataset = SLDataset(tmp_path)
        assert len(dataset) == 15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sl_pipeline.py::TestSLDataset -v`
Expected: FAIL

- [ ] **Step 3: Write the implementation**

```python
# keisei/sl/dataset.py
"""Memory-mapped SL dataset for training positions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Per-position record layout in shard:
# observation: float32[50*81] = 16200 bytes
# policy_target: int64         = 8 bytes
# value_target: int64          = 8 bytes
# score_target: float32        = 4 bytes
# Total: 16220 bytes per position

OBS_SIZE = 50 * 81
OBS_BYTES = OBS_SIZE * 4  # float32
RECORD_SIZE = OBS_BYTES + 8 + 8 + 4  # 16220 bytes


def write_shard(
    path: Path,
    observations: np.ndarray,
    policy_targets: np.ndarray,
    value_targets: np.ndarray,
    score_targets: np.ndarray,
) -> None:
    """Write positions to a binary shard file."""
    n = observations.shape[0]
    assert observations.shape == (n, OBS_SIZE)
    assert policy_targets.shape == (n,)
    assert value_targets.shape == (n,)
    assert score_targets.shape == (n,)

    # NOTE: Per-record writes (4 syscalls per position). For initial implementation
    # this is acceptable. For production scale (millions of positions), interleave
    # into a single contiguous buffer and write in one call to reduce I/O overhead.
    with open(path, "wb") as f:
        for i in range(n):
            f.write(observations[i].astype(np.float32).tobytes())
            f.write(policy_targets[i].astype(np.int64).tobytes())
            f.write(value_targets[i].astype(np.int64).tobytes())
            f.write(score_targets[i].astype(np.float32).tobytes())


class SLDataset(Dataset):
    """Memory-mapped dataset reading from binary shard files."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.shards: list[tuple[Path, int]] = []  # (path, num_positions)
        self._cumulative: list[int] = []

        shard_files = sorted(data_dir.glob("shard_*.bin"))
        total = 0
        for shard_path in shard_files:
            file_size = shard_path.stat().st_size
            n_positions = file_size // RECORD_SIZE
            if n_positions > 0:
                self.shards.append((shard_path, n_positions))
                total += n_positions
                self._cumulative.append(total)

        self._total = total
        # Cache grows to one entry per shard (typically 10-100 shards).
        # If shard count becomes very large, add LRU eviction.
        self._mmap_cache: dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return self._total

    def _get_mmap(self, path: Path) -> np.ndarray:
        if path not in self._mmap_cache:
            self._mmap_cache[path] = np.memmap(path, dtype=np.uint8, mode="r")
        return self._mmap_cache[path]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= self._total:
            raise IndexError(f"index {idx} out of range for dataset with {self._total} positions")

        # Find which shard this index belongs to — O(log n) via bisect
        import bisect
        shard_idx = bisect.bisect_right(self._cumulative, idx)
        local_idx = idx - (self._cumulative[shard_idx - 1] if shard_idx > 0 else 0)

        shard_path, _ = self.shards[shard_idx]
        mmap = self._get_mmap(shard_path)

        offset = local_idx * RECORD_SIZE

        # Read observation
        obs_bytes = mmap[offset : offset + OBS_BYTES]
        obs = np.frombuffer(obs_bytes, dtype=np.float32).reshape(50, 9, 9).copy()

        # Read targets
        offset += OBS_BYTES
        policy = np.frombuffer(mmap[offset : offset + 8], dtype=np.int64).copy()
        offset += 8
        value = np.frombuffer(mmap[offset : offset + 8], dtype=np.int64).copy()
        offset += 8
        score = np.frombuffer(mmap[offset : offset + 4], dtype=np.float32).copy()

        return {
            "observation": torch.from_numpy(obs),
            "policy_target": torch.tensor(policy[0], dtype=torch.long),
            "value_target": torch.tensor(value[0], dtype=torch.long),
            "score_target": torch.tensor(score[0], dtype=torch.float32),
        }
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_sl_pipeline.py::TestSLDataset -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/sl/dataset.py tests/test_sl_pipeline.py
git commit -m "feat: add SLDataset with memory-mapped binary shards"
```

---

### Task 8: SLTrainer

**Files:**
- Create: `keisei/sl/trainer.py`
- Modify: `tests/test_sl_pipeline.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_sl_pipeline.py`:

```python
from keisei.sl.trainer import SLTrainer, SLConfig
from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams


class TestSLTrainer:
    @pytest.fixture
    def small_model(self):
        params = SEResNetParams(
            num_blocks=2, channels=32, se_reduction=8,
            global_pool_channels=16, policy_channels=8,
            value_fc_size=32, score_fc_size=16, obs_channels=50,
        )
        return SEResNetModel(params)

    def test_train_one_epoch(self, small_model, tmp_path):
        # Create a small dataset
        n = 16
        write_shard(
            tmp_path / "shard_000.bin",
            np.random.randn(n, 50 * 81).astype(np.float32),
            np.random.randint(0, 11259, size=n).astype(np.int64),
            np.random.randint(0, 3, size=n).astype(np.int64),
            np.random.randn(n).astype(np.float32),
        )

        config = SLConfig(data_dir=str(tmp_path), batch_size=8, learning_rate=1e-3, total_epochs=1)
        trainer = SLTrainer(small_model, config)
        metrics = trainer.train_epoch()

        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "score_loss" in metrics
        assert all(not np.isnan(v) for v in metrics.values()), f"NaN in metrics: {metrics}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sl_pipeline.py::TestSLTrainer -v`
Expected: FAIL

- [ ] **Step 3: Write the implementation**

```python
# keisei/sl/trainer.py
"""Supervised learning trainer for KataGo models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from keisei.sl.dataset import SLDataset
from keisei.training.models.katago_base import KataGoBaseModel


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


logger = logging.getLogger(__name__)


class SLTrainer:
    """Supervised learning trainer. Trains one epoch at a time.

    Checkpoint management is the caller's responsibility — call
    save_checkpoint() between epochs and load_checkpoint() to resume.
    This keeps the trainer simple and composable with different
    training scripts (CLI, notebook, distributed).
    """

    def __init__(self, model: KataGoBaseModel, config: SLConfig) -> None:
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.total_epochs, eta_min=1e-6
        )
        self.dataset = SLDataset(Path(config.data_dir))
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )

    def train_epoch(self) -> dict[str, float]:
        self.model.train()
        total_policy = 0.0
        total_value = 0.0
        total_score = 0.0
        num_batches = 0

        for batch in self.dataloader:
            obs = batch["observation"].to(self.device)
            policy_targets = batch["policy_target"].to(self.device)
            value_targets = batch["value_target"].to(self.device)
            score_targets = batch["score_target"].to(self.device)

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
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            total_policy += policy_loss.item()
            total_value += value_loss.item()
            total_score += score_loss.item()
            num_batches += 1

        self.scheduler.step()

        denom = max(num_batches, 1)
        metrics = {
            "policy_loss": total_policy / denom,
            "value_loss": total_value / denom,
            "score_loss": total_score / denom,
        }
        logger.info(
            "SL epoch | policy=%.4f value=%.4f score=%.4f lr=%.6f",
            metrics["policy_loss"], metrics["value_loss"],
            metrics["score_loss"], self.optimizer.param_groups[0]["lr"],
        )
        return metrics
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_sl_pipeline.py::TestSLTrainer -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/sl/trainer.py tests/test_sl_pipeline.py
git commit -m "feat: add SLTrainer for supervised learning warmup"
```

---

### Task 9: Full Test Suite Verification

**Files:** None (verification only)

- [ ] **Step 1: Run all Python tests**

Run: `uv run pytest -v`
Expected: All tests PASS — existing + new.

- [ ] **Step 2: Verify end-to-end KataGo training loop**

`AppConfig` is a frozen dataclass (can't use `_replace`). Create a minimal TOML for the smoke test:

Run: `uv run python -c "
import tempfile
from pathlib import Path
from keisei.config import load_config
from keisei.training.katago_loop import KataGoTrainingLoop

# Write a small test config (2 envs, tiny model)
toml = '''
[model]
display_name = \"smoke-test\"
architecture = \"se_resnet\"
[model.params]
num_blocks = 2
channels = 32
se_reduction = 8
global_pool_channels = 16
policy_channels = 8
value_fc_size = 32
score_fc_size = 16
obs_channels = 50
[training]
algorithm = \"katago_ppo\"
num_games = 2
max_ply = 50
checkpoint_interval = 100
checkpoint_dir = \"/tmp/katago_smoke/\"
[training.algorithm_params]
learning_rate = 0.0002
score_normalization = 76.0
grad_clip = 1.0
[display]
moves_per_minute = 0
db_path = \"/tmp/katago_smoke.db\"
'''
with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
    f.write(toml)
    config = load_config(Path(f.name))
loop = KataGoTrainingLoop(config)
loop.run(num_epochs=1, steps_per_epoch=4)
print(f'Completed: epoch={loop.epoch}, step={loop.global_step}')
"`
Expected: `Completed: epoch=0, step=4`

- [ ] **Step 3: Final commit if any fixes were needed**

```bash
git add -u
git commit -m "fix: address issues found in full integration verification"
```
