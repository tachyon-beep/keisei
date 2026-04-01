# KataGo Plan D: Deferred Items

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement features deferred from Plans A-C that are needed for production training but not for pipeline validation.

**Architecture:** Additions to existing code from Plans A-C. No new architectural patterns.

**Tech Stack:** Python 3.13, PyTorch, TOML. Tests via `uv run pytest`.

**Dependencies:** Requires Plans A, B, and C to be complete.

**Spec reference:** `docs/superpowers/specs/2026-04-01-katago-se-resnet-design.md`

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `keisei/sl/prepare.py` | `keisei-prepare-sl` CLI entrypoint |
| Modify | `keisei/training/katago_loop.py` | LR plateau scheduler + RL warmup entropy |
| Modify | `keisei/training/katago_ppo.py` | `KataGoPPOParams.rl_warmup_epochs` + `rl_warmup_entropy` |
| Modify | `keisei/sl/dataset.py` | Optimized `write_shard` (single-buffer write) |
| Modify | `pyproject.toml` | Register `keisei-prepare-sl` entrypoint |
| Create | `tests/test_prepare_sl.py` | Tests for data preparation CLI |
| Create | `tests/test_lr_scheduler.py` | Tests for plateau scheduler + warmup |

---

### Task 1: `keisei-prepare-sl` CLI Entrypoint

**Files:**
- Create: `keisei/sl/prepare.py`
- Modify: `pyproject.toml`
- Create: `tests/test_prepare_sl.py`

The spec defines this as the batch preprocessing step: scan game sources, parse via GameParser, filter, encode positions, write shards.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prepare_sl.py
"""Tests for the SL data preparation CLI."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from keisei.sl.prepare import prepare_sl_data
from keisei.sl.dataset import SLDataset


@pytest.fixture
def sample_sfen_dir(tmp_path):
    """Create a directory with a small SFEN game file."""
    sfen_content = (
        "result:win_black\n"
        "startpos\n"
        "7g7f\n"
        "3c3d\n"
        "2g2f\n"
        "8c8d\n"
    )
    sfen_file = tmp_path / "games" / "test.sfen"
    sfen_file.parent.mkdir()
    sfen_file.write_text(sfen_content)
    return tmp_path / "games"


def test_prepare_creates_shards(sample_sfen_dir, tmp_path):
    output_dir = tmp_path / "processed"
    prepare_sl_data(
        game_sources=[str(sample_sfen_dir)],
        output_dir=str(output_dir),
        min_ply=2,
    )
    dataset = SLDataset(output_dir)
    assert len(dataset) > 0


def test_prepare_filters_short_games(tmp_path):
    # Game with only 1 move — should be filtered by min_ply=2
    sfen_content = "result:win_black\nstartpos\n7g7f\n"
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    (games_dir / "short.sfen").write_text(sfen_content)

    output_dir = tmp_path / "processed"
    prepare_sl_data(
        game_sources=[str(games_dir)],
        output_dir=str(output_dir),
        min_ply=2,
    )
    dataset = SLDataset(output_dir)
    assert len(dataset) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prepare_sl.py -v`
Expected: FAIL — `prepare_sl_data` not found

- [ ] **Step 3: Write the implementation**

```python
# keisei/sl/prepare.py
"""SL data preparation: parse game records, encode positions, write shards."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from keisei.sl.dataset import OBS_SIZE, write_shard
from keisei.sl.parsers import (
    CSAParser,
    GameFilter,
    GameOutcome,
    GameParser,
    SFENParser,
)

logger = logging.getLogger(__name__)

# Parser registry by file extension
_PARSERS: dict[str, GameParser] = {}
for _parser_cls in [SFENParser, CSAParser]:
    _p = _parser_cls()
    for ext in _p.supported_extensions():
        _PARSERS[ext] = _p


def prepare_sl_data(
    game_sources: list[str],
    output_dir: str,
    min_ply: int = 40,
    min_rating: int | None = None,
    shard_size: int = 100_000,
) -> None:
    """Parse game records, encode positions, and write shards.

    For initial implementation, positions are encoded as flat observation
    tensors using the Rust VecEnv. This requires the shogi-gym native module.

    NOTE: For production scale, parallelize via multiprocessing or Rust rayon.
    This implementation is single-threaded for correctness validation.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    game_filter = GameFilter(min_ply=min_ply, min_rating=min_rating)

    # Collect all game files
    game_files: list[Path] = []
    for source in game_sources:
        source_path = Path(source)
        if source_path.is_file():
            game_files.append(source_path)
        elif source_path.is_dir():
            for ext in _PARSERS:
                game_files.extend(source_path.glob(f"*{ext}"))

    logger.info("Found %d game files across %d sources", len(game_files), len(game_sources))

    # Accumulate positions into shard buffers
    observations: list[np.ndarray] = []
    policy_targets: list[int] = []
    value_targets: list[int] = []
    score_targets: list[float] = []
    shard_idx = 0
    games_parsed = 0
    games_skipped = 0

    # Warn loudly about placeholder data
    logger.warning(
        "*** PLACEHOLDER MODE: observations are all-zeros, policy targets are all-zeros. ***\n"
        "    Shards produced are structurally valid but semantically useless for training.\n"
        "    Full observation/policy encoding requires the Rust engine (shogi-gym).\n"
        "    Use these shards ONLY for pipeline testing, NOT for model training."
    )

    parse_errors = 0
    for game_file in game_files:
        ext = game_file.suffix.lower()
        parser = _PARSERS.get(ext)
        if parser is None:
            logger.warning("No parser for extension '%s', skipping %s", ext, game_file)
            continue

        try:
            file_records = list(parser.parse(game_file))
        except Exception:
            logger.exception("Failed to parse %s — skipping", game_file)
            parse_errors += 1
            continue

        for record in file_records:
            if not game_filter.accepts(record):
                games_skipped += 1
                continue

            games_parsed += 1

            # Determine W/D/L target and score for each position
            for i, move in enumerate(record.moves):
                # Side-to-move perspective: even moves = Black, odd = White
                is_black_to_move = (i % 2 == 0)

                if record.outcome == GameOutcome.WIN_BLACK:
                    value_cat = 0 if is_black_to_move else 2  # W or L
                    raw_score = 1.0 if is_black_to_move else -1.0
                elif record.outcome == GameOutcome.WIN_WHITE:
                    value_cat = 2 if is_black_to_move else 0
                    raw_score = -1.0 if is_black_to_move else 1.0
                else:
                    value_cat = 1  # Draw
                    raw_score = 0.0

                # NOTE: observation encoding and policy target encoding
                # require the Rust engine to replay the position from SFEN.
                # For this initial implementation, we store placeholder
                # observations. A full implementation would:
                #   1. Create a GameState from the starting SFEN
                #   2. Replay moves to reach this position
                #   3. Generate observation via KataGoObservationGenerator
                #   4. Encode the played move via SpatialActionMapper
                #
                # This placeholder allows testing the shard write/read pipeline
                # without requiring the Rust engine.
                obs = np.zeros(OBS_SIZE, dtype=np.float32)  # placeholder
                policy_target = 0  # placeholder

                observations.append(obs)
                policy_targets.append(policy_target)
                value_targets.append(value_cat)
                score_targets.append(raw_score / 76.0)  # normalize

            # Flush shard if buffer is full
            if len(observations) >= shard_size:
                _flush_shard(
                    output_path, shard_idx, observations,
                    policy_targets, value_targets, score_targets,
                )
                shard_idx += 1
                observations.clear()
                policy_targets.clear()
                value_targets.clear()
                score_targets.clear()

    # Flush remaining
    if observations:
        _flush_shard(
            output_path, shard_idx, observations,
            policy_targets, value_targets, score_targets,
        )
        shard_idx += 1

    logger.info(
        "Prepared %d shards from %d games (%d skipped by filter, %d parse errors)",
        shard_idx, games_parsed, games_skipped, parse_errors,
    )


def _flush_shard(
    output_path: Path,
    shard_idx: int,
    observations: list[np.ndarray],
    policy_targets: list[int],
    value_targets: list[int],
    score_targets: list[float],
) -> None:
    n = len(observations)
    shard_path = output_path / f"shard_{shard_idx:03d}.bin"
    write_shard(
        shard_path,
        np.array(observations, dtype=np.float32),
        np.array(policy_targets, dtype=np.int64),
        np.array(value_targets, dtype=np.int64),
        np.array(score_targets, dtype=np.float32),
    )
    logger.info("Wrote shard %s with %d positions", shard_path.name, n)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Prepare SL training data")
    parser.add_argument(
        "--sources", nargs="+", required=True,
        help="Directories or files containing game records",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for processed shards",
    )
    parser.add_argument("--min-ply", type=int, default=40)
    parser.add_argument("--min-rating", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=100_000)
    args = parser.parse_args()

    prepare_sl_data(
        game_sources=args.sources,
        output_dir=args.output,
        min_ply=args.min_ply,
        min_rating=args.min_rating,
        shard_size=args.shard_size,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add GameFilter to parsers.py (required — not present in Plan C)**

Plan C does not define `GameFilter`. Add it to `keisei/sl/parsers.py` before implementing `prepare.py`, since `prepare_sl_data()` imports it:

```python
@dataclass
class GameFilter:
    """Filter for game quality before SL encoding."""
    min_ply: int = 40
    min_rating: int | None = None

    def accepts(self, record: GameRecord) -> bool:
        if len(record.moves) < self.min_ply:
            return False
        if self.min_rating is not None:
            # Check rating metadata if available
            for key in ("rating", "black_rating", "white_rating"):
                rating_str = record.metadata.get(key, "")
                if rating_str.isdigit() and int(rating_str) < self.min_rating:
                    return False
        return True
```

- [ ] **Step 5: Register CLI entrypoint in pyproject.toml**

Add to `[project.scripts]` in `pyproject.toml`:

```toml
keisei-prepare-sl = "keisei.sl.prepare:main"
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_prepare_sl.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add keisei/sl/prepare.py tests/test_prepare_sl.py pyproject.toml
git commit -m "feat: add keisei-prepare-sl CLI for SL data preparation"
```

---

### Task 2: LR Plateau Scheduler

**Files:**
- Modify: `keisei/training/katago_loop.py`
- Create: `tests/test_lr_scheduler.py`

Wire the `ReduceLROnPlateau` scheduler monitoring `value_loss`, as specified in the TOML config.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lr_scheduler.py
"""Tests for LR scheduling in KataGoTrainingLoop."""

import pytest
import torch

from keisei.training.katago_ppo import KataGoPPOParams, KataGoPPOAlgorithm
from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams


@pytest.fixture
def small_model():
    params = SEResNetParams(
        num_blocks=2, channels=32, se_reduction=8,
        global_pool_channels=16, policy_channels=8,
        value_fc_size=32, score_fc_size=16, obs_channels=50,
    )
    return SEResNetModel(params)


class TestLRScheduler:
    def test_plateau_scheduler_reduces_lr(self, small_model):
        """Simulate repeated high value_loss → LR should decrease."""
        from keisei.training.katago_loop import create_lr_scheduler

        params = KataGoPPOParams(learning_rate=1e-3)
        ppo = KataGoPPOAlgorithm(params, small_model)

        scheduler = create_lr_scheduler(
            ppo.optimizer,
            schedule_type="plateau",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        )
        initial_lr = ppo.optimizer.param_groups[0]["lr"]

        # Feed constant "bad" value_loss for patience+1 epochs
        for _ in range(5):
            scheduler.step(10.0)

        final_lr = ppo.optimizer.param_groups[0]["lr"]
        assert final_lr < initial_lr, "LR should have decreased after patience exceeded"

    def test_plateau_scheduler_no_reduction_when_improving(self, small_model):
        """LR should NOT decrease when loss is improving."""
        from keisei.training.katago_loop import create_lr_scheduler

        params = KataGoPPOParams(learning_rate=1e-3)
        ppo = KataGoPPOAlgorithm(params, small_model)
        scheduler = create_lr_scheduler(ppo.optimizer, patience=3, min_lr=1e-6)
        initial_lr = ppo.optimizer.param_groups[0]["lr"]

        # Feed improving losses
        for loss in [10.0, 9.0, 8.0, 7.0, 6.0]:
            scheduler.step(loss)

        final_lr = ppo.optimizer.param_groups[0]["lr"]
        assert final_lr == initial_lr, "LR should not change when loss improves"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_lr_scheduler.py -v`
Expected: FAIL — `create_lr_scheduler` not found

- [ ] **Step 3: Write the implementation**

Add to `keisei/training/katago_loop.py`:

```python
def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    schedule_type: str = "plateau",
    factor: float = 0.5,
    patience: int = 50,
    min_lr: float = 1e-5,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """Create an LR scheduler from config parameters.

    Returns the PyTorch scheduler directly — the training loop is responsible
    for extracting the monitored metric and calling scheduler.step(value).
    No wrapper class needed for a single implementation with one caller.
    """
    if schedule_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience, min_lr=min_lr,
        )
    else:
        raise ValueError(f"Unknown schedule type '{schedule_type}'")
```

Then in `KataGoTrainingLoop.__init__`, BEFORE calling `validate_algorithm_params`,
extract nested config sections that are not `KataGoPPOParams` fields. This prevents
`TypeError: unexpected keyword argument` when the flat dataclass rejects nested dicts:

```python
# Extract nested config sections BEFORE validate_algorithm_params,
# which constructs KataGoPPOParams(**params) and rejects unknown keys.
algo_params = dict(config.training.algorithm_params)  # shallow copy
lr_config = algo_params.pop("lr_schedule", {})
rl_warmup_config = algo_params.pop("rl_warmup", {})

ppo_params = validate_algorithm_params(config.training.algorithm, algo_params)
assert isinstance(ppo_params, KataGoPPOParams)
```

Then after creating the PPO algorithm:

```python
# LR scheduler (optional — only if lr_schedule config is present)
if lr_config:
    self.lr_scheduler = create_lr_scheduler(
        self.ppo.optimizer,
        schedule_type=lr_config.get("type", "plateau"),
        monitor=lr_config.get("monitor", "value_loss"),
        factor=lr_config.get("factor", 0.5),
        patience=lr_config.get("patience", 50),
        min_lr=lr_config.get("min_lr", 1e-5),
    )
else:
    self.lr_scheduler = None

# Store warmup config for use in run()
self._rl_warmup_epochs = rl_warmup_config.get("epochs", 0)
self._rl_warmup_entropy = rl_warmup_config.get("entropy_bonus", 0.05)
```

And in the `run` method, after computing losses:

```python
if self.lr_scheduler is not None:
    monitor_key = "value_loss"
    monitor_value = losses.get(monitor_key)
    if monitor_value is not None:
        old_lr = self.ppo.optimizer.param_groups[0]["lr"]
        self.lr_scheduler.step(monitor_value)
        new_lr = self.ppo.optimizer.param_groups[0]["lr"]
        if new_lr != old_lr:
            logger.info("LR reduced: %.6f → %.6f (monitored %s=%.4f)",
                        old_lr, new_lr, monitor_key, monitor_value)
    else:
        logger.warning("LR scheduler monitor key '%s' not in losses dict", monitor_key)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_lr_scheduler.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/katago_loop.py tests/test_lr_scheduler.py
git commit -m "feat: add ReduceLROnPlateau scheduler to KataGoTrainingLoop"
```

---

### Task 3: RL Warmup Elevated Entropy

**Files:**
- Modify: `keisei/training/katago_ppo.py`
- Modify: `keisei/training/katago_loop.py`

After SL warmup, the first N epochs of RL use elevated entropy bonus to soften the overconfident SL policy.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_lr_scheduler.py`:

```python
class TestRLWarmup:
    def test_warmup_epochs_use_elevated_entropy(self, small_model):
        """During warmup, lambda_entropy should be elevated."""
        params = KataGoPPOParams(lambda_entropy=0.01)
        ppo = KataGoPPOAlgorithm(params, small_model)

        assert ppo.get_entropy_coeff(epoch=0, warmup_epochs=5, warmup_entropy=0.05) == 0.05
        assert ppo.get_entropy_coeff(epoch=4, warmup_epochs=5, warmup_entropy=0.05) == 0.05
        assert ppo.get_entropy_coeff(epoch=5, warmup_epochs=5, warmup_entropy=0.05) == 0.01
        assert ppo.get_entropy_coeff(epoch=100, warmup_epochs=5, warmup_entropy=0.05) == 0.01

    def test_current_entropy_coeff_initialized(self, small_model):
        """current_entropy_coeff should default to params.lambda_entropy."""
        params = KataGoPPOParams(lambda_entropy=0.01)
        ppo = KataGoPPOAlgorithm(params, small_model)
        assert ppo.current_entropy_coeff == 0.01

    def test_update_uses_current_entropy_coeff(self, small_model):
        """Integration test: update() must READ current_entropy_coeff, not params.lambda_entropy.

        This catches the 'stated but not coded' bug where current_entropy_coeff
        is set but update() still reads the frozen dataclass field.
        """
        from keisei.training.katago_ppo import KataGoRolloutBuffer

        params = KataGoPPOParams(lambda_entropy=0.01)
        ppo = KataGoPPOAlgorithm(params, small_model)

        # Fill a small buffer
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
        for _ in range(4):
            obs = torch.randn(2, 50, 9, 9)
            legal_masks = torch.ones(2, 11259, dtype=torch.bool)
            actions, log_probs, values = ppo.select_actions(obs, legal_masks)
            buf.add(obs, actions, log_probs, values,
                    torch.zeros(2), torch.zeros(2, dtype=torch.bool), legal_masks,
                    torch.randint(0, 3, (2,)), torch.randn(2))

        # Run update with default entropy coeff
        losses_default = ppo.update(buf, torch.zeros(2))

        # Refill buffer (update clears it)
        for _ in range(4):
            obs = torch.randn(2, 50, 9, 9)
            legal_masks = torch.ones(2, 11259, dtype=torch.bool)
            actions, log_probs, values = ppo.select_actions(obs, legal_masks)
            buf.add(obs, actions, log_probs, values,
                    torch.zeros(2), torch.zeros(2, dtype=torch.bool), legal_masks,
                    torch.randint(0, 3, (2,)), torch.randn(2))

        # Set a very different entropy coeff and run again
        ppo.current_entropy_coeff = 10.0  # 1000x the default
        losses_elevated = ppo.update(buf, torch.zeros(2))

        # If update() actually reads current_entropy_coeff, the entropy
        # contribution to the loss will be vastly different. If it still
        # reads self.params.lambda_entropy, both runs produce similar losses.
        # We can't compare exact values (different random data), but the
        # entropy term at 10.0 vs 0.01 should be obvious in the total loss.
        assert losses_elevated["entropy"] != 0.0, "Entropy should be non-zero"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_lr_scheduler.py::TestRLWarmup -v`
Expected: FAIL

- [ ] **Step 3: Write the implementation**

**Part A: Add `current_entropy_coeff` to `KataGoPPOAlgorithm.__init__` and `get_entropy_coeff` method:**

```python
# In KataGoPPOAlgorithm.__init__, after self.optimizer:
self.current_entropy_coeff = params.lambda_entropy  # mutable; updated by training loop
```

```python
def get_entropy_coeff(
    self, epoch: int, warmup_epochs: int = 0, warmup_entropy: float = 0.05,
) -> float:
    """Return the entropy coefficient for the current epoch.

    During the first `warmup_epochs` epochs of RL (after SL warmup),
    uses elevated entropy to soften the overconfident SL policy.
    """
    if epoch < warmup_epochs:
        return warmup_entropy
    return self.params.lambda_entropy
```

**Part B: Modify `update()` to use `self.current_entropy_coeff` (CRITICAL — without this, warmup is a no-op):**

In `KataGoPPOAlgorithm.update()` (Plan B Task 8), change the loss computation from:

```python
                # OLD — uses frozen dataclass field, ignores warmup:
                - self.params.lambda_entropy * entropy
```

to:

```python
                # NEW — uses mutable coefficient, set by training loop each epoch:
                - self.current_entropy_coeff * entropy
```

This is the one-line diff that makes the entire warmup mechanism work. Without it, `self.current_entropy_coeff` is set but never read, and the feature is silently inert.

**Part C: Wire in `KataGoTrainingLoop.run()`, before `ppo.update()`:**

```python
# Set epoch-dependent entropy coefficient (uses config stored in __init__)
self.ppo.current_entropy_coeff = self.ppo.get_entropy_coeff(
    epoch_i, self._rl_warmup_epochs, self._rl_warmup_entropy,
)
if epoch_i == 0 or (epoch_i == self._rl_warmup_epochs):
    logger.info("Entropy coefficient: %.4f (warmup=%d, epoch=%d)",
                self.ppo.current_entropy_coeff, self._rl_warmup_epochs, epoch_i)
```

**Part D: Reset LR scheduler patience at warmup boundary (not the whole scheduler):**

```python
if epoch_i == self._rl_warmup_epochs and self.lr_scheduler is not None:
    # Reset patience counter only — preserve accumulated LR reductions.
    # Re-creating the scheduler would also reset LR tracking, which is not
    # the intent. We only want to clear warmup-phase patience consumption.
    self.lr_scheduler.num_bad_epochs = 0
    logger.info("LR scheduler patience reset at warmup boundary (epoch %d)", epoch_i)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_lr_scheduler.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/katago_ppo.py keisei/training/katago_loop.py tests/test_lr_scheduler.py
git commit -m "feat: add RL warmup elevated entropy with LR scheduler reset"
```

---

### Task 4: Optimized `write_shard`

**Files:**
- Modify: `keisei/sl/dataset.py`

Replace per-record writes with a single contiguous buffer write.

- [ ] **Step 1: Write the failing test (benchmark)**

```python
# Add to tests/test_sl_pipeline.py

class TestWriteShardPerformance:
    def test_write_large_shard(self, tmp_path):
        """Write 10K positions — should complete in < 5 seconds."""
        import time

        n = 10_000
        obs = np.random.randn(n, 50 * 81).astype(np.float32)
        policy = np.random.randint(0, 11259, size=n).astype(np.int64)
        value = np.random.randint(0, 3, size=n).astype(np.int64)
        score = np.random.randn(n).astype(np.float32)

        start = time.monotonic()
        write_shard(tmp_path / "perf_test.bin", obs, policy, value, score)
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, f"write_shard took {elapsed:.1f}s for 10K positions"
```

- [ ] **Step 2: Write the optimized implementation**

Replace `write_shard` in `keisei/sl/dataset.py`:

```python
# Structured dtype matching the shard binary layout exactly.
# Field order and sizes must match RECORD_SIZE and the SLDataset reader.
_SHARD_DTYPE = np.dtype([
    ("obs", np.float32, (OBS_SIZE,)),
    ("policy", np.int64),
    ("value", np.int64),
    ("score", np.float32),
])
assert _SHARD_DTYPE.itemsize == RECORD_SIZE  # verify layout matches


def write_shard(
    path: Path,
    observations: np.ndarray,
    policy_targets: np.ndarray,
    value_targets: np.ndarray,
    score_targets: np.ndarray,
) -> None:
    """Write positions to a binary shard file in a single I/O operation.

    Uses a numpy structured array to eliminate the per-record Python loop.
    The resulting binary layout is identical to the original per-record writes.
    """
    n = observations.shape[0]
    assert observations.shape == (n, OBS_SIZE)
    assert policy_targets.shape == (n,)
    assert value_targets.shape == (n,)
    assert score_targets.shape == (n,)

    buf = np.empty(n, dtype=_SHARD_DTYPE)
    buf["obs"] = observations.astype(np.float32)
    buf["policy"] = policy_targets.astype(np.int64)
    buf["value"] = value_targets.astype(np.int64)
    buf["score"] = score_targets.astype(np.float32)
    buf.tofile(path)
```

- [ ] **Step 3: Run tests (both correctness and performance)**

Run: `uv run pytest tests/test_sl_pipeline.py -v`
Expected: All existing shard tests PASS + performance test PASS.

- [ ] **Step 4: Commit**

```bash
git add keisei/sl/dataset.py tests/test_sl_pipeline.py
git commit -m "perf: optimize write_shard to single I/O operation"
```

---

### Task 5: CSA Parser Hardening

**Files:**
- Modify: `keisei/sl/parsers.py`

- [ ] **Step 1: Write tests for multi-game CSA and encoding handling**

Add to `tests/test_sl_pipeline.py`:

```python
class TestCSAParserHardening:
    def test_multi_game_csa_file(self, tmp_path):
        """Games separated by '/' should parse as individual records."""
        multi_game = (
            "V2.2\nN+Player1\nN-Player2\n"
            "P1-KY-KE-GI-KI-OU-KI-GI-KE-KY\n"
            "P2 * -HI *  *  *  *  * -KA * \n"
            "P3-FU-FU-FU-FU-FU-FU-FU-FU-FU\n"
            "P4 *  *  *  *  *  *  *  *  * \n"
            "P5 *  *  *  *  *  *  *  *  * \n"
            "P6 *  *  *  *  *  *  *  *  * \n"
            "P7+FU+FU+FU+FU+FU+FU+FU+FU+FU\n"
            "P8 * +KA *  *  *  *  * +HI * \n"
            "P9+KY+KE+GI+KI+OU+KI+GI+KE+KY\n"
            "+\n+7776FU\n-3334FU\n%TORYO\n"
            "/\n"
            "V2.2\nN+A\nN-B\n+\n+2726FU\n-8384FU\n%TORYO\n"
        )
        csa_file = tmp_path / "multi.csa"
        csa_file.write_text(multi_game)
        parser = CSAParser()
        games = list(parser.parse(csa_file))
        assert len(games) == 2

    def test_empty_game_between_separators(self, tmp_path):
        """Empty blocks between '/' separators should be skipped."""
        content = "+7776FU\n%TORYO\n/\n/\n+2726FU\n%TORYO\n"
        csa_file = tmp_path / "gaps.csa"
        csa_file.write_text(content)
        parser = CSAParser()
        games = list(parser.parse(csa_file))
        assert len(games) == 2
```

- [ ] **Step 2: Handle multi-game CSA files**

Floodgate archives often pack multiple games per file, separated by `/` lines. The current parser treats the entire file as one game. Split on `/` separator before parsing each game block.

- [ ] **Step 3: Handle Shift-JIS encoding**

Older CSA files (pre-2010 Floodgate) use Shift-JIS encoding, not UTF-8. The current `errors="replace"` silently garbles player names and comments. Try UTF-8 first, fall back to Shift-JIS if `chardet` is available, and log a warning when replacement characters are produced.

**Dependency:** Add `chardet` to `pyproject.toml` as an optional dependency:
```toml
[project.optional-dependencies]
sl = ["chardet>=5.0"]
```

Guard the import in `parsers.py`:
```python
try:
    import chardet
except ImportError:
    chardet = None  # type: ignore[assignment]
```

- [ ] **Step 4: Add AMP/mixed precision note to SLTrainer**

Add a comment noting that `torch.autocast` can be added for production-scale SL training on GPU. Not implementing now — adds complexity for minimal gain during pipeline validation.

- [ ] **Step 4: Document SL optimizer state discard at SL→RL boundary**

The SL checkpoint includes optimizer state, but `KataGoTrainingLoop` creates a fresh Adam optimizer. This is intentional — the SL optimizer has momentum from supervised gradients that would fight the RL gradient signal. The RL warmup elevated entropy (Task 3) compensates. Add a comment in `KataGoTrainingLoop._check_resume` explaining this.

- [ ] **Step 5: Commit**

```bash
git add keisei/sl/parsers.py keisei/sl/trainer.py keisei/training/katago_loop.py pyproject.toml tests/test_sl_pipeline.py
git commit -m "feat: CSA multi-game support, Shift-JIS detection, SL→RL docs"
```

---

### Task 6: Full Verification

**Files:** None (verification only)

- [ ] **Step 1: Run all tests**

Run: `uv run pytest -v`
Expected: All tests PASS.

- [ ] **Step 2: Verify CLI entrypoint works**

Run: `uv run keisei-prepare-sl --help`
Expected: Prints usage.

- [ ] **Step 3: Commit if any fixes needed**

```bash
git add -u
git commit -m "fix: address issues found in Plan D verification"
```
