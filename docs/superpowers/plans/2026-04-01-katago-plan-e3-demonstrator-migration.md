# KataGo Plan E-3: Demonstrator, Evaluation CLI & Migration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add demonstrator exhibition games, the `keisei-evaluate` head-to-head CLI, migrate test imports from old modules, and delete the old training loop and PPO code (3-PR phasing: PR 1 add, PR 2 migrate, PR 3 delete).

**Architecture:** New `demonstrator.py` and `evaluate.py` modules. Test migration updates `conftest.py` and registry test imports. Deletion of `loop.py`, `ppo.py`, and their test files.

**Tech Stack:** Python 3.13, PyTorch, threading, argparse. Tests via `uv run pytest`.

**Dependencies:** Requires Plans A-D, E-1, and E-2 complete. Verify before starting:

```bash
uv run python -c "
from keisei.training.katago_loop import KataGoTrainingLoop, split_merge_step, main
from keisei.training.league import OpponentPool, OpponentSampler
from keisei.training.value_adapter import get_value_adapter
print('Plans E-1 and E-2 ready (including katago_loop.main)')
"
```

**Spec reference:** `docs/superpowers/specs/2026-04-01-plan-e-league-consolidation-design.md` — Demonstrator Games, Evaluation Entrypoint, Deletion Sequencing sections.

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `keisei/training/demonstrator.py` | `DemonstratorRunner` — threaded inference-only exhibition matches |
| Create | `keisei/training/evaluate.py` | `keisei-evaluate` CLI for head-to-head comparison |
| Create | `tests/test_demonstrator.py` | DemonstratorRunner tests |
| Create | `tests/test_evaluate.py` | Evaluation CLI tests |
| Create | `tests/test_gae.py` | GAE regression tests (migrated from test_ppo.py) — if not already created by E-1 |
| Modify | `tests/conftest.py` | Update imports: PPOAlgorithm → KataGoPPOAlgorithm, RolloutBuffer → KataGoRolloutBuffer |
| Modify | `tests/test_registries.py` | Update PPOParams → KataGoPPOParams |
| Modify | `tests/test_registry_gaps.py` | Same |
| Modify | `pyproject.toml` | Add `keisei-evaluate` entrypoint, rewire `keisei-train` |
| Delete | `keisei/training/loop.py` | Old training loop (PR 3 only) |
| Delete | `keisei/training/ppo.py` | Old PPO (PR 3 only — compute_gae already in gae.py) |
| Delete | `tests/test_loop.py` | PR 3 only — coverage in test_pipeline_consolidation.py |
| Delete | `tests/test_loop_gaps.py` | PR 3 only |
| Delete | `tests/test_ppo.py` | PR 3 only — GAE tests migrated to test_gae.py |
| Delete | `tests/test_ppo_gaps.py` | PR 3 only |

---

### Task 1: DemonstratorRunner

**Files:**
- Create: `keisei/training/demonstrator.py`
- Create: `tests/test_demonstrator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_demonstrator.py
"""Tests for the DemonstratorRunner — inference-only exhibition matches."""

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from keisei.training.demonstrator import DemonstratorRunner


def _make_mock_model():
    """Create a minimal mock model for demonstrator inference."""
    model = MagicMock()
    def forward(obs):
        batch = obs.shape[0]
        output = MagicMock()
        output.policy_logits = torch.randn(batch, 9, 9, 139)
        return output
    model.__call__ = forward
    model.eval = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)
    return model


def _make_mock_pool(num_entries=3):
    """Create a mock OpponentPool with fake entries."""
    from keisei.training.league import OpponentEntry
    pool = MagicMock()
    entries = [
        OpponentEntry(
            id=i, architecture="resnet", model_params={"hidden_size": 16},
            checkpoint_path=f"/fake/ckpt_{i}.pt", elo_rating=1000.0 + i * 50,
            created_epoch=i * 10, games_played=0, created_at="2026-01-01",
        )
        for i in range(num_entries)
    ]
    pool.list_entries.return_value = entries
    pool.load_opponent.return_value = _make_mock_model()
    pool.pin = MagicMock()
    pool.unpin = MagicMock()
    return pool


class TestDemonstratorRunner:
    def test_init_creates_runner(self):
        pool = _make_mock_pool()
        runner = DemonstratorRunner(
            pool=pool,
            db_path="/tmp/test.db",
            num_slots=3,
            moves_per_minute=600,  # fast for testing
            device="cpu",
        )
        assert runner.num_slots == 3
        assert not runner.is_alive()

    def test_start_and_stop(self):
        pool = _make_mock_pool()
        runner = DemonstratorRunner(
            pool=pool,
            db_path="/tmp/test.db",
            num_slots=1,
            moves_per_minute=6000,
            device="cpu",
        )
        runner.start()
        assert runner.is_alive()
        time.sleep(0.1)  # let thread run briefly
        runner.stop()
        runner.join(timeout=2.0)
        assert not runner.is_alive()

    def test_crash_is_non_fatal(self):
        """If the runner thread crashes, it should log and stop, not propagate."""
        pool = _make_mock_pool(num_entries=3)
        # Force a real crash by making load_opponent raise
        pool.load_opponent.side_effect = RuntimeError("simulated crash")
        runner = DemonstratorRunner(
            pool=pool,
            db_path="/tmp/test.db",
            num_slots=1,
            moves_per_minute=6000,
            device="cpu",
        )
        runner.start()
        runner.join(timeout=3.0)
        # Thread should have died (crash caught by run()'s try/except)
        assert not runner.is_alive(), "Thread should have stopped after crash"

    def test_slot_fallback_insufficient_entries(self):
        """With < 2 entries, slots should be inactive."""
        pool = _make_mock_pool(num_entries=1)
        runner = DemonstratorRunner(
            pool=pool,
            db_path="/tmp/test.db",
            num_slots=3,
            moves_per_minute=6000,
            device="cpu",
        )
        matchups = runner._select_matchups()
        # Need >= 2 entries for any slot
        assert len(matchups) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_demonstrator.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# keisei/training/demonstrator.py
"""DemonstratorRunner — threaded inference-only exhibition matches."""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from contextlib import nullcontext

from keisei.training.league import OpponentEntry, OpponentPool

logger = logging.getLogger(__name__)


def _get_policy_flat(model_output, batch_size: int) -> torch.Tensor:
    """Extract flat policy logits from either BaseModel (tuple) or KataGoBaseModel (dataclass).

    BaseModel.forward() returns (policy_logits, value) — policy already flat (B, action_space).
    KataGoBaseModel.forward() returns KataGoOutput — policy is (B, 9, 9, 139), needs reshape.
    """
    if isinstance(model_output, tuple):
        # BaseModel contract: (policy_logits, value)
        return model_output[0]
    else:
        # KataGoBaseModel contract: KataGoOutput with .policy_logits
        return model_output.policy_logits.reshape(batch_size, -1)


@dataclass
class DemoMatchup:
    """A demonstrator game matchup."""
    slot: int
    entry_a: OpponentEntry
    entry_b: OpponentEntry


class DemonstratorRunner(threading.Thread):
    """Runs inference-only exhibition matches in a background thread.

    Error policy: crashes are non-fatal. The main loop wraps run() in
    try/except, logs the exception, and stops. The training loop checks
    is_alive() at epoch boundaries and logs a WARNING if the thread died.
    """

    def __init__(
        self,
        pool: OpponentPool,
        db_path: str,
        num_slots: int = 3,
        moves_per_minute: int = 60,
        device: str = "cpu",
    ) -> None:
        super().__init__(daemon=True, name="DemonstratorRunner")
        self.pool = pool
        self.db_path = db_path
        self.num_slots = num_slots
        self.move_delay = 60.0 / max(moves_per_minute, 1)
        self.device = device
        self._stop_event = threading.Event()

        # CUDA stream isolation (if on GPU)
        self._stream = None
        if device.startswith("cuda") and torch.cuda.is_available():
            self._stream = torch.cuda.Stream()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        """Thread main loop — wrapped in try/except for non-fatal crash policy."""
        try:
            self._run_loop()
        except Exception:
            logger.exception("DemonstratorRunner crashed — thread stopping")

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            matchups = self._select_matchups()
            if not matchups:
                # Not enough pool entries for any matchup
                self._stop_event.wait(timeout=5.0)
                continue

            for matchup in matchups:
                if self._stop_event.is_set():
                    return
                try:
                    self._play_game(matchup)
                except Exception:
                    # Per-slot crash isolation: one bad matchup doesn't kill other slots
                    logger.exception("Demo slot %d game failed — skipping", matchup.slot)

    def _select_matchups(self) -> list[DemoMatchup]:
        """Select matchups for active demo slots based on pool state."""
        entries = self.pool.list_entries()
        if len(entries) < 2:
            return []

        matchups = []
        sorted_by_elo = sorted(entries, key=lambda e: e.elo_rating, reverse=True)

        # Slot 1: #1 Elo vs #2 Elo (the championship match)
        if len(sorted_by_elo) >= 2 and self.num_slots >= 1:
            matchups.append(DemoMatchup(
                slot=1, entry_a=sorted_by_elo[0], entry_b=sorted_by_elo[1]
            ))

        # Slot 2: Cross-architecture (if available), else random
        if self.num_slots >= 2 and len(entries) >= 2:
            archs = {}
            for e in entries:
                archs.setdefault(e.architecture, []).append(e)
            if len(archs) >= 2:
                arch_names = list(archs.keys())
                a = random.choice(archs[arch_names[0]])
                b = random.choice(archs[arch_names[1]])
                matchups.append(DemoMatchup(slot=2, entry_a=a, entry_b=b))
            else:
                pair = random.sample(entries, 2)
                matchups.append(DemoMatchup(slot=2, entry_a=pair[0], entry_b=pair[1]))

        # Slot 3: Random pairing
        if self.num_slots >= 3 and len(entries) >= 2:
            pair = random.sample(entries, 2)
            matchups.append(DemoMatchup(slot=3, entry_a=pair[0], entry_b=pair[1]))

        return matchups

    def _play_game(self, matchup: DemoMatchup) -> None:
        """Play a single demonstrator game to completion."""
        # Pin entries to prevent eviction during load
        self.pool.pin(matchup.entry_a.id)
        self.pool.pin(matchup.entry_b.id)
        try:
            model_a = self.pool.load_opponent(matchup.entry_a, device=self.device)
            model_b = self.pool.load_opponent(matchup.entry_b, device=self.device)
        except FileNotFoundError:
            logger.warning("Checkpoint missing for demo slot %d — skipping game", matchup.slot)
            return
        finally:
            self.pool.unpin(matchup.entry_a.id)
            self.pool.unpin(matchup.entry_b.id)

        logger.info(
            "Demo slot %d: %s (elo=%.0f) vs %s (elo=%.0f)",
            matchup.slot,
            matchup.entry_a.architecture, matchup.entry_a.elo_rating,
            matchup.entry_b.architecture, matchup.entry_b.elo_rating,
        )

        # Create a single-env VecEnv with the correct observation mode.
        # Must match the models' expected obs_channels — both models in the
        # pool are 50-channel post-Plan-A, so use "katago" mode.
        try:
            from shogi_gym import VecEnv
            env = VecEnv(
                num_envs=1, max_ply=512,
                observation_mode="katago", action_mode="spatial",
            )
        except ImportError:
            logger.warning("shogi_gym not available — demo slot %d inactive", matchup.slot)
            return

        reset_result = env.reset()
        obs = torch.from_numpy(np.asarray(reset_result.observations)).to(self.device)
        legal_masks = torch.from_numpy(np.asarray(reset_result.legal_masks)).to(self.device)
        current_player = 0  # Black starts

        models = [model_a, model_b]  # index by current_player
        done = False

        while not done and not self._stop_event.is_set():
            model = models[current_player]

            ctx = torch.cuda.stream(self._stream) if self._stream else nullcontext()
            with ctx:
                with torch.no_grad():
                    output = model(obs)
                    flat = _get_policy_flat(output, obs.shape[0])
                    masked = flat.masked_fill(~legal_masks, float("-inf"))
                    probs = F.softmax(masked, dim=-1)
                    action = torch.distributions.Categorical(probs).sample()

            step_result = env.step(action.tolist())
            done = bool(step_result.terminated[0] or step_result.truncated[0])

            obs = torch.from_numpy(np.asarray(step_result.observations)).to(self.device)
            legal_masks = torch.from_numpy(np.asarray(step_result.legal_masks)).to(self.device)
            current_player = int(step_result.current_players[0])

            # Pace the game for watchability
            time.sleep(self.move_delay)

        # Write final game state to DB with game_type="demo"
        try:
            from keisei.db import write_game_snapshots
            write_game_snapshots(self.db_path, [{
                "game_id": 1000 + matchup.slot,  # demo slots use IDs 1001-1003
                "board_json": "{}",  # simplified — real implementation would serialize board
                "hands_json": "{}",
                "current_player": str(current_player),
                "ply": int(step_result.step_metadata.ply_count[0]) if hasattr(step_result, 'step_metadata') else 0,
                "is_over": 1,
                "result": "demo_complete",
                "sfen": "",
                "in_check": 0,
                "move_history_json": "[]",
                "value_estimate": 0.0,
                "game_type": "demo",
                "demo_slot": matchup.slot,
            }])
        except Exception:
            logger.warning("Failed to write demo game snapshot for slot %d", matchup.slot)

        logger.info("Demo slot %d game completed", matchup.slot)
```

Note: `nullcontext` is from `contextlib` (Python 3.7+). Add `from contextlib import nullcontext` at the top.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_demonstrator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/demonstrator.py tests/test_demonstrator.py
git commit -m "feat: add DemonstratorRunner for inference-only exhibition matches"
```

---

### Task 2: Evaluation CLI

**Files:**
- Create: `keisei/training/evaluate.py`
- Create: `tests/test_evaluate.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py
"""Tests for the keisei-evaluate head-to-head CLI."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from keisei.training.evaluate import run_evaluation, EvalResult


class TestEvalResult:
    def test_win_rate(self):
        result = EvalResult(wins=60, losses=30, draws=10)
        assert result.total_games == 100
        assert abs(result.win_rate - 0.6) < 1e-6

    def test_elo_delta(self):
        result = EvalResult(wins=60, losses=30, draws=10)
        delta = result.elo_delta()
        # 60% win rate → positive Elo delta
        assert delta > 0

    def test_confidence_interval(self):
        result = EvalResult(wins=200, losses=150, draws=50)
        low, high = result.win_rate_ci(confidence=0.95)
        assert low < result.win_rate
        assert high > result.win_rate
        assert high - low < 0.15  # 400 games → CI < ±7.5%


class TestRunEvaluation:
    def test_returns_eval_result(self):
        """run_evaluation should return an EvalResult with W/L/D counts."""
        with patch("keisei.training.evaluate._play_evaluation_games") as mock_play:
            mock_play.return_value = EvalResult(wins=5, losses=3, draws=2)
            result = run_evaluation(
                checkpoint_a="/fake/a.pt",
                arch_a="resnet",
                checkpoint_b="/fake/b.pt",
                arch_b="se_resnet",
                games=10,
                max_ply=100,
            )
            assert result.total_games == 10
            assert result.wins == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# keisei/training/evaluate.py
"""keisei-evaluate: head-to-head evaluation between two checkpoints."""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from keisei.training.demonstrator import _get_policy_flat
from keisei.training.model_registry import build_model

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of a head-to-head evaluation."""
    wins: int
    losses: int
    draws: int

    @property
    def total_games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        if self.total_games == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.total_games

    def elo_delta(self) -> float:
        """Estimated Elo difference: positive means A is stronger."""
        wr = self.win_rate
        if wr <= 0.0 or wr >= 1.0:
            return float("inf") if wr >= 1.0 else float("-inf")
        return -400.0 * math.log10(1.0 / wr - 1.0)

    def win_rate_ci(self, confidence: float = 0.95) -> tuple[float, float]:
        """Wilson score confidence interval for win rate."""
        n = self.total_games
        if n == 0:
            return (0.0, 1.0)
        p = self.win_rate
        # z-score for confidence level
        z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
        return (max(0.0, centre - margin), min(1.0, centre + margin))


def run_evaluation(
    checkpoint_a: str,
    arch_a: str,
    checkpoint_b: str,
    arch_b: str,
    games: int = 400,
    max_ply: int = 500,
    params_a: dict | None = None,
    params_b: dict | None = None,
) -> EvalResult:
    """Run head-to-head evaluation between two checkpoints."""
    return _play_evaluation_games(
        checkpoint_a, arch_a, checkpoint_b, arch_b,
        games, max_ply, params_a or {}, params_b or {},
    )


def _play_evaluation_games(
    checkpoint_a: str, arch_a: str,
    checkpoint_b: str, arch_b: str,
    games: int, max_ply: int,
    params_a: dict, params_b: dict,
) -> EvalResult:
    """Play the actual games. Separated for testability (can be mocked)."""
    model_a = build_model(arch_a, params_a)
    model_a.load_state_dict(torch.load(checkpoint_a, map_location="cpu", weights_only=True))
    model_a.eval()

    model_b = build_model(arch_b, params_b)
    model_b.load_state_dict(torch.load(checkpoint_b, map_location="cpu", weights_only=True))
    model_b.eval()

    from shogi_gym import VecEnv

    wins, losses, draws = 0, 0, 0

    # Create VecEnv once (reuse via reset between games)
    env = VecEnv(num_envs=1, max_ply=max_ply,
                 observation_mode="katago", action_mode="spatial")

    for game_i in range(games):
        # Alternate sides: A plays Black on even games, White on odd
        a_is_black = (game_i % 2 == 0)
        models = [model_a, model_b] if a_is_black else [model_b, model_a]

        reset_result = env.reset()
        obs = torch.from_numpy(np.asarray(reset_result.observations))
        legal_masks = torch.from_numpy(np.asarray(reset_result.legal_masks))
        current_player = 0
        done = False

        while not done:
            model = models[current_player]
            with torch.no_grad():
                output = model(obs)
                flat = _get_policy_flat(output, obs.shape[0])
                masked = flat.masked_fill(~legal_masks, float("-inf"))
                probs = F.softmax(masked, dim=-1)
                action = torch.distributions.Categorical(probs).sample()

            step_result = env.step(action.tolist())
            done = bool(step_result.terminated[0] or step_result.truncated[0])
            obs = torch.from_numpy(np.asarray(step_result.observations))
            legal_masks = torch.from_numpy(np.asarray(step_result.legal_masks))
            current_player = int(step_result.current_players[0])

        reward = float(step_result.rewards[0])
        # Reward is from Black's perspective
        a_reward = reward if a_is_black else -reward
        if a_reward > 0:
            wins += 1
        elif a_reward < 0:
            losses += 1
        else:
            draws += 1

        if (game_i + 1) % 50 == 0:
            logger.info("Evaluation: %d/%d games (W=%d L=%d D=%d)",
                        game_i + 1, games, wins, losses, draws)

    return EvalResult(wins=wins, losses=losses, draws=draws)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Head-to-head evaluation of two checkpoints")
    parser.add_argument("--checkpoint-a", required=True, help="Path to checkpoint A")
    parser.add_argument("--arch-a", required=True, help="Architecture name for A")
    parser.add_argument("--checkpoint-b", required=True, help="Path to checkpoint B")
    parser.add_argument("--arch-b", required=True, help="Architecture name for B")
    parser.add_argument("--games", type=int, default=400,
                        help="Number of games (default 400; <200 produces low-precision Elo)")
    parser.add_argument("--max-ply", type=int, default=500)
    args = parser.parse_args()

    if args.games < 200:
        logger.warning("--games=%d is below 200: Elo estimates will have wide confidence intervals",
                        args.games)

    result = run_evaluation(
        checkpoint_a=args.checkpoint_a, arch_a=args.arch_a,
        checkpoint_b=args.checkpoint_b, arch_b=args.arch_b,
        games=args.games, max_ply=args.max_ply,
    )

    low, high = result.win_rate_ci()
    print(f"\nResults ({result.total_games} games):")
    print(f"  A wins: {result.wins}  losses: {result.losses}  draws: {result.draws}")
    print(f"  Win rate: {result.win_rate:.1%} (95% CI: [{low:.1%}, {high:.1%}])")
    print(f"  Elo delta: {result.elo_delta():+.0f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Register CLI entrypoint**

Add to `pyproject.toml` `[project.scripts]`:

```toml
keisei-evaluate = "keisei.training.evaluate:main"
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add keisei/training/evaluate.py tests/test_evaluate.py pyproject.toml
git commit -m "feat: add keisei-evaluate CLI for head-to-head checkpoint comparison"
```

---

### Task 3: Rewire `keisei-train` Entrypoint

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update the entrypoint**

Change in `pyproject.toml`:

```toml
# Old:
keisei-train = "keisei.training.loop:main"

# New:
keisei-train = "keisei.training.katago_loop:main"
```

This requires `katago_loop.py` to have a `main()` function. Verify it exists (Plan C creates it as part of `KataGoTrainingLoop`). If not, add one:

```python
def main() -> None:
    import argparse
    from keisei.config import load_config
    parser = argparse.ArgumentParser(description="Keisei training")
    parser.add_argument("config", type=Path, help="Path to TOML config file")
    args = parser.parse_args()
    config = load_config(args.config)
    loop = KataGoTrainingLoop(config)
    loop.run(num_epochs=1000, steps_per_epoch=config.training.max_ply)
```

- [ ] **Step 2: Verify entrypoint resolves**

Run: `uv run keisei-train --help`
Expected: Prints usage (or at least doesn't crash with ImportError)

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml keisei/training/katago_loop.py
git commit -m "feat: rewire keisei-train entrypoint to unified KataGoTrainingLoop"
```

---

### Task 4: Test Import Migration (PR 2)

**Files:**
- Modify: `tests/conftest.py`
- Modify: `tests/test_registries.py`
- Modify: `tests/test_registry_gaps.py`

This is PR 2 of the 3-PR sequence. All new tests from E-1/E-2/E-3 are already passing. Now update old test imports to point at the new modules.

**CRITICAL ORDERING:** `test_ppo_gaps.py` imports `fill_buffer` from conftest and uses a locally-created `PPOAlgorithm`. When conftest's `fill_buffer` is migrated to `KataGoPPOAlgorithm`, `test_ppo_gaps.py` breaks. Fix: also update `test_ppo_gaps.py` in this same PR to use `KataGoPPOAlgorithm`, OR keep a thin compatibility shim in `fill_buffer` that accepts either type until PR 3 deletes the file.

Also update the stale shape constants in conftest (`obs_shape=(46, 9, 9)` → `(50, 9, 9)`, `action_space=13527` → `11259`) to match post-Plan-A KataGoRolloutBuffer expectations.

- [ ] **Step 1: Update `conftest.py`**

Change imports from:
```python
from keisei.training.ppo import PPOAlgorithm, RolloutBuffer
```
to:
```python
from keisei.training.katago_ppo import KataGoPPOAlgorithm, KataGoRolloutBuffer
```

Update fixtures that create `PPOAlgorithm` → `KataGoPPOAlgorithm`:
```python
@pytest.fixture
def small_ppo(small_resnet):
    from keisei.training.katago_ppo import KataGoPPOParams
    params = KataGoPPOParams(learning_rate=1e-3, batch_size=4, epochs_per_batch=1)
    return KataGoPPOAlgorithm(params, small_resnet)
```

Update `fill_buffer` to use `KataGoRolloutBuffer` shape.

- [ ] **Step 2: Update `test_registries.py`**

Change:
```python
from keisei.training.algorithm_registry import PPOParams
```
to:
```python
from keisei.training.katago_ppo import KataGoPPOParams
```

Update any assertions that reference `PPOParams` → `KataGoPPOParams`.

- [ ] **Step 3: Update `test_registry_gaps.py`**

Same import change as above.

- [ ] **Step 4: Run all tests**

Run: `uv run pytest -v`
Expected: ALL tests pass — both old test files (now importing from new modules) and new test files.

This is the critical gate: if any old test fails after the import migration, fix it before proceeding to PR 3.

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/test_registries.py tests/test_registry_gaps.py
git commit -m "refactor: migrate test imports from old PPO/loop to KataGo equivalents"
```

---

### Task 5: Delete Old Code (PR 3)

**Files:**
- Delete: `keisei/training/loop.py`
- Delete: `keisei/training/ppo.py`
- Delete: `tests/test_loop.py`
- Delete: `tests/test_loop_gaps.py`
- Delete: `tests/test_ppo.py`
- Delete: `tests/test_ppo_gaps.py`

**CRITICAL SEQUENCING:** Only execute this task after Task 4's tests pass. This is a one-way door — once deleted, the old code is only recoverable from git history.

**Test coverage disposition (verify before deleting):**

| Old Test File | Coverage Target | New Location |
|---------------|-----------------|--------------|
| `test_loop.py` (10 tests) | Training loop integration | `test_pipeline_consolidation.py` (Plan E-2) |
| `test_loop_gaps.py` (6 tests) | Heartbeat, resume, VecEnv | `test_katago_loop.py` (Plan C/E-2) |
| `test_ppo.py` GAE tests (6 tests) | GAE computation | `test_gae.py` (Plan E-1) |
| `test_ppo.py` PPO tests (18 tests) | select_actions, update, entropy | `test_katago_ppo.py` (Plan B) |
| `test_ppo_gaps.py` (6 tests) | Edge cases, fill_buffer | `test_katago_ppo.py` (Plan B) |

A human should verify that the new test files cover the equivalent behavioral surface before signing off on PR 3.

- [ ] **Step 1: Verify no remaining imports from old modules**

Run: `uv run python -c "
import ast, sys
from pathlib import Path
old_imports = ['keisei.training.loop', 'keisei.training.ppo']
found = []
for py in Path('keisei').rglob('*.py'):
    text = py.read_text()
    for old in old_imports:
        if old in text:
            found.append((str(py), old))
for py in Path('tests').rglob('*.py'):
    text = py.read_text()
    for old in old_imports:
        if old in text:
            found.append((str(py), old))
if found:
    for f, imp in found:
        print(f'BLOCKED: {f} still imports {imp}')
    sys.exit(1)
else:
    print('No remaining imports from old modules — safe to delete')
"`
Expected: `No remaining imports from old modules — safe to delete`

- [ ] **Step 2: Delete old source files**

```bash
rm keisei/training/loop.py keisei/training/ppo.py
```

- [ ] **Step 3: Delete old test files**

```bash
rm tests/test_loop.py tests/test_loop_gaps.py tests/test_ppo.py tests/test_ppo_gaps.py
```

- [ ] **Step 4: Run all tests**

Run: `uv run pytest -v`
Expected: ALL tests PASS. No import errors, no missing fixtures.

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "chore: delete old TrainingLoop, PPOAlgorithm, and their tests

Replaced by KataGoTrainingLoop (unified loop with league support)
and KataGoPPOAlgorithm (multi-head PPO with value adapter).
compute_gae extracted to gae.py in Plan E-1."
```

---

### Task 6: Full Verification

**Files:** None (verification only)

- [ ] **Step 1: Run all tests**

Run: `uv run pytest -v`
Expected: ALL tests PASS.

- [ ] **Step 2: Verify CLI entrypoints**

Run: `uv run keisei-train --help`
Expected: Prints usage for the unified training loop.

Run: `uv run keisei-evaluate --help`
Expected: Prints evaluation CLI usage.

- [ ] **Step 3: Verify old modules are gone**

Run: `python -c "import keisei.training.loop" 2>&1`
Expected: `ModuleNotFoundError`

Run: `python -c "import keisei.training.ppo" 2>&1`
Expected: `ModuleNotFoundError`

- [ ] **Step 4: Commit if any fixes needed**

```bash
git add -u
git commit -m "fix: address issues found in Plan E-3 verification"
```
