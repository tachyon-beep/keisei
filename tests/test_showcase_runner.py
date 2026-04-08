"""Tests for the showcase sidecar runner."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from keisei.db import init_db
from keisei.showcase.db_ops import (
    queue_match,
    claim_next_match,
    read_active_showcase_game,
    read_showcase_moves_since,
    read_heartbeat,
    cleanup_orphaned_games,
    create_showcase_game,
)
from keisei.showcase.runner import ShowcaseRunner


@pytest.fixture
def db(tmp_path: Path) -> str:
    path = str(tmp_path / "test.db")
    init_db(path)
    return path


@pytest.fixture
def mock_spectator_env() -> MagicMock:
    """Mock SpectatorEnv that plays a 3-move game."""
    env = MagicMock()
    env.action_space_size = 13527  # property, not method

    move_count = 0
    def mock_step(action: int) -> dict:
        nonlocal move_count
        move_count += 1
        return {
            "board": [None] * 81,
            "hands": {"black": {}, "white": {}},
            "current_player": "white" if move_count % 2 == 1 else "black",
            "ply": move_count,
            "is_over": move_count >= 3,
            "result": "checkmate" if move_count >= 3 else "in_progress",
            "in_check": False,
            "sfen": "startpos",
            "move_history": [{"action": i, "notation": f"move{i}"} for i in range(1, move_count + 1)],
        }

    def mock_reset() -> dict:
        nonlocal move_count
        move_count = 0
        return {
            "board": [None] * 81,
            "hands": {"black": {}, "white": {}},
            "current_player": "black",
            "ply": 0,
            "is_over": False,
            "result": "in_progress",
            "in_check": False,
            "sfen": "startpos",
            "move_history": [],
        }

    env.step.side_effect = mock_step
    env.reset.side_effect = mock_reset
    env.legal_actions.return_value = [42, 100, 200]
    env.get_observation.return_value = np.zeros((46, 9, 9), dtype=np.float32)
    # is_over is a @property (#[getter]) on real SpectatorEnv — use PropertyMock
    type(env).is_over = property(lambda self: move_count >= 3)
    return env


@pytest.fixture
def mock_model() -> MagicMock:
    """Mock model returning tuple (policy_logits, value)."""
    model = MagicMock()
    model.training = False
    model.eval.return_value = model

    def mock_forward(obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = obs.shape[0]
        policy = torch.randn(batch, 13527)
        value = torch.tensor([[0.3]])
        return policy, value

    model.side_effect = mock_forward
    model.named_parameters.return_value = []
    return model


class TestShowcaseRunner:
    def test_cleanup_on_startup(self, db: str) -> None:
        """Runner cleans up orphaned games from previous crashes."""
        qid = queue_match(db, "e1", "e2", "normal")
        claim_next_match(db)
        create_showcase_game(db, queue_id=qid, entry_id_black="e1", entry_id_white="e2",
            elo_black=1500.0, elo_white=1480.0, name_black="A", name_white="B")

        runner = ShowcaseRunner(db_path=db)
        runner._startup_cleanup()

        assert read_active_showcase_game(db) is None

    def test_run_single_game(self, db: str, mock_spectator_env: MagicMock, mock_model: MagicMock) -> None:
        """Runner plays a complete game and writes moves to DB."""
        qid = queue_match(db, "e1", "e2", "normal")

        runner = ShowcaseRunner(db_path=db)

        # Seed numpy RNG for deterministic test (temperature sampling)
        np.random.seed(42)

        with patch.object(runner, "_create_env", return_value=mock_spectator_env), \
             patch.object(runner, "_load_models", return_value=(mock_model, mock_model, "resnet", "resnet")):
            match = claim_next_match(db)
            runner._run_game(match)

        game = read_active_showcase_game(db)
        assert game is None  # game completed

        # Moves were written — mock plays 3 moves before checkmate
        moves = read_showcase_moves_since(db, 1, since_ply=0)
        assert len(moves) == 3

    def test_heartbeat_written(self, db: str) -> None:
        runner = ShowcaseRunner(db_path=db)
        runner._write_heartbeat()
        hb = read_heartbeat(db)
        assert hb is not None

    def test_speed_from_queue(self, db: str) -> None:
        """Runner reads speed from the queue row."""
        runner = ShowcaseRunner(db_path=db)
        assert runner._get_delay("slow") == 4.0
        assert runner._get_delay("normal") == 2.0
        assert runner._get_delay("fast") == 0.5


class TestShowcaseIntegration:
    """Integration tests using real (tiny) models."""

    @pytest.fixture
    def tiny_model_checkpoint(self, tmp_path: Path) -> tuple[str, Path]:
        """Create a tiny MLP model checkpoint."""
        from keisei.training.model_registry import build_model
        arch = "mlp"
        params = {"hidden_sizes": [64, 64]}  # MLPParams.hidden_sizes has no default
        model = build_model(arch, params)
        ckpt = tmp_path / "tiny.pt"
        torch.save(model.state_dict(), ckpt)
        return arch, ckpt

    def test_inference_cpu_only(self, tiny_model_checkpoint: tuple[str, Path]) -> None:
        """Verify inference runs on CPU and produces valid output."""
        from keisei.showcase.inference import load_model_for_showcase, run_inference
        arch, ckpt = tiny_model_checkpoint
        model = load_model_for_showcase(ckpt, arch, {"hidden_sizes": [64, 64]})

        # All params should be on CPU
        for name, param in model.named_parameters():
            assert param.device == torch.device("cpu"), f"{name} on {param.device}"

        # Run inference
        obs = np.random.randn(46, 9, 9).astype(np.float32)  # SpectatorEnv channels
        policy, win_prob = run_inference(model, obs, arch)
        assert policy.shape[0] > 0
        assert 0.0 <= win_prob <= 1.0

    def test_cuda_not_available_in_showcase(self) -> None:
        """After enforce_cpu_only, CUDA should not be available."""
        from keisei.showcase.inference import enforce_cpu_only
        enforce_cpu_only(cpu_threads=1)
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
