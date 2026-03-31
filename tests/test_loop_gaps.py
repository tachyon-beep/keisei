"""Gap-analysis tests for TrainingLoop: resume fidelity, heartbeat debounce,
missing vecenv attrs."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from keisei.config import AppConfig, load_config
from keisei.db import (
    init_db,
    read_metrics_since,
    read_training_state,
    update_heartbeat,
    write_training_state,
)
from keisei.training.loop import TrainingLoop

from conftest import make_mock_vecenv


# ===================================================================
# C3 — _check_resume overwrites training state with correct new values
# ===================================================================


class TestCheckResumeOverwrite:
    """When _check_resume falls through (no valid checkpoint), it must
    overwrite the DB row with the *current* config's values, not stale ones."""

    def test_overwrite_replaces_display_name(self, make_config) -> None:
        config = make_config(display_name="NewBot")
        init_db(config.display.db_path)

        # Pre-populate with an old training state
        write_training_state(config.display.db_path, {
            "config_json": '{"old": true}',
            "display_name": "OldBot",
            "model_arch": "resnet",
            "algorithm_name": "ppo",
            "started_at": "2025-01-01T00:00:00Z",
            "current_epoch": 42,
            "current_step": 9999,
        })

        mock_vecenv = make_mock_vecenv(num_envs=config.training.num_games)
        loop = TrainingLoop(config, vecenv=mock_vecenv)

        # Fresh start
        assert loop.epoch == 0

        # The DB row should now reflect the NEW config, not the old one
        state = read_training_state(config.display.db_path)
        assert state is not None
        assert state["display_name"] == "NewBot", (
            f"Expected 'NewBot', got '{state['display_name']}'"
        )
        assert state["model_arch"] == "resnet"
        assert state["algorithm_name"] == "ppo"
        # current_epoch should be reset to 0 (default for new state)
        assert state["current_epoch"] == 0

    def test_overwrite_updates_started_at(self, make_config) -> None:
        config = make_config()
        init_db(config.display.db_path)

        write_training_state(config.display.db_path, {
            "config_json": '{}',
            "display_name": "OldBot",
            "model_arch": "resnet",
            "algorithm_name": "ppo",
            "started_at": "2020-01-01T00:00:00Z",
            "current_epoch": 10,
            "current_step": 500,
        })

        mock_vecenv = make_mock_vecenv(num_envs=config.training.num_games)
        TrainingLoop(config, vecenv=mock_vecenv)

        state = read_training_state(config.display.db_path)
        assert state is not None
        # started_at should be refreshed (not 2020)
        assert state["started_at"] > "2020-01-01T00:00:00Z"


# ===================================================================
# H5 — Heartbeat 10-second debounce
# ===================================================================


class TestHeartbeatDebounce:
    """_maybe_update_heartbeat should fire at most once per 10 seconds."""

    def test_heartbeat_not_called_within_10s(self, make_config) -> None:
        config = make_config()
        mock_vecenv = make_mock_vecenv(num_envs=config.training.num_games)
        loop = TrainingLoop(config, vecenv=mock_vecenv)

        with patch("keisei.training.loop.update_heartbeat") as mock_hb:
            # Simulate time barely advancing (< 10s)
            with patch("keisei.training.loop.time") as mock_time:
                mock_time.monotonic.return_value = loop._last_heartbeat + 5.0
                loop._maybe_update_heartbeat()
                mock_hb.assert_not_called()

    def test_heartbeat_called_after_10s(self, make_config) -> None:
        config = make_config()
        mock_vecenv = make_mock_vecenv(num_envs=config.training.num_games)
        loop = TrainingLoop(config, vecenv=mock_vecenv)

        with patch("keisei.training.loop.update_heartbeat") as mock_hb:
            with patch("keisei.training.loop.time") as mock_time:
                mock_time.monotonic.return_value = loop._last_heartbeat + 11.0
                loop._maybe_update_heartbeat()
                mock_hb.assert_called_once_with(loop.db_path)

    def test_heartbeat_resets_timer(self, make_config) -> None:
        config = make_config()
        mock_vecenv = make_mock_vecenv(num_envs=config.training.num_games)
        loop = TrainingLoop(config, vecenv=mock_vecenv)

        with patch("keisei.training.loop.update_heartbeat"):
            with patch("keisei.training.loop.time") as mock_time:
                t = loop._last_heartbeat + 11.0
                mock_time.monotonic.return_value = t
                loop._maybe_update_heartbeat()
                # After firing, _last_heartbeat should be updated
                assert loop._last_heartbeat == t


# ===================================================================
# M5 — VecEnv without optional attributes
# ===================================================================


class TestVecEnvMissingAttrs:
    """The training loop accesses draw_rate, truncation_rate,
    mean_episode_length, and reset_stats via getattr/hasattr.
    A vecenv that lacks these must not crash the loop."""

    @staticmethod
    def _make_bare_vecenv(num_envs: int) -> MagicMock:
        """VecEnv mock without optional stat attributes."""
        mock = MagicMock(spec=[
            "reset", "step", "num_envs",
        ])
        mock.num_envs = num_envs

        reset_result = MagicMock()
        reset_result.observations = np.zeros(
            (num_envs, 46, 9, 9), dtype=np.float32
        )
        reset_result.legal_masks = np.ones((num_envs, 13527), dtype=bool)
        mock.reset.return_value = reset_result

        step_result = MagicMock()
        step_result.observations = np.zeros(
            (num_envs, 46, 9, 9), dtype=np.float32
        )
        step_result.legal_masks = np.ones((num_envs, 13527), dtype=bool)
        step_result.rewards = np.zeros(num_envs, dtype=np.float32)
        step_result.terminated = np.zeros(num_envs, dtype=bool)
        step_result.truncated = np.zeros(num_envs, dtype=bool)
        mock.step.return_value = step_result

        return mock

    def test_loop_runs_without_reset_stats(self, make_config) -> None:
        config = make_config()
        bare = self._make_bare_vecenv(num_envs=config.training.num_games)
        loop = TrainingLoop(config, vecenv=bare)
        loop.run(num_epochs=1, steps_per_epoch=4)

        rows = read_metrics_since(config.display.db_path, since_id=0)
        assert len(rows) == 1
        # Optional metrics should be None when attrs missing
        assert rows[0]["draw_rate"] is None
        assert rows[0]["truncation_rate"] is None
        assert rows[0]["avg_episode_length"] is None
