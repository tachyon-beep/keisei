"""Tests for keisei.webui.streamlit_manager."""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from keisei.config_schema import WebUIConfig
from keisei.webui.streamlit_manager import StreamlitManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    return WebUIConfig(enabled=True, port=8501, host="localhost")


@pytest.fixture
def state_dir(tmp_path):
    return tmp_path / "webui_state"


@pytest.fixture
def manager(config, state_dir):
    return StreamlitManager(config, state_dir=state_dir)


@pytest.fixture
def mock_trainer():
    """Minimal mock trainer for snapshot building."""
    t = MagicMock()
    t.game = None  # No game — simplest case
    t.step_manager = None
    t.experience_buffer = None
    t.last_gradient_norm = 0.1

    mm = MagicMock()
    mm.global_timestep = 100
    mm.total_episodes_completed = 5
    mm.black_wins = 3
    mm.white_wins = 1
    mm.draws = 1
    mm.processing = False
    mm.get_hot_squares.return_value = []

    history = MagicMock()
    history.policy_losses = []
    history.value_losses = []
    history.entropies = []
    history.kl_divergences = []
    history.clip_fractions = []
    history.learning_rates = []
    history.episode_lengths = []
    history.episode_rewards = []
    history.win_rates_history = []
    mm.history = history
    t.metrics_manager = mm
    return t


# ---------------------------------------------------------------------------
# start()
# ---------------------------------------------------------------------------


class TestStart:
    @patch("keisei.webui.streamlit_manager.STREAMLIT_AVAILABLE", False)
    def test_returns_false_when_streamlit_unavailable(self, config, state_dir):
        mgr = StreamlitManager(config, state_dir=state_dir)
        assert mgr.start() is False

    @patch("keisei.webui.streamlit_manager.STREAMLIT_AVAILABLE", True)
    @patch("subprocess.Popen")
    def test_launches_subprocess(self, mock_popen, manager):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        assert manager.start() is True
        assert mock_popen.called
        # Verify the command includes streamlit run
        cmd = mock_popen.call_args[0][0]
        assert "streamlit" in cmd
        assert "run" in cmd

    @patch("keisei.webui.streamlit_manager.STREAMLIT_AVAILABLE", True)
    @patch("subprocess.Popen")
    def test_idempotent_when_already_running(self, mock_popen, manager):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        assert manager.start() is True
        assert manager.start() is True  # Second call should not re-launch
        assert mock_popen.call_count == 1


# ---------------------------------------------------------------------------
# stop()
# ---------------------------------------------------------------------------


class TestStop:
    @patch("keisei.webui.streamlit_manager.STREAMLIT_AVAILABLE", True)
    @patch("subprocess.Popen")
    def test_terminates_subprocess(self, mock_popen, manager):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc
        manager.start()

        manager.stop()
        mock_proc.terminate.assert_called_once()

    def test_stop_without_start_is_safe(self, manager):
        manager.stop()  # Should not raise


# ---------------------------------------------------------------------------
# update_progress / refresh_dashboard_panels
# ---------------------------------------------------------------------------


class TestStateWriting:
    def test_update_progress_writes_state_file(self, manager, mock_trainer, state_dir):
        manager.update_progress(mock_trainer, speed=50.0, pending_updates={})
        state_path = state_dir / "state.json"
        assert state_path.exists()
        with open(state_path) as f:
            data = json.load(f)
        assert data["speed"] == 50.0

    def test_refresh_dashboard_panels_writes_state(self, manager, mock_trainer, state_dir):
        manager.refresh_dashboard_panels(mock_trainer)
        state_path = state_dir / "state.json"
        assert state_path.exists()

    def test_rate_limiting_prevents_rapid_writes(self, manager, mock_trainer, state_dir):
        state_path = state_dir / "state.json"

        manager.update_progress(mock_trainer, speed=1.0, pending_updates={})
        assert state_path.exists()
        first_mtime = state_path.stat().st_mtime

        # Immediate second call should be rate-limited
        manager.update_progress(mock_trainer, speed=2.0, pending_updates={})
        second_mtime = state_path.stat().st_mtime
        assert first_mtime == second_mtime  # File not rewritten

    def test_write_after_interval(self, manager, mock_trainer, state_dir):
        state_path = state_dir / "state.json"
        manager._min_write_interval = 0.01  # Shorten for test

        manager.update_progress(mock_trainer, speed=1.0, pending_updates={})
        time.sleep(0.02)
        manager.update_progress(mock_trainer, speed=2.0, pending_updates={})

        with open(state_path) as f:
            data = json.load(f)
        assert data["speed"] == 2.0

    def test_state_file_is_valid_json(self, manager, mock_trainer, state_dir):
        manager.update_progress(mock_trainer, speed=99.0, pending_updates={"ep_metrics": "L:5 R:0.3"})
        state_path = state_dir / "state.json"
        with open(state_path) as f:
            data = json.load(f)
        assert "timestamp" in data
        assert "metrics" in data


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_default_state_dir(self, config):
        """When no state_dir is provided, uses cwd/.keisei_webui."""
        mgr = StreamlitManager(config)
        assert mgr._state_dir == Path.cwd() / ".keisei_webui"

    @patch("keisei.webui.streamlit_manager.STREAMLIT_AVAILABLE", True)
    @patch("subprocess.Popen", side_effect=OSError("no such file"))
    def test_start_handles_popen_failure(self, mock_popen, manager):
        assert manager.start() is False
