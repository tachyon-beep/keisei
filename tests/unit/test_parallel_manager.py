"""Unit tests for keisei.training.parallel.parallel_manager: ParallelManager."""

import time
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_parallel_config(
    *,
    num_workers=2,
    batch_size=32,
    enabled=True,
    max_queue_size=100,
    timeout_seconds=30,
    sync_interval=100,
    compression_enabled=False,
    worker_seed_offset=1000,
):
    """Build a minimal parallel config dict."""
    return {
        "num_workers": num_workers,
        "batch_size": batch_size,
        "enabled": enabled,
        "max_queue_size": max_queue_size,
        "timeout_seconds": timeout_seconds,
        "sync_interval": sync_interval,
        "compression_enabled": compression_enabled,
        "worker_seed_offset": worker_seed_offset,
    }


def _make_env_config():
    return {
        "device": "cpu",
        "input_channels": 46,
        "num_actions_total": 13527,
        "seed": 42,
        "input_features": "core46",
    }


def _make_model_config():
    return {
        "model_type": "resnet",
        "tower_depth": 2,
        "tower_width": 64,
        "se_ratio": 0.0,
        "obs_shape": (46, 9, 9),
        "num_actions": 13527,
    }


@pytest.fixture
def parallel_config():
    return _make_parallel_config()


@pytest.fixture
def env_config():
    return _make_env_config()


@pytest.fixture
def model_config():
    return _make_model_config()


# ===========================================================================
# TestInit
# ===========================================================================


class TestInit:
    """Tests for ParallelManager __init__."""

    @patch("keisei.training.parallel.parallel_manager.ModelSynchronizer")
    @patch("keisei.training.parallel.parallel_manager.WorkerCommunicator")
    def test_stores_configs(self, mock_comm_cls, mock_sync_cls, env_config, model_config, parallel_config):
        from keisei.training.parallel.parallel_manager import ParallelManager

        pm = ParallelManager(
            env_config=env_config,
            model_config=model_config,
            parallel_config=parallel_config,
            device="cpu",
        )
        assert pm.env_config is env_config
        assert pm.model_config is model_config
        assert pm.parallel_config is parallel_config

    @patch("keisei.training.parallel.parallel_manager.ModelSynchronizer")
    @patch("keisei.training.parallel.parallel_manager.WorkerCommunicator")
    def test_creates_communicator_and_sync(self, mock_comm_cls, mock_sync_cls, env_config, model_config, parallel_config):
        from keisei.training.parallel.parallel_manager import ParallelManager

        pm = ParallelManager(
            env_config=env_config,
            model_config=model_config,
            parallel_config=parallel_config,
            device="cpu",
        )
        mock_comm_cls.assert_called_once()
        mock_sync_cls.assert_called_once()
        assert pm.communicator is mock_comm_cls.return_value
        assert pm.model_sync is mock_sync_cls.return_value

    @patch("keisei.training.parallel.parallel_manager.ModelSynchronizer")
    @patch("keisei.training.parallel.parallel_manager.WorkerCommunicator")
    def test_initial_state(self, mock_comm_cls, mock_sync_cls, env_config, model_config, parallel_config):
        from keisei.training.parallel.parallel_manager import ParallelManager

        pm = ParallelManager(
            env_config=env_config,
            model_config=model_config,
            parallel_config=parallel_config,
            device="cpu",
        )
        assert pm.num_workers == 2
        assert pm.batch_size == 32
        assert pm.enabled is True
        assert pm.workers == []
        assert pm.is_running is False
        assert pm.total_steps_collected == 0
        assert pm.total_batches_received == 0


# ===========================================================================
# TestStartWorkers
# ===========================================================================


class TestStartWorkers:
    """Tests for start_workers."""

    @patch("keisei.training.parallel.parallel_manager.SelfPlayWorker")
    @patch("keisei.training.parallel.parallel_manager.ModelSynchronizer")
    @patch("keisei.training.parallel.parallel_manager.WorkerCommunicator")
    def test_creates_correct_number_of_workers(self, mock_comm_cls, mock_sync_cls, mock_worker_cls, env_config, model_config, parallel_config):
        from keisei.training.parallel.parallel_manager import ParallelManager

        mock_comm = mock_comm_cls.return_value
        mock_comm.experience_queues = [MagicMock(), MagicMock()]
        mock_comm.model_queues = [MagicMock(), MagicMock()]
        mock_comm.control_queues = [MagicMock(), MagicMock()]

        mock_worker = MagicMock()
        mock_worker.pid = 12345
        mock_worker_cls.return_value = mock_worker

        pm = ParallelManager(
            env_config=env_config,
            model_config=model_config,
            parallel_config=parallel_config,
            device="cpu",
        )

        mock_model = MagicMock(spec=nn.Module)
        result = pm.start_workers(mock_model)

        assert result is True
        assert len(pm.workers) == 2
        assert pm.is_running is True
        assert mock_worker.start.call_count == 2

    @patch("keisei.training.parallel.parallel_manager.ModelSynchronizer")
    @patch("keisei.training.parallel.parallel_manager.WorkerCommunicator")
    def test_returns_true_when_disabled(self, mock_comm_cls, mock_sync_cls, env_config, model_config):
        from keisei.training.parallel.parallel_manager import ParallelManager

        config = _make_parallel_config(enabled=False)
        pm = ParallelManager(
            env_config=env_config,
            model_config=model_config,
            parallel_config=config,
            device="cpu",
        )

        mock_model = MagicMock(spec=nn.Module)
        result = pm.start_workers(mock_model)

        assert result is True
        assert len(pm.workers) == 0

    @patch("keisei.training.parallel.parallel_manager.SelfPlayWorker")
    @patch("keisei.training.parallel.parallel_manager.ModelSynchronizer")
    @patch("keisei.training.parallel.parallel_manager.WorkerCommunicator")
    def test_returns_false_on_os_error(self, mock_comm_cls, mock_sync_cls, mock_worker_cls, env_config, model_config, parallel_config):
        from keisei.training.parallel.parallel_manager import ParallelManager

        mock_comm = mock_comm_cls.return_value
        mock_comm.experience_queues = [MagicMock(), MagicMock()]
        mock_comm.model_queues = [MagicMock(), MagicMock()]
        mock_comm.control_queues = [MagicMock(), MagicMock()]

        mock_worker = MagicMock()
        mock_worker.start.side_effect = OSError("cannot fork")
        mock_worker_cls.return_value = mock_worker

        pm = ParallelManager(
            env_config=env_config,
            model_config=model_config,
            parallel_config=parallel_config,
            device="cpu",
        )

        mock_model = MagicMock(spec=nn.Module)
        result = pm.start_workers(mock_model)

        assert result is False


# ===========================================================================
# TestCollectExperiences
# ===========================================================================


class TestCollectExperiences:
    """Tests for collect_experiences."""

    @patch("keisei.training.parallel.parallel_manager.ModelSynchronizer")
    @patch("keisei.training.parallel.parallel_manager.WorkerCommunicator")
    def test_collects_from_worker_queues(self, mock_comm_cls, mock_sync_cls, env_config, model_config, parallel_config):
        from keisei.training.parallel.parallel_manager import ParallelManager

        mock_comm = mock_comm_cls.return_value
        mock_comm.collect_experiences.return_value = [
            (0, {"experiences": [1, 2, 3], "batch_size": 3, "steps_collected": 3, "games_played": 1}),
        ]

        pm = ParallelManager(
            env_config=env_config,
            model_config=model_config,
            parallel_config=parallel_config,
            device="cpu",
        )
        pm.is_running = True

        mock_buffer = MagicMock()
        count = pm.collect_experiences(mock_buffer)

        assert count == 3
        mock_buffer.add_from_worker_batch.assert_called_once()
        assert pm.total_steps_collected == 3
        assert pm.total_batches_received == 1

    @patch("keisei.training.parallel.parallel_manager.ModelSynchronizer")
    @patch("keisei.training.parallel.parallel_manager.WorkerCommunicator")
    def test_returns_zero_when_not_enabled(self, mock_comm_cls, mock_sync_cls, env_config, model_config):
        from keisei.training.parallel.parallel_manager import ParallelManager

        config = _make_parallel_config(enabled=False)
        pm = ParallelManager(
            env_config=env_config,
            model_config=model_config,
            parallel_config=config,
            device="cpu",
        )

        mock_buffer = MagicMock()
        count = pm.collect_experiences(mock_buffer)
        assert count == 0

    @patch("keisei.training.parallel.parallel_manager.ModelSynchronizer")
    @patch("keisei.training.parallel.parallel_manager.WorkerCommunicator")
    def test_returns_zero_when_not_running(self, mock_comm_cls, mock_sync_cls, env_config, model_config, parallel_config):
        from keisei.training.parallel.parallel_manager import ParallelManager

        pm = ParallelManager(
            env_config=env_config,
            model_config=model_config,
            parallel_config=parallel_config,
            device="cpu",
        )
        # is_running defaults to False

        mock_buffer = MagicMock()
        count = pm.collect_experiences(mock_buffer)
        assert count == 0

    @patch("keisei.training.parallel.parallel_manager.ModelSynchronizer")
    @patch("keisei.training.parallel.parallel_manager.WorkerCommunicator")
    def test_handles_collection_error(self, mock_comm_cls, mock_sync_cls, env_config, model_config, parallel_config):
        from keisei.training.parallel.parallel_manager import ParallelManager

        mock_comm = mock_comm_cls.return_value
        mock_comm.collect_experiences.side_effect = RuntimeError("queue broken")

        pm = ParallelManager(
            env_config=env_config,
            model_config=model_config,
            parallel_config=parallel_config,
            device="cpu",
        )
        pm.is_running = True

        mock_buffer = MagicMock()
        count = pm.collect_experiences(mock_buffer)
        assert count == 0

    @patch("keisei.training.parallel.parallel_manager.ModelSynchronizer")
    @patch("keisei.training.parallel.parallel_manager.WorkerCommunicator")
    def test_returns_zero_for_empty_queues(self, mock_comm_cls, mock_sync_cls, env_config, model_config, parallel_config):
        from keisei.training.parallel.parallel_manager import ParallelManager

        mock_comm = mock_comm_cls.return_value
        mock_comm.collect_experiences.return_value = []

        pm = ParallelManager(
            env_config=env_config,
            model_config=model_config,
            parallel_config=parallel_config,
            device="cpu",
        )
        pm.is_running = True

        mock_buffer = MagicMock()
        count = pm.collect_experiences(mock_buffer)
        assert count == 0


# ===========================================================================
# TestSyncModelIfNeeded
# ===========================================================================


class TestSyncModelIfNeeded:
    """Tests for sync_model_if_needed."""

    @patch("keisei.training.parallel.parallel_manager.ModelSynchronizer")
    @patch("keisei.training.parallel.parallel_manager.WorkerCommunicator")
    def test_syncs_when_interval_reached(self, mock_comm_cls, mock_sync_cls, env_config, model_config, parallel_config):
        from keisei.training.parallel.parallel_manager import ParallelManager

        mock_sync = mock_sync_cls.return_value
        mock_sync.should_sync.return_value = True

        pm = ParallelManager(
            env_config=env_config,
            model_config=model_config,
            parallel_config=parallel_config,
            device="cpu",
        )
        pm.is_running = True

        mock_model = MagicMock(spec=nn.Module)
        result = pm.sync_model_if_needed(mock_model, current_step=200)

        assert result is True
        mock_comm_cls.return_value.send_model_weights.assert_called_once()

    @patch("keisei.training.parallel.parallel_manager.ModelSynchronizer")
    @patch("keisei.training.parallel.parallel_manager.WorkerCommunicator")
    def test_skips_when_interval_not_reached(self, mock_comm_cls, mock_sync_cls, env_config, model_config, parallel_config):
        from keisei.training.parallel.parallel_manager import ParallelManager

        mock_sync = mock_sync_cls.return_value
        mock_sync.should_sync.return_value = False

        pm = ParallelManager(
            env_config=env_config,
            model_config=model_config,
            parallel_config=parallel_config,
            device="cpu",
        )
        pm.is_running = True

        mock_model = MagicMock(spec=nn.Module)
        result = pm.sync_model_if_needed(mock_model, current_step=50)

        assert result is False

    @patch("keisei.training.parallel.parallel_manager.ModelSynchronizer")
    @patch("keisei.training.parallel.parallel_manager.WorkerCommunicator")
    def test_returns_false_when_not_running(self, mock_comm_cls, mock_sync_cls, env_config, model_config, parallel_config):
        from keisei.training.parallel.parallel_manager import ParallelManager

        pm = ParallelManager(
            env_config=env_config,
            model_config=model_config,
            parallel_config=parallel_config,
            device="cpu",
        )
        # is_running defaults to False

        mock_model = MagicMock(spec=nn.Module)
        result = pm.sync_model_if_needed(mock_model, current_step=200)
        assert result is False


# ===========================================================================
# TestStopWorkers
# ===========================================================================


class TestStopWorkers:
    """Tests for stop_workers."""

    @patch("keisei.training.parallel.parallel_manager.ModelSynchronizer")
    @patch("keisei.training.parallel.parallel_manager.WorkerCommunicator")
    def test_graceful_shutdown(self, mock_comm_cls, mock_sync_cls, env_config, model_config, parallel_config):
        from keisei.training.parallel.parallel_manager import ParallelManager

        pm = ParallelManager(
            env_config=env_config,
            model_config=model_config,
            parallel_config=parallel_config,
            device="cpu",
        )

        mock_worker = MagicMock()
        mock_worker.is_alive.return_value = False
        pm.workers = [mock_worker]
        pm.is_running = True

        pm.stop_workers()

        mock_comm_cls.return_value.send_control_command.assert_called_once_with("stop")
        mock_comm_cls.return_value.cleanup.assert_called_once()
        assert pm.workers == []
        assert pm.is_running is False

    @patch("keisei.training.parallel.parallel_manager.ModelSynchronizer")
    @patch("keisei.training.parallel.parallel_manager.WorkerCommunicator")
    def test_force_terminate_on_join_timeout(self, mock_comm_cls, mock_sync_cls, env_config, model_config, parallel_config):
        from keisei.training.parallel.parallel_manager import ParallelManager

        pm = ParallelManager(
            env_config=env_config,
            model_config=model_config,
            parallel_config=parallel_config,
            device="cpu",
        )

        mock_worker = MagicMock()
        mock_worker.is_alive.return_value = True  # Always alive → force terminate
        mock_worker.pid = 99999
        pm.workers = [mock_worker]
        pm.is_running = True

        pm.stop_workers()

        mock_worker.terminate.assert_called_once()
        assert pm.workers == []

    @patch("keisei.training.parallel.parallel_manager.ModelSynchronizer")
    @patch("keisei.training.parallel.parallel_manager.WorkerCommunicator")
    def test_noop_when_no_workers(self, mock_comm_cls, mock_sync_cls, env_config, model_config, parallel_config):
        from keisei.training.parallel.parallel_manager import ParallelManager

        pm = ParallelManager(
            env_config=env_config,
            model_config=model_config,
            parallel_config=parallel_config,
            device="cpu",
        )
        pm.workers = []

        # Should not raise or call anything
        pm.stop_workers()
        mock_comm_cls.return_value.send_control_command.assert_not_called()


# ===========================================================================
# TestIsHealthy
# ===========================================================================


class TestIsHealthy:
    """Tests for is_healthy."""

    @patch("keisei.training.parallel.parallel_manager.ModelSynchronizer")
    @patch("keisei.training.parallel.parallel_manager.WorkerCommunicator")
    def test_healthy_when_disabled(self, mock_comm_cls, mock_sync_cls, env_config, model_config):
        from keisei.training.parallel.parallel_manager import ParallelManager

        config = _make_parallel_config(enabled=False)
        pm = ParallelManager(
            env_config=env_config,
            model_config=model_config,
            parallel_config=config,
            device="cpu",
        )
        assert pm.is_healthy() is True

    @patch("keisei.training.parallel.parallel_manager.ModelSynchronizer")
    @patch("keisei.training.parallel.parallel_manager.WorkerCommunicator")
    def test_unhealthy_when_not_running(self, mock_comm_cls, mock_sync_cls, env_config, model_config, parallel_config):
        from keisei.training.parallel.parallel_manager import ParallelManager

        pm = ParallelManager(
            env_config=env_config,
            model_config=model_config,
            parallel_config=parallel_config,
            device="cpu",
        )
        pm.is_running = False
        assert pm.is_healthy() is False

    @patch("keisei.training.parallel.parallel_manager.ModelSynchronizer")
    @patch("keisei.training.parallel.parallel_manager.WorkerCommunicator")
    def test_unhealthy_when_worker_dead(self, mock_comm_cls, mock_sync_cls, env_config, model_config, parallel_config):
        from keisei.training.parallel.parallel_manager import ParallelManager

        pm = ParallelManager(
            env_config=env_config,
            model_config=model_config,
            parallel_config=parallel_config,
            device="cpu",
        )
        pm.is_running = True

        mock_worker = MagicMock()
        mock_worker.is_alive.return_value = False
        pm.workers = [mock_worker, mock_worker]  # 2 workers, both dead

        assert pm.is_healthy() is False


# ===========================================================================
# TestGetParallelStats
# ===========================================================================


class TestGetParallelStats:
    """Tests for get_parallel_stats."""

    @patch("keisei.training.parallel.parallel_manager.ModelSynchronizer")
    @patch("keisei.training.parallel.parallel_manager.WorkerCommunicator")
    def test_stats_structure(self, mock_comm_cls, mock_sync_cls, env_config, model_config, parallel_config):
        from keisei.training.parallel.parallel_manager import ParallelManager

        mock_sync = mock_sync_cls.return_value
        mock_sync.get_sync_stats.return_value = {"syncs": 0}

        pm = ParallelManager(
            env_config=env_config,
            model_config=model_config,
            parallel_config=parallel_config,
            device="cpu",
        )

        stats = pm.get_parallel_stats()

        assert "enabled" in stats
        assert "running" in stats
        assert "num_workers" in stats
        assert "total_steps_collected" in stats
        assert "total_batches_received" in stats
        assert "worker_stats" in stats
        assert "sync_stats" in stats
        assert "collection_rate" in stats

    @patch("keisei.training.parallel.parallel_manager.ModelSynchronizer")
    @patch("keisei.training.parallel.parallel_manager.WorkerCommunicator")
    def test_stats_include_queue_info_when_running(self, mock_comm_cls, mock_sync_cls, env_config, model_config, parallel_config):
        from keisei.training.parallel.parallel_manager import ParallelManager

        mock_comm = mock_comm_cls.return_value
        mock_comm.get_queue_info.return_value = {"queue_sizes": [0, 0]}
        mock_sync = mock_sync_cls.return_value
        mock_sync.get_sync_stats.return_value = {}

        pm = ParallelManager(
            env_config=env_config,
            model_config=model_config,
            parallel_config=parallel_config,
            device="cpu",
        )
        pm.is_running = True

        stats = pm.get_parallel_stats()
        assert stats["queue_info"] == {"queue_sizes": [0, 0]}


# ===========================================================================
# TestContextManager
# ===========================================================================


class TestContextManager:
    """Tests for __enter__/__exit__ behavior."""

    @patch("keisei.training.parallel.parallel_manager.ModelSynchronizer")
    @patch("keisei.training.parallel.parallel_manager.WorkerCommunicator")
    def test_enter_returns_self(self, mock_comm_cls, mock_sync_cls, env_config, model_config, parallel_config):
        from keisei.training.parallel.parallel_manager import ParallelManager

        pm = ParallelManager(
            env_config=env_config,
            model_config=model_config,
            parallel_config=parallel_config,
            device="cpu",
        )
        assert pm.__enter__() is pm

    @patch("keisei.training.parallel.parallel_manager.ModelSynchronizer")
    @patch("keisei.training.parallel.parallel_manager.WorkerCommunicator")
    def test_exit_stops_workers(self, mock_comm_cls, mock_sync_cls, env_config, model_config, parallel_config):
        from keisei.training.parallel.parallel_manager import ParallelManager

        pm = ParallelManager(
            env_config=env_config,
            model_config=model_config,
            parallel_config=parallel_config,
            device="cpu",
        )
        mock_worker = MagicMock()
        mock_worker.is_alive.return_value = False
        pm.workers = [mock_worker]
        pm.is_running = True

        pm.__exit__(None, None, None)

        assert pm.workers == []
        assert pm.is_running is False
