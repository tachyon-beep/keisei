"""Unit tests for keisei.training.parallel.self_play_worker.SelfPlayWorker."""

import queue
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from keisei.core.experience_buffer import Experience
from keisei.training.parallel.self_play_worker import SelfPlayWorker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_worker(**overrides):
    """Create a SelfPlayWorker with mock queues (no real multiprocessing)."""
    defaults = dict(
        worker_id=0,
        env_config={"seed": 42, "input_channels": 46, "num_actions": 13527},
        model_config={
            "model_type": "resnet",
            "input_channels": 46,
            "num_actions": 13527,
            "tower_depth": 9,
            "tower_width": 256,
            "se_ratio": 0.25,
        },
        parallel_config={"batch_size": 32, "timeout_seconds": 5.0},
        experience_queue=MagicMock(),
        model_queue=MagicMock(),
        control_queue=MagicMock(),
        seed_offset=1000,
    )
    defaults.update(overrides)
    # Use object.__new__ to skip mp.Process.__init__ side effects,
    # then call SelfPlayWorker.__init__ with mock queues.
    worker = object.__new__(SelfPlayWorker)
    SelfPlayWorker.__init__(worker, **defaults)
    return worker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def worker():
    return _make_worker()


@pytest.fixture
def small_model():
    """A tiny model for weight-update tests."""
    return nn.Sequential(nn.Linear(10, 5))


# ---------------------------------------------------------------------------
# TestSelfPlayWorkerInit
# ---------------------------------------------------------------------------


class TestSelfPlayWorkerInit:
    def test_worker_id_stored(self, worker):
        assert worker.worker_id == 0

    def test_seed_calculation(self):
        w = _make_worker(worker_id=3, seed_offset=500)
        # seed = env_config["seed"] + seed_offset + worker_id = 42 + 500 + 3
        assert w.seed == 545

    def test_initial_state(self, worker):
        assert worker.running is True
        assert worker.steps_collected == 0
        assert worker.games_played == 0

    def test_queues_stored(self, worker):
        assert worker.experience_queue is not None
        assert worker.model_queue is not None
        assert worker.control_queue is not None


# ---------------------------------------------------------------------------
# TestSetupWorker
# ---------------------------------------------------------------------------


class TestSetupWorker:
    @patch("keisei.training.parallel.self_play_worker.np.random.seed")
    @patch("keisei.training.parallel.self_play_worker.torch.manual_seed")
    @patch("keisei.training.models.model_factory")
    def test_seeds_set(self, mock_factory, mock_torch_seed, mock_np_seed):
        w = _make_worker(worker_id=2, seed_offset=1000)
        mock_model = MagicMock()
        mock_factory.return_value = mock_model
        w._setup_worker()
        expected_seed = 42 + 1000 + 2
        mock_np_seed.assert_called_once_with(expected_seed)
        mock_torch_seed.assert_called_once_with(expected_seed)

    @patch("keisei.training.models.model_factory")
    def test_game_created(self, mock_factory):
        mock_factory.return_value = MagicMock()
        w = _make_worker()
        w._setup_worker()
        from keisei.shogi.shogi_game import ShogiGame

        assert isinstance(w.game, ShogiGame)

    @patch("keisei.training.models.model_factory")
    def test_policy_mapper_created(self, mock_factory):
        mock_factory.return_value = MagicMock()
        w = _make_worker()
        w._setup_worker()
        from keisei.utils.utils import PolicyOutputMapper

        assert isinstance(w.policy_mapper, PolicyOutputMapper)

    @patch("keisei.training.models.model_factory")
    def test_model_created_via_factory(self, mock_factory):
        mock_model = MagicMock()
        mock_factory.return_value = mock_model
        w = _make_worker()
        w._setup_worker()
        mock_factory.assert_called_once_with(
            model_type="resnet",
            obs_shape=(46, 9, 9),
            num_actions=13527,
            tower_depth=9,
            tower_width=256,
            se_ratio=0.25,
        )

    @patch("keisei.training.models.model_factory")
    def test_model_in_eval_mode(self, mock_factory):
        mock_model = MagicMock()
        mock_factory.return_value = mock_model
        w = _make_worker()
        w._setup_worker()
        mock_model.eval.assert_called_once()


# ---------------------------------------------------------------------------
# TestHandleControlCommand
# ---------------------------------------------------------------------------


class TestHandleControlCommand:
    def test_stop_sets_running_false(self, worker):
        worker._handle_control_command({"command": "stop"})
        assert worker.running is False

    def test_reset_clears_current_obs(self, worker):
        worker.game = MagicMock()
        worker._current_obs = np.zeros((46, 9, 9))
        worker._handle_control_command({"command": "reset"})
        assert worker._current_obs is None

    def test_unknown_command_logs_warning(self, worker, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            worker._handle_control_command({"command": "xyz"})
        assert "unknown command" in caplog.text.lower()


# ---------------------------------------------------------------------------
# TestUpdateModel
# ---------------------------------------------------------------------------


class TestUpdateModel:
    def test_loads_uncompressed_weights(self, worker, small_model):
        worker.model = small_model
        # Build sync data in uncompressed format
        sd = small_model.state_dict()
        model_data = {}
        for key, tensor in sd.items():
            np_arr = tensor.cpu().numpy().copy()  # .copy() to break memory sharing
            model_data[key] = {
                "data": np_arr,
                "shape": np_arr.shape,
                "dtype": str(np_arr.dtype),
                "compressed": False,
            }

        original_weight = sd["0.weight"].clone()
        # Mutate the model weights first to confirm they get restored
        with torch.no_grad():
            for p in small_model.parameters():
                p.fill_(0.0)

        worker._update_model({"model_data": model_data})
        assert torch.allclose(small_model.state_dict()["0.weight"], original_weight)

    def test_loads_compressed_weights(self, worker, small_model):
        from keisei.training.parallel.utils import compress_array

        worker.model = small_model
        sd = small_model.state_dict()
        original_sd = {k: v.clone() for k, v in sd.items()}
        model_data = {}
        for key, tensor in sd.items():
            np_arr = tensor.cpu().numpy()
            model_data[key] = compress_array(np_arr)

        # Mutate model
        with torch.no_grad():
            for p in small_model.parameters():
                p.fill_(0.0)

        worker._update_model({"model_data": model_data})
        for key in original_sd:
            assert torch.allclose(small_model.state_dict()[key], original_sd[key])

    def test_no_model_logs_warning(self, worker, caplog):
        import logging

        worker.model = None
        with caplog.at_level(logging.WARNING):
            worker._update_model({"model_data": {}})
        assert "not initialized" in caplog.text.lower()

    def test_corrupted_data_logs_error(self, worker, small_model, caplog):
        import logging

        worker.model = small_model
        with caplog.at_level(logging.ERROR):
            worker._update_model({"model_data": {"bad": "data"}})
        assert "failed" in caplog.text.lower()


# ---------------------------------------------------------------------------
# TestExperiencesToBatch
# ---------------------------------------------------------------------------


class TestExperiencesToBatch:
    def _make_experience(self, obs_shape=(46, 9, 9), action=0):
        return Experience(
            obs=torch.randn(*obs_shape),
            action=action,
            reward=1.0,
            log_prob=-0.5,
            value=0.8,
            done=False,
            legal_mask=torch.ones(13527),
        )

    def test_correct_tensor_shapes(self, worker):
        exps = [self._make_experience(action=i) for i in range(4)]
        batch = worker._experiences_to_batch(exps)
        assert batch["obs"].shape == (4, 46, 9, 9)
        assert batch["actions"].shape == (4,)
        assert batch["rewards"].shape == (4,)
        assert batch["log_probs"].shape == (4,)
        assert batch["values"].shape == (4,)
        assert batch["dones"].shape == (4,)
        assert batch["legal_masks"].shape == (4, 13527)

    def test_correct_dtypes(self, worker):
        exps = [self._make_experience() for _ in range(2)]
        batch = worker._experiences_to_batch(exps)
        assert batch["actions"].dtype == torch.int64
        assert batch["rewards"].dtype == torch.float32
        assert batch["dones"].dtype == torch.bool

    def test_values_preserved(self, worker):
        exp = Experience(
            obs=torch.ones(46, 9, 9),
            action=42,
            reward=3.14,
            log_prob=-1.23,
            value=0.99,
            done=True,
            legal_mask=torch.zeros(13527),
        )
        batch = worker._experiences_to_batch([exp])
        assert batch["actions"].item() == 42
        assert batch["rewards"].item() == pytest.approx(3.14)
        assert batch["log_probs"].item() == pytest.approx(-1.23)
        assert batch["values"].item() == pytest.approx(0.99)
        assert batch["dones"].item() is True


# ---------------------------------------------------------------------------
# TestSendExperienceBatch
# ---------------------------------------------------------------------------


class TestSendExperienceBatch:
    def _make_experience(self):
        return Experience(
            obs=torch.randn(46, 9, 9),
            action=0,
            reward=0.0,
            log_prob=-0.1,
            value=0.5,
            done=False,
            legal_mask=torch.ones(13527),
        )

    def test_sends_to_experience_queue(self, worker):
        exps = [self._make_experience()]
        worker._send_experience_batch(exps)
        worker.experience_queue.put.assert_called_once()
        call_args = worker.experience_queue.put.call_args
        msg = call_args[0][0]
        assert msg["worker_id"] == 0
        assert "experiences" in msg
        assert msg["batch_size"] == 1

    def test_queue_full_drops_batch(self, worker, caplog):
        import logging

        worker.experience_queue.put.side_effect = queue.Full()
        exps = [self._make_experience()]
        with caplog.at_level(logging.WARNING):
            worker._send_experience_batch(exps)
        assert "full" in caplog.text.lower()

    def test_batch_message_structure(self, worker):
        exps = [self._make_experience(), self._make_experience()]
        worker._send_experience_batch(exps)
        msg = worker.experience_queue.put.call_args[0][0]
        assert "worker_id" in msg
        assert "experiences" in msg
        assert "batch_size" in msg
        assert "timestamp" in msg
        assert msg["batch_size"] == 2


# ---------------------------------------------------------------------------
# TestCheckControlCommands
# ---------------------------------------------------------------------------


class TestCheckControlCommands:
    def test_processes_all_pending_commands(self, worker):
        # Simulate 2 commands then Empty
        worker.control_queue.get_nowait.side_effect = [
            {"command": "reset"},
            {"command": "stop"},
            queue.Empty(),
        ]
        worker.game = MagicMock()
        worker._check_control_commands()
        assert worker.running is False  # stop was processed

    def test_empty_queue_no_error(self, worker):
        worker.control_queue.get_nowait.side_effect = queue.Empty()
        worker._check_control_commands()  # Should not raise


# ---------------------------------------------------------------------------
# TestCheckModelUpdates
# ---------------------------------------------------------------------------


class TestCheckModelUpdates:
    def test_applies_pending_update(self, worker, small_model):
        worker.model = small_model
        sd = small_model.state_dict()
        model_data = {}
        for key, tensor in sd.items():
            np_arr = tensor.cpu().numpy()
            model_data[key] = {
                "data": np_arr,
                "shape": np_arr.shape,
                "dtype": str(np_arr.dtype),
                "compressed": False,
            }
        worker.model_queue.get_nowait.side_effect = [
            {"model_data": model_data},
            queue.Empty(),
        ]
        worker._check_model_updates()
        # Should have been called without error — model still intact
        assert worker.model is not None

    def test_empty_queue_no_error(self, worker):
        worker.model_queue.get_nowait.side_effect = queue.Empty()
        worker._check_model_updates()  # Should not raise


# ---------------------------------------------------------------------------
# TestGetWorkerStats
# ---------------------------------------------------------------------------


class TestGetWorkerStats:
    def test_returns_expected_keys(self, worker):
        stats = worker.get_worker_stats()
        expected_keys = {"worker_id", "steps_collected", "games_played", "running", "seed"}
        assert set(stats.keys()) == expected_keys

    def test_initial_values(self, worker):
        stats = worker.get_worker_stats()
        assert stats["worker_id"] == 0
        assert stats["steps_collected"] == 0
        assert stats["games_played"] == 0
        assert stats["running"] is True
        assert stats["seed"] == 42 + 1000 + 0
