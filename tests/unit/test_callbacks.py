"""Unit tests for keisei.training.callbacks: CheckpointCallback, EvaluationCallback, AsyncEvaluationCallback."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from keisei.training.callbacks import (
    AsyncEvaluationCallback,
    Callback,
    CheckpointCallback,
    EvaluationCallback,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trainer(global_timestep=0):
    """Build a minimal mock trainer for callback testing."""
    trainer = MagicMock()
    trainer.metrics_manager = MagicMock()
    trainer.metrics_manager.global_timestep = global_timestep
    trainer.metrics_manager.total_episodes_completed = 10
    trainer.metrics_manager.black_wins = 5
    trainer.metrics_manager.white_wins = 3
    trainer.metrics_manager.draws = 2
    trainer.agent = MagicMock()
    trainer.agent.model = MagicMock()
    trainer.model_manager = MagicMock()
    trainer.model_manager.save_checkpoint.return_value = (True, "/tmp/ckpt.pt")
    trainer.evaluation_manager = MagicMock()
    trainer.evaluation_manager.opponent_pool = MagicMock()
    trainer.evaluation_manager.opponent_pool.add_checkpoint = MagicMock()
    trainer.evaluation_manager.opponent_pool.sample.return_value = "/tmp/old.pt"
    trainer.evaluation_manager.evaluate_current_agent.return_value = MagicMock(
        summary_stats=MagicMock(win_rate=0.6, loss_rate=0.3, total_games=10)
    )
    trainer.log_both = MagicMock()
    trainer.run_name = "test_run"
    trainer.is_train_wandb_active = False
    trainer.session_manager = MagicMock()
    trainer.session_manager.run_artifact_dir = "/tmp/artifacts"
    trainer.evaluation_elo_snapshot = None
    return trainer


# ---------------------------------------------------------------------------
# Callback base class
# ---------------------------------------------------------------------------


class TestCallbackBase:
    """Tests for the Callback base class."""

    def test_on_step_end_is_noop(self):
        cb = Callback()
        # Should not raise
        cb.on_step_end(MagicMock())


# ---------------------------------------------------------------------------
# CheckpointCallback
# ---------------------------------------------------------------------------


class TestCheckpointCallback:
    """Tests for CheckpointCallback."""

    def test_fires_at_correct_interval(self):
        cb = CheckpointCallback(interval=100, model_dir="/tmp/models")
        trainer = _make_trainer(global_timestep=99)  # (99 + 1) % 100 == 0
        cb.on_step_end(trainer)
        trainer.model_manager.save_checkpoint.assert_called_once()

    def test_does_not_fire_before_interval(self):
        cb = CheckpointCallback(interval=100, model_dir="/tmp/models")
        trainer = _make_trainer(global_timestep=98)  # (98 + 1) % 100 == 99
        cb.on_step_end(trainer)
        trainer.model_manager.save_checkpoint.assert_not_called()

    def test_fires_at_first_interval(self):
        cb = CheckpointCallback(interval=10, model_dir="/tmp/models")
        trainer = _make_trainer(global_timestep=9)  # (9 + 1) % 10 == 0
        cb.on_step_end(trainer)
        trainer.model_manager.save_checkpoint.assert_called_once()

    def test_handles_missing_agent(self):
        cb = CheckpointCallback(interval=10, model_dir="/tmp/models")
        trainer = _make_trainer(global_timestep=9)
        trainer.agent = None
        cb.on_step_end(trainer)
        # Should log error and not crash
        trainer.log_both.assert_called()
        trainer.model_manager.save_checkpoint.assert_not_called()

    def test_adds_to_opponent_pool_on_success(self):
        cb = CheckpointCallback(interval=10, model_dir="/tmp/models")
        trainer = _make_trainer(global_timestep=9)
        cb.on_step_end(trainer)
        trainer.evaluation_manager.opponent_pool.add_checkpoint.assert_called_once_with(
            "/tmp/ckpt.pt"
        )

    def test_logs_error_on_save_failure(self):
        cb = CheckpointCallback(interval=10, model_dir="/tmp/models")
        trainer = _make_trainer(global_timestep=9)
        trainer.model_manager.save_checkpoint.return_value = (False, None)
        cb.on_step_end(trainer)
        # Should log error
        assert any(
            "[ERROR]" in str(call) for call in trainer.log_both.call_args_list
        )


# ---------------------------------------------------------------------------
# EvaluationCallback
# ---------------------------------------------------------------------------


class TestEvaluationCallback:
    """Tests for EvaluationCallback."""

    def _make_eval_cfg(self, enable=True):
        return SimpleNamespace(
            enable_periodic_evaluation=enable,
            elo_registry_path=None,
        )

    def test_fires_at_correct_interval(self):
        cfg = self._make_eval_cfg()
        cb = EvaluationCallback(cfg, interval=100)
        trainer = _make_trainer(global_timestep=99)
        cb.on_step_end(trainer)
        trainer.evaluation_manager.evaluate_current_agent.assert_called_once()

    def test_does_not_fire_when_disabled(self):
        cfg = self._make_eval_cfg(enable=False)
        cb = EvaluationCallback(cfg, interval=100)
        trainer = _make_trainer(global_timestep=99)
        cb.on_step_end(trainer)
        trainer.evaluation_manager.evaluate_current_agent.assert_not_called()

    def test_does_not_fire_before_interval(self):
        cfg = self._make_eval_cfg()
        cb = EvaluationCallback(cfg, interval=100)
        trainer = _make_trainer(global_timestep=50)
        cb.on_step_end(trainer)
        trainer.evaluation_manager.evaluate_current_agent.assert_not_called()

    def test_bootstrap_when_no_previous_checkpoints(self):
        cfg = self._make_eval_cfg()
        cb = EvaluationCallback(cfg, interval=10)
        trainer = _make_trainer(global_timestep=9)
        trainer.evaluation_manager.opponent_pool.sample.return_value = None
        cb.on_step_end(trainer)
        # Should still run evaluation (bootstrap)
        trainer.evaluation_manager.evaluate_current_agent.assert_called_once()
        # Should save initial checkpoint
        trainer.model_manager.save_checkpoint.assert_called_once()

    def test_handles_missing_agent(self):
        cfg = self._make_eval_cfg()
        cb = EvaluationCallback(cfg, interval=10)
        trainer = _make_trainer(global_timestep=9)
        trainer.agent = None
        cb.on_step_end(trainer)
        trainer.evaluation_manager.evaluate_current_agent.assert_not_called()

    def test_sets_model_to_eval_and_back_to_train(self):
        cfg = self._make_eval_cfg()
        cb = EvaluationCallback(cfg, interval=10)
        trainer = _make_trainer(global_timestep=9)
        cb.on_step_end(trainer)
        trainer.agent.model.eval.assert_called()
        trainer.agent.model.train.assert_called()

    def test_boundary_one_before_interval(self):
        cfg = self._make_eval_cfg()
        cb = EvaluationCallback(cfg, interval=100)
        trainer = _make_trainer(global_timestep=98)
        cb.on_step_end(trainer)
        trainer.evaluation_manager.evaluate_current_agent.assert_not_called()

    def test_boundary_one_after_interval(self):
        cfg = self._make_eval_cfg()
        cb = EvaluationCallback(cfg, interval=100)
        trainer = _make_trainer(global_timestep=100)
        cb.on_step_end(trainer)
        trainer.evaluation_manager.evaluate_current_agent.assert_not_called()


# ---------------------------------------------------------------------------
# AsyncEvaluationCallback
# ---------------------------------------------------------------------------


class TestAsyncEvaluationCallback:
    """Tests for AsyncEvaluationCallback."""

    def _make_eval_cfg(self, enable=True):
        return SimpleNamespace(
            enable_periodic_evaluation=enable,
            elo_registry_path=None,
        )

    @pytest.mark.asyncio
    async def test_returns_none_when_disabled(self):
        cfg = self._make_eval_cfg(enable=False)
        cb = AsyncEvaluationCallback(cfg, interval=10)
        trainer = _make_trainer(global_timestep=9)
        result = await cb.on_step_end_async(trainer)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_checkpoints(self):
        cfg = self._make_eval_cfg()
        cb = AsyncEvaluationCallback(cfg, interval=10)
        trainer = _make_trainer(global_timestep=9)
        trainer.evaluation_manager.opponent_pool.sample.return_value = None
        result = await cb.on_step_end_async(trainer)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_not_at_interval(self):
        cfg = self._make_eval_cfg()
        cb = AsyncEvaluationCallback(cfg, interval=100)
        trainer = _make_trainer(global_timestep=50)
        result = await cb.on_step_end_async(trainer)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_metrics_dict_on_success(self):
        cfg = self._make_eval_cfg()
        cb = AsyncEvaluationCallback(cfg, interval=10)
        trainer = _make_trainer(global_timestep=9)

        eval_result = MagicMock()
        eval_result.summary_stats.win_rate = 0.7
        eval_result.summary_stats.loss_rate = 0.2
        eval_result.summary_stats.total_games = 10
        eval_result.summary_stats.avg_game_length = 50.0

        async def mock_eval(agent):
            return eval_result

        trainer.evaluation_manager.evaluate_current_agent_async = mock_eval

        result = await cb.on_step_end_async(trainer)
        assert result is not None
        assert "evaluation/win_rate" in result
        assert result["evaluation/win_rate"] == 0.7
