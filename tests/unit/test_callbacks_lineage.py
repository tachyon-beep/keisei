"""
Tests for lineage event emission from EvaluationCallback.

Validates that evaluation callbacks emit match_completed and model_promoted
events through ModelManager when evaluation results are available.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from keisei.lineage.event_schema import validate_event
from keisei.lineage.registry import LineageRegistry
from keisei.training.callbacks import EvaluationCallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_eval_results(win_rate=0.7, loss_rate=0.2, draw_rate=0.1, total_games=20):
    """Build a mock eval_results with summary_stats."""
    results = MagicMock()
    results.summary_stats.win_rate = win_rate
    results.summary_stats.loss_rate = loss_rate
    results.summary_stats.draw_rate = draw_rate
    results.summary_stats.total_games = total_games
    return results


def _make_trainer(tmp_path, lineage_enabled=True):
    """Build a mock trainer with model_manager and lineage registry."""
    trainer = MagicMock()
    trainer.run_name = "test-run"
    trainer.is_train_wandb_active = False

    # Metrics manager with controllable timestep
    trainer.metrics_manager.global_timestep = 999  # +1 in callback = 1000

    # Agent and model
    trainer.agent = MagicMock()
    trainer.agent.model = MagicMock()

    # Evaluation manager with opponent pool
    trainer.evaluation_manager.opponent_pool.sample.return_value = (
        str(tmp_path / "models" / "checkpoint_ts3000.pth")
    )

    # Model manager with real lineage methods (patched construction)
    from keisei.training.model_manager import ModelManager

    config = MagicMock()
    config.training.mixed_precision = False
    config.training.enable_torch_compile = False
    config.training.enable_compilation_benchmarking = False
    config.training.input_features = "core46"
    config.training.model_type = "resnet"
    config.training.tower_depth = 9
    config.training.tower_width = 256
    config.training.se_ratio = 0.25
    config.env.device = "cpu"
    config.env.num_actions_total = 13527

    args = MagicMock()
    args.input_features = None
    args.model = None
    args.tower_depth = None
    args.tower_width = None
    args.se_ratio = None
    args.resume = None

    import torch

    device = torch.device("cpu")

    with (
        patch.object(ModelManager, "_setup_feature_spec"),
        patch.object(ModelManager, "_setup_mixed_precision"),
        patch.object(ModelManager, "_setup_compilation_infrastructure"),
    ):
        mm = ModelManager(config, args, device, logger_func=lambda msg: None)

    registry = None
    if lineage_enabled:
        registry_path = tmp_path / "lineage.jsonl"
        registry = LineageRegistry(registry_path)
        mm.set_lineage_registry(registry, "test-run")

    trainer.model_manager = mm
    return trainer, registry


def _make_eval_cfg(tmp_path, elo_enabled=False):
    """Build a mock eval config."""
    cfg = MagicMock()
    cfg.enable_periodic_evaluation = True
    if elo_enabled:
        cfg.elo_registry_path = str(tmp_path / "elo_ratings.json")
    else:
        cfg.elo_registry_path = None
    return cfg


# ---------------------------------------------------------------------------
# EvaluationCallback emits match_completed
# ---------------------------------------------------------------------------


class TestEvalCallbackMatchCompleted:
    def test_emits_match_completed_on_evaluation(self, tmp_path):
        """Successful evaluation emits a match_completed lineage event."""
        trainer, registry = _make_trainer(tmp_path)
        eval_cfg = _make_eval_cfg(tmp_path)
        callback = EvaluationCallback(eval_cfg, interval=1000)

        eval_results = _make_eval_results(win_rate=0.65, loss_rate=0.25, total_games=20)

        with patch.object(
            trainer.evaluation_manager,
            "evaluate_current_agent",
            return_value=eval_results,
        ):
            callback.on_step_end(trainer)

        assert registry.event_count == 1
        event = registry.load_all()[0]
        assert event["event_type"] == "match_completed"
        assert event["payload"]["result"] == "win"
        assert event["payload"]["num_games"] == 20
        assert event["payload"]["win_rate"] == 0.65
        assert event["payload"]["opponent_model_id"] == "checkpoint_ts3000.pth"
        assert validate_event(event) == []

    def test_loss_result_recorded(self, tmp_path):
        trainer, registry = _make_trainer(tmp_path)
        eval_cfg = _make_eval_cfg(tmp_path)
        callback = EvaluationCallback(eval_cfg, interval=1000)

        eval_results = _make_eval_results(win_rate=0.2, loss_rate=0.7, total_games=10)

        with patch.object(
            trainer.evaluation_manager,
            "evaluate_current_agent",
            return_value=eval_results,
        ):
            callback.on_step_end(trainer)

        event = registry.load_all()[0]
        assert event["payload"]["result"] == "loss"

    def test_draw_result_recorded(self, tmp_path):
        trainer, registry = _make_trainer(tmp_path)
        eval_cfg = _make_eval_cfg(tmp_path)
        callback = EvaluationCallback(eval_cfg, interval=1000)

        eval_results = _make_eval_results(
            win_rate=0.5, loss_rate=0.5, total_games=10
        )

        with patch.object(
            trainer.evaluation_manager,
            "evaluate_current_agent",
            return_value=eval_results,
        ):
            callback.on_step_end(trainer)

        event = registry.load_all()[0]
        assert event["payload"]["result"] == "draw"

    def test_no_emission_when_no_opponent(self, tmp_path):
        """Initial evaluation (no opponent) should not emit match_completed."""
        trainer, registry = _make_trainer(tmp_path)
        trainer.evaluation_manager.opponent_pool.sample.return_value = None
        eval_cfg = _make_eval_cfg(tmp_path)
        callback = EvaluationCallback(eval_cfg, interval=1000)

        # Patch save_checkpoint to avoid file-system side effects
        trainer.model_manager.save_checkpoint = MagicMock(
            return_value=(True, "/fake/path")
        )

        eval_results = _make_eval_results(win_rate=0.8, total_games=10)
        with patch.object(
            trainer.evaluation_manager,
            "evaluate_current_agent",
            return_value=eval_results,
        ):
            callback.on_step_end(trainer)

        # No match_completed should be emitted for initial eval (no opponent)
        assert registry.event_count == 0

    def test_no_emission_without_registry(self, tmp_path):
        """Without lineage registry, evaluation still works, no events emitted."""
        trainer, _ = _make_trainer(tmp_path, lineage_enabled=False)
        eval_cfg = _make_eval_cfg(tmp_path)
        callback = EvaluationCallback(eval_cfg, interval=1000)

        eval_results = _make_eval_results()

        with patch.object(
            trainer.evaluation_manager,
            "evaluate_current_agent",
            return_value=eval_results,
        ):
            # Should not raise
            callback.on_step_end(trainer)

    def test_no_emission_when_eval_disabled(self, tmp_path):
        """When evaluation is disabled, no events should be emitted."""
        trainer, registry = _make_trainer(tmp_path)
        eval_cfg = MagicMock()
        eval_cfg.enable_periodic_evaluation = False
        callback = EvaluationCallback(eval_cfg, interval=1000)

        callback.on_step_end(trainer)

        assert registry.event_count == 0

    def test_timestep_in_event_payload(self, tmp_path):
        """The global_timestep in the event should be the callback timestep (+1)."""
        trainer, registry = _make_trainer(tmp_path)
        trainer.metrics_manager.global_timestep = 4999  # +1 = 5000
        eval_cfg = _make_eval_cfg(tmp_path)
        callback = EvaluationCallback(eval_cfg, interval=5000)

        eval_results = _make_eval_results()

        with patch.object(
            trainer.evaluation_manager,
            "evaluate_current_agent",
            return_value=eval_results,
        ):
            callback.on_step_end(trainer)

        event = registry.load_all()[0]
        assert "5000" in event["model_id"]


# ---------------------------------------------------------------------------
# EvaluationCallback emits model_promoted on Elo improvement
# ---------------------------------------------------------------------------


class TestEvalCallbackModelPromoted:
    def test_no_promotion_without_elo_registry(self, tmp_path):
        """Without elo_registry_path, no promotion event is emitted."""
        trainer, registry = _make_trainer(tmp_path)
        eval_cfg = _make_eval_cfg(tmp_path, elo_enabled=False)
        callback = EvaluationCallback(eval_cfg, interval=1000)

        eval_results = _make_eval_results(win_rate=0.8, loss_rate=0.1)

        with patch.object(
            trainer.evaluation_manager,
            "evaluate_current_agent",
            return_value=eval_results,
        ):
            callback.on_step_end(trainer)

        # Only match_completed, no model_promoted
        events = registry.load_all()
        types = [e["event_type"] for e in events]
        assert "model_promoted" not in types
        assert "match_completed" in types
