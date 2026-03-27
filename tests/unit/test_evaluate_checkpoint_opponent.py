"""Tests for opponent_checkpoint parameter in EvaluationManager.

Verifies that evaluate_checkpoint() and evaluate_checkpoint_async() properly
accept and use the opponent_checkpoint parameter (not underscore-prefixed).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from keisei.evaluation.core_manager import EvaluationManager


@pytest.fixture
def mock_config():
    """Create a minimal mock EvaluationConfig."""
    config = MagicMock()
    config.strategy = "single"
    config.temp_agent_device = "cpu"
    config.model_weight_cache_size = 5
    config.enable_in_memory_evaluation = False
    config.max_concurrent_evaluations = 4
    config.evaluation_timeout_seconds = 300
    config.enable_performance_safeguards = False
    return config


@pytest.fixture
def manager(mock_config):
    """Create an EvaluationManager with safeguards disabled."""
    mgr = EvaluationManager(mock_config, run_name="test_run")
    mgr.setup(device="cpu", policy_mapper=None, model_dir="/tmp", wandb_active=False)
    return mgr


@pytest.fixture
def fake_agent_checkpoint(tmp_path):
    """Create a fake agent checkpoint file."""
    cp = tmp_path / "agent.pt"
    cp.write_bytes(b"fake")
    return str(cp)


@pytest.fixture
def fake_opponent_checkpoint(tmp_path):
    """Create a fake opponent checkpoint file."""
    cp = tmp_path / "opponent_model.pt"
    cp.write_bytes(b"fake")
    return str(cp)


class TestEvaluateCheckpointOpponent:
    """Tests for opponent_checkpoint in evaluate_checkpoint()."""

    def test_parameter_not_underscore_prefixed(self):
        """The parameter should be named opponent_checkpoint, not _opponent_checkpoint."""
        import inspect

        sig = inspect.signature(EvaluationManager.evaluate_checkpoint)
        params = list(sig.parameters.keys())
        assert "opponent_checkpoint" in params, (
            f"Expected 'opponent_checkpoint' parameter, got: {params}"
        )
        assert "_opponent_checkpoint" not in params, (
            "Parameter should not have underscore prefix"
        )

    @patch("keisei.evaluation.core_manager.EvaluatorFactory")
    @patch("keisei.evaluation.core_manager.torch.load" if False else "torch.load")
    def test_missing_opponent_file_raises(
        self, mock_torch_load, mock_factory, manager, fake_agent_checkpoint
    ):
        """Passing a nonexistent opponent checkpoint should raise FileNotFoundError."""
        mock_torch_load.return_value = {"model_state_dict": {}}
        with pytest.raises(FileNotFoundError, match="Opponent checkpoint not found"):
            manager.evaluate_checkpoint(
                fake_agent_checkpoint,
                opponent_checkpoint="/nonexistent/opponent.pt",
            )

    @patch("keisei.evaluation.core_manager.EvaluatorFactory")
    def test_valid_opponent_adds_info_to_context(
        self,
        mock_factory,
        manager,
        fake_agent_checkpoint,
        fake_opponent_checkpoint,
    ):
        """A valid opponent checkpoint should populate environment_info in the context."""
        # Mock torch.load for both checkpoints
        mock_result = MagicMock()
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate = AsyncMock(return_value=mock_result)
        mock_factory.create.return_value = mock_evaluator

        captured_context = {}

        async def capture_evaluate(agent_info, context):
            captured_context.update(context.environment_info)
            return mock_result

        mock_evaluator.evaluate = AsyncMock(side_effect=capture_evaluate)

        with patch("torch.load", return_value={"model_state_dict": {}}):
            manager.evaluate_checkpoint(
                fake_agent_checkpoint,
                opponent_checkpoint=fake_opponent_checkpoint,
            )

        assert "opponent_checkpoint" in captured_context
        assert captured_context["opponent_checkpoint"] == fake_opponent_checkpoint
        assert "opponent_info" in captured_context
        opp = captured_context["opponent_info"]
        assert opp.name == "opponent_model"
        assert opp.type == "ppo"
        assert opp.checkpoint_path == fake_opponent_checkpoint


class TestEvaluateCheckpointAsyncOpponent:
    """Tests for opponent_checkpoint in evaluate_checkpoint_async()."""

    def test_parameter_not_underscore_prefixed(self):
        """The parameter should be named opponent_checkpoint, not _opponent_checkpoint."""
        import inspect

        sig = inspect.signature(EvaluationManager.evaluate_checkpoint_async)
        params = list(sig.parameters.keys())
        assert "opponent_checkpoint" in params, (
            f"Expected 'opponent_checkpoint' parameter, got: {params}"
        )
        assert "_opponent_checkpoint" not in params, (
            "Parameter should not have underscore prefix"
        )

    @patch("keisei.evaluation.core_manager.EvaluatorFactory")
    def test_missing_opponent_file_raises(
        self, mock_factory, manager, fake_agent_checkpoint
    ):
        """Passing a nonexistent opponent checkpoint should raise FileNotFoundError."""
        with patch("torch.load", return_value={"model_state_dict": {}}):
            with pytest.raises(
                FileNotFoundError, match="Opponent checkpoint not found"
            ):
                asyncio.run(
                    manager.evaluate_checkpoint_async(
                        fake_agent_checkpoint,
                        opponent_checkpoint="/nonexistent/opponent.pt",
                    )
                )

    @patch("keisei.evaluation.core_manager.EvaluatorFactory")
    def test_valid_opponent_adds_info_to_context(
        self,
        mock_factory,
        manager,
        fake_agent_checkpoint,
        fake_opponent_checkpoint,
    ):
        """A valid opponent checkpoint should populate environment_info in the context."""
        mock_result = MagicMock()
        mock_evaluator = MagicMock()
        mock_factory.create.return_value = mock_evaluator

        captured_context = {}

        async def capture_evaluate(agent_info, context):
            captured_context.update(context.environment_info)
            return mock_result

        mock_evaluator.evaluate = AsyncMock(side_effect=capture_evaluate)

        with patch("torch.load", return_value={"model_state_dict": {}}):
            asyncio.run(
                manager.evaluate_checkpoint_async(
                    fake_agent_checkpoint,
                    opponent_checkpoint=fake_opponent_checkpoint,
                )
            )

        assert "opponent_checkpoint" in captured_context
        assert captured_context["opponent_checkpoint"] == fake_opponent_checkpoint
        assert "opponent_info" in captured_context
        opp = captured_context["opponent_info"]
        assert opp.name == "opponent_model"
        assert opp.type == "ppo"
        assert opp.checkpoint_path == fake_opponent_checkpoint
