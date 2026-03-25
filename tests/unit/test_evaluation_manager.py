"""Regression tests for EvaluationManager runtime safety."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from keisei.evaluation.core_manager import EvaluationManager

pytestmark = pytest.mark.unit


def _make_manager():
    config = SimpleNamespace(enable_performance_safeguards=False)
    return EvaluationManager(config=config, run_name="test-run")


def _make_agent():
    agent = MagicMock()
    agent.name = "agent-under-test"
    agent.model = MagicMock()
    return agent


class TestEvaluateCurrentAgentModeRestoration:
    """Model training mode is restored even when evaluation fails."""

    def test_sync_path_restores_training_mode_when_runner_raises(self):
        manager = _make_manager()
        agent = _make_agent()
        evaluator = MagicMock()

        with patch(
            "keisei.evaluation.core_manager.EvaluatorFactory.create",
            return_value=evaluator,
        ), patch(
            "keisei.evaluation.core_manager.asyncio.run",
            side_effect=RuntimeError("evaluation failed"),
        ):
            with pytest.raises(RuntimeError, match="evaluation failed"):
                manager.evaluate_current_agent(agent)

        agent.model.eval.assert_called_once()
        agent.model.train.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_path_restores_training_mode_when_evaluation_raises(self):
        manager = _make_manager()
        agent = _make_agent()
        evaluator = MagicMock()
        evaluator.evaluate = AsyncMock(side_effect=RuntimeError("async evaluation failed"))

        with patch(
            "keisei.evaluation.core_manager.EvaluatorFactory.create",
            return_value=evaluator,
        ):
            with pytest.raises(RuntimeError, match="async evaluation failed"):
                await manager.evaluate_current_agent_async(agent)

        agent.model.eval.assert_called_once()
        agent.model.train.assert_called_once()
