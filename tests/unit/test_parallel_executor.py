"""Regression tests for parallel evaluation execution."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from keisei.evaluation.core.parallel_executor import (
    ParallelGameExecutor,
    ParallelGameTask,
)

pytestmark = pytest.mark.unit


class _ImmediateFuture:
    def __init__(self, value):
        self._value = value

    def result(self, timeout=None):
        return self._value


class _ImmediateExecutor:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def submit(self, fn):
        return _ImmediateFuture(fn())


class TestParallelExecutorLoopSafety:
    """Synchronous helpers should not deadlock when a loop is already running."""

    def test_execute_single_game_avoids_run_coroutine_threadsafe_on_active_loop(self):
        executor = ParallelGameExecutor(timeout_per_game_seconds=7)
        task = ParallelGameTask(
            task_id="game-1",
            agent_info=MagicMock(),
            opponent_info=MagicMock(),
            context=SimpleNamespace(
                configuration=SimpleNamespace(enable_in_memory_evaluation=False)
            ),
            game_executor=self._make_async_executor("ok"),
        )

        with patch(
            "keisei.evaluation.core.parallel_executor.asyncio.get_running_loop",
            return_value=object(),
        ), patch(
            "keisei.evaluation.core.parallel_executor.ThreadPoolExecutor",
            return_value=_ImmediateExecutor(),
        ), patch(
            "keisei.evaluation.core.parallel_executor.asyncio.run_coroutine_threadsafe"
        ) as run_threadsafe:
            result = executor._execute_single_game(task)

        assert result == "ok"
        run_threadsafe.assert_not_called()

    @staticmethod
    def _make_async_executor(result):
        async def _executor(agent_info, opponent_info, context):
            return result

        return _executor
