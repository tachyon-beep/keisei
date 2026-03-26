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

    def test_run_coroutine_blocking_no_loop_uses_asyncio_run(self):
        """When no event loop is running, asyncio.run() is used directly."""
        executor = ParallelGameExecutor(timeout_per_game_seconds=5)

        async def coro():
            return "direct"

        with patch(
            "keisei.evaluation.core.parallel_executor.asyncio.get_running_loop",
            side_effect=RuntimeError,
        ), patch(
            "keisei.evaluation.core.parallel_executor.asyncio.run",
            return_value="direct",
        ) as mock_run:
            result = executor._run_coroutine_blocking(coro)

        assert result == "direct"
        mock_run.assert_called_once()

    def test_run_coroutine_blocking_timeout_cancels_future(self):
        """When the thread pool times out, the future is cancelled and TimeoutError raised."""
        executor = ParallelGameExecutor(timeout_per_game_seconds=1)

        async def slow_coro():
            return "never"

        mock_future = MagicMock()
        mock_future.result.side_effect = TimeoutError("timed out")
        mock_pool = MagicMock()
        mock_pool.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool.__exit__ = MagicMock(return_value=False)
        mock_pool.submit.return_value = mock_future

        with patch(
            "keisei.evaluation.core.parallel_executor.asyncio.get_running_loop",
            return_value=MagicMock(),
        ), patch(
            "keisei.evaluation.core.parallel_executor.ThreadPoolExecutor",
            return_value=mock_pool,
        ):
            with pytest.raises(TimeoutError):
                executor._run_coroutine_blocking(slow_coro)

        mock_future.cancel.assert_called_once()

    @staticmethod
    def _make_async_executor(result):
        async def _executor(agent_info, opponent_info, context):
            return result

        return _executor
