"""
Parallel game execution framework for evaluation performance optimization.
Handles concurrent game execution with resource management and error handling.
"""

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..core import (
    AgentInfo,
    EvaluationContext,
    GameResult,
    OpponentInfo,
)

logger = logging.getLogger(__name__)


class ParallelGameTask:
    """Represents a single game execution task."""

    def __init__(
        self,
        task_id: str,
        agent_info: AgentInfo,
        opponent_info: OpponentInfo,
        context: EvaluationContext,
        game_executor: Callable,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.task_id = task_id
        self.agent_info = agent_info
        self.opponent_info = opponent_info
        self.context = context
        self.game_executor = game_executor
        self.metadata = metadata or {}
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.result: Optional[GameResult] = None
        self.error: Optional[Exception] = None


class ParallelGameExecutor:
    """
    Manages parallel execution of multiple games with resource constraints.
    """

    def __init__(
        self,
        max_concurrent_games: int = 4,
        max_memory_usage_mb: int = 2048,
        timeout_per_game_seconds: int = 300,
    ):
        self.max_concurrent_games = max_concurrent_games
        self.max_memory_usage_mb = max_memory_usage_mb
        self.timeout_per_game_seconds = timeout_per_game_seconds
        self.active_tasks: Dict[str, ParallelGameTask] = {}
        self.completed_tasks: List[ParallelGameTask] = []
        self.failed_tasks: List[ParallelGameTask] = []
        self._executor: Optional[ThreadPoolExecutor] = None
        self._memory_monitor_active = False

    def __enter__(self):
        self._executor = ThreadPoolExecutor(max_workers=self.max_concurrent_games)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

    async def execute_games_parallel(
        self,
        tasks: List[ParallelGameTask],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[List[GameResult], List[str]]:
        """
        Execute multiple games in parallel with resource management.

        Args:
            tasks: List of game tasks to execute
            progress_callback: Optional callback for progress updates (completed, total)

        Returns:
            Tuple of (successful_results, error_messages)
        """
        if not self._executor:
            raise RuntimeError("ParallelGameExecutor must be used as context manager")

        logger.info(
            f"Starting parallel execution of {len(tasks)} games with max {self.max_concurrent_games} concurrent"
        )

        successful_results: List[GameResult] = []
        error_messages: List[str] = []

        # Start memory monitoring
        self._start_memory_monitoring()

        try:
            loop = asyncio.get_event_loop()
            completed_count = 0

            async def _run_task(task: ParallelGameTask) -> None:
                """Run a single game task in the thread pool, tracking results."""
                nonlocal completed_count
                self.active_tasks[task.task_id] = task
                try:
                    result = await loop.run_in_executor(
                        self._executor, self._execute_single_game, task
                    )
                    if result:
                        successful_results.append(result)
                        self.completed_tasks.append(task)
                        logger.debug(f"Game {task.task_id} completed successfully")
                    else:
                        error_msg = f"Game {task.task_id} returned no result"
                        error_messages.append(error_msg)
                        self.failed_tasks.append(task)
                        logger.warning(error_msg)
                except Exception as e:
                    error_msg = f"Game {task.task_id} failed with error: {str(e)}"
                    error_messages.append(error_msg)
                    task.error = e
                    self.failed_tasks.append(task)
                    logger.error(error_msg, exc_info=True)
                finally:
                    self.active_tasks.pop(task.task_id, None)
                    completed_count += 1
                    if progress_callback:
                        progress_callback(completed_count, len(tasks))

            await asyncio.wait_for(
                asyncio.gather(*[_run_task(task) for task in tasks]),
                timeout=self.timeout_per_game_seconds * len(tasks),
            )

        finally:
            self._stop_memory_monitoring()

        logger.info(
            f"Parallel execution completed: {len(successful_results)} successful, {len(error_messages)} failed"
        )
        return successful_results, error_messages

    def _execute_single_game(self, task: ParallelGameTask) -> Optional[GameResult]:
        """Execute a single game task."""
        task.start_time = time.time()

        try:
            # Check if we should use in-memory evaluation
            use_in_memory = getattr(
                task.context.configuration, "enable_in_memory_evaluation", False
            )

            if (
                use_in_memory
                and hasattr(task.game_executor, "__self__")
                and hasattr(task.game_executor.__self__, "evaluate_step_in_memory")
            ):
                # Use in-memory evaluation if available
                logger.debug(f"Using in-memory evaluation for game {task.task_id}")
                try:
                    # Check if we're in an existing event loop
                    loop = asyncio.get_running_loop()
                    # Use asyncio.run_coroutine_threadsafe to run from thread pool
                    future = asyncio.run_coroutine_threadsafe(
                        task.game_executor.__self__.evaluate_step_in_memory(
                            task.agent_info, task.opponent_info, task.context
                        ),
                        loop,
                    )
                    result = future.result(timeout=300)  # 5 minutes timeout
                except RuntimeError:
                    # No running loop, safe to use asyncio.run()
                    result = asyncio.run(
                        task.game_executor.__self__.evaluate_step_in_memory(
                            task.agent_info, task.opponent_info, task.context
                        )
                    )
            else:
                # Use regular evaluation
                logger.debug(f"Using regular evaluation for game {task.task_id}")
                try:
                    # Check if we're in an existing event loop
                    loop = asyncio.get_running_loop()
                    # Use asyncio.run_coroutine_threadsafe to run from thread pool
                    future = asyncio.run_coroutine_threadsafe(
                        task.game_executor(
                            task.agent_info, task.opponent_info, task.context
                        ),
                        loop,
                    )
                    result = future.result(timeout=300)  # 5 minutes timeout
                except RuntimeError:
                    # No running loop, safe to use asyncio.run()
                    result = asyncio.run(
                        task.game_executor(
                            task.agent_info, task.opponent_info, task.context
                        )
                    )

            task.result = result
            return result

        except Exception as e:
            task.error = e
            logger.error(
                f"Error executing game {task.task_id}: {str(e)}", exc_info=True
            )
            return None

        finally:
            task.end_time = time.time()

    def _start_memory_monitoring(self):
        """Start monitoring memory usage during parallel execution."""
        self._memory_monitor_active = True
        # Implementation would track memory usage and potentially throttle
        # execution if memory usage gets too high
        logger.debug("Memory monitoring started")

    def _stop_memory_monitoring(self):
        """Stop memory monitoring."""
        self._memory_monitor_active = False
        logger.debug("Memory monitoring stopped")

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get statistics about the parallel execution."""
        total_tasks = len(self.completed_tasks) + len(self.failed_tasks)

        if self.completed_tasks:
            avg_duration = sum(
                (task.end_time or 0) - (task.start_time or 0)
                for task in self.completed_tasks
                if task.start_time and task.end_time
            ) / len(self.completed_tasks)
        else:
            avg_duration = 0

        return {
            "total_tasks": total_tasks,
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "success_rate": (
                len(self.completed_tasks) / total_tasks if total_tasks > 0 else 0
            ),
            "average_game_duration": avg_duration,
            "max_concurrent_games": self.max_concurrent_games,
        }


class BatchGameExecutor:
    """
    Executes games in batches for better resource management.
    """

    def __init__(
        self,
        batch_size: int = 8,
        max_concurrent_games: int = 4,
        timeout_per_batch_seconds: int = 600,
    ):
        self.batch_size = batch_size
        self.max_concurrent_games = max_concurrent_games
        self.timeout_per_batch_seconds = timeout_per_batch_seconds

    async def execute_games_in_batches(
        self,
        tasks: List[ParallelGameTask],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[List[GameResult], List[str]]:
        """
        Execute games in batches to manage memory and resources.

        Args:
            tasks: List of game tasks to execute
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (successful_results, error_messages)
        """
        all_results: List[GameResult] = []
        all_errors: List[str] = []

        # Split tasks into batches
        batches = [
            tasks[i : i + self.batch_size]
            for i in range(0, len(tasks), self.batch_size)
        ]

        logger.info(
            f"Executing {len(tasks)} games in {len(batches)} batches of size {self.batch_size}"
        )

        completed_games = 0
        total_games = len(tasks)

        for batch_idx, batch_tasks in enumerate(batches):
            logger.info(
                f"Executing batch {batch_idx + 1}/{len(batches)} with {len(batch_tasks)} games"
            )

            with ParallelGameExecutor(
                max_concurrent_games=self.max_concurrent_games,
                timeout_per_game_seconds=300,  # 5 minutes per game
            ) as executor:

                batch_results, batch_errors = await executor.execute_games_parallel(
                    batch_tasks,
                    progress_callback=None,  # We'll handle progress at batch level
                )

                all_results.extend(batch_results)
                all_errors.extend(batch_errors)

                completed_games += len(batch_tasks)
                if progress_callback:
                    progress_callback(completed_games, total_games)

                # Log batch statistics
                stats = executor.get_execution_stats()
                logger.info(
                    f"Batch {batch_idx + 1} completed: "
                    f"{stats['completed_tasks']}/{stats['total_tasks']} successful "
                    f"({stats['success_rate']:.2%} success rate)"
                )

        logger.info(
            f"All batches completed: {len(all_results)} successful games, {len(all_errors)} errors"
        )
        return all_results, all_errors


def create_parallel_game_tasks(
    agent_info: AgentInfo,
    opponents: List[OpponentInfo],
    games_per_opponent: int,
    context: EvaluationContext,
    game_executor: Callable,
) -> List[ParallelGameTask]:
    """
    Create parallel game tasks for a set of opponents.

    Args:
        agent_info: Information about the agent being evaluated
        opponents: List of opponents to play against
        games_per_opponent: Number of games to play against each opponent
        context: Evaluation context
        game_executor: Function to execute individual games

    Returns:
        List of ParallelGameTask objects
    """
    tasks = []

    for opponent in opponents:
        for game_idx in range(games_per_opponent):
            # Alternate agent color for balanced evaluation
            agent_plays_sente = game_idx < (games_per_opponent + 1) // 2

            # Create a copy of opponent info with game-specific metadata
            game_opponent = OpponentInfo.from_dict(opponent.to_dict())
            if not game_opponent.metadata:
                game_opponent.metadata = {}
            game_opponent.metadata.update(
                {
                    "agent_plays_sente_in_eval_step": agent_plays_sente,
                    "game_index": game_idx,
                    "total_games_vs_opponent": games_per_opponent,
                }
            )

            task_id = f"parallel_{context.session_id}_{opponent.name}_{game_idx}_{uuid.uuid4().hex[:8]}"

            task = ParallelGameTask(
                task_id=task_id,
                agent_info=agent_info,
                opponent_info=game_opponent,
                context=context,
                game_executor=game_executor,
                metadata={
                    "opponent_name": opponent.name,
                    "game_index": game_idx,
                    "agent_plays_sente": agent_plays_sente,
                },
            )

            tasks.append(task)

    return tasks
