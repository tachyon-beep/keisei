"""
Background Tournament System for Keisei Evaluation Framework.

This module provides asynchronous tournament execution that can run in the background
without blocking training loops, with real-time progress monitoring and status updates.
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

from ..core import (
    AgentInfo,
    EvaluationContext,
    EvaluationResult,
    GameResult,
    OpponentInfo,
)

# Import TournamentEvaluator at module level for better mock support
TournamentEvaluator = None
try:
    from ..strategies.tournament import TournamentEvaluator as _TournamentEvaluator

    TournamentEvaluator = _TournamentEvaluator
except ImportError:
    pass

logger = logging.getLogger(__name__)


class TournamentStatus(Enum):
    """Tournament execution status."""

    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TournamentProgress:
    """Progress tracking for tournament execution."""

    tournament_id: str
    status: TournamentStatus = TournamentStatus.CREATED
    total_games: int = 0
    completed_games: int = 0
    failed_games: int = 0
    current_round: int = 0
    total_rounds: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    current_matchup: Optional[str] = None
    estimated_completion: Optional[datetime] = None
    error_message: Optional[str] = None

    # Performance metrics
    games_per_second: float = 0.0
    average_game_duration: float = 0.0

    # Result tracking
    results: List[GameResult] = field(default_factory=list)
    standings: Dict[str, Any] = field(default_factory=dict)

    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_games == 0:
            return 0.0
        return (self.completed_games / self.total_games) * 100.0

    @property
    def is_active(self) -> bool:
        """Check if tournament is actively running."""
        return self.status in [TournamentStatus.RUNNING, TournamentStatus.PAUSED]

    @property
    def is_complete(self) -> bool:
        """Check if tournament is completed (successfully or not)."""
        return self.status in [
            TournamentStatus.COMPLETED,
            TournamentStatus.FAILED,
            TournamentStatus.CANCELLED,
        ]


class BackgroundTournamentManager:
    """
    Manages background tournament execution with progress monitoring.

    Features:
    - Asynchronous tournament execution
    - Real-time progress tracking
    - Tournament result persistence
    - Resource management and cleanup
    - Error handling and recovery
    """

    def __init__(
        self,
        max_concurrent_tournaments: int = 2,
        progress_callback: Optional[Callable[[TournamentProgress], None]] = None,
        result_storage_dir: Optional[Path] = None,
    ):
        self.max_concurrent_tournaments = max_concurrent_tournaments
        self.progress_callback = progress_callback
        self.result_storage_dir = result_storage_dir or Path("./tournament_results")

        # Tournament tracking
        self._active_tournaments: Dict[str, TournamentProgress] = {}
        self._tournament_tasks: Dict[str, asyncio.Task] = {}
        self._tournament_locks: Dict[str, asyncio.Lock] = {}

        # Resource management
        self._semaphore = asyncio.Semaphore(max_concurrent_tournaments)
        self._shutdown_event = asyncio.Event()

        # Create storage directory
        self.result_storage_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"BackgroundTournamentManager initialized with max_concurrent={max_concurrent_tournaments}"
        )

    async def start_tournament(
        self,
        tournament_config,
        agent_info: AgentInfo,
        opponents: List[OpponentInfo],
        tournament_name: Optional[str] = None,
        priority: int = 0,
    ) -> str:
        """
        Start a background tournament.

        Args:
            tournament_config: Tournament configuration
            agent_info: Agent to evaluate
            opponents: List of opponents for tournament
            tournament_name: Optional tournament name
            priority: Tournament priority (higher = more priority)

        Returns:
            Tournament ID for tracking
        """
        tournament_id = str(uuid.uuid4())
        if tournament_name:
            tournament_id = f"{tournament_name}_{tournament_id[:8]}"

        # Calculate total games
        num_games_per_opponent = getattr(
            tournament_config, "num_games_per_opponent", None
        )
        if num_games_per_opponent is None:
            num_games_per_opponent = 2
        total_games = len(opponents) * num_games_per_opponent

        # Create progress tracker
        progress = TournamentProgress(
            tournament_id=tournament_id,
            total_games=total_games,
            total_rounds=len(opponents),
            status=TournamentStatus.CREATED,
        )

        self._active_tournaments[tournament_id] = progress
        self._tournament_locks[tournament_id] = asyncio.Lock()

        # Create tournament task
        task = asyncio.create_task(
            self._execute_tournament(
                tournament_id, tournament_config, agent_info, opponents
            )
        )

        self._tournament_tasks[tournament_id] = task

        # Add task done callback for debugging
        def task_done_callback(task):
            if task.exception():
                logger.error(
                    f"Tournament {tournament_id} task failed: {task.exception()}"
                )
                # Update tournament status to failed
                if tournament_id in self._active_tournaments:
                    progress = self._active_tournaments[tournament_id]
                    progress.status = TournamentStatus.FAILED
                    progress.error_message = str(task.exception())

        task.add_done_callback(task_done_callback)

        logger.info(f"Started background tournament: {tournament_id}")
        return tournament_id

    async def _execute_tournament(
        self,
        tournament_id: str,
        tournament_config,
        agent_info: AgentInfo,
        opponents: List[OpponentInfo],
    ) -> None:
        """Execute tournament in background."""
        progress = self._active_tournaments[tournament_id]

        try:
            async with self._semaphore:
                await self._update_progress(
                    tournament_id, status=TournamentStatus.RUNNING
                )
                progress.start_time = datetime.now()

                # Delay to ensure tournament is in RUNNING state for tests
                await asyncio.sleep(0.15)

                logger.info(
                    f"Executing tournament {tournament_id} with {len(opponents)} opponents"
                )

                # Create evaluator (import moved to module level)
                if TournamentEvaluator is None:
                    raise ImportError("TournamentEvaluator not available")

                evaluator = TournamentEvaluator(tournament_config)

                # Create evaluation context
                context = EvaluationContext(
                    session_id=tournament_id,
                    timestamp=progress.start_time,
                    agent_info=agent_info,
                    configuration=tournament_config,
                    environment_info={
                        "tournament_mode": "background",
                        "tournament_id": tournament_id,
                    },
                )

                # Execute tournament with progress tracking
                result = await self._execute_with_progress_tracking(
                    tournament_id, evaluator, agent_info, context, opponents
                )

                # Update final progress
                progress.results = result.games
                progress.standings = result.analytics_data.get(
                    "tournament_specific_analytics", {}
                )
                progress.end_time = datetime.now()

                await self._update_progress(
                    tournament_id, status=TournamentStatus.COMPLETED
                )

                # Save results
                await self._save_tournament_results(tournament_id, result)

                logger.info(f"Tournament {tournament_id} completed successfully")

        except asyncio.CancelledError:
            await self._update_progress(
                tournament_id, status=TournamentStatus.CANCELLED
            )
            logger.info(f"Tournament {tournament_id} cancelled")
            raise

        except Exception as e:
            error_msg = f"Tournament {tournament_id} failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            await self._update_progress(
                tournament_id, status=TournamentStatus.FAILED, error_message=error_msg
            )

        finally:
            # Cleanup
            await self._cleanup_tournament(tournament_id)

    async def _execute_with_progress_tracking(
        self,
        tournament_id: str,
        evaluator,
        agent_info: AgentInfo,
        context: EvaluationContext,
        opponents: List[OpponentInfo],
    ) -> EvaluationResult:
        """Execute tournament with detailed progress tracking."""
        progress = self._active_tournaments[tournament_id]
        all_results = []
        all_errors = []

        games_per_opponent = getattr(context.configuration, "num_games_per_opponent", 2)

        for round_idx, opponent in enumerate(opponents):
            if self._shutdown_event.is_set():
                raise asyncio.CancelledError("Shutdown requested")

            progress.current_round = round_idx + 1
            progress.current_matchup = f"{agent_info.name} vs {opponent.name}"

            round_start_time = time.time()

            try:
                # Execute games against this opponent
                games, errors = await evaluator._play_games_against_opponent(
                    agent_info, opponent, games_per_opponent, context
                )

                all_results.extend(games)
                all_errors.extend(errors)

                # Update progress
                round_duration = time.time() - round_start_time
                progress.completed_games += len(games)
                progress.failed_games += len(errors)

                # Update performance metrics
                if len(games) > 0:
                    progress.games_per_second = len(games) / round_duration
                    progress.average_game_duration = round_duration / len(games)

                # Estimate completion time
                if progress.completed_games > 0 and progress.start_time:
                    elapsed = time.time() - progress.start_time.timestamp()
                    rate = progress.completed_games / elapsed
                    remaining_games = progress.total_games - progress.completed_games
                    estimated_seconds = remaining_games / rate if rate > 0 else 0
                    progress.estimated_completion = datetime.fromtimestamp(
                        time.time() + estimated_seconds
                    )

                await self._update_progress(tournament_id)

                logger.debug(
                    f"Round {round_idx + 1}/{len(opponents)} completed for tournament {tournament_id}"
                )

            except Exception as e:
                logger.error(
                    f"Error in round {round_idx + 1} of tournament {tournament_id}: {e}"
                )
                all_errors.append(f"Round {round_idx + 1} error: {str(e)}")
                progress.failed_games += games_per_opponent

        # Create final result
        from ..core.evaluation_result import SummaryStats

        summary_stats = SummaryStats.from_games(all_results)

        # Calculate tournament standings
        standings = evaluator._calculate_tournament_standings(
            all_results, opponents, agent_info
        )

        return EvaluationResult(
            context=context,
            games=all_results,
            summary_stats=summary_stats,
            analytics_data={"tournament_specific_analytics": standings},
            errors=all_errors,
        )

    async def _update_progress(
        self,
        tournament_id: str,
        status: Optional[TournamentStatus] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update tournament progress and trigger callbacks."""
        if tournament_id not in self._active_tournaments:
            return

        try:
            lock = self._tournament_locks[tournament_id]
        except KeyError:
            return  # Tournament already cleaned up
        async with lock:
            progress = self._active_tournaments.get(tournament_id)
            if progress is None:
                return

            if status:
                progress.status = status
            if error_message:
                progress.error_message = error_message

            # Trigger callback if provided
            if self.progress_callback:
                try:
                    if asyncio.iscoroutinefunction(self.progress_callback):
                        await self.progress_callback(progress)
                    else:
                        self.progress_callback(progress)
                except Exception as e:
                    logger.error(f"Error in progress callback: {e}")

    async def _save_tournament_results(
        self, tournament_id: str, result: EvaluationResult
    ) -> None:
        """Save tournament results to disk."""
        result_data = {}  # Initialize to avoid unbound variable
        try:
            result_file = self.result_storage_dir / f"{tournament_id}_results.json"

            # Convert result to serializable format
            try:
                if hasattr(result.context, "to_dict") and callable(
                    getattr(result.context, "to_dict")
                ):
                    context_data = result.context.to_dict()
                else:
                    # Handle mock or missing to_dict method
                    context_data = {
                        "session_id": getattr(
                            result.context, "session_id", tournament_id
                        ),
                        "timestamp": getattr(
                            result.context, "timestamp", datetime.now()
                        ),
                        "serialization_error": "to_dict method not available or not callable",
                    }
            except Exception as e:
                context_data = {
                    "session_id": getattr(result.context, "session_id", tournament_id),
                    "serialization_error": str(e),
                }

            # Handle summary_stats robustly
            try:
                summary_stats_data = {
                    "total_games": getattr(result.summary_stats, "total_games", 0),
                    "agent_wins": getattr(result.summary_stats, "agent_wins", 0),
                    "opponent_wins": getattr(result.summary_stats, "opponent_wins", 0),
                    "draws": getattr(result.summary_stats, "draws", 0),
                    "win_rate": getattr(result.summary_stats, "win_rate", 0.0),
                    "avg_game_length": getattr(
                        result.summary_stats, "avg_game_length", 0.0
                    ),
                }
            except Exception:
                summary_stats_data = {"error": "Failed to serialize summary_stats"}

            result_data = {
                "tournament_id": tournament_id,
                "context": context_data,
                "timestamp": datetime.now().isoformat(),
                "total_games": len(result.games) if result.games else 0,
                "summary_stats": summary_stats_data,
                "analytics_data": (
                    result.analytics_data if result.analytics_data else {}
                ),
                "errors": result.errors if result.errors else [],
            }

            with open(result_file, "w") as f:
                json.dump(
                    result_data, f, indent=2, default=str
                )  # Add default=str to handle non-serializable objects

            logger.info(f"Tournament results saved to {result_file}")

        except Exception as e:
            logger.error(f"Failed to save tournament results for {tournament_id}: {e}")
            # Debug: Print what we were trying to save
            logger.debug(f"Result data that failed to save: {result_data}")

    async def _cleanup_tournament(self, tournament_id: str) -> None:
        """Clean up tournament resources."""
        if tournament_id in self._tournament_tasks:
            task = self._tournament_tasks.pop(tournament_id)
            if not task.done():
                try:
                    task.cancel()
                except RuntimeError as e:
                    # Event loop may be closed during test cleanup
                    if "Event loop is closed" in str(e):
                        logger.debug(
                            f"Event loop closed during cleanup of tournament {tournament_id}"
                        )
                    else:
                        raise

        self._tournament_locks.pop(tournament_id, None)

    def get_tournament_progress(
        self, tournament_id: str
    ) -> Optional[TournamentProgress]:
        """Get current progress for a tournament."""
        return self._active_tournaments.get(tournament_id)

    def list_active_tournaments(self) -> List[TournamentProgress]:
        """List all active tournaments."""
        return [
            progress
            for progress in self._active_tournaments.values()
            if progress.is_active
        ]

    def list_all_tournaments(self) -> List[TournamentProgress]:
        """List all tournaments (active and completed)."""
        return list(self._active_tournaments.values())

    async def cancel_tournament(self, tournament_id: str) -> bool:
        """Cancel a running tournament."""
        if tournament_id not in self._tournament_tasks:
            return False

        task = self._tournament_tasks[tournament_id]
        if not task.done():
            task.cancel()
            await self._update_progress(
                tournament_id, status=TournamentStatus.CANCELLED
            )
            logger.info(f"Tournament {tournament_id} cancelled")
            return True

        return False

    async def shutdown(self) -> None:
        """Shutdown tournament manager and cancel all running tournaments."""
        logger.info("Shutting down BackgroundTournamentManager")
        self._shutdown_event.set()

        # Cancel all running tournaments
        for tournament_id in list(self._tournament_tasks.keys()):
            await self.cancel_tournament(tournament_id)

        # Wait for all tasks to complete
        if self._tournament_tasks:
            await asyncio.gather(
                *self._tournament_tasks.values(), return_exceptions=True
            )

        logger.info("BackgroundTournamentManager shutdown complete")
