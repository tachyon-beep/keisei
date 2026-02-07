"""
Benchmark evaluation strategy implementation.
Evaluates against a fixed set of benchmark opponents/scenarios.
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import torch

from keisei.shogi.shogi_core_definitions import Color
from keisei.shogi.shogi_game import ShogiGame
from keisei.utils import PolicyOutputMapper
from keisei.utils.agent_loading import initialize_opponent, load_evaluation_agent

from ..core import (
    AgentInfo,
    BaseEvaluator,
    EvaluationContext,
    EvaluationResult,
    GameResult,
    OpponentInfo,
    SummaryStats,
)
from keisei.config_schema import EvaluationConfig

# Define constants for termination reasons
TERMINATION_REASON_MAX_MOVES = "Max moves reached"
TERMINATION_REASON_ILLEGAL_MOVE = "Illegal/No move"
TERMINATION_REASON_MOVE_EXECUTION_ERROR = "Move execution error"
TERMINATION_REASON_ACTION_SELECTION_ERROR = "Action selection error"
TERMINATION_REASON_NO_LEGAL_MOVES_UNDETERMINED = (
    "No Legal Moves (Outcome Undetermined by ShogiGame)"
)
TERMINATION_REASON_GAME_ENDED_UNSPECIFIED = (
    "Game ended (Reason not specified by ShogiGame)"
)
TERMINATION_REASON_GAME_ENDED_UNSPECIFIED_FALLBACK = (
    "Game ended (Reason not specified by ShogiGame) - Fallback"
)
TERMINATION_REASON_UNKNOWN_LOOP_TERMINATION = "Unknown (Loop terminated unexpectedly)"
TERMINATION_REASON_EVAL_STEP_ERROR = "Benchmark evaluate_step error"


class BenchmarkEvaluator(BaseEvaluator):
    """Evaluates agent against a fixed suite of benchmark opponents or scenarios."""

    def __init__(self, config: EvaluationConfig):  # type: ignore
        super().__init__(config)
        self.config: EvaluationConfig = config  # type: ignore
        self.benchmark_suite: List[OpponentInfo] = []
        self.policy_mapper = PolicyOutputMapper()
        if not hasattr(self, "logger") or self.logger is None:
            self.logger = logging.getLogger(__name__)

    def get_opponents(self, context: EvaluationContext) -> List[OpponentInfo]:
        """Get benchmark suite opponents from configuration."""
        # Get suite configuration from strategy parameters
        suite_config = self.config.get_strategy_param("suite_config", [])

        opponents = []
        if not suite_config:
            # Provide default benchmark suite
            opponents = [
                OpponentInfo(
                    name="benchmark_random",
                    type="random",
                    checkpoint_path=None,
                    metadata={"difficulty": "low", "description": "Random baseline"},
                ),
                OpponentInfo(
                    name="benchmark_heuristic",
                    type="heuristic",
                    checkpoint_path=None,
                    metadata={
                        "difficulty": "medium",
                        "description": "Heuristic baseline",
                    },
                ),
            ]
        else:
            for i, benchmark_case_data in enumerate(suite_config):
                name = benchmark_case_data.get("name", f"benchmark_case_{i}")
                opp_type = benchmark_case_data.get("type", "random")
                path = benchmark_case_data.get("checkpoint_path", None)
                metadata = benchmark_case_data.get("metadata", {})

                opponents.append(
                    OpponentInfo(
                        name=name,
                        type=opp_type,
                        checkpoint_path=path,
                        metadata=metadata,
                    )
                )

        return opponents

    async def _load_benchmark_suite(self, context: EvaluationContext):
        """Load the benchmark suite from configuration."""
        # Get suite configuration from strategy parameters
        suite_config = self.config.get_strategy_param("suite_config", [])
        if not suite_config:
            self.logger.warning("No benchmark suite configured. Using placeholders.")
            self.benchmark_suite = [
                OpponentInfo(
                    name="benchmark_standard_strong_v1",
                    type="ppo",
                    checkpoint_path="/path/to/strong_v1_model.ptk",
                    metadata={"difficulty": "high"},
                ),
                OpponentInfo(
                    name="benchmark_opening_test_A",
                    type="scripted",
                    metadata={"scenario": "opening_A"},
                ),
            ]
        else:
            self.benchmark_suite = []
            for i, benchmark_case_data in enumerate(suite_config):
                name = benchmark_case_data.get("name", f"benchmark_case_{i}")
                opp_type = benchmark_case_data.get(
                    "type", "ppo"
                )  # Could be an agent, or a specific scenario
                path = benchmark_case_data.get("checkpoint_path", None)
                metadata = benchmark_case_data.get(
                    "metadata", {}
                )  # Could include specific board setups, etc.
                self.benchmark_suite.append(
                    OpponentInfo(
                        name=name,
                        type=opp_type,
                        checkpoint_path=path,
                        metadata=metadata,
                    )
                )
        self.logger.info(
            f"Loaded benchmark suite with {len(self.benchmark_suite)} cases."
        )

    async def evaluate(
        self, agent_info: AgentInfo, context: Optional[EvaluationContext] = None
    ) -> EvaluationResult:
        """
        Run benchmark evaluation.
        """
        if context is None:
            context = self.setup_context(agent_info)

        self.log_evaluation_start(agent_info, context)
        await self._load_benchmark_suite(context)

        all_game_results: List[GameResult] = []
        errors: List[str] = []

        num_games_per_case = self.config.get_strategy_param("num_games_per_benchmark_case", 1)

        for i_case, benchmark_case_opponent_info in enumerate(self.benchmark_suite):
            try:
                case_results, case_errors = await self._process_benchmark_case(
                    agent_info,
                    benchmark_case_opponent_info,
                    i_case,
                    num_games_per_case,
                    context,
                )
                all_game_results.extend(case_results)
                errors.extend(case_errors)
            except Exception as e:
                error_msg = f"Critical error processing benchmark case {benchmark_case_opponent_info.name}: {e}"
                self.logger.error(error_msg, exc_info=True)
                errors.append(error_msg)

        summary_stats = SummaryStats.from_games(all_game_results)
        benchmark_analytics = self._calculate_benchmark_performance(
            all_game_results, self.benchmark_suite, agent_info
        )

        evaluation_result = EvaluationResult(
            context=context,
            games=all_game_results,
            summary_stats=summary_stats,
            analytics_data=benchmark_analytics,
            errors=errors,
        )

        self.log_evaluation_complete(evaluation_result)
        return evaluation_result

    # --- Game Playing Helper Methods (adapted from LadderEvaluator) ---

    async def _game_load_evaluation_entity(
        self,
        entity_info: AgentInfo | OpponentInfo,
        device_str: str,
        input_channels: int,
    ) -> Any:
        """Helper to load an agent or opponent for a game."""
        if isinstance(entity_info, AgentInfo):
            if "agent_instance" in entity_info.metadata:
                return entity_info.metadata["agent_instance"]
            return load_evaluation_agent(
                checkpoint_path=entity_info.checkpoint_path or "",
                device_str=device_str,
                policy_mapper=self.policy_mapper,
                input_channels=input_channels,
            )
        if isinstance(entity_info, OpponentInfo) and entity_info.type == "ppo_agent":
            return load_evaluation_agent(
                checkpoint_path=entity_info.checkpoint_path or "",
                device_str=device_str,
                policy_mapper=self.policy_mapper,
                input_channels=input_channels,
            )
        if isinstance(entity_info, OpponentInfo):
            # For benchmark, opponent might be a script, or another agent model
            return initialize_opponent(
                opponent_type=entity_info.type,
                opponent_path=entity_info.checkpoint_path,
                device_str=device_str,
                policy_mapper=self.policy_mapper,
                input_channels=input_channels,
            )
        raise ValueError(f"Unknown entity type for loading: {type(entity_info)}")

    async def _game_get_player_action(
        self, player_entity: Any, game: ShogiGame, legal_mask: Any
    ) -> Any:
        """Gets an action from the player entity (agent or opponent)."""
        move = None
        if hasattr(player_entity, "select_action"):  # PPOAgent-like
            move_tuple = player_entity.select_action(
                game.get_observation(),
                legal_mask,
                is_training=False,
            )
            if move_tuple is not None:
                move = move_tuple[0] if isinstance(move_tuple, tuple) else move_tuple
        elif hasattr(player_entity, "select_move"):  # Heuristic or other BaseOpponent
            move = player_entity.select_move(game)
        else:
            self.logger.error(
                "Player entity of type %s does not have a recognized action selection method.",
                type(player_entity).__name__,
            )
            raise TypeError(f"Unsupported player entity type: {type(player_entity)}")
        return move

    async def _game_validate_and_make_move(
        self,
        game: ShogiGame,
        move: Any,
        legal_moves: List[Any],
        current_player_color_value: int,
        player_entity_type_name: str,
    ) -> bool:
        """Validates the selected move and makes it in the game. Returns True if successful."""
        if move is None or move not in legal_moves:
            self.logger.warning(
                "Player %s (%s) made an illegal move (%s) or no move. Legal moves: %s. Game ending.",
                current_player_color_value,
                player_entity_type_name,
                move,
                len(legal_moves) if legal_moves else "None",
            )
            game.game_over = True
            game.winner = Color(1 - current_player_color_value)
            game.termination_reason = TERMINATION_REASON_ILLEGAL_MOVE
            return False

        # Use test_move for validation before making the move
        if not game.test_move(move):
            self.logger.warning(
                "Player %s (%s) made an invalid move (%s). Game ending due to invalid move.",
                current_player_color_value,
                player_entity_type_name,
                move,
            )
            game.game_over = True
            game.winner = Color(1 - current_player_color_value)
            game.termination_reason = "Invalid move"
            return False

        try:
            game.make_move(move)
            return True
        except Exception as e:
            self.logger.error(
                "Error making move %s for player %s: %s",
                move,
                current_player_color_value,
                e,
                exc_info=True,
            )
            game.game_over = True
            game.winner = Color(1 - current_player_color_value)
            game.termination_reason = (
                f"{TERMINATION_REASON_MOVE_EXECUTION_ERROR}: {str(e)}"
            )
            return False

    async def _handle_no_legal_moves(self, game: ShogiGame) -> None:
        """Handles the situation where a player has no legal moves."""
        game.game_over = True
        if game.winner is not None or game.termination_reason:
            self.logger.info(
                "Game ended: No legal moves for player %s. ShogiGame state: winner=%s, termination=%s",
                game.current_player.value,
                game.winner.name if game.winner else "None",
                game.termination_reason,
            )
        else:
            self.logger.warning(
                "No legal moves for player %s, and ShogiGame did not set a winner or termination_reason. "
                "Defaulting to draw. Current player: %s",
                game.current_player.value,
                game.current_player.name,
            )
            game.termination_reason = TERMINATION_REASON_NO_LEGAL_MOVES_UNDETERMINED

    async def _game_process_one_turn(
        self, game: ShogiGame, player_entity: Any, context: EvaluationContext
    ) -> bool:
        """
        Processes one turn for the current player.
        Returns True if the game continues, False if the game ended this turn.
        """
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            await self._handle_no_legal_moves(game)
            return False

        current_player_color_value = game.current_player.value
        player_entity_type_name = type(player_entity).__name__

        device_obj: torch.device
        if hasattr(player_entity, "device") and isinstance(
            getattr(player_entity, "device"), torch.device
        ):
            device_obj = getattr(player_entity, "device")
        else:
            default_device = "cpu"
            if context and context.configuration:  # Access via context.configuration
                default_device = getattr(context.configuration, "default_device", "cpu")
            device_obj = torch.device(default_device)

        legal_mask = self.policy_mapper.get_legal_mask(legal_moves, device_obj)
        move = None
        try:
            move = await self._game_get_player_action(player_entity, game, legal_mask)
        except Exception as e:
            self.logger.error(
                "Error during action selection for player %s (%s): %s",
                current_player_color_value,
                player_entity_type_name,
                e,
                exc_info=True,
            )
            game.game_over = True
            game.winner = Color(1 - current_player_color_value)
            game.termination_reason = (
                f"{TERMINATION_REASON_ACTION_SELECTION_ERROR}: {str(e)}"
            )
            return False

        if not await self._game_validate_and_make_move(
            game, move, legal_moves, current_player_color_value, player_entity_type_name
        ):
            return False

        return True

    async def _determine_game_loop_termination_reason(
        self, game: ShogiGame, move_count: int, max_moves: int
    ) -> str:
        """Determines the termination reason if not already set by ShogiGame after the loop."""
        if game.termination_reason:
            return game.termination_reason

        if move_count >= max_moves:
            return TERMINATION_REASON_MAX_MOVES
        elif game.game_over:
            # ShogiGame should set a reason if game_over is true. This is a fallback.
            self.logger.warning(
                "_determine_game_loop_termination_reason: Game is over but ShogiGame did not set a termination reason. Using fallback."
            )
            return TERMINATION_REASON_GAME_ENDED_UNSPECIFIED_FALLBACK
        else:
            # This case should ideally not be reached if the loop terminates correctly.
            self.logger.error(
                "_determine_game_loop_termination_reason: Loop terminated unexpectedly. Moves: %s, Max: %s, Game Over: %s",
                move_count,
                max_moves,
                game.game_over,
            )
            return TERMINATION_REASON_UNKNOWN_LOOP_TERMINATION

    async def _game_run_game_loop(
        self,
        sente_player: Any,
        gote_player: Any,
        context: EvaluationContext,
        initial_fen: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Runs the Shogi game loop and returns outcome. Sente is player 0, Gote is player 1."""
        max_moves = 500
        if context and context.configuration:
            max_moves = getattr(context.configuration, "max_moves_per_game", 500)

        player_map = {0: sente_player, 1: gote_player}

        if initial_fen:
            try:
                game = ShogiGame.from_sfen(
                    initial_fen, max_moves_for_game_instance=max_moves
                )
            except ValueError as e:
                self.logger.error(
                    f"Invalid SFEN string for benchmark: {initial_fen}. Error: {e}",
                    exc_info=True,
                )
                game = ShogiGame(max_moves_per_game=max_moves)
        else:
            game = ShogiGame(max_moves_per_game=max_moves)

        move_count = 0
        while not game.game_over and move_count < max_moves:
            current_player_entity = player_map[game.current_player.value]
            if not await self._game_process_one_turn(
                game, current_player_entity, context
            ):
                break
            move_count += 1

        # Determine and set termination reason if not already set by ShogiGame
        if not game.termination_reason:
            game.termination_reason = (
                await self._determine_game_loop_termination_reason(
                    game, move_count, max_moves
                )
            )

        winner_val = game.winner.value if game.winner is not None else None

        return {
            "winner": winner_val,
            "moves_count": move_count,
            "termination_reason": game.termination_reason,
            "final_fen": game.to_sfen_string(),
        }

    # --- End of Game Playing Helper Methods ---

    # --- Helper methods for evaluate_step (Benchmark specific) ---
    def _determine_final_winner_for_benchmark(
        self, game_outcome: Dict[str, Any], agent_plays_sente: bool
    ) -> Optional[int]:
        """Determines the winner from the agent's perspective (0 for agent win, 1 for opponent win, None for draw)."""
        if game_outcome["winner"] is None:  # Draw
            return None
        sente_won = game_outcome["winner"] == 0  # 0 for Sente
        if agent_plays_sente:
            return 0 if sente_won else 1
        else:  # Agent plays Gote
            return 1 if sente_won else 0

    def _prepare_benchmark_game_metadata(
        self,
        benchmark_opponent_info: OpponentInfo,
        agent_info: AgentInfo,
        game_outcome: Dict[str, Any],
        agent_plays_sente: bool,
    ) -> Dict[str, Any]:
        """Prepares metadata for a successfully completed benchmark game."""
        metadata = (
            benchmark_opponent_info.metadata or {}
        ).copy()  # Start with benchmark case metadata
        metadata["agent_color"] = "Sente" if agent_plays_sente else "Gote"
        metadata["sente_player_name"] = (
            agent_info.name if agent_plays_sente else benchmark_opponent_info.name
        )
        metadata["gote_player_name"] = (
            benchmark_opponent_info.name if agent_plays_sente else agent_info.name
        )
        metadata["termination_reason"] = game_outcome["termination_reason"]
        metadata["final_fen"] = game_outcome.get("final_fen")
        # Remove internal flags if they were added
        metadata.pop("agent_plays_sente_in_eval_step", None)
        return metadata

    def _prepare_benchmark_error_metadata(
        self,
        benchmark_opponent_info: OpponentInfo,
        error: Exception,
        agent_plays_sente: bool,  # Best guess for agent color
    ) -> Dict[str, Any]:
        """Prepares metadata for a benchmark game that ended in an error during play."""
        error_meta = (benchmark_opponent_info.metadata or {}).copy()
        error_meta["error"] = str(error)
        error_meta["agent_color"] = "Sente" if agent_plays_sente else "Gote"
        error_meta["termination_reason"] = (
            f"{TERMINATION_REASON_EVAL_STEP_ERROR}: {str(error)}"
        )
        error_meta.pop("agent_plays_sente_in_eval_step", None)
        return error_meta

    async def _setup_benchmark_game_entities_and_context(
        self,
        agent_info: AgentInfo,
        benchmark_opponent_info: OpponentInfo,
        context: EvaluationContext,
    ) -> Tuple[Any, Any, str, int, bool, Optional[str]]:
        """Helper to load entities and get game parameters from context for benchmark."""
        agent_plays_sente = benchmark_opponent_info.metadata.get(
            "agent_plays_sente_in_eval_step", True
        )
        self.logger.info(
            f"Benchmark Game: Agent ({agent_info.name}) as {'Sente' if agent_plays_sente else 'Gote'} vs Benchmark Opponent ({benchmark_opponent_info.name})"
        )

        device_str = "cpu"
        input_channels = 46
        initial_fen: Optional[str] = None

        if context and context.configuration:  # Access via context.configuration
            # These should ideally come from BenchmarkConfig if they can vary per benchmark
            device_str = getattr(context.configuration, "default_device", "cpu")
            input_channels = getattr(context.configuration, "input_channels", 46)

        # Benchmark specific: initial FEN might be in opponent_info.metadata
        initial_fen = benchmark_opponent_info.metadata.get("initial_fen", None)
        if initial_fen:
            self.logger.info(f"Starting benchmark game from FEN: {initial_fen}")

        logical_agent_entity = await self._game_load_evaluation_entity(
            agent_info, device_str, input_channels
        )
        # The benchmark_opponent_info itself defines the opponent (could be an agent, script, etc.)
        logical_benchmark_opponent_entity = await self._game_load_evaluation_entity(
            benchmark_opponent_info, device_str, input_channels
        )

        return (
            logical_agent_entity,
            logical_benchmark_opponent_entity,
            device_str,
            input_channels,
            agent_plays_sente,
            initial_fen,
        )

    # --- End of evaluate_step helpers ---

    async def evaluate_step(
        self,
        agent_info: AgentInfo,
        opponent_info: OpponentInfo,
        context: EvaluationContext,
    ) -> GameResult:
        """
        Evaluate a single game for a benchmark case.
        Agent's color is determined by opponent_info.metadata["agent_plays_sente_in_eval_step"].
        Initial board setup can be defined by opponent_info.metadata["initial_fen"].
        """
        # opponent_info here is the current_benchmark_opponent_info from the evaluate method
        game_id_suffix = opponent_info.metadata.get("benchmark_case_index", "X")
        game_id_suffix += f"_{opponent_info.metadata.get('game_index_in_case', 'Y')}"
        game_id = f"benchmark_game_{context.session_id}_{game_id_suffix}_{uuid.uuid4().hex[:6]}"
        start_time = time.time()

        # Pass opponent_info (which is current_benchmark_opponent_info) to setup
        (
            logical_agent_entity,
            logical_benchmark_opponent_entity,
            _,
            _,
            agent_plays_sente,
            initial_fen,
        ) = await self._setup_benchmark_game_entities_and_context(
            agent_info, opponent_info, context
        )

        try:
            sente_player, gote_player = (
                (logical_agent_entity, logical_benchmark_opponent_entity)
                if agent_plays_sente
                else (logical_benchmark_opponent_entity, logical_agent_entity)
            )

            game_outcome = await self._game_run_game_loop(
                sente_player, gote_player, context, initial_fen=initial_fen
            )
            duration = time.time() - start_time

            final_winner_code = self._determine_final_winner_for_benchmark(
                game_outcome, agent_plays_sente
            )
            # Pass opponent_info to prepare metadata
            game_metadata = self._prepare_benchmark_game_metadata(
                opponent_info, agent_info, game_outcome, agent_plays_sente
            )

            return GameResult(
                game_id=game_id,
                agent_info=agent_info,
                opponent_info=opponent_info,
                winner=final_winner_code,
                moves_count=game_outcome["moves_count"],
                duration_seconds=duration,
                metadata=game_metadata,
            )

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(
                f"Critical error in BenchmarkEvaluator.evaluate_step for game {game_id} (benchmark: {opponent_info.name}): {e}",
                exc_info=True,
            )
            agent_sente_at_error = opponent_info.metadata.get(
                "agent_plays_sente_in_eval_step", True
            )
            # Pass opponent_info to prepare error metadata
            error_metadata = self._prepare_benchmark_error_metadata(
                opponent_info, e, agent_sente_at_error
            )

            return GameResult(
                game_id=game_id,
                agent_info=agent_info,
                opponent_info=opponent_info,
                winner=None,
                moves_count=0,
                duration_seconds=duration,
                metadata=error_metadata,
            )

    def _calculate_benchmark_performance(
        self,
        results: List[GameResult],
        benchmark_suite: List[OpponentInfo],
        agent_info: AgentInfo,
    ) -> Dict[str, Any]:
        """Calculate performance against each benchmark case."""
        performance: Dict[str, Any] = {"per_benchmark_case_results": {}}

        for case_info in benchmark_suite:
            case_games = [g for g in results if g.opponent_info.name == case_info.name]
            if not case_games:
                performance["per_benchmark_case_results"][case_info.name] = {
                    "played": 0,
                    "wins_or_passes": 0,
                    "pass_rate": 0,
                    "details": "No games played",
                }
                continue

            # Define "win" or "pass" condition for a benchmark. Could be simple win, or meeting specific criteria.
            # For now, assume standard win (winner == 0)
            passes = sum(1 for g in case_games if g.winner == 0)

            performance["per_benchmark_case_results"][case_info.name] = {
                "played": len(case_games),
                "wins_or_passes": passes,
                "pass_rate": passes / len(case_games) if case_games else 0,
                # Could add more details like average moves, duration per case
            }

        overall_pass_rate = (
            sum(
                p["wins_or_passes"]
                for p in performance["per_benchmark_case_results"].values()
            )
            / sum(
                p["played"] for p in performance["per_benchmark_case_results"].values()
            )
            if results
            else 0
        )
        performance["overall_benchmark_pass_rate"] = overall_pass_rate

        self.logger.info(
            f"Benchmark performance calculated for agent {agent_info.name}. Overall pass rate: {overall_pass_rate:.2%}"
        )
        return performance

    def validate_config(self) -> bool:
        if not super().validate_config():
            return False
        # Add benchmark-specific config validation
        suite_config = self.config.get_strategy_param("suite_config", [])
        if not isinstance(suite_config, list):
            self.logger.warning(
                "suite_config should be a list. Evaluation might not run as expected."
            )
        num_games_per_case = self.config.get_strategy_param(
            "num_games_per_benchmark_case", 1
        )
        if num_games_per_case <= 0:
            self.logger.error("num_games_per_benchmark_case must be positive.")
            return False
        return True

    async def _process_benchmark_case(
        self,
        agent_info: AgentInfo,
        benchmark_case_opponent_info: OpponentInfo,
        i_case: int,
        num_games_per_case: int,
        context: EvaluationContext,
    ) -> Tuple[List[GameResult], List[str]]:
        """Processes a single benchmark case, playing the required number of games."""
        case_game_results: List[GameResult] = []
        case_errors: List[str] = []

        for i_game in range(num_games_per_case):
            agent_plays_sente_in_this_game = True
            if num_games_per_case > 1:
                agent_plays_sente_in_this_game = i_game % 2 == 0

            current_benchmark_opponent_info = OpponentInfo.from_dict(
                benchmark_case_opponent_info.to_dict()
            )
            current_benchmark_opponent_info.metadata = (
                benchmark_case_opponent_info.metadata or {}
            ).copy()
            current_benchmark_opponent_info.metadata[
                "agent_plays_sente_in_eval_step"
            ] = agent_plays_sente_in_this_game
            current_benchmark_opponent_info.metadata["benchmark_case_index"] = i_case
            current_benchmark_opponent_info.metadata["game_index_in_case"] = i_game

            game_desc = f"game {i_game+1}/{num_games_per_case} for benchmark case {benchmark_case_opponent_info.name} (Agent as {'Sente' if agent_plays_sente_in_this_game else 'Gote'})"
            try:
                self.logger.debug(f"Starting {game_desc}")
                game_result = await self.evaluate_step(
                    agent_info, current_benchmark_opponent_info, context
                )
                case_game_results.append(game_result)
            except Exception as e:
                error_msg = f"Error during game orchestration for {game_desc}: {e}"
                self.logger.error(error_msg, exc_info=True)
                case_errors.append(error_msg)
        return case_game_results, case_errors


# Register this evaluator with the factory
from ..core import EvaluationStrategy, EvaluatorFactory

EvaluatorFactory.register(
    EvaluationStrategy.BENCHMARK, BenchmarkEvaluator  # type: ignore
)
