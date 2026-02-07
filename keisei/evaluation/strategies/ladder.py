"""
Ladder (ELO-based) evaluation strategy implementation.
"""

import logging  # Add logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import torch  # Add torch

from keisei.shogi.shogi_core_definitions import Color
from keisei.shogi.shogi_game import ShogiGame
from keisei.utils import PolicyOutputMapper
from keisei.utils.agent_loading import initialize_opponent, load_evaluation_agent

from keisei.config_schema import EvaluationConfig
from ..core import (
    AgentInfo,
    BaseEvaluator,
    EvaluationContext,
    EvaluationResult,
    GameResult,
    OpponentInfo,
    SummaryStats,
)

# Explicitly import create_game_result if it's a direct function, or use GameResult constructor
# from ..core.evaluation_result import create_game_result # Assuming GameResult constructor is used

# Define constants for termination reasons (copied from tournament.py for now)
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
TERMINATION_REASON_EVAL_STEP_ERROR = "Ladder evaluate_step error"


# Placeholder for an ELO management system
# from ..analytics.elo_tracker import EloTracker


# For now, let's define a placeholder if not available
class EloTracker:  # pragma: no cover
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.ratings: Dict[str, float] = {}
        self.default_rating = 1500.0
        self.k_factor = 32  # Example K-factor

    def get_agent_rating(self, agent_id: str) -> float:
        return self.ratings.get(agent_id, self.default_rating)

    def update_ratings(
        self, agent_id: str, opponent_id: str, game_results: List[GameResult]
    ) -> None:
        # Simplified ELO update logic
        # In a real system, this would be more sophisticated
        agent_rating = self.get_agent_rating(agent_id)
        opponent_rating = self.get_agent_rating(opponent_id)

        for game in game_results:
            expected_score_agent = 1 / (
                1 + 10 ** ((opponent_rating - agent_rating) / 400)
            )

            actual_score_agent = 0.0
            if game.winner == 0:  # Agent won
                actual_score_agent = 1.0
            elif game.winner is None:  # Draw
                actual_score_agent = 0.5

            agent_rating_change = self.k_factor * (
                actual_score_agent - expected_score_agent
            )
            opponent_rating_change = self.k_factor * (
                (1 - actual_score_agent) - (1 - expected_score_agent)
            )

            agent_rating += agent_rating_change
            opponent_rating += opponent_rating_change

        self.ratings[agent_id] = agent_rating
        self.ratings[opponent_id] = opponent_rating
        # Persist ratings if needed

    def get_elo_snapshot(self) -> Dict[str, float]:
        return self.ratings.copy()


class LadderEvaluator(BaseEvaluator):
    """ELO ladder system with adaptive opponent selection."""

    def __init__(self, config: EvaluationConfig):  # type: ignore
        super().__init__(config)
        self.config: EvaluationConfig = config  # type: ignore
        self.elo_tracker = EloTracker(self.config.get_strategy_param("elo_config", {}))
        self.opponent_pool: List[OpponentInfo] = []
        self.policy_mapper = PolicyOutputMapper()  # Add PolicyOutputMapper instance
        # Ensure self.logger is initialized by BaseEvaluator or here
        if not hasattr(self, "logger") or self.logger is None:
            self.logger = logging.getLogger(__name__)  # Fallback logger

    def get_opponents(self, context: EvaluationContext) -> List[OpponentInfo]:
        """Get ladder opponents from configuration and pool."""
        # If opponent pool is empty, initialize from config
        if not self.opponent_pool:
            opponent_pool_config = self.config.get_strategy_param(
                "opponent_pool_config", []
            )

            for i, opp_config in enumerate(opponent_pool_config):
                name = opp_config.get("name", f"ladder_opponent_{i}")
                opp_type = opp_config.get("type", "random")
                checkpoint_path = opp_config.get("checkpoint_path")
                metadata = opp_config.get("metadata", {})

                self.opponent_pool.append(
                    OpponentInfo(
                        name=name,
                        type=opp_type,
                        checkpoint_path=checkpoint_path,
                        metadata=metadata,
                    )
                )

        # If still no opponents, provide defaults
        if not self.opponent_pool:
            self.opponent_pool = [
                OpponentInfo(
                    name="ladder_random",
                    type="random",
                    checkpoint_path=None,
                    metadata={"description": "Default random opponent for ladder"},
                ),
                OpponentInfo(
                    name="ladder_heuristic",
                    type="heuristic",
                    checkpoint_path=None,
                    metadata={"description": "Default heuristic opponent for ladder"},
                ),
            ]

        # For ladder, select subset based on rating range
        num_opponents = self.config.get_strategy_param(
            "num_opponents_per_evaluation", 3
        )
        return self.opponent_pool[:num_opponents]

    # --- Game Playing Helper Methods (adapted from TournamentEvaluator) ---

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
            return initialize_opponent(
                opponent_type=entity_info.type,
                opponent_path=entity_info.checkpoint_path,
                device_str=device_str,
                policy_mapper=self.policy_mapper,
                input_channels=input_channels,
                # config_params removed as it's not supported by initialize_opponent
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
        except (
            Exception
        ) as e:  # Broad exception to catch any issue during move execution by ShogiGame
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
            return False  # Game ended

        current_player_color_value = game.current_player.value
        player_entity_type_name = type(player_entity).__name__

        device_obj: torch.device
        if hasattr(player_entity, "device") and isinstance(
            getattr(player_entity, "device"), torch.device
        ):
            device_obj = getattr(player_entity, "device")
        else:
            default_device = "cpu"
            if context and context.configuration:
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

    async def _game_run_game_loop(
        self, sente_player: Any, gote_player: Any, context: EvaluationContext
    ) -> Dict[str, Any]:
        """Runs the Shogi game loop and returns outcome. Sente is player 0, Gote is player 1."""
        max_moves = 500
        if context and context.configuration:
            max_moves = getattr(context.configuration, "max_moves_per_game", 500)

        player_map = {0: sente_player, 1: gote_player}
        game = ShogiGame(max_moves_per_game=max_moves)
        move_count = 0

        while not game.game_over and move_count < max_moves:
            current_player_entity = player_map[game.current_player.value]
            if not await self._game_process_one_turn(
                game, current_player_entity, context
            ):
                break
            move_count += 1

        if not game.termination_reason:
            if move_count >= max_moves:
                game.termination_reason = TERMINATION_REASON_MAX_MOVES
            elif game.game_over:
                game.termination_reason = TERMINATION_REASON_GAME_ENDED_UNSPECIFIED
            else:
                game.termination_reason = TERMINATION_REASON_UNKNOWN_LOOP_TERMINATION
                self.logger.warning(
                    "_game_run_game_loop: Reached '%s'. Moves: %s, Game Over: %s, ShogiGame Reason: %s",
                    game.termination_reason,
                    move_count,
                    game.game_over,
                    game.termination_reason,
                )

        winner_val = game.winner.value if game.winner is not None else None

        return {
            "winner": winner_val,
            "moves_count": move_count,
            "termination_reason": game.termination_reason,
        }

    # --- End of Game Playing Helper Methods ---

    # --- Helper methods for evaluate_step ---
    def _determine_final_winner(
        self, game_outcome: Dict[str, Any], agent_plays_sente: bool
    ) -> Optional[int]:
        """Determines the winner from the agent's perspective."""
        if game_outcome["winner"] is None:
            return None
        sente_won = game_outcome["winner"] == 0  # 0 for Sente
        if agent_plays_sente:
            return 0 if sente_won else 1
        else:  # Agent plays Gote
            return 1 if sente_won else 0

    def _prepare_game_metadata(
        self,
        opponent_info: OpponentInfo,
        agent_info: AgentInfo,
        game_outcome: Dict[str, Any],
        agent_plays_sente: bool,
        # game_id: str # Not currently used but could be for more detailed metadata
    ) -> Dict[str, Any]:
        """Prepares metadata for a successfully completed game."""
        metadata = (opponent_info.metadata or {}).copy()
        metadata["agent_color"] = "Sente" if agent_plays_sente else "Gote"
        metadata["sente_player_name"] = (
            agent_info.name if agent_plays_sente else opponent_info.name
        )
        metadata["gote_player_name"] = (
            opponent_info.name if agent_plays_sente else agent_info.name
        )
        metadata["termination_reason"] = game_outcome["termination_reason"]
        metadata.pop("agent_plays_sente_in_eval_step", None)  # Clean up temporary flag
        return metadata

    def _prepare_error_metadata(
        self,
        opponent_info: OpponentInfo,
        error: Exception,
        agent_plays_sente: bool,
        # game_id: str # Not currently used
    ) -> Dict[str, Any]:
        """Prepares metadata for a game that ended in an error."""
        error_meta = (opponent_info.metadata or {}).copy()
        error_meta["error"] = str(error)
        error_meta["agent_color"] = (
            "Sente" if agent_plays_sente else "Gote"
        )  # Best guess
        error_meta["termination_reason"] = (
            f"{TERMINATION_REASON_EVAL_STEP_ERROR}: {str(error)}"
        )
        error_meta.pop(
            "agent_plays_sente_in_eval_step", None
        )  # Clean up temporary flag
        return error_meta

    # --- End of evaluate_step helpers ---

    async def _initialize_opponent_pool(self, context: EvaluationContext):
        """Initializes or loads the pool of opponents for the ladder."""
        # This could load from a config file, a database, or a predefined list in LadderConfig
        pool_configs = self.config.strategy_params.get("opponent_pool_config", [])
        if not pool_configs:
            self.logger.warning(
                "No opponent pool configured for ladder. Using placeholders."
            )
            self.opponent_pool = [
                OpponentInfo(
                    name="ladder_opp_random_1",
                    type="random",
                    metadata={"initial_rating": 1400},
                ),
                OpponentInfo(
                    name="ladder_opp_heuristic_1",
                    type="heuristic",
                    metadata={"initial_rating": 1600},
                ),
            ]
        else:
            self.opponent_pool = []
            for i, opp_config_data in enumerate(pool_configs):
                name = opp_config_data.get("name", f"ladder_opponent_{i}")
                opp_type = opp_config_data.get("type", "random")
                path = opp_config_data.get("checkpoint_path", None)
                initial_rating = opp_config_data.get(
                    "initial_rating", self.elo_tracker.default_rating
                )
                # Ensure opponent is in EloTracker
                if name not in self.elo_tracker.ratings:
                    self.elo_tracker.ratings[name] = initial_rating
                self.opponent_pool.append(
                    OpponentInfo(
                        name=name,
                        type=opp_type,
                        checkpoint_path=path,
                        metadata={"initial_rating": initial_rating},
                    )
                )
        self.logger.info(
            f"Initialized ladder opponent pool with {len(self.opponent_pool)} opponents."
        )

    async def evaluate(
        self, agent_info: AgentInfo, context: Optional[EvaluationContext] = None
    ) -> EvaluationResult:
        """
        Run ladder evaluation.
        """
        if context is None:
            context = self.setup_context(agent_info)

        self.log_evaluation_start(agent_info, context)
        await self._initialize_opponent_pool(context)

        initial_agent_rating = self.elo_tracker.get_agent_rating(agent_info.name)
        if agent_info.name not in self.elo_tracker.ratings:
            self.elo_tracker.ratings[agent_info.name] = initial_agent_rating

        selected_opponents = self._select_ladder_opponents(
            initial_agent_rating, context
        )

        all_game_results: List[GameResult] = []
        errors: List[str] = []
        num_games_per_match = self.config.get_strategy_param("num_games_per_match", 2)

        for opponent_info in selected_opponents:
            try:
                match_games, match_errors = await self._play_match_against_opponent(
                    agent_info, opponent_info, num_games_per_match, context
                )
                all_game_results.extend(match_games)
                errors.extend(match_errors)

                if match_games:
                    self.elo_tracker.update_ratings(
                        agent_info.name, opponent_info.name, match_games
                    )
                    self.logger.info(
                        f"Ratings updated after match: Agent({agent_info.name}): {self.elo_tracker.get_agent_rating(agent_info.name):.2f}, Opponent({opponent_info.name}): {self.elo_tracker.get_agent_rating(opponent_info.name):.2f}"
                    )

            except (
                Exception
            ) as e:  # Catch errors from _play_match_against_opponent itself or Elo update
                error_msg = f"Critical error during match processing for opponent {opponent_info.name}: {e}"
                self.logger.error(error_msg, exc_info=True)
                errors.append(error_msg)

        summary_stats = SummaryStats.from_games(all_game_results)
        final_agent_rating = self.elo_tracker.get_agent_rating(agent_info.name)
        ladder_specific_analytics = {
            "initial_agent_rating": initial_agent_rating,
            "final_agent_rating": final_agent_rating,
            "rating_change": final_agent_rating - initial_agent_rating,
        }

        evaluation_result = EvaluationResult(
            context=context,
            games=all_game_results,
            summary_stats=summary_stats,
            analytics_data={
                "ladder_specific_analytics": ladder_specific_analytics,
                "final_elo_snapshot": self.elo_tracker.get_elo_snapshot(),
            },
            errors=errors,
            elo_tracker=self.elo_tracker,  # type: ignore # Placeholder EloTracker, real one from analytics needed
        )

        self.log_evaluation_complete(evaluation_result)
        return evaluation_result

    async def _play_match_against_opponent(
        self,
        agent_info: AgentInfo,
        opponent_info: OpponentInfo,
        num_games_per_match: int,
        context: EvaluationContext,
    ) -> Tuple[List[GameResult], List[str]]:
        """Plays a match (set of games) against a single opponent, balancing colors."""
        match_games: List[GameResult] = []
        match_errors: List[str] = []

        for i in range(num_games_per_match):
            current_opponent_info_for_game = OpponentInfo.from_dict(
                opponent_info.to_dict()
            )
            current_opponent_info_for_game.metadata = (
                opponent_info.metadata or {}
            ).copy()

            agent_plays_sente_in_this_game = i < (num_games_per_match + 1) // 2
            current_opponent_info_for_game.metadata[
                "agent_plays_sente_in_eval_step"
            ] = agent_plays_sente_in_this_game

            game_desc = f"game {i+1}/{num_games_per_match} vs {opponent_info.name} (Agent as {'Sente' if agent_plays_sente_in_this_game else 'Gote'})"
            try:
                self.logger.debug(f"Starting {game_desc}")
                game_result = await self.evaluate_step(
                    agent_info, current_opponent_info_for_game, context
                )
                match_games.append(game_result)
            except Exception as e:
                error_msg = f"Error during game orchestration for {game_desc}: {e}"
                self.logger.error(error_msg, exc_info=True)
                match_errors.append(error_msg)
        return match_games, match_errors

    async def _setup_game_entities_and_context(
        self,
        agent_info: AgentInfo,
        opponent_info: OpponentInfo,
        context: EvaluationContext,
    ) -> Tuple[Any, Any, str, int, bool]:
        """Helper to load entities and get game parameters from context."""
        agent_plays_sente = opponent_info.metadata.get(
            "agent_plays_sente_in_eval_step", True
        )
        self.logger.info(
            f"Agent ({agent_info.name}) as {'Sente' if agent_plays_sente else 'Gote'} vs Opponent ({opponent_info.name})"
        )

        device_str = "cpu"
        input_channels = 46
        if context and context.configuration:
            device_str = getattr(context.configuration, "default_device", "cpu")
            input_channels = getattr(context.configuration, "input_channels", 46)

        logical_agent_entity = await self._game_load_evaluation_entity(
            agent_info, device_str, input_channels
        )
        logical_opponent_entity = await self._game_load_evaluation_entity(
            opponent_info, device_str, input_channels
        )

        return (
            logical_agent_entity,
            logical_opponent_entity,
            device_str,
            input_channels,
            agent_plays_sente,
        )

    async def evaluate_step(
        self,
        agent_info: AgentInfo,
        opponent_info: OpponentInfo,
        context: EvaluationContext,
    ) -> GameResult:
        """
        Evaluate a single game step (one game) against a specific opponent on the ladder.
        Agent's color is determined by opponent_info.metadata["agent_plays_sente_in_eval_step"].
        """
        game_id = f"ladder_game_{context.session_id}_{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        logical_agent_entity, logical_opponent_entity, _, _, agent_plays_sente = (
            await self._setup_game_entities_and_context(
                agent_info, opponent_info, context
            )
        )

        try:
            sente_player, gote_player = (
                (logical_agent_entity, logical_opponent_entity)
                if agent_plays_sente
                else (logical_opponent_entity, logical_agent_entity)
            )

            game_outcome = await self._game_run_game_loop(
                sente_player, gote_player, context
            )
            duration = time.time() - start_time

            final_winner_code = self._determine_final_winner(
                game_outcome, agent_plays_sente
            )
            game_metadata = self._prepare_game_metadata(
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
                f"Critical error in LadderEvaluator.evaluate_step for game {game_id}: {e}",
                exc_info=True,
            )
            error_metadata = self._prepare_error_metadata(
                opponent_info, e, agent_plays_sente
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

    def _select_ladder_opponents(
        self, agent_rating: float, context: EvaluationContext
    ) -> List[OpponentInfo]:
        """Selects opponents from the pool for the ladder, based on the agent's current rating."""
        # For now, a simple placeholder logic: select all opponents with a rating range
        if not self.opponent_pool:
            self.logger.warning(
                "Opponent pool is empty. Cannot select opponents for ladder."
            )
            return []

        # Basic filtering by rating range (exclude the agent itself by name)
        agent_name = context.agent_info.name
        filtered_opponents = [
            opp
            for opp in self.opponent_pool
            if opp.name != agent_name
            and opp.metadata.get("initial_rating", 1500) <= agent_rating + 400
            and opp.metadata.get("initial_rating", 1500) >= agent_rating - 400
        ]

        # Sort by rating (ascending)
        filtered_opponents.sort(
            key=lambda opp: opp.metadata.get("initial_rating", 1500)
        )

        # Select top N opponents for the ladder, where N is configurable
        num_opponents_to_select = self.config.get_strategy_param("num_opponents_to_select", 5)
        selected_opponents = filtered_opponents[:num_opponents_to_select]

        self.logger.info(
            f"Selected {len(selected_opponents)} opponents for ladder: {[opp.name for opp in selected_opponents]}"
        )
        return selected_opponents


# Register this evaluator with the factory
from ..core import EvaluationStrategy, EvaluatorFactory

EvaluatorFactory.register(EvaluationStrategy.LADDER, LadderEvaluator)  # type: ignore
