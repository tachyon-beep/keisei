"""
Single opponent evaluation strategy implementation.

This module implements evaluation against a single opponent, which is the most
common evaluation strategy used during training.
"""

import logging
import statistics
import uuid
from typing import (  # Ensure Union is imported; Added List
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Union,
)

import torch  # Import torch at the module level

if TYPE_CHECKING:
    pass  # No need for torch here if imported above

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
    create_game_result,
)
from keisei.config_schema import EvaluationConfig

logger = logging.getLogger(__name__)


class SingleOpponentEvaluator(BaseEvaluator):
    """
    Evaluator for single opponent evaluation strategy.

    This evaluator runs a specified number of games against a single opponent,
    with optional color balancing and game distribution features.
    """

    def __init__(self, config: EvaluationConfig):
        """Initialize single opponent evaluator."""
        super().__init__(config)
        self.config: EvaluationConfig = config
        self.policy_mapper = PolicyOutputMapper()

        # In-memory evaluation support
        self.agent_weights: Optional[Dict[str, torch.Tensor]] = None
        self.opponent_weights: Optional[Dict[str, torch.Tensor]] = None
        self.temp_agent_config = None

    async def _load_evaluation_entity(
        self,
        entity_info: Union[AgentInfo, OpponentInfo],
        device_str: str,
        input_channels: int,
    ) -> Any:  # Added return type hint
        """Helper to load an agent or opponent."""
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
        elif isinstance(entity_info, OpponentInfo):
            return initialize_opponent(
                opponent_type=entity_info.type,
                opponent_path=entity_info.checkpoint_path,
                device_str=device_str,
                policy_mapper=self.policy_mapper,
                input_channels=input_channels,
            )
        raise ValueError(f"Unknown entity type for loading: {type(entity_info)}")

    async def _get_player_action(
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
            # This case should ideally not be reached if entities are loaded correctly
            logger.error(
                f"Player entity of type {type(player_entity)} does not have a recognized action selection method."
            )
            raise TypeError(f"Unsupported player entity type: {type(player_entity)}")
        return move

    async def _validate_and_make_move(
        self,
        game: ShogiGame,
        move: Any,
        legal_moves: List[Any],
        current_player_color_value: int,
        player_entity_type_name: str,
    ) -> bool:
        """Validates the selected move and makes it in the game. Returns True if successful."""
        if move is None or move not in legal_moves:
            logger.warning(
                f"Player {current_player_color_value} ({player_entity_type_name}) made an illegal move ({move}) or no move. "
                f"Legal moves: {len(legal_moves) if legal_moves else 'None'}. Game ending."
            )
            game.game_over = True
            game.winner = Color(1 - current_player_color_value)  # Other player wins
            game.termination_reason = "Illegal/No move"
            return False

        # Use test_move for validation before making the move
        if not game.test_move(move):
            logger.warning(
                f"Player {current_player_color_value} ({player_entity_type_name}) made an invalid move ({move}). "
                f"Game ending due to invalid move."
            )
            game.game_over = True
            game.winner = Color(1 - current_player_color_value)  # Other player wins
            game.termination_reason = "Invalid move"
            return False

        try:
            game.make_move(move)
            return True
        except Exception as e:
            logger.error(
                f"Error making move {move} for player {current_player_color_value}: {e}",
                exc_info=True,
            )
            game.game_over = True
            game.winner = Color(1 - current_player_color_value)  # Other player wins
            game.termination_reason = f"Move execution error: {str(e)}"
            return False

    async def _run_game_loop(
        self, agent_player: Any, opponent_player: Any, context: EvaluationContext
    ) -> Dict[str, Any]:
        """Runs the Shogi game loop and returns outcome."""
        max_moves = getattr(context.configuration, "max_moves_per_game", 500)

        player_map = {0: agent_player, 1: opponent_player}  # Sente (0), Gote (1)

        game = ShogiGame(max_moves_per_game=max_moves)
        move_count = 0

        while not game.game_over and move_count < max_moves:
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                game.game_over = True
                logger.info(
                    f"Game ended: No legal moves for player {game.current_player.value}."
                )
                break

            current_player_color_value = game.current_player.value
            current_player_entity = player_map[current_player_color_value]
            player_entity_type_name = type(current_player_entity).__name__

            device_obj: torch.device
            if hasattr(current_player_entity, "device") and isinstance(
                getattr(current_player_entity, "device"), torch.device
            ):
                device_obj = getattr(current_player_entity, "device")
            else:
                device_obj = torch.device(
                    getattr(current_player_entity, "device", "cpu")
                )

            legal_mask = self.policy_mapper.get_legal_mask(legal_moves, device_obj)

            move = None
            try:
                move = await self._get_player_action(
                    current_player_entity, game, legal_mask
                )
            except Exception as e:
                logger.error(
                    f"Error during action selection for player {current_player_color_value} ({player_entity_type_name}): {e}",
                    exc_info=True,
                )
                game.game_over = True
                game.winner = Color(1 - current_player_color_value)
                game.termination_reason = f"Action selection error: {str(e)}"
                break

            if not await self._validate_and_make_move(
                game,
                move,
                legal_moves,
                current_player_color_value,
                player_entity_type_name,
            ):
                break  # Game ended due to illegal move or error during make_move

            move_count += 1

        winner_val = None
        if game.winner is not None:
            winner_val = game.winner.value

        return {
            "winner": winner_val,  # This is 0 if Sente (player_map[0]) wins, 1 if Gote (player_map[1]) wins
            "moves_count": move_count,
            "termination_reason": game.termination_reason,
        }

    async def evaluate(
        self, agent_info: AgentInfo, context: Optional[EvaluationContext] = None
    ) -> EvaluationResult:
        """
        Run single opponent evaluation.

        Args:
            agent_info: Information about the agent to evaluate
            context: Optional evaluation context

        Returns:
            Complete evaluation results
        """
        # Validate inputs
        if not self.validate_agent(agent_info):
            raise ValueError("Invalid agent configuration")

        if not self.validate_config():
            raise ValueError("Invalid evaluator configuration")

        # Set up context
        if context is None:
            context = self.setup_context(agent_info)
        else:
            context = self.setup_context(agent_info, context)

        self.log_evaluation_start(agent_info, context)

        # Create opponent info
        # Map opponent name to opponent type for supported opponents
        opponent_type_mapping = {
            "random": "random",
            "heuristic": "heuristic",
            "default_opponent": "random",  # Default fallback
        }

        opponent_name = self.config.get_strategy_param(
            "opponent_name", "default_opponent"
        )
        opponent_type = opponent_type_mapping.get(opponent_name, "random")

        opponent_info = OpponentInfo(
            name=opponent_name,
            type=opponent_type,
            checkpoint_path=self.config.get_strategy_param("opponent_path"),
            metadata=self.config.get_strategy_param("opponent_params", {}),
        )

        # Run games
        games = []
        errors = []

        try:
            # Determine game distribution
            games_per_color = self._calculate_game_distribution()

            # Run games with agent as first player
            for i in range(games_per_color["agent_first"]):
                try:
                    # Pass opponent_info directly
                    game = await self.evaluate_step(agent_info, opponent_info, context)
                    games.append(game)
                except Exception as e:
                    logger.error(f"Error in game {i} (agent first): {e}", exc_info=True)
                    errors.append(f"Game {i} (agent first): {str(e)}")

            # Run games with agent as second player (if color balancing enabled)
            if games_per_color["agent_second"] > 0:
                for i in range(games_per_color["agent_second"]):
                    game_index = games_per_color["agent_first"] + i
                    try:
                        # Create modified opponent for color-swapped game
                        # For single opponent, the 'opponent_info' is for the designated opponent.
                        # To play as second, the game logic itself should handle colors.
                        # The OpponentInfo here should still refer to the same opponent.
                        # The game playing logic (_run_game_loop) will use game.current_player.
                        # We need a way to tell the game to start with the opponent as the first player (Sente).
                        # This is typically handled by the environment or game setup.
                        # For now, assuming the ShogiGame can be initialized or a flag passed
                        # to indicate who starts. If not, this part needs more thought on how
                        # the agent plays as Gote (White).
                        # A simple way is to swap agent and opponent roles when calling _run_game_loop
                        # if the game itself doesn't have a parameter for starting player.
                        # However, the current `_run_game_loop` assumes `agent` is player 0.
                        # This needs careful handling.
                        # The current `opponent_info.metadata` with "agent_plays_second" is a good hint.
                        # Let's assume `evaluate_step` can use this metadata.

                        # Create a context or pass a flag to evaluate_step indicating agent plays second.
                        # For now, the existing swapped_opponent logic seems to try this via metadata.
                        swapped_opponent_meta = opponent_info.metadata.copy()
                        swapped_opponent_meta["agent_plays_second"] = True

                        # This `swapped_opponent` is actually just a modified OpponentInfo for the same opponent.
                        # The key is how `evaluate_step` or `_run_game_loop` interprets this.
                        # If `evaluate_step` is adjusted to handle `agent_plays_second`, this should work.
                        current_opponent_info = OpponentInfo(
                            name=opponent_info.name,
                            type=opponent_info.type,
                            checkpoint_path=opponent_info.checkpoint_path,
                            metadata=swapped_opponent_meta,
                        )
                        game = await self.evaluate_step(
                            agent_info, current_opponent_info, context
                        )
                        games.append(game)
                    except Exception as e:
                        logger.error(
                            f"Error in game {game_index} (agent second): {e}",
                            exc_info=True,
                        )
                        errors.append(f"Game {game_index} (agent second): {str(e)}")

        except Exception as e:
            logger.error(
                "Critical error during evaluation: %s", e
            )  # Use module-level logger
            errors.append(f"Critical error: {str(e)}")

        # Calculate summary statistics
        summary_stats = SummaryStats.from_games(games)

        # Create result
        result = EvaluationResult(
            context=context,
            games=games,
            summary_stats=summary_stats,
            analytics_data=self._calculate_analytics(games),
            errors=errors,
            elo_tracker=None,  # Will be filled by ELO system if enabled
        )

        self.log_evaluation_complete(result)
        return result

    async def evaluate_in_memory(
        self,
        agent_info: AgentInfo,
        context: Optional[EvaluationContext] = None,
        *,
        agent_weights: Optional[Dict[str, torch.Tensor]] = None,
        opponent_weights: Optional[Dict[str, torch.Tensor]] = None,
        opponent_info: Optional[OpponentInfo] = None,
    ) -> EvaluationResult:
        """
        Run single opponent evaluation using in-memory weights.

        Args:
            agent_info: Information about the agent to evaluate
            context: Optional evaluation context
            agent_weights: Pre-extracted agent model weights
            opponent_weights: Pre-extracted opponent model weights
            opponent_info: Opponent information

        Returns:
            Complete evaluation results
        """
        # Validate inputs
        if not self.validate_agent(agent_info):
            raise ValueError("Invalid agent configuration")

        if not self.validate_config():
            raise ValueError("Invalid evaluator configuration")

        # Store weights for use during evaluation
        self.agent_weights = agent_weights
        self.opponent_weights = opponent_weights

        try:
            # Set up context
            if context is None:
                context = self.setup_context(agent_info)
            else:
                context = self.setup_context(agent_info, context)

            self.log_evaluation_start(agent_info, context)

            # Create opponent info if not provided
            if opponent_info is None:
                opponent_info = OpponentInfo(
                    name=self.config.get_strategy_param(
                        "opponent_name", "default_opponent"
                    ),
                    type="ppo_agent" if self.opponent_weights else "unknown",
                    checkpoint_path=self.config.get_strategy_param("opponent_path"),
                    metadata=self.config.get_strategy_param("opponent_params", {}),
                )

            # Run games with in-memory weights
            games = []
            errors = []

            try:
                # Determine game distribution
                games_per_color = self._calculate_game_distribution()

                # Run games with agent as first player
                for i in range(games_per_color["agent_first"]):
                    try:
                        game = await self.evaluate_step_in_memory(
                            agent_info, opponent_info, context
                        )
                        games.append(game)
                    except Exception as e:
                        logger.error(
                            f"Error in game {i} (agent first): {e}", exc_info=True
                        )
                        errors.append(f"Game {i} (agent first): {str(e)}")

                # Run games with agent as second player (if color balancing enabled)
                if games_per_color["agent_second"] > 0:
                    for i in range(games_per_color["agent_second"]):
                        game_index = games_per_color["agent_first"] + i
                        try:
                            # Create modified opponent for color-swapped game
                            swapped_opponent_meta = opponent_info.metadata.copy()
                            swapped_opponent_meta["agent_plays_second"] = True

                            current_opponent_info = OpponentInfo(
                                name=opponent_info.name,
                                type=opponent_info.type,
                                checkpoint_path=opponent_info.checkpoint_path,
                                metadata=swapped_opponent_meta,
                            )
                            game = await self.evaluate_step_in_memory(
                                agent_info, current_opponent_info, context
                            )
                            games.append(game)
                        except Exception as e:
                            logger.error(
                                f"Error in game {game_index} (agent second): {e}",
                                exc_info=True,
                            )
                            errors.append(f"Game {game_index} (agent second): {str(e)}")

            except Exception as e:
                logger.error("Critical error during in-memory evaluation: %s", e)
                errors.append(f"Critical error: {str(e)}")

            # Calculate summary statistics
            summary_stats = SummaryStats.from_games(games)

            # Create result
            result = EvaluationResult(
                context=context,
                games=games,
                summary_stats=summary_stats,
                analytics_data=self._calculate_analytics(games),
                errors=errors,
                elo_tracker=None,  # Will be filled by ELO system if enabled
            )

            self.log_evaluation_complete(result)
            return result

        finally:
            # Clean up references
            self.agent_weights = None
            self.opponent_weights = None

    async def _load_evaluation_entity_in_memory(
        self,
        entity_info: Union[AgentInfo, OpponentInfo],
        device_str: str,
        input_channels: int,
    ) -> Any:
        """Helper to load an agent or opponent from in-memory weights."""
        if isinstance(entity_info, AgentInfo):
            return await self._load_agent_in_memory(
                entity_info, device_str, input_channels
            )
        elif isinstance(entity_info, OpponentInfo):
            return await self._load_opponent_in_memory(
                entity_info, device_str, input_channels
            )

        raise ValueError(f"Unknown entity type for loading: {type(entity_info)}")

    async def _load_agent_in_memory(
        self,
        agent_info: AgentInfo,
        device_str: str,
        input_channels: int,
    ) -> Any:
        """Load agent using in-memory weights if available."""
        # Check for direct agent instance
        if "agent_instance" in agent_info.metadata:
            return agent_info.metadata["agent_instance"]

        # Try to create from in-memory weights
        if self.agent_weights is not None:
            try:
                from ..core.model_manager import ModelWeightManager

                manager = ModelWeightManager(device=device_str)
                agent = manager.create_agent_from_weights(
                    weights=self.agent_weights, device=device_str
                )
                logger.debug("Successfully created agent from in-memory weights")
                return agent
            except Exception as e:
                logger.warning("Failed to create agent from in-memory weights: %s", e)

        # Fallback to regular loading
        return load_evaluation_agent(
            checkpoint_path=agent_info.checkpoint_path or "",
            device_str=device_str,
            policy_mapper=self.policy_mapper,
            input_channels=input_channels,
        )

    async def _load_opponent_in_memory(
        self,
        opponent_info: OpponentInfo,
        device_str: str,
        input_channels: int,
    ) -> Any:
        """Load opponent using in-memory weights if available."""
        # Try to create PPO opponent from in-memory weights
        if opponent_info.type == "ppo_agent" and self.opponent_weights is not None:
            try:
                from ..core.model_manager import ModelWeightManager

                manager = ModelWeightManager(device=device_str)
                opponent = manager.create_agent_from_weights(
                    weights=self.opponent_weights, device=device_str
                )
                logger.debug("Successfully created opponent from in-memory weights")
                return opponent
            except Exception as e:
                logger.warning(
                    "Failed to create opponent from in-memory weights: %s", e
                )

        # Fallback to regular loading
        if opponent_info.type == "ppo_agent":
            return load_evaluation_agent(
                checkpoint_path=opponent_info.checkpoint_path or "",
                device_str=device_str,
                policy_mapper=self.policy_mapper,
                input_channels=input_channels,
            )
        else:
            return initialize_opponent(
                opponent_type=opponent_info.type,
                opponent_path=opponent_info.checkpoint_path,
                device_str=device_str,
                policy_mapper=self.policy_mapper,
                input_channels=input_channels,
            )

    async def evaluate_step(
        self,
        agent_info: AgentInfo,
        opponent_info: OpponentInfo,
        context: EvaluationContext,
    ) -> GameResult:
        """
        Evaluate a single game between agent and opponent.
        """
        import time

        game_id = f"single_{context.session_id}_{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        agent_plays_sente = not opponent_info.metadata.get("agent_plays_second", False)

        try:
            device_str = "cpu"
            input_channels = 46

            # Load the logical agent and opponent entities
            logical_agent_entity = await self._load_evaluation_entity(
                agent_info, device_str, input_channels
            )
            logical_opponent_entity = await self._load_evaluation_entity(
                opponent_info, device_str, input_channels
            )

            # Determine who plays Sente (player 0) and Gote (player 1) for this game
            sente_player = (
                logical_agent_entity if agent_plays_sente else logical_opponent_entity
            )
            gote_player = (
                logical_opponent_entity if agent_plays_sente else logical_agent_entity
            )

            # The _run_game_loop always assumes the first arg is Sente, second is Gote.
            game_outcome = await self._run_game_loop(sente_player, gote_player, context)

            duration = time.time() - start_time

            # game_outcome["winner"] is 0 if Sente won, 1 if Gote won, None if draw.
            # We need to map this to: 0 if logical_agent_entity won, 1 if logical_opponent_entity won.

            final_winner_code = None  # For agent vs opponent perspective
            if game_outcome["winner"] is not None:  # Not a draw from game error
                sente_won = game_outcome["winner"] == 0
                if agent_plays_sente:  # Agent was Sente
                    final_winner_code = 0 if sente_won else 1
                else:  # Agent was Gote
                    final_winner_code = (
                        0 if not sente_won else 1
                    )  # Agent wins if Gote won (sente_won is false)

            return create_game_result(
                game_id=game_id,
                agent_info=agent_info,
                opponent_info=opponent_info,
                winner=final_winner_code,
                moves_count=game_outcome["moves_count"],
                duration_seconds=duration,
                metadata={"termination_reason": game_outcome["termination_reason"]},
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                "Critical error in evaluate_step for game %s: %s",
                game_id,
                e,
                exc_info=True,
            )

            return create_game_result(
                game_id=game_id,
                agent_info=agent_info,
                opponent_info=opponent_info,
                winner=None,
                moves_count=0,
                duration_seconds=duration,
                metadata={"termination_reason": f"Evaluate_step error: {str(e)}"},
            )

    async def evaluate_step_in_memory(
        self,
        agent_info: AgentInfo,
        opponent_info: OpponentInfo,
        context: EvaluationContext,
    ) -> GameResult:
        """
        Evaluate a single game between agent and opponent using in-memory weights.
        """
        import time

        game_id = f"inmem_{context.session_id}_{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        agent_plays_sente = not opponent_info.metadata.get("agent_plays_second", False)

        try:
            device_str = "cpu"
            input_channels = 46

            # Load the logical agent and opponent entities with in-memory weights
            logical_agent_entity = await self._load_evaluation_entity_in_memory(
                agent_info, device_str, input_channels
            )
            logical_opponent_entity = await self._load_evaluation_entity_in_memory(
                opponent_info, device_str, input_channels
            )

            # Determine who plays Sente (player 0) and Gote (player 1) for this game
            sente_player = (
                logical_agent_entity if agent_plays_sente else logical_opponent_entity
            )
            gote_player = (
                logical_opponent_entity if agent_plays_sente else logical_agent_entity
            )

            # The _run_game_loop always assumes the first arg is Sente, second is Gote.
            game_outcome = await self._run_game_loop(sente_player, gote_player, context)

            duration = time.time() - start_time

            # game_outcome["winner"] is 0 if Sente won, 1 if Gote won, None if draw.
            # We need to map this to: 0 if logical_agent_entity won, 1 if logical_opponent_entity won.

            final_winner_code = None  # For agent vs opponent perspective
            if game_outcome["winner"] is not None:  # Not a draw from game error
                sente_won = game_outcome["winner"] == 0
                if agent_plays_sente:  # Agent was Sente
                    final_winner_code = 0 if sente_won else 1
                else:  # Agent was Gote
                    final_winner_code = (
                        0 if not sente_won else 1
                    )  # Agent wins if Gote won (sente_won is false)

            return create_game_result(
                game_id=game_id,
                agent_info=agent_info,
                opponent_info=opponent_info,
                winner=final_winner_code,
                moves_count=game_outcome["moves_count"],
                duration_seconds=duration,
                metadata={
                    "termination_reason": game_outcome["termination_reason"],
                    "in_memory": True,
                },
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Critical error in evaluate_step_in_memory for game {game_id}: {e}",
                exc_info=True,
            )

            return create_game_result(
                game_id=game_id,
                agent_info=agent_info,
                opponent_info=opponent_info,
                winner=None,
                moves_count=0,
                duration_seconds=duration,
                metadata={
                    "termination_reason": f"Evaluate_step_in_memory error: {str(e)}",
                    "in_memory": True,
                },
            )

    def get_opponents(self, context: EvaluationContext) -> List[OpponentInfo]:
        """Get the single opponent for this evaluation."""
        # Map opponent names to proper types based on supported opponent types
        opponent_type_mapping = {
            "random": "random",
            "heuristic": "heuristic",
            # If opponent_path is provided, it's likely a PPO agent
        }

        # Determine opponent type based on name and configuration
        opponent_name = self.config.get_strategy_param(
            "opponent_name", "default_opponent"
        )
        opponent_type = opponent_type_mapping.get(opponent_name.lower(), "unknown")

        # If we have a checkpoint path but the type is still unknown, assume it's a PPO agent
        opponent_path = self.config.get_strategy_param("opponent_path")
        if opponent_type == "unknown" and opponent_path:
            opponent_type = "ppo"

        return [
            OpponentInfo(
                name=self.config.get_strategy_param(
                    "opponent_name", "default_opponent"
                ),
                type=opponent_type,
                checkpoint_path=self.config.get_strategy_param("opponent_path"),
                metadata=self.config.get_strategy_param("opponent_params", {}),
            )
        ]

    def _calculate_game_distribution(self) -> dict:
        """
        Calculate how many games to play with each color configuration.

        Returns:
            Dictionary with game counts for each configuration
        """
        total_games = self.config.num_games

        if not self.config.get_strategy_param("play_as_both_colors", True):
            # All games with agent as first player
            return {"agent_first": total_games, "agent_second": 0}

        # Try to balance colors
        half_games = total_games // 2
        remainder = total_games % 2

        # Check if the distribution is within tolerance
        if total_games > 0:
            imbalance = abs(half_games - (half_games + remainder)) / total_games
            color_balance_tolerance = self.config.get_strategy_param(
                "color_balance_tolerance", 0.1
            )
            if imbalance > color_balance_tolerance:
                logger.warning(
                    "Color balance tolerance exceeded: %.3f > %.3f",
                    imbalance,
                    color_balance_tolerance,
                )

        return {
            "agent_first": half_games + remainder,  # Give extra game to agent_first
            "agent_second": half_games,
        }

    def _calculate_analytics(self, games: List[GameResult]) -> dict:
        """Calculate additional analytics for the evaluation."""
        if not games:
            return {}

        # Calculate color-specific statistics if applicable
        analytics = {}

        if self.config.get_strategy_param("play_as_both_colors", True):
            # Separate games by color (approximation based on metadata)
            first_player_games = [
                g
                for g in games
                if not g.opponent_info.metadata.get("agent_plays_second", False)
            ]
            second_player_games = [
                g
                for g in games
                if g.opponent_info.metadata.get("agent_plays_second", False)
            ]

            if first_player_games:
                first_wins = sum(1 for g in first_player_games if g.is_agent_win)
                analytics["first_player_win_rate"] = first_wins / len(
                    first_player_games
                )
                analytics["first_player_games"] = len(first_player_games)

            if second_player_games:
                second_wins = sum(1 for g in second_player_games if g.is_agent_win)
                analytics["second_player_win_rate"] = second_wins / len(
                    second_player_games
                )
                analytics["second_player_games"] = len(second_player_games)

        # Game length statistics
        moves_list = [g.moves_count for g in games]
        if moves_list:
            analytics["min_game_length"] = min(moves_list)
            analytics["max_game_length"] = max(moves_list)
            analytics["median_game_length"] = statistics.median(moves_list)

        # Duration statistics
        durations = [g.duration_seconds for g in games]
        if durations:
            analytics["min_duration"] = min(durations)
            analytics["max_duration"] = max(durations)
            analytics["median_duration"] = statistics.median(durations)

        return analytics

    def validate_config(self) -> bool:
        """Validate single opponent specific configuration."""
        if not super().validate_config():
            return False

        if not self.config.get_strategy_param("opponent_name", "default_opponent"):
            logger.error("Opponent name is required for single opponent evaluation")
            return False

        color_balance_tolerance = self.config.get_strategy_param(
            "color_balance_tolerance", 0.1
        )
        if color_balance_tolerance < 0 or color_balance_tolerance > 0.5:
            logger.error("Color balance tolerance must be between 0 and 0.5")
            return False

        return True


# Register this evaluator with the factory
from ..core import EvaluationStrategy, EvaluatorFactory

EvaluatorFactory.register(EvaluationStrategy.SINGLE_OPPONENT, SingleOpponentEvaluator)
