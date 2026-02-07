"""
Custom evaluation strategy implementation.

This strategy allows for flexible, user-defined evaluation configurations
that can be tailored to specific research or testing needs.
"""

import logging
from typing import Dict, List, Optional

import torch

from ..core import (
    AgentInfo,
    BaseEvaluator,
    EvaluationContext,
    EvaluationResult,
    EvaluationStrategy,
    EvaluatorFactory,
    GameResult,
    OpponentInfo,
    SummaryStats,
)
from keisei.config_schema import EvaluationConfig

logger = logging.getLogger(__name__)


class CustomEvaluator(BaseEvaluator):
    """
    Custom evaluation strategy with flexible configuration.

    This evaluator allows users to define custom evaluation scenarios
    through configuration parameters, making it ideal for research
    and experimentation.
    """

    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        self.config: EvaluationConfig = config

    def get_opponents(self, context: EvaluationContext) -> List[OpponentInfo]:
        """Get custom opponents from configuration."""
        # Custom strategy can define opponents in multiple ways
        custom_opponents_config = self.config.get_strategy_param("custom_opponents", [])

        opponents = []

        # Method 1: Direct opponent configuration list
        if isinstance(custom_opponents_config, list):
            for i, opp_config in enumerate(custom_opponents_config):
                if isinstance(opp_config, dict):
                    name = opp_config.get("name", f"custom_opponent_{i}")
                    opp_type = opp_config.get("type", "random")
                    checkpoint_path = opp_config.get("checkpoint_path")
                    metadata = opp_config.get("metadata", {})

                    opponents.append(
                        OpponentInfo(
                            name=name,
                            type=opp_type,
                            checkpoint_path=checkpoint_path,
                            metadata=metadata,
                        )
                    )

        # Method 2: Opponent pool reference
        opponent_pool_size = self.config.get_strategy_param("opponent_pool_size", 0)
        if opponent_pool_size > 0 and not opponents:
            # Create opponents from pool configuration
            pool_type = self.config.get_strategy_param("opponent_pool_type", "random")
            for i in range(opponent_pool_size):
                opponents.append(
                    OpponentInfo(
                        name=f"pool_opponent_{i}",
                        type=pool_type,
                        checkpoint_path=None,
                        metadata={"pool_index": i},
                    )
                )

        # Method 3: Single custom opponent
        single_opponent_config = self.config.get_strategy_param("single_opponent", None)
        if single_opponent_config and not opponents:
            if isinstance(single_opponent_config, dict):
                opponents.append(
                    OpponentInfo(
                        name=single_opponent_config.get("name", "custom_single"),
                        type=single_opponent_config.get("type", "random"),
                        checkpoint_path=single_opponent_config.get("checkpoint_path"),
                        metadata=single_opponent_config.get("metadata", {}),
                    )
                )

        # Fallback: If no opponents configured, provide a default
        if not opponents:
            logger.warning(
                "No custom opponents configured, using default random opponent"
            )
            opponents.append(
                OpponentInfo(
                    name="default_custom_random",
                    type="random",
                    checkpoint_path=None,
                    metadata={"description": "Default opponent for custom evaluation"},
                )
            )

        logger.info(f"Custom evaluation configured with {len(opponents)} opponents")
        return opponents

    async def evaluate(
        self, agent_info: AgentInfo, context: Optional[EvaluationContext] = None
    ) -> EvaluationResult:
        """Run custom evaluation with the specified agent."""
        if context is None:
            context = self.setup_context(agent_info)

        self.log_evaluation_start(agent_info, context)

        # Get opponents from configuration
        opponents = self.get_opponents(context)

        # Determine evaluation mode
        evaluation_mode = self.config.get_strategy_param(
            "evaluation_mode", "round_robin"
        )

        if evaluation_mode == "round_robin":
            results = await self._run_round_robin_evaluation(
                agent_info, opponents, context
            )
        elif evaluation_mode == "single_elimination":
            results = await self._run_single_elimination_evaluation(
                agent_info, opponents, context
            )
        elif evaluation_mode == "custom_sequence":
            results = await self._run_custom_sequence_evaluation(
                agent_info, opponents, context
            )
        else:
            logger.warning(
                f"Unknown evaluation mode: {evaluation_mode}, using round_robin"
            )
            results = await self._run_round_robin_evaluation(
                agent_info, opponents, context
            )

        evaluation_result = EvaluationResult(
            context=context,
            games=results,
            summary_stats=SummaryStats.from_games(results),
            analytics_data={
                "custom_evaluation_analytics": {
                    "evaluation_mode": evaluation_mode,
                    "total_opponents": len(opponents),
                    "configuration": self.config.strategy_params,
                }
            },
            errors=[],
        )

        self.log_evaluation_complete(evaluation_result)
        return evaluation_result

    async def evaluate_in_memory(
        self,
        agent_info: AgentInfo,
        context: Optional[EvaluationContext] = None,
        *,
        agent_weights: Optional[Dict[str, torch.Tensor]] = None,
        opponent_weights: Optional[Dict[str, torch.Tensor]] = None,
        opponent_info: Optional[OpponentInfo] = None,
    ) -> EvaluationResult:
        """Run custom evaluation using in-memory model weights."""
        if context is None:
            context = self.setup_context(agent_info)

        self.log_evaluation_start(agent_info, context)

        # Create in-memory agent with provided weights
        if agent_weights is not None:
            in_memory_agent = AgentInfo(
                name=agent_info.name,
                checkpoint_path=agent_info.checkpoint_path,
                metadata={
                    **(agent_info.metadata or {}),
                    "agent_weights": agent_weights,
                    "use_in_memory": True,
                },
            )
        else:
            in_memory_agent = agent_info

        # Get opponents (potentially using in-memory configuration)
        opponents = self.get_opponents(context)

        # If specific opponent info and weights provided, use those
        if opponent_info is not None and opponent_weights is not None:
            in_memory_opponent = OpponentInfo(
                name=opponent_info.name,
                type=opponent_info.type,
                checkpoint_path=opponent_info.checkpoint_path,
                metadata={
                    **(opponent_info.metadata or {}),
                    "opponent_weights": opponent_weights,
                    "use_in_memory": True,
                },
            )
            # Use this single opponent for evaluation
            opponents = [in_memory_opponent]

        # Run evaluation with in-memory configuration
        evaluation_mode = self.config.get_strategy_param(
            "evaluation_mode", "round_robin"
        )

        if evaluation_mode == "round_robin":
            results = await self._run_round_robin_evaluation(
                in_memory_agent, opponents, context
            )
        elif evaluation_mode == "single_elimination":
            results = await self._run_single_elimination_evaluation(
                in_memory_agent, opponents, context
            )
        elif evaluation_mode == "custom_sequence":
            results = await self._run_custom_sequence_evaluation(
                in_memory_agent, opponents, context
            )
        else:
            results = await self._run_round_robin_evaluation(
                in_memory_agent, opponents, context
            )

        evaluation_result = EvaluationResult(
            context=context,
            games=results,
            summary_stats=SummaryStats.from_games(results),
            analytics_data={
                "custom_evaluation_analytics": {
                    "evaluation_mode": evaluation_mode,
                    "total_opponents": len(opponents),
                    "in_memory_evaluation": True,
                    "configuration": self.config.strategy_params,
                }
            },
            errors=[],
        )

        self.log_evaluation_complete(evaluation_result)
        return evaluation_result

    async def evaluate_step(
        self,
        agent_info: AgentInfo,
        opponent_info: OpponentInfo,
        context: EvaluationContext,
    ) -> GameResult:
        """Evaluate a single step (game) against an opponent."""
        raise NotImplementedError(
            "CustomEvaluator.evaluate_step() must be implemented with actual game "
            "execution logic. Subclass CustomEvaluator and override this method."
        )

    async def _run_round_robin_evaluation(
        self,
        agent_info: AgentInfo,
        opponents: List[OpponentInfo],
        context: EvaluationContext,
    ) -> List[GameResult]:
        """Run round-robin evaluation against all opponents."""
        results = []
        games_per_opponent = self.config.get_strategy_param("games_per_opponent", 1)

        for opponent in opponents:
            for game_num in range(games_per_opponent):
                try:
                    # Create opponent with game-specific metadata
                    game_opponent = OpponentInfo(
                        name=opponent.name,
                        type=opponent.type,
                        checkpoint_path=opponent.checkpoint_path,
                        metadata={
                            **(opponent.metadata or {}),
                            "game_index": game_num,
                            "agent_plays_sente_in_eval_step": (game_num % 2) == 0,
                        },
                    )

                    result = await self.evaluate_step(
                        agent_info, game_opponent, context
                    )
                    results.append(result)

                except Exception as e:
                    logger.error(
                        f"Error in round-robin game {game_num} vs {opponent.name}: {e}"
                    )

        return results

    async def _run_single_elimination_evaluation(
        self,
        agent_info: AgentInfo,
        opponents: List[OpponentInfo],
        context: EvaluationContext,
    ) -> List[GameResult]:
        """Run single-elimination tournament evaluation."""
        results = []

        # Single elimination: agent plays against each opponent until first loss
        for opponent in opponents:
            try:
                result = await self.evaluate_step(agent_info, opponent, context)
                results.append(result)

                # If agent lost, stop the elimination
                if result.winner == 1:  # Opponent won
                    logger.info(f"Agent eliminated by {opponent.name}")
                    break

            except Exception as e:
                logger.error(f"Error in elimination game vs {opponent.name}: {e}")
                break

        return results

    async def _run_custom_sequence_evaluation(
        self,
        agent_info: AgentInfo,
        opponents: List[OpponentInfo],
        context: EvaluationContext,
    ) -> List[GameResult]:
        """Run custom sequence evaluation based on configuration."""
        results = []

        # Get custom sequence configuration
        custom_sequence = self.config.get_strategy_param("custom_sequence", [])

        if not custom_sequence:
            logger.warning("No custom sequence defined, falling back to round-robin")
            return await self._run_round_robin_evaluation(
                agent_info, opponents, context
            )

        # Execute custom sequence
        for step in custom_sequence:
            if isinstance(step, dict):
                opponent_name = step.get("opponent")
                games = step.get("games", 1)

                # Find the specified opponent
                opponent = next(
                    (opp for opp in opponents if opp.name == opponent_name), None
                )
                if opponent is None:
                    logger.warning(
                        f"Opponent '{opponent_name}' not found in sequence step"
                    )
                    continue

                # Play the specified number of games
                for game_num in range(games):
                    try:
                        game_opponent = OpponentInfo(
                            name=opponent.name,
                            type=opponent.type,
                            checkpoint_path=opponent.checkpoint_path,
                            metadata={
                                **(opponent.metadata or {}),
                                "sequence_step": len(results),
                                "game_index": game_num,
                                "agent_plays_sente_in_eval_step": (game_num % 2) == 0,
                            },
                        )

                        result = await self.evaluate_step(
                            agent_info, game_opponent, context
                        )
                        results.append(result)

                    except Exception as e:
                        logger.error(
                            f"Error in custom sequence game vs {opponent_name}: {e}"
                        )

        return results


# Register this evaluator with the factory
EvaluatorFactory.register(EvaluationStrategy.CUSTOM, CustomEvaluator)
