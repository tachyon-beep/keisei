"""
training/env_manager.py: Environment management for Shogi RL training.

This module handles environment-related concerns including:
- Game environment initialization and configuration
- Policy output mapper setup and validation
- Action space configuration and validation
- Environment seeding
- Observation space setup
"""

from typing import Callable, Optional, Tuple

import numpy as np

from keisei.config_schema import AppConfig
from keisei.constants import SHOGI_BOARD_SIZE
from keisei.shogi import ShogiGame
from keisei.utils import PolicyOutputMapper


class EnvManager:
    """Manages environment setup and configuration for training runs."""

    def __init__(self, config: AppConfig, logger_func: Optional[Callable] = None):
        """
        Initialize the EnvManager.

        Args:
            config: Application configuration
            logger_func: Optional logging function for status messages
        """
        self.config = config
        self.logger_func = logger_func or (lambda msg: None)

        # Initialize environment components (will be set by setup_environment)
        self.game: Optional[ShogiGame] = None
        self.policy_output_mapper: Optional[PolicyOutputMapper] = None
        self.action_space_size: int = 0
        self.obs_space_shape: Optional[Tuple[int, int, int]] = None

        # Environment setup is now called explicitly by Trainer

    def setup_environment(self) -> Tuple[ShogiGame, PolicyOutputMapper]:
        """Initialize game environment and related components.

        All components are built into local variables first and only assigned
        to ``self`` once everything succeeds, avoiding partial initialization.

        Returns:
            Tuple[ShogiGame, PolicyOutputMapper]: The initialized game and policy mapper.
        """
        try:
            game = ShogiGame(max_moves_per_game=self.config.env.max_moves_per_game)

            if hasattr(game, "seed") and self.config.env.seed is not None:
                try:
                    game.seed(self.config.env.seed)
                    self.logger_func(f"Environment seeded with: {self.config.env.seed}")
                except Exception as e:
                    self.logger_func(f"Warning: Failed to seed environment: {e}")

            obs_space_shape = (self.config.env.input_channels, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE)

        except (RuntimeError, ValueError, OSError) as e:
            self.logger_func(f"Error initializing ShogiGame: {e}. Aborting.")
            raise RuntimeError(f"Failed to initialize ShogiGame: {e}") from e

        try:
            mapper = PolicyOutputMapper()
            action_space_size = mapper.get_total_actions()

            config_num_actions = self.config.env.num_actions_total
            if config_num_actions != action_space_size:
                error_msg = (
                    f"Action space mismatch: config specifies {config_num_actions} "
                    f"actions but PolicyOutputMapper provides {action_space_size} actions"
                )
                self.logger_func(f"CRITICAL: {error_msg}")
                raise ValueError(error_msg)

            self.logger_func(f"Action space validated: {action_space_size} total actions")

        except (RuntimeError, ValueError) as e:
            self.logger_func(f"Error initializing PolicyOutputMapper: {e}")
            raise RuntimeError(f"Failed to initialize PolicyOutputMapper: {e}") from e

        # Commit all state atomically
        self.game = game
        self.policy_output_mapper = mapper
        self.action_space_size = action_space_size
        self.obs_space_shape = obs_space_shape

        return self.game, self.policy_output_mapper
    def get_environment_info(self) -> dict:
        """Get information about the current environment configuration."""
        return {
            "game": self.game,
            "policy_mapper": self.policy_output_mapper,
            "action_space_size": self.action_space_size,
            "obs_space_shape": self.obs_space_shape,
            "input_channels": self.config.env.input_channels,
            "num_actions_total": self.config.env.num_actions_total,
            "seed": self.config.env.seed,
            "game_type": type(self.game).__name__,
            "policy_mapper_type": type(self.policy_output_mapper).__name__,
        }

    def reset_game(self):
        """Reset the game environment to initial state."""
        if not self.game:
            self.logger_func("Error: Game not initialized. Cannot reset.")
            return False
        try:
            self.game.reset()
            return True
        except Exception as e:
            self.logger_func(f"Error resetting game: {e}")
            return False

    def initialize_game_state(self) -> Optional[np.ndarray]:
        """Reset the game and return the initial observation.

        Returns:
            The initial observation, or None if the game is not initialized
            or the reset fails.
        """
        if not self.reset_game():
            return None
        try:
            return self.game.get_observation()
        except Exception as e:
            self.logger_func(f"Error getting initial observation: {e}")
            return None

    def validate_environment(self) -> bool:
        """Validate that the environment is properly configured.

        This is a non-destructive check — it does not reset or modify game state.

        Returns:
            True if the environment is valid, False otherwise.
        """
        try:
            if self.game is None:
                self.logger_func("Environment validation failed: game not initialized")
                return False

            if self.policy_output_mapper is None:
                self.logger_func(
                    "Environment validation failed: policy mapper not initialized"
                )
                return False

            if self.action_space_size <= 0:
                self.logger_func(
                    "Environment validation failed: invalid action space size"
                )
                return False

            if self.obs_space_shape is None or len(self.obs_space_shape) != 3:
                self.logger_func(
                    "Environment validation failed: invalid observation space shape"
                )
                return False

            self.logger_func("Environment validation passed")
            return True

        except Exception as e:
            self.logger_func(f"Environment validation failed with exception: {e}")
            return False

    def get_legal_moves_count(self) -> int:
        """Get the number of legal moves in the current game state."""
        if not self.game:
            self.logger_func(
                "Error: Game not initialized. Cannot get legal moves count."
            )
            return 0
        try:
            legal_moves = self.game.get_legal_moves()
            return len(legal_moves) if legal_moves else 0
        except Exception as e:
            self.logger_func(f"Error getting legal moves count: {e}")
            return 0

    def setup_seeding(self, seed: Optional[int] = None):
        """
        Setup seeding for the environment.

        Args:
            seed: Optional seed value. If None, uses config seed.
        """
        seed_value = seed if seed is not None else self.config.env.seed

        if not self.game:
            self.logger_func("Error: Game not initialized. Cannot set seed.")
            return False

        if seed_value is not None and hasattr(self.game, "seed"):
            try:
                self.game.seed(seed_value)
                self.logger_func(f"Environment re-seeded with: {seed_value}")
                return True
            except Exception as e:
                self.logger_func(f"Error setting environment seed: {e}")
                return False
        elif seed_value is None:
            self.logger_func("No seed value provided for re-seeding.")
            return False  # Or True if no-op is considered success
        else:  # game does not have seed method
            self.logger_func(
                f"Warning: Game object does not have a 'seed' method. Cannot re-seed with {seed_value}."
            )
            return False
