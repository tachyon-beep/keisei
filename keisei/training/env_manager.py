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

        Returns:
            Tuple[ShogiGame, PolicyOutputMapper]: The initialized game and policy mapper.
        """
        try:
            # Initialize the Shogi game
            self.game = ShogiGame(max_moves_per_game=self.config.env.max_moves_per_game)

            # Setup seeding if specified
            if hasattr(self.game, "seed") and self.config.env.seed is not None:
                try:
                    self.game.seed(self.config.env.seed)
                    self.logger_func(f"Environment seeded with: {self.config.env.seed}")
                except Exception as e:
                    self.logger_func(f"Warning: Failed to seed environment: {e}")

            # Setup observation space shape
            self.obs_space_shape = (self.config.env.input_channels, SHOGI_BOARD_SIZE, SHOGI_BOARD_SIZE)

        except (RuntimeError, ValueError, OSError) as e:
            self.logger_func(f"Error initializing ShogiGame: {e}. Aborting.")
            raise RuntimeError(f"Failed to initialize ShogiGame: {e}") from e

        # Initialize policy output mapper
        try:
            self.policy_output_mapper = PolicyOutputMapper()
            self.action_space_size = self.policy_output_mapper.get_total_actions()

            # Validate action space consistency
            self._validate_action_space()

        except (RuntimeError, ValueError) as e:
            self.logger_func(f"Error initializing PolicyOutputMapper: {e}")
            raise RuntimeError(f"Failed to initialize PolicyOutputMapper: {e}") from e

        return self.game, self.policy_output_mapper

    def _validate_action_space(self):
        """Validate that action space configuration is consistent."""
        if self.policy_output_mapper is None:
            # This should not happen if called after policy_output_mapper is initialized
            self.logger_func(
                "CRITICAL: PolicyOutputMapper not initialized before _validate_action_space."
            )
            raise ValueError("PolicyOutputMapper not initialized.")

        config_num_actions = self.config.env.num_actions_total
        mapper_num_actions = self.policy_output_mapper.get_total_actions()

        if config_num_actions != mapper_num_actions:
            error_msg = (
                f"Action space mismatch: config specifies {config_num_actions} "
                f"actions but PolicyOutputMapper provides {mapper_num_actions} actions"
            )
            self.logger_func(f"CRITICAL: {error_msg}")
            raise ValueError(error_msg)

        self.logger_func(f"Action space validated: {mapper_num_actions} total actions")

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
        """
        Resets the game environment and returns the initial observation.
        ShogiGame.reset() is expected to return the initial observation.

        Returns:
            Optional[np.ndarray]: The initial observation from the environment, or None on error.
        """
        if not self.game:
            self.logger_func("Error: Game not initialized. Cannot get initial state.")
            return None
        try:
            # ShogiGame.reset() now returns the observation directly.
            initial_obs = self.game.reset()
            self.logger_func(
                "Game state initialized and initial observation obtained from game.reset()."
            )
            return initial_obs
        except Exception as e:
            self.logger_func(f"Error initializing game state: {e}")
            return None

    def validate_environment(self) -> bool:
        """
        Validate that the environment is properly configured and functional.

        Returns:
            bool: True if environment is valid, False otherwise
        """
        try:
            # Check game initialization
            if self.game is None:
                self.logger_func("Environment validation failed: game not initialized")
                return False

            # Check policy mapper
            if self.policy_output_mapper is None:
                self.logger_func(
                    "Environment validation failed: policy mapper not initialized"
                )
                return False

            # Check action space consistency
            if self.action_space_size <= 0:
                self.logger_func(
                    "Environment validation failed: invalid action space size"
                )
                return False

            # Test game reset functionality
            # Get initial observation, reset, then get another and compare
            # This assumes get_observation() is available and returns a comparable state.
            if not self.game:  # Should be caught by earlier check, but good practice
                self.logger_func(
                    "Environment validation failed: game not initialized for reset test."
                )
                return False

            obs1 = self.game.get_observation()
            if not self.reset_game():  # Resets the game
                self.logger_func("Environment validation failed: game reset failed")
                return False
            obs2_after_reset = self.game.get_observation()  # Get obs after reset

            # Simple comparison; for complex objects, a more robust comparison might be needed
            if not np.array_equal(obs1, obs2_after_reset):
                self.logger_func(
                    "Environment validation warning: Observation after reset differs from initial observation. This might be expected if seeding is not deterministic or initial state has randomness."
                )
            # Depending on game logic, obs1 and obs2_after_reset should ideally be the same if reset is deterministic.
            # For now, we just check if reset_game() itself succeeded.

            # Test observation space
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
