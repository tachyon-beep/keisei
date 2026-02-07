"""
Model weight management for in-memory evaluation.

This module provides functionality to extract, cache, and manage model weights
for efficient in-memory evaluation without file I/O overhead.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# Top-level imports
from keisei.config_schema import (
    AppConfig,
    DemoConfig,
    EnvConfig,
    EvaluationConfig,
    LoggingConfig,
    ParallelConfig,
    TrainingConfig,
    WandBConfig,
)
from keisei.core.neural_network import ActorCritic
from keisei.core.ppo_agent import PPOAgent
from keisei.utils import PolicyOutputMapper

logger = logging.getLogger(__name__)


class ModelWeightManager:
    """Manages model weights for in-memory evaluation."""

    def __init__(self, device: str = "cpu", max_cache_size: int = 5):
        """
        Initialize the ModelWeightManager.

        Args:
            device: Device to store cached weights on
            max_cache_size: Maximum number of opponent weights to cache
        """
        self.device = torch.device(device)
        self.max_cache_size = max_cache_size
        self._weight_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self._cache_order: List[str] = []  # For LRU eviction
        self._cache_hits = 0  # Track cache hits
        self._cache_misses = 0  # Track cache misses

    def extract_agent_weights(self, agent: PPOAgent) -> Dict[str, torch.Tensor]:
        """
        Extract and clone current agent weights for evaluation.

        Args:
            agent: PPOAgent instance to extract weights from

        Returns:
            Dictionary of cloned model weights on CPU

        Raises:
            ValueError: If agent doesn't have a model attribute
        """
        if not hasattr(agent, "model") or agent.model is None:
            raise ValueError("Agent must have a model attribute")

        # Clone weights to avoid interference with training
        weights = {}
        for name, param in agent.model.state_dict().items():
            weights[name] = param.clone().detach().cpu()

        logger.debug("Extracted %d weight tensors from agent", len(weights))
        return weights

    def cache_opponent_weights(
        self, opponent_id: str, checkpoint_path: Path
    ) -> Dict[str, torch.Tensor]:
        """
        Load and cache opponent weights from checkpoint.

        Args:
            opponent_id: Unique identifier for the opponent
            checkpoint_path: Path to the checkpoint file

        Returns:
            Dictionary of model weights

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            RuntimeError: If checkpoint loading fails
        """
        # Check if already cached
        if opponent_id in self._weight_cache:
            # Move to end for LRU
            self._cache_order.remove(opponent_id)
            self._cache_order.append(opponent_id)
            self._cache_hits += 1  # Track cache hit
            logger.debug("Using cached weights for opponent %s", opponent_id)
            return self._weight_cache[opponent_id]

        # Evict oldest if cache is full
        while len(self._weight_cache) >= self.max_cache_size:
            oldest_id = self._cache_order.pop(0)
            del self._weight_cache[oldest_id]
            logger.debug("Evicted cached weights for opponent %s", oldest_id)

        # Load checkpoint
        try:
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )

            # Extract model weights from checkpoint
            if "model_state_dict" in checkpoint:
                weights = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                weights = checkpoint["state_dict"]
            else:
                weights = checkpoint

            # Ensure weights are on CPU
            cpu_weights = {}
            for name, param in weights.items():
                if isinstance(param, torch.Tensor):
                    cpu_weights[name] = param.cpu()
                else:
                    cpu_weights[name] = param

            # Cache the weights
            self._weight_cache[opponent_id] = cpu_weights
            self._cache_order.append(opponent_id)

            # Increment miss counter only after successful cache operation
            self._cache_misses += 1

            logger.debug(
                "Cached weights for opponent %s (%d tensors)",
                opponent_id,
                len(cpu_weights),
            )
            return cpu_weights

        except Exception as e:
            logger.error("Failed to load weights from %s: %s", checkpoint_path, e)
            raise RuntimeError(f"Checkpoint loading failed: {e}") from e

    def create_agent_from_weights(
        self,
        weights: Dict[str, torch.Tensor],
        agent_class=PPOAgent,
        config: Any = None,
        device: Optional[str] = None,
    ) -> PPOAgent:
        """
        Create an agent instance from weights.

        Args:
            weights: Dictionary of model weights
            agent_class: Agent class to instantiate
            config: Configuration for agent creation
            device: Device to place the agent on

        Returns:
            Agent instance with loaded weights

        Raises:
            ValueError: If required parameters are missing
            RuntimeError: If agent creation fails
        """
        try:
            # Determine device
            target_device = torch.device(device or self.device)

            # Infer model architecture from weights
            input_channels = self._infer_input_channels_from_weights(weights)
            total_actions = self._infer_total_actions_from_weights(weights)
            conv_out_channels = self._infer_conv_out_channels_from_weights(weights)
            flattened_size = self._infer_flattened_size_from_weights(weights)

            # Create minimal config if not provided
            if config is None:
                config = self._create_minimal_config(
                    input_channels, total_actions, target_device
                )

            # Create a dynamic ActorCritic model that matches the weights architecture
            model = self._create_dynamic_actor_critic(
                input_channels, total_actions, conv_out_channels, flattened_size
            ).to(target_device)

            # Create agent with model using dependency injection
            agent = agent_class(
                model=model,
                config=config,
                device=target_device,
                name="WeightReconstructedAgent",
            )

            # Load weights into model
            # Ensure weights are on correct device
            device_weights = {}
            for tensor_name, tensor in weights.items():
                if isinstance(tensor, torch.Tensor):
                    device_weights[tensor_name] = tensor.to(target_device)
                else:
                    device_weights[tensor_name] = tensor

            agent.model.load_state_dict(device_weights, strict=True)
            agent.model.eval()  # Set to evaluation mode

            logger.debug(
                "Successfully created agent from %d weight tensors", len(weights)
            )
            return agent

        except Exception as e:
            logger.error("Failed to create agent from weights: %s", e)
            raise RuntimeError(f"Agent creation from weights failed: {e}") from e

    def _infer_total_actions_from_weights(
        self, weights: Dict[str, torch.Tensor]
    ) -> int:
        """
        Infer the total number of actions from model weights.

        Args:
            weights: Dictionary of model weights

        Returns:
            Total number of actions
        """
        # Look for policy head layer to infer number of actions
        policy_layer_names = [
            "policy_head.weight",
            "actor_head.weight",
            "policy.weight",
        ]

        for layer_name in policy_layer_names:
            if layer_name in weights and isinstance(weights[layer_name], torch.Tensor):
                tensor = weights[layer_name]
                if (
                    len(tensor.shape) == 2
                ):  # Linear layer weight shape: [out_features, in_features]
                    return tensor.shape[0]

        # Fallback to default action space from PolicyOutputMapper
        policy_mapper = PolicyOutputMapper()
        logger.warning(
            "Could not infer total actions from weights, using PolicyOutputMapper default"
        )
        return policy_mapper.get_total_actions()

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "cache_size": len(self._weight_cache),
            "max_cache_size": self.max_cache_size,
            "device": str(self.device),
            "cached_opponents": list(self._weight_cache.keys()),
            "cache_order": self._cache_order.copy(),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
        }

    def clear_cache(self) -> None:
        """Clear the weight cache."""
        self._weight_cache.clear()
        self._cache_order.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.debug("Cleared weight cache")

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get estimated memory usage of cached weights.

        Returns:
            Dictionary with memory usage statistics in MB
        """
        total_size = 0
        sizes_by_opponent = {}

        for opponent_id, weights in self._weight_cache.items():
            opponent_size = 0
            for _, tensor in weights.items():
                if isinstance(tensor, torch.Tensor):
                    opponent_size += tensor.numel() * tensor.element_size()

            sizes_by_opponent[opponent_id] = opponent_size / (
                1024 * 1024
            )  # Convert to MB
            total_size += opponent_size

        return {
            "total_mb": total_size / (1024 * 1024),
            "by_opponent_mb": sizes_by_opponent,
        }

    def _infer_input_channels_from_weights(
        self, weights: Dict[str, torch.Tensor]
    ) -> int:
        """
        Infer the number of input channels from model weights.

        Args:
            weights: Dictionary of model weights

        Returns:
            Number of input channels

        Raises:
            ValueError: If input channels cannot be inferred
        """
        # Look for common first layer names to infer input channels
        first_layer_names = ["conv.weight", "stem.weight", "conv1.weight"]

        for layer_name in first_layer_names:
            if layer_name in weights and isinstance(weights[layer_name], torch.Tensor):
                tensor = weights[layer_name]
                if (
                    len(tensor.shape) == 4
                ):  # Conv2d weight shape: [out_channels, in_channels, kernel_h, kernel_w]
                    return tensor.shape[1]

        # Fallback: look for any conv layer that might be the first layer
        for name, tensor in weights.items():
            if "conv" in name and "weight" in name and isinstance(tensor, torch.Tensor):
                if len(tensor.shape) == 4:  # Conv2d weight
                    return tensor.shape[1]

        # Default fallback
        logger.warning("Could not infer input channels from weights, using default 46")
        return 46

    def _infer_conv_out_channels_from_weights(
        self, weights: Dict[str, torch.Tensor]
    ) -> int:
        """Infer conv output channels from weights."""
        if "conv.weight" in weights:
            # conv.weight shape is [out_channels, in_channels, kernel_h, kernel_w]
            return weights["conv.weight"].shape[0]
        return 16  # Default fallback

    def _infer_flattened_size_from_weights(
        self, weights: Dict[str, torch.Tensor]
    ) -> int:
        """Infer flattened size from policy_head weight dimensions."""
        if "policy_head.weight" in weights:
            # policy_head.weight shape is [total_actions, flattened_size]
            return weights["policy_head.weight"].shape[1]
        return 1296  # Default fallback (16 * 9 * 9)

    def _create_dynamic_actor_critic(
        self,
        input_channels: int,
        total_actions: int,
        conv_out_channels: int,
        flattened_size: int,
    ) -> torch.nn.Module:
        """Create a dynamic ActorCritic model with inferred architecture."""
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.distributions import Categorical

        class DynamicActorCritic(nn.Module):
            def __init__(
                self, input_channels, total_actions, conv_out_channels, flattened_size
            ):
                super().__init__()
                self.conv = nn.Conv2d(
                    input_channels, conv_out_channels, kernel_size=3, padding=1
                )
                self.relu = nn.ReLU()
                self.flatten = nn.Flatten()

                self.policy_head = nn.Linear(flattened_size, total_actions)
                self.value_head = nn.Linear(flattened_size, 1)

            def forward(self, x):
                x = self.conv(x)
                x = self.relu(x)
                x = self.flatten(x)
                policy = self.policy_head(x)
                value = self.value_head(x)
                return policy, value

            def get_action_and_value(
                self,
                obs: torch.Tensor,
                legal_mask: Optional[torch.Tensor] = None,
                deterministic: bool = False,
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                """Get action and value for compatibility with ActorCriticProtocol."""
                logits, value = self.forward(obs)

                if legal_mask is not None:
                    logits = logits.masked_fill(legal_mask == 0, float("-inf"))

                probs = Categorical(logits=logits)
                if deterministic:
                    action = torch.argmax(logits, dim=-1)
                else:
                    action = probs.sample()

                return action, probs.log_prob(action), value.squeeze(-1)

            def evaluate_actions(
                self,
                obs: torch.Tensor,
                actions: torch.Tensor,
                legal_mask: Optional[torch.Tensor] = None,
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                """Evaluate actions for compatibility with ActorCriticProtocol."""
                logits, value = self.forward(obs)

                if legal_mask is not None:
                    logits = logits.masked_fill(legal_mask == 0, float("-inf"))

                probs = Categorical(logits=logits)
                return probs.log_prob(actions), probs.entropy(), value.squeeze(-1)

        return DynamicActorCritic(
            input_channels, total_actions, conv_out_channels, flattened_size
        )

    def _create_minimal_config(
        self, input_channels: int, total_actions: int, device: torch.device
    ) -> AppConfig:  # Changed return type hint
        """
        Create a minimal AppConfig for agent initialization.

        Args:
            input_channels: Number of input channels
            total_actions: Total number of actions
            device: Target device

        Returns:
            Minimal AppConfig instance
        """
        return AppConfig(
            parallel=ParallelConfig(
                enabled=False,
                num_workers=1,
                batch_size=32,
                sync_interval=100,
                compression_enabled=True,
                timeout_seconds=10.0,
                max_queue_size=1000,
                worker_seed_offset=1000,
            ),
            env=EnvConfig(
                device=str(device),
                input_channels=input_channels,
                num_actions_total=total_actions,
                seed=42,
                max_moves_per_game=500,
            ),
            training=TrainingConfig(
                total_timesteps=1,
                steps_per_epoch=1,
                ppo_epochs=1,
                minibatch_size=2,
                learning_rate=1e-4,
                gamma=0.99,
                clip_epsilon=0.2,
                value_loss_coeff=0.5,
                entropy_coef=0.01,
                input_features="core46",
                render_every_steps=1,
                refresh_per_second=4,
                enable_spinner=True,
                tower_depth=9,
                tower_width=256,
                se_ratio=0.25,
                model_type="resnet",
                mixed_precision=False,
                ddp=False,
                gradient_clip_max_norm=0.5,
                lambda_gae=0.95,
                checkpoint_interval_timesteps=10000,
                evaluation_interval_timesteps=50000,
                weight_decay=0.0,
                normalize_advantages=True,
                enable_value_clipping=False,  # Added missing argument
                lr_schedule_type=None,
                lr_schedule_kwargs=None,
                lr_schedule_step_on="epoch",
            ),
            evaluation=EvaluationConfig(
                num_games=1,
                opponent_type="random",
                evaluation_interval_timesteps=50000,
                enable_periodic_evaluation=False,
                max_moves_per_game=500,
                log_file_path_eval="/tmp/eval.log",
                wandb_log_eval=False,
                elo_registry_path=None,
                agent_id=None,
                opponent_id=None,
                previous_model_pool_size=5,
                # Added missing arguments with default/minimal values
                strategy="single_opponent",
                max_concurrent_games=1,
                timeout_per_game=600.0,
                randomize_positions=False,
                random_seed=None,
                save_games=False,
                save_path="/tmp/eval_games",
                log_level="INFO",
                update_elo=False,
                enable_in_memory_evaluation=True,  # Assuming True for this context
                model_weight_cache_size=5,  # Default from ModelWeightManager
                enable_parallel_execution=False,
                process_restart_threshold=3,
                temp_agent_device=str(device),
                clear_cache_after_evaluation=True,
            ),
            logging=LoggingConfig(
                log_file="/tmp/eval.log", model_dir="/tmp/", run_name="eval-run"
            ),
            wandb=WandBConfig(
                enabled=False,
                project="eval",
                entity=None,
                run_name_prefix="eval-run",
                watch_model=False,
                watch_log_freq=1000,
                watch_log_type="all",
                log_model_artifact=False,
            ),
            demo=DemoConfig(enable_demo_mode=False, demo_mode_delay=0.0),
        )
