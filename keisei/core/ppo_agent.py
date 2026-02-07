"""
Minimal PPOAgent for DRL Shogi Client.
"""

import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

from keisei.config_schema import AppConfig
from keisei.core.actor_critic_protocol import ActorCriticProtocol
from keisei.core.experience_buffer import ExperienceBuffer
from keisei.core.scheduler_factory import SchedulerFactory
from keisei.utils import PolicyOutputMapper
from keisei.utils.unified_logger import log_error_to_stderr, log_info_to_stderr

if TYPE_CHECKING:
    from keisei.shogi.shogi_core_definitions import MoveTuple


class PPOAgent:
    """Proximal Policy Optimization agent for Shogi (PPO logic)."""

    def __init__(
        self,
        model: ActorCriticProtocol,
        config: AppConfig,
        device: torch.device,
        name: str = "PPOAgent",
        scaler=None,
        use_mixed_precision: bool = False,
    ):
        """
        Initialize the PPOAgent with model, optimizer, and PPO hyperparameters.

        Args:
            model: ActorCritic model instance conforming to ActorCriticProtocol
            config: Application configuration
            device: PyTorch device for training
            name: Agent name for identification
        """
        self.config = config.model_copy(
            deep=True
        )  # Ensure the agent stores a deep copy
        self.device = device
        self.name = name

        # Mixed precision support
        self.scaler = scaler
        self.use_mixed_precision = use_mixed_precision

        # Direct model assignment (dependency injection)
        self.model: ActorCriticProtocol = model.to(self.device)

        # Initialize policy mapper
        policy_output_mapper = PolicyOutputMapper()
        self.policy_output_mapper = policy_output_mapper
        self.num_actions_total = self.policy_output_mapper.get_total_actions()
        # Add weight_decay from config if present, else default to 0.0
        weight_decay = getattr(config.training, "weight_decay", 0.0)
        # Initialize optimizer, handle invalid learning rate gracefully
        try:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=weight_decay,
            )
        except Exception as e:
            # Fallback to default learning rate on error
            log_error_to_stderr(
                "PPOAgent",
                f"Could not initialize optimizer with lr={config.training.learning_rate}, using default lr=1e-3: {e}",
            )
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=1e-3, weight_decay=weight_decay
            )

        # PPO hyperparameters
        self.gamma = config.training.gamma
        self.clip_epsilon = config.training.clip_epsilon
        self.value_loss_coeff = config.training.value_loss_coeff
        self.entropy_coef = config.training.entropy_coef
        self.ppo_epochs = config.training.ppo_epochs
        self.minibatch_size = config.training.minibatch_size
        # Normalization options
        self.normalize_advantages = getattr(
            config.training, "normalize_advantages", True
        )

        self.last_kl_div = 0.0  # Initialize KL divergence tracker
        self.gradient_clip_max_norm = (
            config.training.gradient_clip_max_norm
        )  # Added from config

        # Use a numpy Generator for shuffling
        self._rng = np.random.default_rng(getattr(config.env, "seed", None))

        # Learning rate scheduler configuration
        self.lr_schedule_type = config.training.lr_schedule_type
        self.lr_schedule_step_on = config.training.lr_schedule_step_on
        total_steps = self._calculate_total_scheduler_steps(config)
        self.scheduler = SchedulerFactory.create_scheduler(
            optimizer=self.optimizer,
            schedule_type=self.lr_schedule_type,
            total_steps=total_steps,
            schedule_kwargs=config.training.lr_schedule_kwargs,
        )

        # Metrics exposed for UI instrumentation
        self.last_gradient_norm: float = 0.0

    def _calculate_total_scheduler_steps(self, config: AppConfig) -> int:
        """Calculate total number of scheduler steps based on configuration."""
        if config.training.lr_schedule_step_on == "epoch":
            # epochs = total_timesteps // steps_per_epoch
            return (
                config.training.total_timesteps // config.training.steps_per_epoch
            ) * config.training.ppo_epochs
        else:
            # updates per epoch = steps_per_epoch // minibatch_size * ppo_epochs
            updates_per_epoch = (
                config.training.steps_per_epoch // config.training.minibatch_size
            ) * config.training.ppo_epochs
            # number of epochs = total_timesteps // steps_per_epoch
            num_epochs = (
                config.training.total_timesteps // config.training.steps_per_epoch
            )
            return num_epochs * updates_per_epoch

    def select_action(
        self,
        obs: np.ndarray,
        legal_mask: torch.Tensor,
        *,  # Force is_training to be keyword-only
        is_training: bool = True,
    ) -> Tuple[
        Optional["MoveTuple"],
        int,
        float,
        float,
    ]:
        """
        Select an action given an observation, legal Shogi moves, and a precomputed legal_mask.
        Returns the selected Shogi move, its policy index, log probability, and value estimate.
        """
        self.model.train(is_training)

        obs_tensor = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # Apply observation normalization if a compatible scaler is provided
        if self.scaler is not None and not isinstance(self.scaler, GradScaler):
            if hasattr(self.scaler, "transform"):
                obs_tensor = self.scaler.transform(obs_tensor)
            else:
                obs_tensor = self.scaler(obs_tensor)

        if not legal_mask.any():
            log_error_to_stderr(
                "PPOAgent",
                "select_action called with no legal moves (based on input legal_mask)",
            )
            # Fallback behavior might be needed if model.get_action_and_value can't handle all-False mask.
            # neural_network.py's get_action_and_value attempts to handle this.
            # If this path is hit, it implies the caller might not have checked for no legal moves.
            # The train.py logic should ideally prevent calling select_action if no legal_moves.
            # Let it proceed, model.get_action_and_value will use the all-false mask.

        # Get action, log_prob, and value from the ActorCritic model
        # Pass deterministic based on not is_training
        if is_training:
            with torch.no_grad():
                (
                    selected_policy_index_tensor,
                    log_prob_tensor,
                    value_tensor,
                ) = self.model.get_action_and_value(
                    obs_tensor,
                    legal_mask=legal_mask,
                    deterministic=not is_training,
                )
        else:
            (
                selected_policy_index_tensor,
                log_prob_tensor,
                value_tensor,
            ) = self.model.get_action_and_value(
                obs_tensor,
                legal_mask=legal_mask,
                deterministic=not is_training,
            )

        selected_policy_index_val = int(selected_policy_index_tensor.item())
        log_prob_val = float(log_prob_tensor.item())
        value_float = float(
            value_tensor.item()
        )  # Value is already squeezed in get_action_and_value

        selected_shogi_move: Optional["MoveTuple"] = None
        try:
            selected_shogi_move = self.policy_output_mapper.policy_index_to_shogi_move(
                selected_policy_index_val
            )
        except IndexError as e:
            log_error_to_stderr(
                "PPOAgent",
                f"Policy index {selected_policy_index_val} out of bounds in select_action: {e}",
            )
            # Handle by returning no move or re-raising, depending on desired robustness.
            return None, -1, 0.0, value_float
            # Or raise the error

        return (
            selected_shogi_move,
            selected_policy_index_val,
            log_prob_val,
            value_float,
        )

    def get_value(self, obs_np: np.ndarray) -> float:
        """Get the value prediction from the critic for a given NumPy observation."""
        self.model.eval()
        obs_tensor = torch.as_tensor(
            obs_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        if self.scaler is not None and not isinstance(self.scaler, GradScaler):
            if hasattr(self.scaler, "transform"):
                obs_tensor = self.scaler.transform(obs_tensor)
            else:
                obs_tensor = self.scaler(obs_tensor)
        with torch.no_grad():
            _, _, value_estimate = self.model.get_action_and_value(
                obs_tensor, deterministic=True
            )
        return float(value_estimate.item())

    def learn(self, experience_buffer: ExperienceBuffer) -> Dict[str, float]:
        """
        Perform PPO update using experiences from the buffer.
        Returns a dictionary of logging metrics.
        """
        self.model.train()

        batch_data = experience_buffer.get_batch()
        # It's good practice for ExperienceBuffer.get_batch() to already place tensors on self.device
        # or for the buffer itself to live on self.device if memory allows.
        # If not, the .to(self.device) calls below are necessary.

        current_lr = self.optimizer.param_groups[0]["lr"]

        if not batch_data or batch_data["obs"].shape[0] == 0:
            log_error_to_stderr("PPOAgent", "learn called with empty batch_data")
            return {
                "ppo/policy_loss": 0.0,
                "ppo/value_loss": 0.0,
                "ppo/entropy": 0.0,
                "ppo/kl_divergence_approx": self.last_kl_div,
                "ppo/learning_rate": current_lr,
            }

        obs_batch = batch_data["obs"].to(self.device)
        actions_batch = batch_data["actions"].to(self.device)
        old_log_probs_batch = batch_data["log_probs"].to(self.device)
        old_values_batch = batch_data["values"].to(self.device)
        advantages_batch = batch_data["advantages"].to(self.device)
        returns_batch = batch_data["returns"].to(self.device)
        legal_masks_batch = batch_data["legal_masks"].to(self.device)

        # Conditionally normalize advantages based on configuration
        if self.normalize_advantages:
            advantage_std = advantages_batch.std()
            # Only normalize if we have multiple samples and non-zero std
            if advantage_std > 1e-8 and advantages_batch.shape[0] > 1:
                advantages_batch = (
                    advantages_batch - advantages_batch.mean()
                ) / advantage_std
            # For single sample or zero std, skip normalization to avoid numerical issues

        num_samples = obs_batch.shape[0]
        indices = np.arange(num_samples)

        total_policy_loss_epoch, total_value_loss_epoch, total_entropy_epoch = (
            0.0,
            0.0,
            0.0,
        )
        total_kl_div = 0.0
        total_clip_frac = 0.0
        num_updates = 0

        for _ in range(self.ppo_epochs):
            self._rng.shuffle(indices)
            for start_idx in range(0, num_samples, self.minibatch_size):
                end_idx = start_idx + self.minibatch_size
                minibatch_indices = indices[start_idx:end_idx]

                obs_minibatch = obs_batch[minibatch_indices]
                actions_minibatch = actions_batch[minibatch_indices]
                old_log_probs_minibatch = old_log_probs_batch[minibatch_indices]
                old_values_minibatch = old_values_batch[minibatch_indices]
                advantages_minibatch = advantages_batch[minibatch_indices]
                returns_minibatch = returns_batch[minibatch_indices]
                legal_masks_minibatch = legal_masks_batch[minibatch_indices]

                # Apply observation normalization if scaler provided
                if self.scaler is not None and not isinstance(self.scaler, GradScaler):
                    if hasattr(self.scaler, "transform"):
                        obs_minibatch = self.scaler.transform(obs_minibatch)
                    else:
                        obs_minibatch = self.scaler(obs_minibatch)

                # Get new log_probs, entropy, and value from the model
                # Note on entropy: legal_mask is now passed here. Entropy is calculated
                # over legal actions only.
                if self.use_mixed_precision and isinstance(self.scaler, GradScaler):
                    # Mixed precision forward pass
                    with torch.cuda.amp.autocast():
                        new_log_probs, entropy, new_values = (
                            self.model.evaluate_actions(
                                obs_minibatch,
                                actions_minibatch,
                                legal_mask=legal_masks_minibatch,
                            )
                        )
                else:
                    # Standard precision forward pass
                    new_log_probs, entropy, new_values = self.model.evaluate_actions(
                        obs_minibatch,
                        actions_minibatch,
                        legal_mask=legal_masks_minibatch,
                    )

                # PPO Loss Calculation
                ratio = torch.exp(new_log_probs - old_log_probs_minibatch)

                clip_mask = (ratio - 1.0).abs() > self.clip_epsilon
                total_clip_frac += clip_mask.float().mean().item()

                # Calculate KL divergence approximation
                kl_div = (old_log_probs_minibatch - new_log_probs).mean()

                # Clipped surrogate objective
                surr1 = ratio * advantages_minibatch
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                    * advantages_minibatch
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss with optional clipping
                if self.config.training.enable_value_clipping:
                    values_pred_clipped = old_values_minibatch + torch.clamp(
                        new_values - old_values_minibatch,
                        -self.clip_epsilon,
                        self.clip_epsilon,
                    )
                    value_loss_unclipped = F.mse_loss(
                        new_values.squeeze(), returns_minibatch.squeeze()
                    )
                    value_loss_clipped = F.mse_loss(
                        values_pred_clipped.squeeze(), returns_minibatch.squeeze()
                    )
                    value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
                else:
                    value_loss = F.mse_loss(
                        new_values.squeeze(), returns_minibatch.squeeze()
                    )

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_loss_coeff * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()

                # Fix B4: Use mixed precision for backward pass if enabled
                if self.use_mixed_precision and isinstance(self.scaler, GradScaler):
                    # Mixed precision backward pass
                    self.scaler.scale(loss).backward()
                    # Unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.gradient_clip_max_norm,
                    )
                    # Compute gradient norm after unscaling and clipping
                    total = 0.0
                    for param in self.model.parameters():
                        if param.grad is not None:
                            norm_val = param.grad.detach().data.norm(2).item()
                            total += norm_val * norm_val
                    self.last_gradient_norm = total**0.5
                    # Update with scaler
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard precision backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.gradient_clip_max_norm,
                    )
                    total = 0.0
                    for param in self.model.parameters():
                        if param.grad is not None:
                            norm_val = param.grad.detach().data.norm(2).item()
                            total += norm_val * norm_val
                    self.last_gradient_norm = total**0.5
                    self.optimizer.step()

                # Step scheduler on minibatch update if configured
                if self.scheduler is not None and self.lr_schedule_step_on == "update":
                    self.scheduler.step()

                total_policy_loss_epoch += policy_loss.item()
                total_value_loss_epoch += value_loss.item()
                total_entropy_epoch += entropy_loss.item()
                total_kl_div += kl_div.item()
                num_updates += 1

        # Step scheduler on epoch if configured
        if self.scheduler is not None and self.lr_schedule_step_on == "epoch":
            self.scheduler.step()

        # Recalculate current learning rate after scheduler step
        current_lr = self.optimizer.param_groups[0]["lr"]
        # Compute average losses over updates
        avg_policy_loss = (
            total_policy_loss_epoch / num_updates if num_updates > 0 else 0.0
        )
        avg_value_loss = (
            total_value_loss_epoch / num_updates if num_updates > 0 else 0.0
        )
        avg_entropy = total_entropy_epoch / num_updates if num_updates > 0 else 0.0
        avg_kl_div = total_kl_div / num_updates if num_updates > 0 else 0.0
        avg_clip_frac = total_clip_frac / num_updates if num_updates > 0 else 0.0

        # Update tracked KL divergence
        self.last_kl_div = avg_kl_div

        # Compile metrics
        return {
            "ppo/policy_loss": avg_policy_loss,
            "ppo/value_loss": avg_value_loss,
            "ppo/entropy": avg_entropy,
            "ppo/kl_divergence_approx": avg_kl_div,
            "ppo/clip_fraction": avg_clip_frac,
            "ppo/learning_rate": current_lr,
        }

    def save_model(
        self,
        file_path: str,
        global_timestep: int = 0,
        total_episodes_completed: int = 0,
        stats_to_save: Optional[Dict[str, int]] = None,
    ) -> None:
        """Saves the model, optimizer, scheduler, and training state to a file."""
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_timestep": global_timestep,
            "total_episodes_completed": total_episodes_completed,
        }
        if stats_to_save:
            save_dict.update(stats_to_save)
        # Include scheduler state if present
        if self.scheduler is not None:
            save_dict["scheduler_state_dict"] = self.scheduler.state_dict()
            save_dict["lr_schedule_type"] = self.lr_schedule_type
            save_dict["lr_schedule_step_on"] = self.lr_schedule_step_on

        torch.save(save_dict, file_path)
        log_info_to_stderr(
            "PPOAgent", f"Model, optimizer, scheduler, and state saved to {file_path}"
        )

    def load_model(self, file_path: str) -> Dict[str, Any]:
        """Loads the model, optimizer, scheduler, and training state from a file."""
        if not os.path.exists(file_path):
            log_error_to_stderr("PPOAgent", f"Checkpoint file {file_path} not found")
            return {
                "global_timestep": 0,
                "total_episodes_completed": 0,
                "black_wins": 0,
                "white_wins": 0,
                "draws": 0,
                "error": "File not found",
            }
        try:
            checkpoint = torch.load(
                file_path, map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # Load scheduler state if present
            if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            result = {
                "global_timestep": checkpoint.get("global_timestep", 0),
                "total_episodes_completed": checkpoint.get(
                    "total_episodes_completed", 0
                ),
                "black_wins": checkpoint.get("black_wins", 0),
                "white_wins": checkpoint.get("white_wins", 0),
                "draws": checkpoint.get("draws", 0),
                "lr_schedule_type": checkpoint.get("lr_schedule_type", None),
                "lr_schedule_step_on": checkpoint.get("lr_schedule_step_on", "epoch"),
            }
            return result
        except (KeyError, RuntimeError, EOFError) as e:
            log_error_to_stderr(
                "PPOAgent", f"Error loading checkpoint from {file_path}: {e}"
            )
            return {
                "global_timestep": 0,
                "total_episodes_completed": 0,
                "black_wins": 0,
                "white_wins": 0,
                "draws": 0,
                "error": str(e),
            }

    def get_name(self) -> str:  # Added getter for name
        return self.name
