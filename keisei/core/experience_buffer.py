"""
Minimal ExperienceBuffer for DRL Shogi Client.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch  # Ensure torch is imported

from keisei.constants import CORE_OBSERVATION_CHANNELS, FULL_ACTION_SPACE, SHOGI_BOARD_SIZE
from keisei.utils.unified_logger import log_warning_to_stderr


@dataclass
class Experience:
    """Single experience transition for parallel collection."""

    obs: torch.Tensor
    action: int
    reward: float
    log_prob: float
    value: float
    done: bool
    legal_mask: torch.Tensor


class ExperienceBuffer:
    """Experience buffer for storing transitions during RL training."""

    def __init__(
        self,
        buffer_size: int,
        gamma: float,
        lambda_gae: float,
        device: str = "cpu",
        obs_channels: int = CORE_OBSERVATION_CHANNELS,
        board_size: int = SHOGI_BOARD_SIZE,
        num_actions: int = FULL_ACTION_SPACE,
    ):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.device = torch.device(device)  # Store as torch.device

        # Pre-allocate tensors for improved memory efficiency
        # Observations: (buffer_size, C, H, W) - Board representation
        self.obs = torch.zeros(
            (buffer_size, obs_channels, board_size, board_size),
            dtype=torch.float32,
            device=self.device,
        )
        # Actions: (buffer_size,) - Action indices
        self.actions = torch.zeros(buffer_size, dtype=torch.int64, device=self.device)
        # Rewards: (buffer_size,) - Immediate rewards
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=self.device)
        # Log probabilities: (buffer_size,) - Action log probabilities
        self.log_probs = torch.zeros(
            buffer_size, dtype=torch.float32, device=self.device
        )
        # Values: (buffer_size,) - Value function estimates
        self.values = torch.zeros(buffer_size, dtype=torch.float32, device=self.device)
        # Done flags: (buffer_size,) - Episode termination flags
        self.dones = torch.zeros(buffer_size, dtype=torch.bool, device=self.device)
        # Legal masks: (buffer_size, num_actions) - Legal action masks
        self.legal_masks = torch.zeros(
            (buffer_size, num_actions), dtype=torch.bool, device=self.device
        )
        # Advantages and returns: populated by compute_advantages_and_returns
        self.advantages = torch.zeros(
            buffer_size, dtype=torch.float32, device=self.device
        )
        self.returns = torch.zeros(buffer_size, dtype=torch.float32, device=self.device)
        self.ptr = 0
        # Flag to track if advantages and returns have been computed
        self._advantages_computed = False

    def add(
        self,
        obs: torch.Tensor,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
        legal_mask: torch.Tensor,  # Added legal_mask
    ):
        """
        Add a transition to the buffer.
        'obs' is expected to be a PyTorch tensor of shape (C, H, W) on self.device.
        'legal_mask' is expected to be a PyTorch tensor on self.device.
        """
        if self.ptr < self.buffer_size:
            # Store data using tensor indexing for better performance
            self.obs[self.ptr] = obs.to(self.device)
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.log_probs[self.ptr] = log_prob
            self.values[self.ptr] = value
            self.dones[self.ptr] = done
            self.legal_masks[self.ptr] = legal_mask.to(self.device)
            self.ptr += 1
        else:
            # This case should ideally be handled by the training loop,
            # which calls learn() and then clear() when buffer is full.
            log_warning_to_stderr(
                "ExperienceBuffer", "Buffer is full. Cannot add new experience."
            )

    def compute_advantages_and_returns(
        self, last_value: float
    ):  # last_value is a float
        """
        Computes Generalized Advantage Estimation (GAE) and returns for the collected experiences.
        This should be called after the buffer is full (i.e., self.ptr == self.buffer_size).
        Uses PyTorch tensor operations for GAE calculation.
        """
        if self.ptr == 0:
            log_warning_to_stderr(
                "ExperienceBuffer",
                "compute_advantages_and_returns called on an empty buffer.",
            )
            return

        # Use tensor slices for GAE computation (much more efficient)
        rewards_tensor = self.rewards[: self.ptr]
        values_tensor = self.values[: self.ptr]
        dones_tensor = self.dones[: self.ptr].float()  # Convert bool to float for mask
        masks_tensor = 1.0 - dones_tensor

        # Initialize GAE variables
        gae = torch.tensor(0.0, device=self.device)
        next_value_tensor = torch.tensor(
            last_value, dtype=torch.float32, device=self.device
        )

        # Compute advantages and returns using reverse iteration
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                current_next_value = next_value_tensor
            else:
                current_next_value = values_tensor[t + 1]

            delta = (
                rewards_tensor[t]
                + self.gamma * current_next_value * masks_tensor[t]
                - values_tensor[t]
            )
            gae = delta + self.gamma * self.lambda_gae * masks_tensor[t] * gae

            # Store computed values directly in pre-allocated tensors
            self.advantages[t] = gae
            self.returns[t] = gae + values_tensor[t]

        # Mark that advantages have been computed
        self._advantages_computed = True

    def get_batch(self) -> dict:
        """
        Returns all collected experiences as a dictionary of PyTorch tensors on self.device.
        Assumes compute_advantages_and_returns has been called.
        """
        if self.ptr == 0:
            # This should be handled by PPOAgent.learn() checking for empty batch_data
            # For safety, one might return structured empty tensors if needed upstream.
            log_warning_to_stderr(
                "ExperienceBuffer",
                "get_batch called on an empty or not-yet-computed buffer.",
            )
            return {}  # PPOAgent.learn already checks for this

        if not self._advantages_computed:
            raise RuntimeError(
                "Cannot get batch: compute_advantages_and_returns() must be called first"
            )

        num_samples = self.ptr

        # --- Efficient Batching with Pre-allocated Tensors ---
        # All data is already stored as tensors, just slice to get the active portion
        obs_tensor = self.obs[:num_samples]
        actions_tensor = self.actions[:num_samples]
        log_probs_tensor = self.log_probs[:num_samples]
        values_tensor = self.values[:num_samples]
        advantages_tensor = self.advantages[:num_samples]
        returns_tensor = self.returns[:num_samples]
        dones_tensor = self.dones[:num_samples]
        legal_masks_tensor = self.legal_masks[:num_samples]

        return {
            "obs": obs_tensor,
            "actions": actions_tensor,
            "log_probs": log_probs_tensor,  # These are old_log_probs for PPO
            "values": values_tensor,  # These are old_V(s_t) for PPO, used with GAE
            "rewards": self.rewards[:num_samples],  # Include rewards in batch data
            "advantages": advantages_tensor,
            "returns": returns_tensor,  # These are the GAE-based returns (targets for value func)
            "dones": dones_tensor,  # For record keeping or if needed by learn()
            "legal_masks": legal_masks_tensor,  # Added legal_masks_tensor
        }

    def clear(self):
        """Clears all stored experiences from the buffer."""
        # With pre-allocated tensors, we just reset the pointer
        # The tensor memory remains allocated for reuse
        self.ptr = 0
        # Reset the computation flag when clearing buffer
        self._advantages_computed = False

    def __len__(self):
        """Return the number of transitions currently in the buffer."""
        return self.ptr

    def size(self) -> int:
        """Current number of stored transitions."""
        return self.ptr

    def capacity(self) -> int:
        """Maximum capacity of the buffer."""
        return self.buffer_size

    def add_batch(self, experiences: List[Experience]) -> None:
        """
        Add a batch of experiences to the buffer for parallel collection.

        Args:
            experiences: List of Experience objects to add
        """
        for exp in experiences:
            if self.ptr >= self.buffer_size:
                break  # Don't exceed buffer size

            self.add(
                obs=exp.obs,
                action=exp.action,
                reward=exp.reward,
                log_prob=exp.log_prob,
                value=exp.value,
                done=exp.done,
                legal_mask=exp.legal_mask,
            )

    def add_from_worker_batch(self, worker_data: Dict[str, torch.Tensor]) -> None:
        """
        Add experiences from worker batch data (optimized for parallel collection).

        Args:
            worker_data: Dictionary containing batched tensors from workers
        """
        batch_size = worker_data["obs"].shape[0]

        for i in range(batch_size):
            if self.ptr >= self.buffer_size:
                break

            self.add(
                obs=worker_data["obs"][i],
                action=int(worker_data["actions"][i].item()),
                reward=worker_data["rewards"][i].item(),
                log_prob=worker_data["log_probs"][i].item(),
                value=worker_data["values"][i].item(),
                done=bool(worker_data["dones"][i].item()),
                legal_mask=worker_data["legal_masks"][i],
            )

    def get_worker_batch_format(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get current buffer contents in worker batch format for validation.

        Returns:
            Dictionary with batched tensors or None if buffer is empty
        """
        if self.ptr == 0:
            return None

        return {
            "obs": self.obs[: self.ptr],
            "actions": self.actions[: self.ptr],
            "rewards": self.rewards[: self.ptr],
            "log_probs": self.log_probs[: self.ptr],
            "values": self.values[: self.ptr],
            "dones": self.dones[: self.ptr],
            "legal_masks": self.legal_masks[: self.ptr],
        }

    def merge_from_parallel_buffers(
        self, parallel_buffers: List["ExperienceBuffer"]
    ) -> None:
        """
        Merge experiences from multiple parallel buffers.

        Args:
            parallel_buffers: List of ExperienceBuffer instances from workers
        """
        for buffer in parallel_buffers:
            if buffer.ptr == 0:
                continue

            # Add all experiences from this buffer
            for i in range(buffer.ptr):
                if self.ptr >= self.buffer_size:
                    return  # Main buffer is full

                self.add(
                    obs=buffer.obs[i],
                    action=int(buffer.actions[i].item()),
                    reward=float(buffer.rewards[i].item()),
                    log_prob=float(buffer.log_probs[i].item()),
                    value=float(buffer.values[i].item()),
                    done=bool(buffer.dones[i].item()),
                    legal_mask=buffer.legal_masks[i],
                )
