"""
Protocol definition for Actor-Critic models in Keisei.
"""

# pylint: disable=unnecessary-ellipsis

from typing import Any, Dict, Iterator, Optional, Protocol, Tuple

import torch
import torch.nn as nn


class ActorCriticProtocol(Protocol):
    """Protocol defining the interface that all Actor-Critic models must implement."""

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x: Input observation tensor

        Returns:
            Tuple of (policy_logits, value_estimate)
        """
        ...

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action and value from observation.

        Args:
            obs: Input observation tensor
            legal_mask: Optional boolean tensor indicating legal actions
            deterministic: If True, choose action deterministically

        Returns:
            Tuple of (action, log_prob, value)
        """
        ...

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given observations.

        Args:
            obs: Input observation tensor
            actions: Actions to evaluate
            legal_mask: Optional boolean tensor indicating legal actions

        Returns:
            Tuple of (log_probs, entropy, value)
        """
        ...

    # PyTorch Module methods that are used in PPOAgent
    def train(self, mode: bool = True) -> Any:
        """Set the module in training mode."""
        ...

    def eval(self) -> Any:
        """Set the module in evaluation mode."""
        ...

    def parameters(self) -> Iterator[nn.Parameter]:
        """Return an iterator over module parameters."""
        ...

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        """Return a dictionary containing a whole state of the module."""
        ...

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> Any:
        """Copy parameters and buffers from state_dict into this module and its descendants."""
        ...

    def to(self, *args, **kwargs) -> Any:
        """Move and/or cast the parameters and buffers."""
        ...
