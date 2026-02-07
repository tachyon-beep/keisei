"""
Base Actor-Critic model implementation for Keisei.

This module provides a shared base class that implements the common ActorCritic
methods to reduce code duplication between different model architectures.
"""

import sys
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from keisei.utils.unified_logger import log_error_to_stderr

from .actor_critic_protocol import ActorCriticProtocol


class BaseActorCriticModel(nn.Module, ActorCriticProtocol, ABC):
    """
    Abstract base class for Actor-Critic models that implements shared methods.

    This class provides common implementations of get_action_and_value and
    evaluate_actions methods while requiring subclasses to implement the
    forward method.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model. Must be implemented by subclasses.

        Args:
            x: Input observation tensor

        Returns:
            Tuple of (policy_logits, value_estimate)
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given an observation (and optional legal action mask), return a sampled or deterministically chosen action,
        its log probability, and value estimate.

        Args:
            obs: Input observation tensor.
            legal_mask: Optional boolean tensor indicating legal actions.
                        If provided, illegal actions will be masked out before sampling/argmax.
            deterministic: If True, choose the action with the highest probability (argmax).
                           If False, sample from the distribution.
        Returns:
            action: Chosen action tensor.
            log_prob: Log probability of the chosen action.
            value: Value estimate tensor.
        """
        policy_logits, value = self.forward(obs)

        if legal_mask is not None:
            # Apply the legal mask: set logits of illegal actions to -infinity
            # Ensure legal_mask has the same shape as policy_logits or is broadcastable
            if (
                legal_mask.ndim == 1
                and policy_logits.ndim == 2
                and policy_logits.shape[0] == 1
            ):
                legal_mask = legal_mask.unsqueeze(0)  # Adapt for batch size 1

            masked_logits = torch.where(
                legal_mask,
                policy_logits,
                torch.tensor(float("-inf"), device=policy_logits.device),
            )
            # Handle case where all masked_logits are -inf to prevent NaN in softmax
            if not torch.any(legal_mask):  # Or check if masked_logits are all -inf
                # If no legal moves, softmax over original logits might be one option,
                # or let it produce NaNs which should be caught upstream.
                # For now, this will lead to NaNs if all are masked.
                # This situation should ideally be caught by the caller (e.g. PPOAgent)
                pass  # Let it proceed, PPOAgent should handle no legal moves.
            probs = F.softmax(masked_logits, dim=-1)
        else:
            probs = F.softmax(policy_logits, dim=-1)

        # Check for NaNs in probs, which can happen if all logits were -inf
        if torch.isnan(probs).any():
            log_error_to_stderr(
                self.__class__.__name__,
                "NaNs in probabilities in get_action_and_value. Check legal_mask and logits. Defaulting to uniform for affected rows.",
            )
            if probs.dim() > 1:
                nan_rows = torch.isnan(probs).any(dim=-1)
                probs[nan_rows] = torch.ones(probs.shape[-1], device=probs.device) / probs.shape[-1]
            else:
                probs = torch.ones_like(probs) / probs.shape[-1]

        dist = torch.distributions.Categorical(probs=probs)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        # Handle value squeezing - some models squeeze in forward, others don't
        if value.dim() > 1 and value.shape[-1] == 1:
            value = value.squeeze(-1)

        return action, log_prob, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate the log probabilities, entropy, and value for given observations and actions.

        Args:
            obs: Input observation tensor.
            actions: Actions taken tensor.
            legal_mask: Optional boolean tensor indicating legal actions.
                        If provided, it can be used to ensure probabilities are calculated correctly,
                        though for evaluating *taken* actions, it's often assumed they were legal.
                        It primarily affects entropy calculation if used to restrict the distribution.
        Returns:
            log_probs: Log probabilities of the taken actions.
            entropy: Entropy of the action distribution.
            value: Value estimate tensor.
        """
        policy_logits, value = self.forward(obs)

        # If legal_mask is None (e.g., when called from PPOAgent.learn during batch processing),
        # the policy distribution (probs) and entropy are calculated over all possible actions,
        # not just those that were legal in the specific states from which 'actions' were sampled.
        # This is a common approach. To calculate entropy strictly over the legal action space
        # for each state in the batch, legal masks for each observation in obs_minibatch
        # would need to be stored in the experience buffer and passed here.
        if legal_mask is not None:
            # Apply the legal mask for calculating probabilities and entropy correctly
            # Ensure legal_mask has the same shape as policy_logits or is broadcastable

            # The shape of legal_mask should be (batch_size, num_actions)
            # The shape of policy_logits is (batch_size, num_actions)
            # No unsqueezing or broadcasting adjustment should be needed here if shapes are consistent.
            # The previous check for legal_mask.ndim == 1 was more for get_action_and_value with batch_size=1.
            # Here, we expect legal_mask to match policy_logits if provided.

            masked_logits = torch.where(
                legal_mask,
                policy_logits,
                torch.tensor(float("-inf"), device=policy_logits.device),
            )
            probs = F.softmax(masked_logits, dim=-1)
        else:
            probs = F.softmax(policy_logits, dim=-1)

        # Check for NaNs in probs (e.g. if all logits in a row were -inf due to masking)
        # Replace NaNs with uniform distribution for stability in entropy calculation for those rows
        if torch.isnan(probs).any():
            log_error_to_stderr(
                self.__class__.__name__,
                "NaNs in probabilities in evaluate_actions. Check legal_mask and logits. Defaulting to uniform for affected rows.",
            )
            nan_rows = torch.isnan(probs).any(dim=1)
            probs[nan_rows] = torch.ones_like(probs[nan_rows]) / policy_logits.shape[-1]

        dist = torch.distributions.Categorical(probs=probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        # Handle value squeezing - some models squeeze in forward, others don't
        if value.dim() > 1 and value.shape[-1] == 1:
            value = value.squeeze(-1)

        return log_probs, entropy, value
