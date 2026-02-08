"""
Base Actor-Critic model implementation for Keisei.

This module provides a shared base class that implements the common ActorCritic
methods to reduce code duplication between different model architectures.
"""

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

            neg_inf = torch.finfo(policy_logits.dtype).min
            masked_logits = torch.where(legal_mask, policy_logits, neg_inf)
            # No legal actions → terminal state; return safe dummy values.
            if not torch.any(legal_mask):
                log_error_to_stderr(
                    self.__class__.__name__,
                    "No legal actions in get_action_and_value — "
                    "caller should not request actions for terminal states.",
                )
                if policy_logits.dim() > 1:
                    batch = policy_logits.shape[0]
                    action = torch.zeros(
                        batch, dtype=torch.long, device=policy_logits.device
                    )
                    log_prob = torch.zeros(batch, device=policy_logits.device)
                else:
                    action = torch.tensor(
                        0, dtype=torch.long, device=policy_logits.device
                    )
                    log_prob = torch.tensor(0.0, device=policy_logits.device)
                if value.dim() > 1 and value.shape[-1] == 1:
                    value = value.squeeze(-1)
                return action, log_prob, value

            probs = F.softmax(masked_logits, dim=-1)
        else:
            probs = F.softmax(policy_logits, dim=-1)

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
        all_masked = None
        if legal_mask is not None:
            neg_inf = torch.finfo(policy_logits.dtype).min
            masked_logits = torch.where(legal_mask, policy_logits, neg_inf)

            # Detect rows where all actions are masked (terminal states in batch).
            # Replace with uniform logits to avoid NaN from softmax; the
            # resulting log_probs and entropy are zeroed out below.
            all_masked = ~legal_mask.any(dim=-1)
            if all_masked.any():
                log_error_to_stderr(
                    self.__class__.__name__,
                    "All-masked rows in evaluate_actions — "
                    "terminal states should not appear in experience buffer.",
                )
                masked_logits = masked_logits.clone()
                masked_logits[all_masked] = 0.0

            probs = F.softmax(masked_logits, dim=-1)
        else:
            probs = F.softmax(policy_logits, dim=-1)

        dist = torch.distributions.Categorical(probs=probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        # Zero out log_probs and entropy for all-masked rows so PPO ratio
        # is exp(0-0)=1 (no gradient signal for terminal states).
        if all_masked is not None and all_masked.any():
            log_probs = log_probs.clone()
            entropy = entropy.clone()
            log_probs[all_masked] = 0.0
            entropy[all_masked] = 0.0

        # Handle value squeezing - some models squeeze in forward, others don't
        if value.dim() > 1 and value.shape[-1] == 1:
            value = value.squeeze(-1)

        return log_probs, entropy, value
