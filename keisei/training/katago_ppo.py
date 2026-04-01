# keisei/training/katago_ppo.py
"""KataGo-style multi-head PPO: W/D/L value, score lead, spatial policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from keisei.training.models.katago_base import KataGoBaseModel


@dataclass(frozen=True)
class KataGoPPOParams:
    learning_rate: float = 2e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95        # GAE lambda -- exposed as config, not hardcoded
    clip_epsilon: float = 0.2
    epochs_per_batch: int = 4
    batch_size: int = 256
    lambda_policy: float = 1.0
    lambda_value: float = 1.5
    lambda_score: float = 0.02
    lambda_entropy: float = 0.01
    score_normalization: float = 76.0  # used by KataGoTrainingLoop to normalize targets at buffer level
    grad_clip: float = 1.0


# NOTE: Buffer memory at scale (128 steps x 512 envs):
# - legal_masks: 128 x 512 x 11259 x 1 byte = ~740 MB
# - observations: 128 x 512 x 50 x 9 x 9 x 4 bytes = ~1060 MB
# - other fields (7 x ~33 MB each): ~230 MB
# Total: ~2 GB CPU RAM. If memory becomes the binding constraint,
# consider sparse legal_mask storage or regenerating masks from
# game state during update. For now, keep it simple and dense.


class KataGoRolloutBuffer:
    def __init__(self, num_envs: int, obs_shape: tuple[int, ...], action_space: int) -> None:
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_space = action_space
        self.clear()

    def clear(self) -> None:
        self.observations: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.log_probs: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.rewards: list[torch.Tensor] = []
        self.dones: list[torch.Tensor] = []
        self.legal_masks: list[torch.Tensor] = []
        self.value_categories: list[torch.Tensor] = []
        self.score_targets: list[torch.Tensor] = []

    @property
    def size(self) -> int:
        return len(self.observations)

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        legal_masks: torch.Tensor,
        value_categories: torch.Tensor,
        score_targets: torch.Tensor,
    ) -> None:
        """Add a timestep to the buffer.

        Args:
            score_targets: Pre-normalized score estimates in [-1, 1]. The caller
                (KataGoTrainingLoop) divides raw material difference by
                KataGoPPOParams.score_normalization before storing here.
                Raw scores can range from -200 to +200; without normalization,
                the MSE loss would dominate all other loss terms.
        """
        # Guard against unnormalized score targets (catches Plan C integration bugs)
        if score_targets.abs().max() > 2.0:
            raise ValueError(
                f"score_targets appear unnormalized: max abs value = "
                f"{score_targets.abs().max().item():.1f}, expected <= 1.0. "
                f"Divide by score_normalization before storing."
            )
        self.observations.append(obs)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.values.append(values)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.legal_masks.append(legal_masks)
        self.value_categories.append(value_categories)
        self.score_targets.append(score_targets)

    def flatten(self) -> dict[str, torch.Tensor]:
        return {
            "observations": torch.stack(self.observations).reshape(-1, *self.obs_shape),
            "actions": torch.stack(self.actions).reshape(-1),
            "log_probs": torch.stack(self.log_probs).reshape(-1),
            "values": torch.stack(self.values).reshape(-1),
            "rewards": torch.stack(self.rewards).reshape(-1),
            "dones": torch.stack(self.dones).reshape(-1),
            "legal_masks": torch.stack(self.legal_masks).reshape(-1, self.action_space),
            "value_categories": torch.stack(self.value_categories).reshape(-1),
            "score_targets": torch.stack(self.score_targets).reshape(-1),
        }


class KataGoPPOAlgorithm:
    def __init__(
        self,
        params: KataGoPPOParams,
        model: KataGoBaseModel,
        forward_model: torch.nn.Module | None = None,
    ) -> None:
        self.params = params
        self.model = model
        self.forward_model = forward_model or model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
        self.current_entropy_coeff = params.lambda_entropy  # mutable; Plan D warmup updates this

    @staticmethod
    def scalar_value(value_logits: torch.Tensor) -> torch.Tensor:
        """Project W/D/L logits to scalar value: P(W) - P(L).

        Used by both select_actions (rollout) and bootstrap (GAE).
        Centralised here so the formula can't diverge between the two call sites.
        """
        value_probs = F.softmax(value_logits, dim=-1)
        return value_probs[:, 0] - value_probs[:, 2]

    @torch.no_grad()
    def select_actions(
        self, obs: torch.Tensor, legal_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.forward_model.eval()
        output = self.forward_model(obs)

        # Guard: no env should have zero legal actions
        legal_counts = legal_masks.sum(dim=-1)
        if (legal_counts == 0).any():
            zero_envs = (legal_counts == 0).nonzero(as_tuple=True)[0].tolist()
            raise RuntimeError(
                f"Environments {zero_envs} have zero legal actions — "
                f"all-False legal mask would produce NaN"
            )

        # Flatten spatial policy to (B, 11259), apply mask
        flat_logits = output.policy_logits.reshape(obs.shape[0], -1)
        masked_logits = flat_logits.masked_fill(~legal_masks, float("-inf"))

        probs = F.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        # Scalar value for GAE — uses shared projection method
        scalar_values = self.scalar_value(output.value_logits)

        return actions, log_probs, scalar_values
