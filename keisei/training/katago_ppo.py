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

    def update(self, buffer: KataGoRolloutBuffer, next_values: torch.Tensor) -> dict[str, float]:
        # Deferred import to avoid circular: katago_ppo -> ppo -> algorithm_registry -> katago_ppo
        from keisei.training.ppo import compute_gae

        self.forward_model.train()
        data = buffer.flatten()
        T = buffer.size
        N = buffer.num_envs

        # GAE computation (uses scalar P(W)-P(L) values)
        rewards_2d = data["rewards"].reshape(T, N)
        values_2d = data["values"].reshape(T, N)
        dones_2d = data["dones"].reshape(T, N)

        all_advantages = torch.zeros(T, N, device=data["rewards"].device)
        for env_i in range(N):
            all_advantages[:, env_i] = compute_gae(
                rewards_2d[:, env_i], values_2d[:, env_i], dones_2d[:, env_i],
                next_values[env_i], gamma=self.params.gamma, lam=self.params.gae_lambda,
            )

        advantages = all_advantages.reshape(-1)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_samples = T * N
        batch_size = min(self.params.batch_size, total_samples)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_score_loss = 0.0
        total_entropy = 0.0
        total_grad_norm = 0.0
        num_updates = 0

        for _ in range(self.params.epochs_per_batch):
            indices = torch.randperm(total_samples, device=data["rewards"].device)
            for start in range(0, total_samples, batch_size):
                end = min(start + batch_size, total_samples)
                idx = indices[start:end]

                batch_obs = data["observations"][idx]
                batch_actions = data["actions"][idx]
                batch_old_log_probs = data["log_probs"][idx]
                batch_advantages = advantages[idx]
                batch_legal_masks = data["legal_masks"][idx]
                batch_value_cats = data["value_categories"][idx]
                batch_score_targets = data["score_targets"][idx]

                output = self.forward_model(batch_obs)

                # Policy loss (clipped surrogate)
                flat_logits = output.policy_logits.reshape(batch_obs.shape[0], -1)

                # NaN guard: check raw model output BEFORE masking.
                if flat_logits.isnan().any():
                    raise RuntimeError("NaN in raw policy logits from model forward pass")

                masked_logits = flat_logits.masked_fill(~batch_legal_masks, float("-inf"))
                log_probs_all = F.log_softmax(masked_logits, dim=-1)
                new_log_probs = log_probs_all.gather(
                    1, batch_actions.unsqueeze(1)
                ).squeeze(1)

                ratio = (new_log_probs - batch_old_log_probs).exp()
                clip = self.params.clip_epsilon
                surr1 = ratio * batch_advantages
                surr2 = ratio.clamp(1 - clip, 1 + clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Entropy over legal actions only.
                probs = F.softmax(masked_logits, dim=-1)
                safe_log_probs = log_probs_all.masked_fill(~batch_legal_masks, 0.0)
                entropy = -(probs * safe_log_probs).sum(dim=-1).mean()

                # Value loss (cross-entropy on W/D/L)
                value_loss = F.cross_entropy(
                    output.value_logits, batch_value_cats, ignore_index=-1
                )

                # Score loss (MSE on normalized score)
                score_loss = F.mse_loss(output.score_lead.squeeze(-1), batch_score_targets)

                # Combined loss
                loss = (
                    self.params.lambda_policy * policy_loss
                    + self.params.lambda_value * value_loss
                    + self.params.lambda_score * score_loss
                    - self.current_entropy_coeff * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.params.grad_clip
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_score_loss += score_loss.item()
                total_entropy += entropy.item()
                total_grad_norm += float(grad_norm)
                num_updates += 1

        buffer.clear()

        denom = max(num_updates, 1)
        return {
            "policy_loss": total_policy_loss / denom,
            "value_loss": total_value_loss / denom,
            "score_loss": total_score_loss / denom,
            "entropy": total_entropy / denom,
            "gradient_norm": total_grad_norm / denom,
        }
