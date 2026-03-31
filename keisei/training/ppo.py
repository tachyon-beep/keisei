"""PPO algorithm: rollout buffer, GAE, clipped policy update."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from keisei.training.models.base import BaseModel
from keisei.training.algorithm_registry import PPOParams


def compute_gae(
    rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
    next_value: torch.Tensor, gamma: float, lam: float,
) -> torch.Tensor:
    T = len(rewards)
    advantages = torch.zeros(T, device=rewards.device)
    last_gae = torch.tensor(0.0, device=rewards.device)

    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        not_done = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_val * not_done - values[t]
        last_gae = delta + gamma * lam * not_done * last_gae
        advantages[t] = last_gae

    return advantages


class RolloutBuffer:
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

    @property
    def size(self) -> int:
        return len(self.observations)

    def add(self, obs: torch.Tensor, actions: torch.Tensor, log_probs: torch.Tensor,
            values: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor,
            legal_masks: torch.Tensor) -> None:
        self.observations.append(obs)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.values.append(values)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.legal_masks.append(legal_masks)

    def flatten(self) -> dict[str, torch.Tensor]:
        return {
            "observations": torch.stack(self.observations).reshape(-1, *self.obs_shape),
            "actions": torch.stack(self.actions).reshape(-1),
            "log_probs": torch.stack(self.log_probs).reshape(-1),
            "values": torch.stack(self.values).reshape(-1),
            "rewards": torch.stack(self.rewards).reshape(-1),
            "dones": torch.stack(self.dones).reshape(-1),
            "legal_masks": torch.stack(self.legal_masks).reshape(-1, self.action_space),
        }


class PPOAlgorithm:
    def __init__(self, params: PPOParams, model: BaseModel) -> None:
        self.params = params
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    @torch.no_grad()
    def select_actions(self, obs: torch.Tensor, legal_masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        policy_logits, values = self.model(obs)
        masked_logits = policy_logits.masked_fill(~legal_masks, float("-inf"))
        probs = F.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs, values.squeeze(-1)

    def update(self, buffer: RolloutBuffer, next_values: torch.Tensor) -> dict[str, float]:
        data = buffer.flatten()
        T = buffer.size
        N = buffer.num_envs

        rewards_2d = data["rewards"].reshape(T, N)
        values_2d = data["values"].reshape(T, N)
        dones_2d = data["dones"].reshape(T, N)

        all_advantages = torch.zeros(T, N)
        for env_i in range(N):
            all_advantages[:, env_i] = compute_gae(
                rewards_2d[:, env_i], values_2d[:, env_i], dones_2d[:, env_i],
                next_values[env_i], gamma=self.params.gamma, lam=0.95,
            )

        advantages = all_advantages.reshape(-1)
        returns = advantages + data["values"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_samples = T * N
        batch_size = min(self.params.batch_size, total_samples)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_grad_norm = 0.0
        num_updates = 0

        for _ in range(self.params.epochs_per_batch):
            indices = torch.randperm(total_samples)
            for start in range(0, total_samples, batch_size):
                end = min(start + batch_size, total_samples)
                idx = indices[start:end]

                batch_obs = data["observations"][idx]
                batch_actions = data["actions"][idx]
                batch_old_log_probs = data["log_probs"][idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]
                batch_legal_masks = data["legal_masks"][idx]

                policy_logits, values = self.model(batch_obs)
                masked_logits = policy_logits.masked_fill(~batch_legal_masks, float("-inf"))
                log_probs_all = F.log_softmax(masked_logits, dim=-1)
                new_log_probs = log_probs_all.gather(1, batch_actions.unsqueeze(1)).squeeze(1)

                probs = F.softmax(masked_logits, dim=-1)
                entropy = -(probs * log_probs_all).sum(dim=-1).mean()

                ratio = (new_log_probs - batch_old_log_probs).exp()
                clip = self.params.clip_epsilon
                surr1 = ratio * batch_advantages
                surr2 = ratio.clamp(1 - clip, 1 + clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values.squeeze(-1), batch_returns)

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_grad_norm += float(grad_norm)
                num_updates += 1

        buffer.clear()

        return {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
            "gradient_norm": total_grad_norm / max(num_updates, 1),
        }
