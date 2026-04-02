# keisei/training/katago_ppo.py
"""KataGo-style multi-head PPO: W/D/L value, score lead, spatial policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from keisei.sl.dataset import SCORE_NORMALIZATION
from keisei.training.models.katago_base import KataGoBaseModel


def compute_value_metrics(
    value_logits: torch.Tensor, value_targets: torch.Tensor,
) -> dict[str, float]:
    """Compute value prediction metrics for monitoring.

    Args:
        value_logits: (N, 3) raw W/D/L logits
        value_targets: (N,) int targets {0=W, 1=D, 2=L}

    Returns:
        Dict with value_accuracy, frac_predicted_win/draw/loss
    """
    predictions = value_logits.argmax(dim=-1)
    return {
        "value_accuracy": (predictions == value_targets).float().mean().item(),
        "frac_predicted_win": (predictions == 0).float().mean().item(),
        "frac_predicted_draw": (predictions == 1).float().mean().item(),
        "frac_predicted_loss": (predictions == 2).float().mean().item(),
    }


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
    score_normalization: float = SCORE_NORMALIZATION  # used by KataGoTrainingLoop to normalize targets
    grad_clip: float = 1.0
    use_amp: bool = False


# NOTE: Buffer memory at scale (128 steps x 512 envs):
# - legal_masks: 128 x 512 x 11259 x 1 byte = ~740 MB
# - observations: 128 x 512 x 50 x 9 x 9 x 4 bytes = ~1060 MB
# - other fields (7 x ~33 MB each): ~230 MB
# Total: ~2 GB CPU RAM (stored on CPU; mini-batches transferred to GPU).
# If CPU memory becomes the binding constraint, consider sparse
# legal_mask storage or regenerating masks from game state during update.


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
        self.env_ids: list[torch.Tensor] = []

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
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Add a timestep to the buffer.

        Args:
            score_targets: Pre-normalized score estimates in [-1, 1]. The caller
                (KataGoTrainingLoop) divides raw material difference by
                KataGoPPOParams.score_normalization before storing here.
                Raw scores can range from -200 to +200; without normalization,
                the MSE loss would dominate all other loss terms.
        """
        # Detach and move to CPU first — all validation below runs on CPU
        # tensors, avoiding implicit CUDA synchronization on the hot path.
        obs_cpu = obs.detach().cpu()
        actions_cpu = actions.detach().cpu()
        log_probs_cpu = log_probs.detach().cpu()
        values_cpu = values.detach().cpu()
        rewards_cpu = rewards.detach().cpu()
        dones_cpu = dones.detach().cpu()
        legal_masks_cpu = legal_masks.detach().cpu()
        value_cats_cpu = value_categories.detach().cpu()
        score_cpu = score_targets.detach().cpu()

        # Guard against invalid value categories
        valid_cats = {-1, 0, 1, 2}
        unique_cats = set(value_cats_cpu.unique().tolist())
        invalid = unique_cats - valid_cats
        if invalid:
            raise ValueError(
                f"value_categories contains invalid values {invalid}. "
                f"Expected only {{-1=ignore, 0=W, 1=D, 2=L}}."
            )

        # Guard: NaN score targets are no longer valid — every position gets real material balance.
        if score_cpu.isnan().any():
            raise ValueError(
                "score_targets contains NaN. With per-step material balance, "
                "all targets should be real-valued."
            )

        # Guard against unnormalized score targets (catches integration bugs).
        # With per-step material balance / 76.0, typical range is [-1.7, +1.7].
        # Theoretical max: fully promoted one-sided = 196/76 = 2.58. Threshold
        # at 3.5 gives 35% headroom above the theoretical maximum.
        abs_max = score_cpu.abs().max()
        if abs_max > 3.5:
            raise ValueError(
                f"score_targets appear unnormalized: max abs value = "
                f"{abs_max.item():.1f}. "
                f"Expected in [-1.7, +1.7] typical, theoretical max 2.58 (guard 3.5)."
            )

        # Store on CPU to reduce GPU memory pressure during rollout collection.
        # Mini-batches are transferred back to GPU during update().
        self.observations.append(obs_cpu)
        self.actions.append(actions_cpu)
        self.log_probs.append(log_probs_cpu)
        self.values.append(values_cpu)
        self.rewards.append(rewards_cpu)
        self.dones.append(dones_cpu)
        self.legal_masks.append(legal_masks_cpu)
        self.value_categories.append(value_cats_cpu)
        self.score_targets.append(score_cpu)
        if env_ids is not None:
            self.env_ids.append(env_ids.detach().cpu())

    def flatten(self) -> dict[str, torch.Tensor]:
        if self.size == 0:
            raise ValueError(
                "Cannot flatten an empty buffer. Call add() at least once before flatten()."
            )
        # Use cat instead of stack to support variable-sized timesteps
        # (split-merge mode stores only learner envs per step, which varies).
        result = {
            "observations": torch.cat(self.observations, dim=0).reshape(-1, *self.obs_shape),
            "actions": torch.cat(self.actions, dim=0).reshape(-1),
            "log_probs": torch.cat(self.log_probs, dim=0).reshape(-1),
            "values": torch.cat(self.values, dim=0).reshape(-1),
            "rewards": torch.cat(self.rewards, dim=0).reshape(-1),
            "dones": torch.cat(self.dones, dim=0).reshape(-1),
            "legal_masks": torch.cat(self.legal_masks, dim=0).reshape(-1, self.action_space),
            "value_categories": torch.cat(self.value_categories, dim=0).reshape(-1),
            "score_targets": torch.cat(self.score_targets, dim=0).reshape(-1),
        }
        if self.env_ids:
            result["env_ids"] = torch.cat(self.env_ids, dim=0).reshape(-1)
        return result


class KataGoPPOAlgorithm:
    def __init__(
        self,
        params: KataGoPPOParams,
        model: KataGoBaseModel,
        forward_model: torch.nn.Module | None = None,
        warmup_epochs: int = 0,
        warmup_entropy: float = 0.05,
    ) -> None:
        """Create a KataGo PPO algorithm.

        Args:
            params: Frozen hyperparameters.
            model: The unwrapped base model (used for optimizer and grad clipping).
            forward_model: The model used for forward passes. Pass the DataParallel
                wrapper here if using multi-GPU. If None, defaults to `model`.
                Convention: ``base_model = model.module if hasattr(...) else model``,
                then ``KataGoPPOAlgorithm(params, base_model, forward_model=model)``.
            warmup_epochs: Number of RL epochs with elevated entropy after SL.
            warmup_entropy: Entropy coefficient during warmup period.
        """
        self.params = params
        self.model = model
        self.forward_model = forward_model or model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
        self.scaler = GradScaler(enabled=params.use_amp)
        self.warmup_epochs = warmup_epochs
        self.warmup_entropy = warmup_entropy
        self.current_entropy_coeff = params.lambda_entropy

    def get_entropy_coeff(self, epoch: int) -> float:
        """Return the entropy coefficient for the current epoch.

        During the first `warmup_epochs` epochs of RL (after SL warmup),
        uses elevated entropy to soften the overconfident SL policy.
        """
        if epoch < self.warmup_epochs:
            return self.warmup_entropy
        return self.params.lambda_entropy

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
        """Select actions for rollout collection.

        Sets model to eval mode for BatchNorm running stats, then restores
        train mode on exit. The stored log_probs are detached (no_grad);
        PPO recomputes new_log_probs under train() in update().
        """
        self.forward_model.eval()
        try:
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
        finally:
            self.forward_model.train()

    def update(
        self,
        buffer: KataGoRolloutBuffer,
        next_values: torch.Tensor,
        value_adapter: Any | None = None,
    ) -> dict[str, float]:
        from keisei.training.gae import compute_gae

        self.forward_model.train()
        data = buffer.flatten()
        T = buffer.size
        N = buffer.num_envs
        total_samples = data["rewards"].numel()
        device = next(self.model.parameters()).device

        # GAE computation — runs on CPU (buffer stores data on CPU).
        next_values_cpu = next_values.detach().cpu()

        if total_samples == T * N:
            # Vectorized path: batched GAE over (T, N) grid
            rewards_2d = data["rewards"].reshape(T, N)
            values_2d = data["values"].reshape(T, N)
            dones_2d = data["dones"].reshape(T, N)

            advantages = compute_gae(
                rewards_2d, values_2d, dones_2d,
                next_values_cpu, gamma=self.params.gamma, lam=self.params.gae_lambda,
            ).reshape(-1)
        elif "env_ids" in data:
            # Per-env GAE for split-merge mode: pad all envs to (T_max, N) and
            # compute GAE in a single vectorized pass.
            from keisei.training.gae import compute_gae_padded

            env_ids = data["env_ids"]
            unique_envs = env_ids.unique()
            advantages = torch.zeros(total_samples)

            # Collect per-env data and pad into (T_max, N_env) tensors
            env_rewards = []
            env_values = []
            env_dones = []
            env_lengths = []
            env_masks = []

            for env_id in unique_envs:
                mask = env_ids == env_id
                env_rewards.append(data["rewards"][mask])
                env_values.append(data["values"][mask])
                env_dones.append(data["dones"][mask])
                env_lengths.append(mask.sum().item())
                env_masks.append(mask)

            max_T = max(env_lengths)
            N_env = len(unique_envs)

            rewards_pad = torch.zeros(max_T, N_env)
            values_pad = torch.zeros(max_T, N_env)
            dones_pad = torch.ones(max_T, N_env)  # padding = done to zero GAE
            nv = torch.zeros(N_env)

            for i, L in enumerate(env_lengths):
                rewards_pad[:L, i] = env_rewards[i]
                values_pad[:L, i] = env_values[i]
                dones_pad[:L, i] = env_dones[i]
                nv[i] = next_values_cpu[unique_envs[i]]

            lengths_t = torch.tensor(env_lengths)
            padded_adv = compute_gae_padded(
                rewards_pad, values_pad, dones_pad, nv, lengths_t,
                gamma=self.params.gamma, lam=self.params.gae_lambda,
            )

            for i, L in enumerate(env_lengths):
                advantages[env_masks[i]] = padded_adv[:L, i]
        else:
            # Fallback: flat GAE (no env_ids — legacy split-merge behavior)
            bootstrap = next_values_cpu.mean()
            advantages = compute_gae(
                data["rewards"], data["values"], data["dones"],
                bootstrap, gamma=self.params.gamma, lam=self.params.gae_lambda,
            )

        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = min(self.params.batch_size, total_samples)

        # AMP settings: bfloat16 on CUDA if supported, else float16; fallback to cpu autocast
        amp_dtype = (
            torch.bfloat16
            if (self.params.use_amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float16
        )
        autocast_device = "cuda" if device.type == "cuda" else "cpu"

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_score_loss = 0.0
        total_entropy = 0.0
        total_grad_norm = 0.0
        num_updates = 0

        for _ in range(self.params.epochs_per_batch):
            indices = torch.randperm(total_samples)
            for start in range(0, total_samples, batch_size):
                end = min(start + batch_size, total_samples)
                idx = indices[start:end]

                # Transfer mini-batch from CPU to training device
                batch_obs = data["observations"][idx].to(device)
                batch_actions = data["actions"][idx].to(device)
                batch_old_log_probs = data["log_probs"][idx].to(device)
                batch_advantages = advantages[idx].to(device)
                batch_legal_masks = data["legal_masks"][idx].to(device)
                batch_value_cats = data["value_categories"][idx].to(device)
                batch_score_targets = data["score_targets"][idx].to(device)

                with autocast(device_type=autocast_device, dtype=amp_dtype, enabled=self.params.use_amp):
                    output = self.forward_model(batch_obs)

                    # Policy loss (clipped surrogate)
                    flat_logits = output.policy_logits.reshape(batch_obs.shape[0], -1)

                    # NaN guard: check raw model output BEFORE masking.
                    if flat_logits.isnan().any():
                        raise RuntimeError("NaN in raw policy logits from model forward pass")

                    # Guard: no sample in the batch should have an all-False legal mask.
                    # This mirrors the guard in select_actions — an all-illegal mask
                    # would produce all-NaN from log_softmax(-inf) and silently corrupt the loss.
                    if (batch_legal_masks.sum(dim=-1) == 0).any():
                        raise RuntimeError(
                            "Batch contains samples with zero legal actions in update(). "
                            "Check that terminal-state masks are not stored in the buffer."
                        )

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

                    # Value + score loss — dispatch through adapter if provided
                    if value_adapter is not None:
                        value_score_loss = value_adapter.compute_value_loss(
                            output.value_logits,
                            returns=None,
                            value_cats=batch_value_cats,
                            score_targets=batch_score_targets,
                            score_pred=output.score_lead,
                        )
                        # For metrics tracking, decompose (adapter combines them)
                        value_loss = value_score_loss  # combined
                        score_loss = torch.tensor(0.0, device=batch_obs.device)
                    else:
                        # Default: inline KataGo multi-head (backward compatible)
                        has_valid_value_targets = (batch_value_cats >= 0).any()
                        if has_valid_value_targets:
                            value_loss = F.cross_entropy(
                                output.value_logits, batch_value_cats, ignore_index=-1
                            )
                        else:
                            value_loss = output.value_logits.sum() * 0.0

                        # Score loss (MSE on normalized material balance).
                        # Every position has a real target — no NaN masking needed.
                        score_loss = F.mse_loss(
                            output.score_lead.squeeze(-1), batch_score_targets,
                        )

                        value_score_loss = (
                            self.params.lambda_value * value_loss
                            + self.params.lambda_score * score_loss
                        )

                    # Combined loss
                    loss = (
                        self.params.lambda_policy * policy_loss
                        + value_score_loss
                        - self.current_entropy_coeff * entropy
                    )

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.params.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_score_loss += score_loss.item()
                total_entropy += entropy.item()
                total_grad_norm += float(grad_norm)
                num_updates += 1

        buffer.clear()

        denom = max(num_updates, 1)
        metrics = {
            "policy_loss": total_policy_loss / denom,
            "value_loss": total_value_loss / denom,
            "score_loss": total_score_loss / denom,
            "entropy": total_entropy / denom,
            "gradient_norm": total_grad_norm / denom,
        }

        # Value prediction metrics for monitoring degeneracy
        # (e.g., model predicting WIN for every position → value_accuracy plateau)
        valid_mask = data["value_categories"] >= 0  # exclude ignore_index=-1
        if valid_mask.any():
            with torch.no_grad():
                valid_cats = data["value_categories"][valid_mask]
                valid_obs = data["observations"][valid_mask]
                sample_size = min(256, valid_obs.shape[0])
                sample_obs = valid_obs[:sample_size].to(device)
                sample_cats = valid_cats[:sample_size].to(device)
                self.forward_model.eval()
                sample_output = self.forward_model(sample_obs)
                value_metrics = compute_value_metrics(
                    sample_output.value_logits, sample_cats
                )
                metrics.update(value_metrics)
                self.forward_model.train()

        self.forward_model.train()
        return metrics
