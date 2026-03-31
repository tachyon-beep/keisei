import math

import torch

from keisei.training.algorithm_registry import PPOParams
from keisei.training.models.resnet import ResNetModel, ResNetParams
from keisei.training.ppo import PPOAlgorithm, RolloutBuffer, compute_gae


def _make_small_ppo() -> tuple[PPOAlgorithm, ResNetModel]:
    """Create a small PPO instance for testing."""
    params = PPOParams(learning_rate=1e-3, batch_size=4, epochs_per_batch=1)
    model = ResNetModel(ResNetParams(hidden_size=16, num_layers=1))
    return PPOAlgorithm(params, model), model


def _fill_buffer(
    ppo: PPOAlgorithm,
    num_envs: int = 2,
    steps: int = 8,
    *,
    legal_mask: torch.Tensor | None = None,
    all_done: bool = False,
    rewards_val: float = 0.0,
) -> RolloutBuffer:
    """Fill a RolloutBuffer with steps from select_actions."""
    buf = RolloutBuffer(num_envs=num_envs, obs_shape=(46, 9, 9), action_space=13527)
    if legal_mask is None:
        legal_mask = torch.ones(num_envs, 13527, dtype=torch.bool)
    for _ in range(steps):
        obs = torch.randn(num_envs, 46, 9, 9)
        actions, log_probs, values = ppo.select_actions(obs, legal_mask)
        buf.add(
            obs, actions, log_probs, values,
            torch.full((num_envs,), rewards_val),
            torch.ones(num_envs, dtype=torch.bool) if all_done else torch.zeros(num_envs, dtype=torch.bool),
            legal_mask,
        )
    return buf


class TestGAE:
    def test_single_step_terminal(self) -> None:
        rewards = torch.tensor([1.0])
        values = torch.tensor([0.5])
        dones = torch.tensor([True])
        next_value = torch.tensor(0.0)
        advantages = compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95)
        assert abs(advantages[0].item() - 0.5) < 1e-5

    def test_two_steps_no_terminal(self) -> None:
        rewards = torch.tensor([0.0, 0.0])
        values = torch.tensor([0.5, 0.6])
        dones = torch.tensor([False, False])
        next_value = torch.tensor(0.7)
        advantages = compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95)
        assert advantages.shape == (2,)
        assert abs(advantages[1].item() - 0.093) < 1e-3
        assert abs(advantages[0].item() - 0.1815) < 1e-2

    def test_terminal_resets_bootstrap(self) -> None:
        rewards = torch.tensor([1.0, 0.0])
        values = torch.tensor([0.3, 0.4])
        dones = torch.tensor([True, False])
        next_value = torch.tensor(0.5)
        advantages = compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95)
        assert abs(advantages[0].item() - 0.7) < 0.1

    def test_all_done_no_bootstrap(self) -> None:
        """When every timestep is terminal, advantage = reward - value (no bootstrap)."""
        rewards = torch.tensor([1.0, 1.0, 1.0])
        values = torch.tensor([0.0, 0.0, 0.0])
        dones = torch.tensor([True, True, True])
        next_value = torch.tensor(999.0)  # Should be ignored
        advantages = compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95)
        for t in range(3):
            assert abs(advantages[t].item() - 1.0) < 1e-5, (
                f"Timestep {t}: expected 1.0, got {advantages[t].item()}"
            )

    def test_mid_sequence_terminal_isolates_episodes(self) -> None:
        """A done=True at t=1 must prevent value leakage from t=2+ into t=0-1."""
        rewards = torch.tensor([0.0, 1.0, 0.0, 0.0])
        values = torch.tensor([0.5, 0.3, 0.5, 0.5])
        dones = torch.tensor([False, True, False, False])
        next_value = torch.tensor(0.5)
        advantages = compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95)
        # t=1 is terminal: advantage = reward - value = 1.0 - 0.3 = 0.7
        assert abs(advantages[1].item() - 0.7) < 1e-5
        # t=0 bootstraps from t=1 value only (not from t=2+)
        # delta_0 = 0 + 0.99 * 0.3 * 1.0 - 0.5 = -0.203
        # last_gae_0 = delta_0 + 0.99*0.95*1.0 * advantage_1
        #            = -0.203 + 0.9405 * 0.7 = 0.4554
        expected_t0 = -0.203 + 0.9405 * 0.7
        assert abs(advantages[0].item() - expected_t0) < 1e-3

    def test_gae_no_nan_with_large_values(self) -> None:
        """Ensure no NaN when values are large."""
        rewards = torch.tensor([100.0, -100.0])
        values = torch.tensor([50.0, -50.0])
        dones = torch.tensor([False, False])
        next_value = torch.tensor(0.0)
        advantages = compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95)
        assert torch.isfinite(advantages).all()


class TestRolloutBuffer:
    def test_add_and_get(self) -> None:
        buf = RolloutBuffer(num_envs=2, obs_shape=(46, 9, 9), action_space=13527)
        obs = torch.randn(2, 46, 9, 9)
        actions = torch.tensor([0, 1])
        log_probs = torch.tensor([-1.0, -2.0])
        values = torch.tensor([0.5, 0.6])
        rewards = torch.tensor([0.0, 0.0])
        dones = torch.tensor([False, False])
        legal_masks = torch.ones(2, 13527, dtype=torch.bool)

        buf.add(obs, actions, log_probs, values, rewards, dones, legal_masks)
        assert buf.size == 1
        buf.add(obs, actions, log_probs, values, rewards, dones, legal_masks)
        assert buf.size == 2

    def test_clear(self) -> None:
        buf = RolloutBuffer(num_envs=2, obs_shape=(46, 9, 9), action_space=13527)
        obs = torch.randn(2, 46, 9, 9)
        buf.add(
            obs, torch.zeros(2, dtype=torch.long), torch.zeros(2),
            torch.zeros(2), torch.zeros(2),
            torch.zeros(2, dtype=torch.bool), torch.ones(2, 13527, dtype=torch.bool),
        )
        buf.clear()
        assert buf.size == 0

    def test_flatten_shapes(self) -> None:
        """Flatten must produce (T*N, ...) tensors for all fields."""
        T, N = 5, 3
        buf = RolloutBuffer(num_envs=N, obs_shape=(46, 9, 9), action_space=13527)
        for _ in range(T):
            buf.add(
                torch.randn(N, 46, 9, 9),
                torch.randint(0, 13527, (N,)),
                torch.randn(N),
                torch.randn(N),
                torch.randn(N),
                torch.zeros(N, dtype=torch.bool),
                torch.ones(N, 13527, dtype=torch.bool),
            )
        assert buf.size == T
        data = buf.flatten()
        assert data["observations"].shape == (T * N, 46, 9, 9)
        assert data["actions"].shape == (T * N,)
        assert data["log_probs"].shape == (T * N,)
        assert data["values"].shape == (T * N,)
        assert data["rewards"].shape == (T * N,)
        assert data["dones"].shape == (T * N,)
        assert data["legal_masks"].shape == (T * N, 13527)

    def test_flatten_preserves_env_time_layout(self) -> None:
        """Verify stack→reshape(-1)→reshape(T,N) preserves per-env ordering."""
        T, N = 3, 2
        buf = RolloutBuffer(num_envs=N, obs_shape=(46, 9, 9), action_space=13527)
        for t in range(T):
            # env 0 gets reward t, env 1 gets reward t+10
            rewards = torch.tensor([float(t), float(t + 10)])
            buf.add(
                torch.randn(N, 46, 9, 9),
                torch.randint(0, 13527, (N,)),
                torch.randn(N),
                torch.randn(N),
                rewards,
                torch.zeros(N, dtype=torch.bool),
                torch.ones(N, 13527, dtype=torch.bool),
            )
        data = buf.flatten()
        rewards_2d = data["rewards"].reshape(T, N)
        # env 0: [0, 1, 2], env 1: [10, 11, 12]
        assert rewards_2d[0, 0].item() == 0.0
        assert rewards_2d[0, 1].item() == 10.0
        assert rewards_2d[1, 0].item() == 1.0
        assert rewards_2d[2, 1].item() == 12.0


class TestPPOAlgorithm:
    def test_select_actions(self) -> None:
        ppo, _ = _make_small_ppo()
        obs = torch.randn(4, 46, 9, 9)
        legal_masks = torch.ones(4, 13527, dtype=torch.bool)
        actions, log_probs, values = ppo.select_actions(obs, legal_masks)
        assert actions.shape == (4,)
        assert log_probs.shape == (4,)
        assert values.shape == (4,)
        assert (actions >= 0).all()
        assert (actions < 13527).all()

    def test_select_actions_sparse_legal_mask(self) -> None:
        """With only 5 legal actions, sampled actions must always be in that set."""
        ppo, _ = _make_small_ppo()
        num_envs = 8
        legal_indices = [0, 42, 1000, 5000, 13526]
        legal_mask = torch.zeros(num_envs, 13527, dtype=torch.bool)
        for idx in legal_indices:
            legal_mask[:, idx] = True

        obs = torch.randn(num_envs, 46, 9, 9)
        # Sample many times to increase confidence
        for _ in range(20):
            actions, log_probs, values = ppo.select_actions(obs, legal_mask)
            for a in actions.tolist():
                assert a in legal_indices, (
                    f"Action {a} not in legal set {legal_indices}"
                )
            assert torch.isfinite(log_probs).all()

    def test_select_actions_single_legal_action(self) -> None:
        """With exactly one legal action, it must always be selected."""
        ppo, _ = _make_small_ppo()
        legal_mask = torch.zeros(2, 13527, dtype=torch.bool)
        legal_mask[:, 42] = True
        obs = torch.randn(2, 46, 9, 9)
        actions, log_probs, _ = ppo.select_actions(obs, legal_mask)
        assert (actions == 42).all()
        # log_prob of a deterministic action should be 0 (log(1.0))
        assert torch.allclose(log_probs, torch.zeros_like(log_probs), atol=1e-5)

    def test_update_returns_losses(self) -> None:
        ppo, _ = _make_small_ppo()
        buf = _fill_buffer(ppo)
        next_values = torch.zeros(2)
        losses = ppo.update(buf, next_values)
        assert "policy_loss" in losses
        assert "value_loss" in losses
        assert "entropy" in losses
        assert "gradient_norm" in losses
        assert isinstance(losses["policy_loss"], float)

    def test_update_no_nan_in_losses(self) -> None:
        """No NaN or inf should appear in any loss value."""
        ppo, _ = _make_small_ppo()
        buf = _fill_buffer(ppo)
        next_values = torch.zeros(2)
        losses = ppo.update(buf, next_values)
        for key, val in losses.items():
            assert not math.isnan(val), f"{key} is NaN"
            assert not math.isinf(val), f"{key} is inf"

    def test_entropy_is_positive(self) -> None:
        """Entropy of a non-degenerate policy must be positive."""
        ppo, _ = _make_small_ppo()
        buf = _fill_buffer(ppo)
        next_values = torch.zeros(2)
        losses = ppo.update(buf, next_values)
        assert losses["entropy"] > 0, f"Entropy should be positive, got {losses['entropy']}"

    def test_update_with_all_done_no_nan(self) -> None:
        """Update with every timestep terminal should not produce NaN."""
        ppo, _ = _make_small_ppo()
        buf = _fill_buffer(ppo, all_done=True, rewards_val=1.0)
        next_values = torch.zeros(2)
        losses = ppo.update(buf, next_values)
        for key, val in losses.items():
            assert not math.isnan(val), f"{key} is NaN with all-done buffer"

    def test_update_with_sparse_legal_mask_no_nan(self) -> None:
        """PPO update with sparse legal masks must not produce NaN entropy."""
        ppo, _ = _make_small_ppo()
        legal_mask = torch.zeros(2, 13527, dtype=torch.bool)
        legal_mask[:, :5] = True  # Only 5 legal actions
        buf = _fill_buffer(ppo, legal_mask=legal_mask)
        next_values = torch.zeros(2)
        losses = ppo.update(buf, next_values)
        for key, val in losses.items():
            assert not math.isnan(val), f"{key} is NaN with sparse mask"
        assert losses["entropy"] > 0

    def test_update_with_stale_log_probs_no_nan(self) -> None:
        """Stale log_probs (large ratio) should not produce NaN due to clipping."""
        ppo, _ = _make_small_ppo()
        buf = RolloutBuffer(num_envs=2, obs_shape=(46, 9, 9), action_space=13527)
        legal_mask = torch.ones(2, 13527, dtype=torch.bool)
        for _ in range(8):
            obs = torch.randn(2, 46, 9, 9)
            actions, _, values = ppo.select_actions(obs, legal_mask)
            # Use very stale log_probs to create extreme ratios
            stale_log_probs = torch.tensor([-10.0, -10.0])
            buf.add(obs, actions, stale_log_probs, values, torch.zeros(2),
                    torch.zeros(2, dtype=torch.bool), legal_mask)
        next_values = torch.zeros(2)
        losses = ppo.update(buf, next_values)
        for key, val in losses.items():
            assert not math.isnan(val), f"{key} is NaN with stale log_probs"
            assert not math.isinf(val), f"{key} is inf with stale log_probs"

    def test_update_single_step_buffer(self) -> None:
        """A buffer with just 1 step should not crash or produce NaN."""
        ppo, _ = _make_small_ppo()
        buf = _fill_buffer(ppo, steps=1)
        next_values = torch.zeros(2)
        losses = ppo.update(buf, next_values)
        for key, val in losses.items():
            assert not math.isnan(val), f"{key} is NaN with single-step buffer"

    def test_select_actions_sets_eval_mode(self) -> None:
        """select_actions must put model in eval mode (BatchNorm uses running stats)."""
        ppo, model = _make_small_ppo()
        model.train()  # Start in train mode
        obs = torch.randn(2, 46, 9, 9)
        masks = torch.ones(2, 13527, dtype=torch.bool)
        ppo.select_actions(obs, masks)
        assert not model.training, "model should be in eval mode after select_actions"

    def test_update_sets_train_mode(self) -> None:
        """update must put model in train mode (BatchNorm uses batch stats)."""
        ppo, model = _make_small_ppo()
        model.eval()  # Start in eval mode
        buf = _fill_buffer(ppo)
        next_values = torch.zeros(2)
        ppo.update(buf, next_values)
        assert model.training, "model should be in train mode after update"
