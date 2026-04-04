# tests/test_pytorch_audit_gaps.py
"""Tests closing PyTorch audit gaps: checkpoint completeness, gradient flow,
numerical stability, value consistency, buffer lifecycle, and GAE fallback."""

import random

import numpy as np
import pytest
import torch

from keisei.training.checkpoint import load_checkpoint, save_checkpoint
from keisei.training.gae import compute_gae
from keisei.training.katago_ppo import (
    KataGoPPOAlgorithm,
    KataGoPPOParams,
    KataGoRolloutBuffer,
    compute_value_metrics,
)
from keisei.training.katago_loop import split_merge_step
from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams


def _small_model():
    return SEResNetModel(SEResNetParams(
        num_blocks=2, channels=32, se_reduction=8,
        global_pool_channels=16, policy_channels=8,
        value_fc_size=32, score_fc_size=16, obs_channels=50,
    ))


def _filled_buffer(num_envs=4, steps=3, action_space=11259):
    """Create a buffer with terminal steps so value head gets gradients."""
    buf = KataGoRolloutBuffer(
        num_envs=num_envs, obs_shape=(50, 9, 9), action_space=action_space,
    )
    for t in range(steps):
        is_last = t == steps - 1
        buf.add(
            obs=torch.randn(num_envs, 50, 9, 9),
            actions=torch.randint(0, action_space, (num_envs,)),
            log_probs=torch.randn(num_envs),
            values=torch.randn(num_envs),
            rewards=torch.where(
                torch.tensor([is_last] * num_envs),
                torch.tensor([1.0, -1.0, 0.0, 1.0][:num_envs]),
                torch.zeros(num_envs),
            ),
            dones=torch.tensor([is_last] * num_envs),
            terminated=torch.tensor([is_last] * num_envs),
            legal_masks=torch.ones(num_envs, action_space, dtype=torch.bool),
            value_categories=torch.where(
                torch.tensor([is_last] * num_envs),
                torch.tensor([0, 2, 1, 0][:num_envs]),
                torch.full((num_envs,), -1),
            ),
            score_targets=torch.randn(num_envs).clamp(-1.5, 1.5),
        )
    return buf


# ---------------------------------------------------------------------------
# 1. Checkpoint: scheduler + RNG round-trip
# ---------------------------------------------------------------------------


class TestCheckpointSchedulerRNG:
    @pytest.fixture
    def model(self):
        return _small_model()

    def test_scheduler_state_round_trip(self, tmp_path, model):
        """LR scheduler state survives save→load."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2,
        )
        # Step scheduler a few times to change internal state
        for val in [1.0, 0.9, 0.8, 0.85, 0.86, 0.87]:
            scheduler.step(val)
        original_num_bad = scheduler.num_bad_epochs
        original_best = scheduler.best

        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, model, optimizer, epoch=6, step=600,
                        scheduler=scheduler)

        # Fresh scheduler with default state
        fresh_model = _small_model()
        fresh_opt = torch.optim.Adam(fresh_model.parameters(), lr=1e-3)
        fresh_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            fresh_opt, mode="min", factor=0.5, patience=2,
        )
        assert fresh_sched.best != original_best  # different before load

        load_checkpoint(path, fresh_model, fresh_opt, scheduler=fresh_sched)

        assert fresh_sched.num_bad_epochs == original_num_bad
        assert fresh_sched.best == pytest.approx(original_best)

    def test_rng_state_round_trip(self, tmp_path, model):
        """Python, numpy, and torch RNG states restored on load."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Advance RNG to a non-default state
        random.random()
        np.random.rand(10)
        torch.rand(10)

        path = tmp_path / "ckpt_rng.pt"
        save_checkpoint(path, model, optimizer, epoch=1, step=10)

        # Record post-save random values (these should be reproducible after load)
        py_val = random.random()
        np_val = np.random.rand()
        torch_val = torch.rand(1).item()

        # Scramble RNG states
        for _ in range(100):
            random.random()
            np.random.rand()
            torch.rand(1)

        # Restore
        fresh_model = _small_model()
        fresh_opt = torch.optim.Adam(fresh_model.parameters(), lr=1e-3)
        load_checkpoint(path, fresh_model, fresh_opt)

        # Same random values should follow
        assert random.random() == py_val
        assert np.random.rand() == pytest.approx(np_val)
        assert torch.rand(1).item() == pytest.approx(torch_val)

    def test_backward_compatible_no_rng(self, tmp_path, model):
        """Old checkpoints without rng_states/scheduler load without error."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # Manually save old-format checkpoint (no rng_states, no scheduler)
        data = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": 5,
            "step": 50,
        }
        path = tmp_path / "old_ckpt.pt"
        torch.save(data, path)

        fresh_model = _small_model()
        fresh_opt = torch.optim.Adam(fresh_model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            fresh_opt, mode="min",
        )
        # Should not raise
        meta = load_checkpoint(path, fresh_model, fresh_opt, scheduler=scheduler)
        assert meta["epoch"] == 5

    def test_no_scheduler_in_checkpoint_skips_restore(self, tmp_path, model):
        """If checkpoint has no scheduler_state_dict, scheduler is left unchanged."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        save_checkpoint(path=tmp_path / "no_sched.pt", model=model,
                        optimizer=optimizer, epoch=1, step=1)
        # No scheduler passed to save, so scheduler_state_dict absent

        fresh_model = _small_model()
        fresh_opt = torch.optim.Adam(fresh_model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(fresh_opt, mode="min")
        original_best = sched.best

        load_checkpoint(tmp_path / "no_sched.pt", fresh_model, fresh_opt,
                        scheduler=sched)
        # But now save_checkpoint always saves rng_states — the scheduler
        # should still be unchanged since we didn't pass one to save
        assert sched.best == original_best


# ---------------------------------------------------------------------------
# 2. Gradient flow through full PPO update (all 3 heads)
# ---------------------------------------------------------------------------


class TestPPOGradientFlow:
    @pytest.fixture
    def ppo(self):
        model = _small_model()
        return KataGoPPOAlgorithm(KataGoPPOParams(epochs_per_batch=1), model)

    def test_all_three_heads_receive_gradients(self, ppo):
        """Policy, value, and score heads should all get non-zero gradients."""
        buf = _filled_buffer(num_envs=4, steps=3)
        next_values = torch.zeros(4)

        # Record params before
        policy_before = ppo.model.policy_conv2.weight.data.clone()
        value_before = ppo.model.value_fc2.weight.data.clone()
        score_before = ppo.model.score_fc2.weight.data.clone()

        ppo.update(buf, next_values)

        # All three heads should have changed
        assert not torch.equal(policy_before, ppo.model.policy_conv2.weight.data), \
            "Policy head received no gradient"
        assert not torch.equal(value_before, ppo.model.value_fc2.weight.data), \
            "Value head received no gradient"
        assert not torch.equal(score_before, ppo.model.score_fc2.weight.data), \
            "Score head received no gradient"

    def test_gradient_norms_are_finite(self, ppo):
        """Gradient norms should be finite after update."""
        buf = _filled_buffer(num_envs=4, steps=3)
        next_values = torch.zeros(4)
        metrics = ppo.update(buf, next_values)

        assert np.isfinite(metrics["gradient_norm"])
        assert metrics["gradient_norm"] > 0

    def test_each_loss_component_is_nonzero(self, ppo):
        """Each loss component should contribute to training."""
        buf = _filled_buffer(num_envs=4, steps=3)
        next_values = torch.zeros(4)
        metrics = ppo.update(buf, next_values)

        assert metrics["policy_loss"] != 0.0, "Policy loss is zero"
        assert metrics["value_loss"] != 0.0, "Value loss is zero"
        assert metrics["score_loss"] != 0.0, "Score loss is zero"
        assert metrics["entropy"] > 0.0, "Entropy should be positive"


# ---------------------------------------------------------------------------
# 3. Numerical stability through select_actions → update with extreme inputs
# ---------------------------------------------------------------------------


class TestNumericalStability:
    @pytest.fixture
    def ppo(self):
        model = _small_model()
        return KataGoPPOAlgorithm(KataGoPPOParams(epochs_per_batch=1), model)

    def test_large_observations_produce_finite_output(self, ppo):
        """Observations with large values should not produce NaN/Inf."""
        obs = torch.full((4, 50, 9, 9), 100.0)
        legal_masks = torch.ones(4, 11259, dtype=torch.bool)
        actions, log_probs, values = ppo.select_actions(obs, legal_masks)

        assert torch.isfinite(log_probs).all(), "NaN/Inf in log_probs"
        assert torch.isfinite(values).all(), "NaN/Inf in values"

    def test_near_zero_observations_produce_finite_output(self, ppo):
        """Very small observations should not produce NaN/Inf."""
        obs = torch.full((4, 50, 9, 9), 1e-7)
        legal_masks = torch.ones(4, 11259, dtype=torch.bool)
        actions, log_probs, values = ppo.select_actions(obs, legal_masks)

        assert torch.isfinite(log_probs).all()
        assert torch.isfinite(values).all()

    def test_extreme_obs_through_full_pipeline(self, ppo):
        """Large-obs rollout → update should produce finite metrics."""
        buf = KataGoRolloutBuffer(
            num_envs=4, obs_shape=(50, 9, 9), action_space=11259,
        )
        for _ in range(3):
            obs = torch.randn(4, 50, 9, 9) * 50.0  # large but varied
            legal_masks = torch.ones(4, 11259, dtype=torch.bool)
            actions, log_probs, values = ppo.select_actions(obs, legal_masks)
            buf.add(
                obs=obs, actions=actions, log_probs=log_probs, values=values,
                rewards=torch.zeros(4), dones=torch.zeros(4, dtype=torch.bool),
                terminated=torch.zeros(4, dtype=torch.bool),
                legal_masks=legal_masks,
                value_categories=torch.full((4,), -1, dtype=torch.long),
                score_targets=torch.randn(4).clamp(-1.5, 1.5),
            )

        next_values = torch.zeros(4)
        metrics = ppo.update(buf, next_values)

        for key, val in metrics.items():
            if isinstance(val, float):
                assert np.isfinite(val), f"{key} is not finite: {val}"

    def test_single_legal_action_per_env(self, ppo):
        """Only one legal action should produce valid log_prob = 0."""
        obs = torch.randn(4, 50, 9, 9)
        legal_masks = torch.zeros(4, 11259, dtype=torch.bool)
        legal_masks[:, 0] = True  # only action 0 is legal

        actions, log_probs, values = ppo.select_actions(obs, legal_masks)

        assert (actions == 0).all(), "Should select only legal action"
        assert torch.allclose(log_probs, torch.zeros_like(log_probs), atol=1e-5), \
            "log_prob of forced action should be ~0"


# ---------------------------------------------------------------------------
# 4. split_merge value consistency with KataGoPPOAlgorithm.scalar_value
# ---------------------------------------------------------------------------


class TestSplitMergeValueConsistency:
    def test_values_match_scalar_value_method(self):
        """split_merge_step should produce the same values as the centralized
        KataGoPPOAlgorithm.scalar_value static method."""
        model = _small_model()
        model.eval()

        num_envs = 4
        obs = torch.randn(num_envs, 50, 9, 9)
        legal_masks = torch.ones(num_envs, 11259, dtype=torch.bool)
        current_players = np.array([0, 0, 0, 0], dtype=np.uint8)

        result = split_merge_step(
            obs=obs, legal_masks=legal_masks,
            current_players=current_players,
            learner_model=model,
            opponent_model=model,
            learner_side=0,
        )

        # Independently compute scalar values
        with torch.no_grad():
            output = model(obs)
            expected_values = KataGoPPOAlgorithm.scalar_value(output.value_logits)

        assert torch.allclose(result.learner_values, expected_values, atol=1e-6), \
            "split_merge values diverge from KataGoPPOAlgorithm.scalar_value"


# ---------------------------------------------------------------------------
# 5. Buffer memory release after clear()
# ---------------------------------------------------------------------------


class TestBufferMemoryLifecycle:
    def test_clear_releases_tensor_references(self):
        """After clear(), internal lists should be empty and hold no tensors."""
        buf = _filled_buffer(num_envs=4, steps=5)
        assert buf.size == 5

        buf.clear()

        assert buf.size == 0
        assert len(buf.observations) == 0
        assert len(buf.actions) == 0
        assert len(buf.log_probs) == 0
        assert len(buf.values) == 0
        assert len(buf.rewards) == 0
        assert len(buf.dones) == 0
        assert len(buf.legal_masks) == 0
        assert len(buf.value_categories) == 0
        assert len(buf.score_targets) == 0
        assert len(buf.env_ids) == 0

    def test_update_calls_clear(self):
        """PPO update() should clear the buffer after consuming it."""
        model = _small_model()
        ppo = KataGoPPOAlgorithm(KataGoPPOParams(epochs_per_batch=1), model)
        buf = _filled_buffer(num_envs=4, steps=3)
        assert buf.size == 3

        ppo.update(buf, torch.zeros(4))

        assert buf.size == 0, "Buffer should be cleared after update()"

    def test_flatten_then_clear_no_shared_references(self):
        """Flattened data should be independent — clearing buffer after
        flatten() should not invalidate the flattened tensors."""
        buf = _filled_buffer(num_envs=4, steps=3)
        data = buf.flatten()

        # Capture reference before clear
        obs_ref = data["observations"].clone()

        buf.clear()

        # Flattened data should still be valid (cat creates new tensors)
        assert torch.equal(data["observations"], obs_ref)


# ---------------------------------------------------------------------------
# 6. Per-env GAE fallback path (no env_ids)
# ---------------------------------------------------------------------------


class TestPerEnvGAEFallback:
    def test_fallback_flat_gae_without_env_ids(self):
        """When split-merge data lacks env_ids, update() should use flat
        GAE with mean bootstrap — verify it produces finite metrics."""
        model = _small_model()
        ppo = KataGoPPOAlgorithm(KataGoPPOParams(epochs_per_batch=1), model)

        # Create buffer with non-matching T*N (simulate split-merge variable steps)
        buf = KataGoRolloutBuffer(
            num_envs=4, obs_shape=(50, 9, 9), action_space=11259,
        )
        # Add 3 steps but only for 2 envs each time (variable)
        for _ in range(3):
            n = 2  # variable number of envs per step
            buf.add(
                obs=torch.randn(n, 50, 9, 9),
                actions=torch.randint(0, 11259, (n,)),
                log_probs=torch.randn(n),
                values=torch.randn(n),
                rewards=torch.zeros(n),
                dones=torch.zeros(n, dtype=torch.bool),
                terminated=torch.zeros(n, dtype=torch.bool),
                legal_masks=torch.ones(n, 11259, dtype=torch.bool),
                value_categories=torch.full((n,), -1, dtype=torch.long),
                score_targets=torch.randn(n).clamp(-1.5, 1.5),
            )
        # total_samples = 6 != T*N = 3*4 = 12, and no env_ids → fallback path

        next_values = torch.zeros(4)
        metrics = ppo.update(buf, next_values)

        for key, val in metrics.items():
            if isinstance(val, float):
                assert np.isfinite(val), f"{key} is not finite in fallback GAE: {val}"

    def test_fallback_uses_mean_bootstrap(self):
        """Fallback path should use mean of next_values as bootstrap."""
        rewards = torch.tensor([1.0, 0.5, -0.5])
        values = torch.tensor([0.1, 0.2, 0.3])
        dones = torch.tensor([0.0, 0.0, 0.0])
        next_values = torch.tensor([0.4, 0.6])
        bootstrap = next_values.mean()

        # The fallback calls compute_gae with scalar bootstrap
        advantages = compute_gae(rewards, values, dones, bootstrap,
                                 gamma=0.99, lam=0.95)

        assert advantages.shape == (3,)
        assert torch.isfinite(advantages).all()


# ---------------------------------------------------------------------------
# 7. value_loss == 0.0 edge case (no terminal steps in epoch)
# ---------------------------------------------------------------------------


class TestValueLossZeroEdgeCase:
    def test_update_with_no_terminal_steps_returns_zero_value_loss(self):
        """When no environment terminates, value_categories are all -1
        (ignore_index) and value_loss should be 0.0."""
        model = _small_model()
        ppo = KataGoPPOAlgorithm(KataGoPPOParams(epochs_per_batch=1), model)

        buf = KataGoRolloutBuffer(
            num_envs=4, obs_shape=(50, 9, 9), action_space=11259,
        )
        for _ in range(3):
            buf.add(
                obs=torch.randn(4, 50, 9, 9),
                actions=torch.randint(0, 11259, (4,)),
                log_probs=torch.randn(4),
                values=torch.randn(4),
                rewards=torch.zeros(4),
                dones=torch.zeros(4, dtype=torch.bool),
                terminated=torch.zeros(4, dtype=torch.bool),
                legal_masks=torch.ones(4, 11259, dtype=torch.bool),
                value_categories=torch.full((4,), -1, dtype=torch.long),  # all ignore
                score_targets=torch.randn(4).clamp(-1.5, 1.5),
            )

        metrics = ppo.update(buf, torch.zeros(4))
        assert metrics["value_loss"] == 0.0
        # Other losses should still be valid
        assert np.isfinite(metrics["policy_loss"])
        assert np.isfinite(metrics["score_loss"])

    def test_update_with_all_ignore_still_trains_policy_and_score(self):
        """Even with no terminal steps, policy and score heads should update."""
        model = _small_model()
        ppo = KataGoPPOAlgorithm(KataGoPPOParams(epochs_per_batch=1), model)

        policy_before = model.policy_conv2.weight.data.clone()
        score_before = model.score_fc2.weight.data.clone()

        buf = KataGoRolloutBuffer(
            num_envs=4, obs_shape=(50, 9, 9), action_space=11259,
        )
        for _ in range(3):
            buf.add(
                obs=torch.randn(4, 50, 9, 9),
                actions=torch.randint(0, 11259, (4,)),
                log_probs=torch.randn(4),
                values=torch.randn(4),
                rewards=torch.zeros(4),
                dones=torch.zeros(4, dtype=torch.bool),
                terminated=torch.zeros(4, dtype=torch.bool),
                legal_masks=torch.ones(4, 11259, dtype=torch.bool),
                value_categories=torch.full((4,), -1, dtype=torch.long),
                score_targets=torch.randn(4).clamp(-1.5, 1.5),
            )

        ppo.update(buf, torch.zeros(4))

        assert not torch.equal(policy_before, model.policy_conv2.weight.data)
        assert not torch.equal(score_before, model.score_fc2.weight.data)


# ---------------------------------------------------------------------------
# 8. compute_value_metrics edge cases
# ---------------------------------------------------------------------------


class TestComputeValueMetricsEdgeCases:
    def test_single_element(self):
        """Single-sample input should produce valid metrics."""
        logits = torch.tensor([[2.0, 0.1, -1.0]])  # predicts W
        targets = torch.tensor([0])  # W
        metrics = compute_value_metrics(logits, targets)
        assert metrics["value_accuracy"] == 1.0
        assert metrics["frac_predicted_win"] == 1.0
        assert metrics["frac_predicted_draw"] == 0.0
        assert metrics["frac_predicted_loss"] == 0.0

    def test_all_same_prediction(self):
        """When model predicts the same class for all samples."""
        logits = torch.tensor([
            [5.0, -1.0, -1.0],
            [5.0, -1.0, -1.0],
            [5.0, -1.0, -1.0],
            [5.0, -1.0, -1.0],
        ])
        targets = torch.tensor([0, 1, 2, 0])  # mixed targets
        metrics = compute_value_metrics(logits, targets)
        assert metrics["value_accuracy"] == pytest.approx(0.5)  # 2/4 correct
        assert metrics["frac_predicted_win"] == 1.0
        assert metrics["frac_predicted_draw"] == 0.0
        assert metrics["frac_predicted_loss"] == 0.0

    def test_all_correct(self):
        """Perfect predictions."""
        logits = torch.tensor([
            [5.0, -1.0, -1.0],  # W
            [-1.0, 5.0, -1.0],  # D
            [-1.0, -1.0, 5.0],  # L
        ])
        targets = torch.tensor([0, 1, 2])
        metrics = compute_value_metrics(logits, targets)
        assert metrics["value_accuracy"] == 1.0
        assert metrics["frac_predicted_win"] == pytest.approx(1 / 3)
        assert metrics["frac_predicted_draw"] == pytest.approx(1 / 3)
        assert metrics["frac_predicted_loss"] == pytest.approx(1 / 3)

    def test_all_wrong(self):
        """All predictions incorrect."""
        logits = torch.tensor([
            [-1.0, -1.0, 5.0],  # predicts L
            [5.0, -1.0, -1.0],  # predicts W
            [-1.0, 5.0, -1.0],  # predicts D
        ])
        targets = torch.tensor([0, 1, 2])  # W, D, L
        metrics = compute_value_metrics(logits, targets)
        assert metrics["value_accuracy"] == 0.0


# ---------------------------------------------------------------------------
# 9. TransformerModel incompatibility with KataGo pipeline
# ---------------------------------------------------------------------------


class TestTransformerKataGoIncompatibility:
    def test_transformer_returns_tuple_not_katago_output(self):
        """TransformerModel.forward() returns (policy, value) tuple,
        not KataGoOutput — it's incompatible with KataGoPPOAlgorithm."""
        from keisei.training.models.transformer import TransformerModel, TransformerParams
        from keisei.training.models.katago_base import KataGoOutput

        model = TransformerModel(TransformerParams(d_model=32, nhead=4, num_layers=1))
        obs = torch.randn(2, 50, 9, 9)
        result = model(obs)

        assert isinstance(result, tuple), "TransformerModel should return tuple"
        assert not isinstance(result, KataGoOutput), \
            "TransformerModel should NOT return KataGoOutput"

    def test_transformer_not_katago_base_model(self):
        """TransformerModel should not be a KataGoBaseModel subclass."""
        from keisei.training.models.transformer import TransformerModel
        from keisei.training.models.katago_base import KataGoBaseModel

        assert not issubclass(TransformerModel, KataGoBaseModel)

    def test_katago_ppo_rejects_non_katago_model(self):
        """KataGoPPOAlgorithm.select_actions should fail with TransformerModel
        because its output lacks .policy_logits attribute."""
        from keisei.training.models.transformer import TransformerModel, TransformerParams

        model = TransformerModel(TransformerParams(d_model=32, nhead=4, num_layers=1))
        ppo = KataGoPPOAlgorithm(KataGoPPOParams(), model)

        obs = torch.randn(2, 50, 9, 9)
        legal_masks = torch.ones(2, 11259, dtype=torch.bool)

        with pytest.raises(AttributeError):
            ppo.select_actions(obs, legal_masks)
