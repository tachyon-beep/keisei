"""Unit tests for ExperienceBuffer: add, clear, get_batch, GAE computation, edge cases."""

import pytest
import torch
import numpy as np

from keisei.core.experience_buffer import ExperienceBuffer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OBS_CHANNELS = 46
BOARD_SIZE = 9
NUM_ACTIONS = 13527
BUFFER_SIZE = 8
GAMMA = 0.99
LAMBDA_GAE = 0.95


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_dummy_experience(num_actions=NUM_ACTIONS):
    """Create a single dummy experience tuple for testing."""
    obs = torch.randn(OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    action = 42
    reward = 1.0
    log_prob = -0.5
    value = 0.8
    done = False
    legal_mask = torch.zeros(num_actions, dtype=torch.bool)
    legal_mask[action] = True  # At least one legal action
    return obs, action, reward, log_prob, value, done, legal_mask


def _fill_buffer(buf, n=None):
    """Fill *buf* with *n* dummy experiences (defaults to buf.buffer_size)."""
    if n is None:
        n = buf.buffer_size
    for _ in range(n):
        buf.add(*make_dummy_experience())


# ---------------------------------------------------------------------------
# 1. Basic operations
# ---------------------------------------------------------------------------
class TestBasicOperations:
    """Tests for fundamental buffer operations."""

    def test_new_buffer_has_ptr_zero_and_len_zero(self):
        buf = ExperienceBuffer(BUFFER_SIZE, GAMMA, LAMBDA_GAE)
        assert buf.ptr == 0
        assert len(buf) == 0

    def test_add_increments_ptr(self):
        buf = ExperienceBuffer(BUFFER_SIZE, GAMMA, LAMBDA_GAE)
        buf.add(*make_dummy_experience())
        assert buf.ptr == 1

    def test_len_returns_number_of_added_items(self):
        buf = ExperienceBuffer(BUFFER_SIZE, GAMMA, LAMBDA_GAE)
        n = 5
        for _ in range(n):
            buf.add(*make_dummy_experience())
        assert len(buf) == n
        assert buf.size() == n

    def test_capacity_returns_buffer_size(self):
        buf = ExperienceBuffer(BUFFER_SIZE, GAMMA, LAMBDA_GAE)
        assert buf.capacity() == BUFFER_SIZE

    def test_clear_resets_ptr_and_advantages_flag(self):
        buf = ExperienceBuffer(BUFFER_SIZE, GAMMA, LAMBDA_GAE)
        _fill_buffer(buf)
        buf.compute_advantages_and_returns(last_value=0.0)
        assert buf._advantages_computed is True
        buf.clear()
        assert buf.ptr == 0
        assert buf._advantages_computed is False

    def test_buffer_stores_correct_observation_data(self):
        buf = ExperienceBuffer(BUFFER_SIZE, GAMMA, LAMBDA_GAE)
        obs, action, reward, log_prob, value, done, legal_mask = make_dummy_experience()
        buf.add(obs, action, reward, log_prob, value, done, legal_mask)
        assert torch.allclose(buf.obs[0], obs)
        assert buf.actions[0].item() == action
        assert buf.rewards[0].item() == pytest.approx(reward)
        assert buf.log_probs[0].item() == pytest.approx(log_prob)
        assert buf.values[0].item() == pytest.approx(value)
        assert buf.dones[0].item() == done
        assert torch.equal(buf.legal_masks[0], legal_mask)


# ---------------------------------------------------------------------------
# 2. Buffer full behaviour
# ---------------------------------------------------------------------------
class TestBufferFullBehavior:
    """Tests for behaviour when buffer reaches capacity."""

    def test_ptr_equals_buffer_size_when_full(self):
        buf = ExperienceBuffer(BUFFER_SIZE, GAMMA, LAMBDA_GAE)
        _fill_buffer(buf)
        assert buf.ptr == BUFFER_SIZE

    def test_adding_beyond_capacity_does_not_increment_ptr(self):
        buf = ExperienceBuffer(BUFFER_SIZE, GAMMA, LAMBDA_GAE)
        _fill_buffer(buf)
        # Try to add one more
        buf.add(*make_dummy_experience())
        assert buf.ptr == BUFFER_SIZE

    def test_overfilling_does_not_crash(self):
        """Adding more than buffer_size items should not raise an exception."""
        buf = ExperienceBuffer(BUFFER_SIZE, GAMMA, LAMBDA_GAE)
        # Add double the capacity; should just warn, not crash
        for _ in range(BUFFER_SIZE * 2):
            buf.add(*make_dummy_experience())
        assert buf.ptr == BUFFER_SIZE


# ---------------------------------------------------------------------------
# 3. GAE computation
# ---------------------------------------------------------------------------
class TestGAEComputation:
    """Tests for Generalized Advantage Estimation computation."""

    def test_compute_sets_advantages_computed_flag(self):
        buf = ExperienceBuffer(BUFFER_SIZE, GAMMA, LAMBDA_GAE)
        _fill_buffer(buf)
        assert buf._advantages_computed is False
        buf.compute_advantages_and_returns(last_value=0.0)
        assert buf._advantages_computed is True

    def test_advantages_nonzero_for_nontrivial_rewards(self):
        buf = ExperienceBuffer(BUFFER_SIZE, GAMMA, LAMBDA_GAE)
        for i in range(BUFFER_SIZE):
            obs = torch.randn(OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE)
            legal_mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
            legal_mask[0] = True
            buf.add(obs, 0, float(i + 1), -0.5, 0.5, False, legal_mask)
        buf.compute_advantages_and_returns(last_value=0.0)
        advantages = buf.advantages[: buf.ptr]
        assert torch.any(advantages != 0.0)

    def test_returns_equal_advantages_plus_values(self):
        buf = ExperienceBuffer(BUFFER_SIZE, GAMMA, LAMBDA_GAE)
        _fill_buffer(buf)
        buf.compute_advantages_and_returns(last_value=0.0)
        n = buf.ptr
        expected_returns = buf.advantages[:n] + buf.values[:n]
        assert torch.allclose(buf.returns[:n], expected_returns, atol=1e-6)

    def test_gae_with_zero_gamma(self):
        """With gamma=0 advantages equal immediate TD errors: r - V(s)."""
        small_buf_size = 4
        buf = ExperienceBuffer(small_buf_size, gamma=0.0, lambda_gae=LAMBDA_GAE)
        rewards = [1.0, 2.0, 3.0, 0.5]
        values = [0.5, 0.5, 0.5, 0.5]
        for r, v in zip(rewards, values):
            obs = torch.randn(OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE)
            legal_mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
            legal_mask[0] = True
            buf.add(obs, 0, r, -0.5, v, False, legal_mask)
        buf.compute_advantages_and_returns(last_value=0.0)
        # With gamma=0: delta_t = r_t + 0*next_val*mask - v_t = r_t - v_t
        # gae_t = delta_t + 0 * ... = delta_t
        for i in range(small_buf_size):
            expected_adv = rewards[i] - values[i]
            assert buf.advantages[i].item() == pytest.approx(expected_adv, abs=1e-6)

    def test_gae_zero_rewards_and_zero_last_value(self):
        """With all zero rewards and last_value=0, advantages depend only on value predictions."""
        small_buf_size = 4
        buf = ExperienceBuffer(small_buf_size, GAMMA, LAMBDA_GAE)
        values = [0.5, 0.3, 0.2, 0.1]
        for v in values:
            obs = torch.randn(OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE)
            legal_mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
            legal_mask[0] = True
            buf.add(obs, 0, 0.0, -0.5, v, False, legal_mask)
        buf.compute_advantages_and_returns(last_value=0.0)
        # Advantages should be non-zero since values are non-zero
        # (the value function predictions create non-zero deltas)
        advantages = buf.advantages[: buf.ptr]
        assert torch.any(advantages != 0.0)

    def test_gae_done_flag_resets_advantage_propagation(self):
        """An episode boundary (done=True) should prevent GAE from propagating
        across the boundary."""
        buf_size = 2
        buf = ExperienceBuffer(buf_size, GAMMA, LAMBDA_GAE)
        obs = torch.randn(OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        legal_mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
        legal_mask[0] = True
        # Step 0: reward=1.0, value=0.5, done=True  (episode ends)
        buf.add(obs.clone(), 0, 1.0, -0.5, 0.5, True, legal_mask.clone())
        # Step 1: reward=2.0, value=0.5, done=False
        buf.add(obs.clone(), 0, 2.0, -0.5, 0.5, False, legal_mask.clone())
        buf.compute_advantages_and_returns(last_value=0.0)

        # Step 1 (last step, done=False):
        #   delta_1 = 2.0 + 0.99 * 0.0 * 1.0 - 0.5 = 1.5
        #   gae_1 = 1.5
        assert buf.advantages[1].item() == pytest.approx(1.5, abs=1e-5)

        # Step 0 (done=True, mask=0):
        #   delta_0 = 1.0 + 0.99 * v_1 * 0.0 - 0.5 = 0.5
        #   gae_0 = delta_0 + 0.99 * 0.95 * 0.0 * gae_1 = 0.5
        assert buf.advantages[0].item() == pytest.approx(0.5, abs=1e-5)

    def test_gae_manual_computation(self):
        """Verify GAE with a manually computed example.

        buffer_size=2, rewards=[1.0, 2.0], values=[0.5, 0.5],
        dones=[False, False], last_value=0.0

        delta_1 = r_1 + gamma*last_value*mask_1 - v_1
               = 2.0 + 0.99*0.0*1.0 - 0.5 = 1.5
        gae_1 = delta_1 = 1.5

        delta_0 = r_0 + gamma*v_1*mask_0 - v_0
               = 1.0 + 0.99*0.5*1.0 - 0.5 = 0.995
        gae_0 = delta_0 + gamma*lambda*mask_0*gae_1
              = 0.995 + 0.99*0.95*1.0*1.5
              = 0.995 + 1.41075 = 2.40575
        """
        buf = ExperienceBuffer(2, GAMMA, LAMBDA_GAE)
        obs = torch.randn(OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        legal_mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
        legal_mask[0] = True
        buf.add(obs.clone(), 0, 1.0, -0.5, 0.5, False, legal_mask.clone())
        buf.add(obs.clone(), 0, 2.0, -0.5, 0.5, False, legal_mask.clone())
        buf.compute_advantages_and_returns(last_value=0.0)

        assert buf.advantages[1].item() == pytest.approx(1.5, abs=1e-5)
        assert buf.advantages[0].item() == pytest.approx(2.40575, abs=1e-5)

        # Returns = advantages + values
        assert buf.returns[0].item() == pytest.approx(2.40575 + 0.5, abs=1e-5)
        assert buf.returns[1].item() == pytest.approx(1.5 + 0.5, abs=1e-5)


# ---------------------------------------------------------------------------
# 4. get_batch
# ---------------------------------------------------------------------------
class TestGetBatch:
    """Tests for the get_batch method."""

    def test_get_batch_returns_dict_with_all_expected_keys(self):
        buf = ExperienceBuffer(BUFFER_SIZE, GAMMA, LAMBDA_GAE)
        _fill_buffer(buf)
        buf.compute_advantages_and_returns(last_value=0.0)
        batch = buf.get_batch()
        expected_keys = {
            "obs",
            "actions",
            "log_probs",
            "values",
            "advantages",
            "returns",
            "dones",
            "legal_masks",
            "rewards",
        }
        assert set(batch.keys()) == expected_keys

    def test_get_batch_shapes(self):
        buf = ExperienceBuffer(BUFFER_SIZE, GAMMA, LAMBDA_GAE)
        n = 5
        for _ in range(n):
            buf.add(*make_dummy_experience())
        buf.compute_advantages_and_returns(last_value=0.0)
        batch = buf.get_batch()

        assert batch["obs"].shape == (n, OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        assert batch["actions"].shape == (n,)
        assert batch["log_probs"].shape == (n,)
        assert batch["values"].shape == (n,)
        assert batch["advantages"].shape == (n,)
        assert batch["returns"].shape == (n,)
        assert batch["dones"].shape == (n,)
        assert batch["legal_masks"].shape == (n, NUM_ACTIONS)
        assert batch["rewards"].shape == (n,)

    def test_get_batch_raises_without_compute(self):
        buf = ExperienceBuffer(BUFFER_SIZE, GAMMA, LAMBDA_GAE)
        _fill_buffer(buf)
        with pytest.raises(RuntimeError, match="compute_advantages_and_returns"):
            buf.get_batch()


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Tests for boundary and edge-case scenarios."""

    def test_compute_on_empty_buffer_does_not_crash(self):
        """compute_advantages_and_returns on an empty buffer should return
        without error and leave _advantages_computed as False."""
        buf = ExperienceBuffer(BUFFER_SIZE, GAMMA, LAMBDA_GAE)
        buf.compute_advantages_and_returns(last_value=0.0)
        # The method returns early for empty buffer; flag stays False
        assert buf._advantages_computed is False

    def test_get_batch_on_empty_buffer_returns_empty_dict(self):
        """get_batch on an empty buffer should return an empty dict, not raise."""
        buf = ExperienceBuffer(BUFFER_SIZE, GAMMA, LAMBDA_GAE)
        batch = buf.get_batch()
        assert batch == {}
