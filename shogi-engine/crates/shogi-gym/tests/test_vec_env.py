"""Tests for VecEnv Python bindings."""
import numpy as np
import pytest
from shogi_gym import VecEnv


class TestVecEnvConstruction:
    def test_create(self):
        env = VecEnv(num_envs=4, max_ply=100)
        assert env.num_envs == 4
        assert env.action_space_size == 13_527
        assert env.observation_channels == 46

    def test_reset_shapes(self):
        env = VecEnv(num_envs=4, max_ply=100)
        result = env.reset()
        obs = np.asarray(result.observations)
        masks = np.asarray(result.legal_masks)
        assert obs.shape == (4, 46, 9, 9)
        assert obs.dtype == np.float32
        assert masks.shape == (4, 13_527)
        assert masks.dtype == np.bool_

    def test_reset_legal_masks_nonzero(self):
        env = VecEnv(num_envs=2, max_ply=100)
        result = env.reset()
        masks = np.asarray(result.legal_masks)
        for i in range(2):
            assert masks[i].sum() > 0, f"env {i} has no legal moves at start"


class TestVecEnvStepping:
    def test_step_shapes(self):
        env = VecEnv(num_envs=2, max_ply=100)
        result = env.reset()
        masks = np.asarray(result.legal_masks)
        actions = [int(np.where(masks[i])[0][0]) for i in range(2)]

        step_result = env.step(actions)
        obs = np.asarray(step_result.observations)
        masks = np.asarray(step_result.legal_masks)
        rewards = np.asarray(step_result.rewards)
        terminated = np.asarray(step_result.terminated)
        truncated = np.asarray(step_result.truncated)

        assert obs.shape == (2, 46, 9, 9)
        assert masks.shape == (2, 13_527)
        assert rewards.shape == (2,)
        assert terminated.shape == (2,)
        assert truncated.shape == (2,)

    def test_step_wrong_num_actions(self):
        env = VecEnv(num_envs=2, max_ply=100)
        env.reset()
        with pytest.raises(ValueError):
            env.step([0])

    def test_step_illegal_action(self):
        env = VecEnv(num_envs=1, max_ply=100)
        result = env.reset()
        masks = np.asarray(result.legal_masks)
        illegal_indices = np.where(~masks[0])[0]
        assert len(illegal_indices) > 0
        with pytest.raises(RuntimeError):
            env.step([int(illegal_indices[0])])

    def test_step_metadata(self):
        env = VecEnv(num_envs=2, max_ply=100)
        result = env.reset()
        masks = np.asarray(result.legal_masks)
        actions = [int(np.where(masks[i])[0][0]) for i in range(2)]

        step_result = env.step(actions)
        meta = step_result.step_metadata
        assert np.asarray(meta.captured_piece).shape == (2,)
        assert np.asarray(meta.termination_reason).shape == (2,)
        assert np.asarray(meta.ply_count).shape == (2,)

    def test_multi_step_no_crash(self):
        env = VecEnv(num_envs=4, max_ply=50)
        result = env.reset()
        masks = np.asarray(result.legal_masks)
        for _ in range(20):
            actions = []
            for i in range(4):
                legal = np.where(masks[i])[0]
                actions.append(int(legal[np.random.randint(len(legal))]))
            step_result = env.step(actions)
            masks = np.asarray(step_result.legal_masks)

    def test_autoreset_on_truncation(self):
        """Verify auto-reset produces valid initial state after truncation."""
        env = VecEnv(num_envs=1, max_ply=1)  # forces truncation after 1 move
        result = env.reset()
        masks = np.asarray(result.legal_masks)
        action = [int(np.where(masks[0])[0][0])]
        step_result = env.step(action)
        # Should be truncated
        assert np.asarray(step_result.truncated)[0], "expected truncation at max_ply=1"
        # After auto-reset, legal masks should be the start-position masks (30 legal moves)
        assert np.asarray(step_result.legal_masks)[0].sum() == 30
        # Terminal observations should be non-zero (the game state before reset)
        term_obs = np.asarray(step_result.terminal_observations)
        assert term_obs.shape == (1, 46, 9, 9)
        assert term_obs[0].sum() != 0.0, "terminal obs should be non-zero"
        # Current player should be Black (just reset)
        assert np.asarray(step_result.current_players)[0] == 0, "reset game should start with Black"

    def test_terminal_observations_shape(self):
        """Verify terminal_observations has correct shape."""
        env = VecEnv(num_envs=2, max_ply=100)
        result = env.reset()
        masks = np.asarray(result.legal_masks)
        actions = [int(np.where(masks[i])[0][0]) for i in range(2)]
        step_result = env.step(actions)
        term_obs = np.asarray(step_result.terminal_observations)
        assert term_obs.shape == (2, 46, 9, 9)
        assert term_obs.dtype == np.float32

    def test_current_players_output(self):
        """Verify current_players is in the step result."""
        env = VecEnv(num_envs=2, max_ply=100)
        result = env.reset()
        masks = np.asarray(result.legal_masks)
        actions = [int(np.where(masks[i])[0][0]) for i in range(2)]
        step_result = env.step(actions)
        players = np.asarray(step_result.current_players)
        assert players.shape == (2,)
        assert players.dtype == np.uint8
        # After first move, current player should be White (1)
        for p in players:
            assert p in (0, 1)

    def test_reward_values_in_range(self):
        """Rewards should only be -1.0, 0.0, or 1.0."""
        env = VecEnv(num_envs=4, max_ply=50)
        result = env.reset()
        masks = np.asarray(result.legal_masks)
        for _ in range(30):
            actions = []
            for i in range(4):
                legal = np.where(masks[i])[0]
                actions.append(int(legal[np.random.randint(len(legal))]))
            step_result = env.step(actions)
            rewards = np.asarray(step_result.rewards)
            for r in rewards:
                assert r in (-1.0, 0.0, 1.0), f"unexpected reward: {r}"
            masks = np.asarray(step_result.legal_masks)
