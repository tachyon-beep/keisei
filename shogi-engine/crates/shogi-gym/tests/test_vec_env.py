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
        env = VecEnv(num_envs=2)
        env.reset()
        with pytest.raises((ValueError, RuntimeError)):
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
