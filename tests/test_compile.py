"""Tests for torch.compile integration with KataGoPPO."""

from __future__ import annotations

import logging

import pytest
import torch

from keisei.training.katago_ppo import (
    KataGoPPOAlgorithm,
    KataGoPPOParams,
    KataGoRolloutBuffer,
)
from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams


def _small_model():
    """Create a minimal SE-ResNet for testing (2 blocks, 32 channels)."""
    return SEResNetModel(SEResNetParams(
        num_blocks=2, channels=32, se_reduction=8,
        global_pool_channels=16, policy_channels=8,
        value_fc_size=32, score_fc_size=16, obs_channels=50,
    ))


def _filled_buffer(num_envs=4, steps=3, action_space=11259):
    """Create a buffer with terminal steps so value head gets gradients.

    Defined at module level so all test classes can use it.
    Kept in sync with tests/test_pytorch_hot_path_gaps.py::_filled_buffer.
    """
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
            legal_masks=torch.ones(num_envs, action_space, dtype=torch.bool),
            value_categories=torch.where(
                torch.tensor([is_last] * num_envs),
                torch.tensor([0, 2, 1, 0][:num_envs]),
                torch.full((num_envs,), -1),
            ),
            score_targets=torch.randn(num_envs).clamp(-1.5, 1.5),
        )
    return buf


class TestParameterIdentity:
    def test_default_forward_model_shares_params(self):
        """When forward_model is None, model and forward_model are the same object."""
        model = _small_model()
        params = KataGoPPOParams()
        ppo = KataGoPPOAlgorithm(params, model)
        assert ppo.forward_model is ppo.model

    def test_explicit_forward_model_shares_params(self):
        """When forward_model is the same object, assertion passes."""
        model = _small_model()
        params = KataGoPPOParams()
        ppo = KataGoPPOAlgorithm(params, model, forward_model=model)
        assert ppo.forward_model is ppo.model

    def test_diverged_forward_model_raises(self):
        """When forward_model is a different model instance, assertion fires."""
        model = _small_model()
        other_model = _small_model()
        params = KataGoPPOParams()
        with pytest.raises(AssertionError, match="forward_model and model must share parameters"):
            KataGoPPOAlgorithm(params, model, forward_model=other_model)


class TestCompileSetup:
    def test_no_compile_by_default(self):
        """compile_mode=None means no compiled models are created."""
        model = _small_model()
        params = KataGoPPOParams()
        ppo = KataGoPPOAlgorithm(params, model)
        assert ppo.compiled_train is None
        assert ppo.compiled_eval is None

    def test_compile_creates_both_models(self):
        """compile_mode set creates compiled_train and compiled_eval.

        Both wrappers should be callable without manual eval()/train() toggles —
        the __init__ sets mode before compiling each (see Note N2: tracing happens
        at first call, not at compile() time, but the mode must be correct at
        first call time).
        """
        model = _small_model()
        params = KataGoPPOParams(compile_mode="default")
        ppo = KataGoPPOAlgorithm(params, model)
        assert ppo.compiled_train is not None
        assert ppo.compiled_eval is not None
        # Both should be callable without mode toggle
        obs = torch.randn(2, 50, 9, 9)
        out_eval = ppo.compiled_eval(obs)
        out_train = ppo.compiled_train(obs)
        assert out_eval.policy_logits.shape == (2, 9, 9, 139)
        assert out_train.policy_logits.shape == (2, 9, 9, 139)

    def test_compiled_wrappers_share_underlying_module(self):
        """compiled_train and compiled_eval wrap the same module, sharing parameters.

        Verify via _orig_mod (the OptimizedModule's reference to the wrapped module).
        Both must resolve to the same object as ppo.forward_model.
        """
        model = _small_model()
        params = KataGoPPOParams(compile_mode="default")
        ppo = KataGoPPOAlgorithm(params, model)
        # torch.compile returns an OptimizedModule with _orig_mod pointing to
        # the original module. Both wrappers should point to the same module.
        assert hasattr(ppo.compiled_train, "_orig_mod"), (
            "compiled_train should be an OptimizedModule with _orig_mod"
        )
        assert ppo.compiled_train._orig_mod is ppo.forward_model
        assert ppo.compiled_eval._orig_mod is ppo.forward_model
        # And the underlying parameters should be the same objects
        train_params = list(ppo.compiled_train._orig_mod.parameters())
        model_params = list(ppo.model.parameters())
        assert len(train_params) == len(model_params)
        for tp, mp in zip(train_params, model_params):
            assert tp.data_ptr() == mp.data_ptr(), (
                "compiled wrapper parameter should share storage with base model"
            )

    def test_reduce_overhead_dynamic_warns(self, caplog):
        """reduce-overhead + compile_dynamic=True should log a warning.

        CUDA graphs (the mechanism behind reduce-overhead) are incompatible with
        dynamic shapes. See Note N5.
        """
        model = _small_model()
        params = KataGoPPOParams(
            compile_mode="reduce-overhead", compile_dynamic=True,
        )
        with caplog.at_level(logging.WARNING):
            KataGoPPOAlgorithm(params, model)
        assert any("reduce-overhead" in msg and "dynamic" in msg for msg in caplog.messages), (
            "Should warn when reduce-overhead is combined with compile_dynamic=True"
        )
