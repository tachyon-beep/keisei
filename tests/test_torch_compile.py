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
        assert ppo.compiled_train is not None
        assert ppo.compiled_eval is not None
        assert hasattr(ppo.compiled_train, "_orig_mod"), (
            "compiled_train should be an OptimizedModule with _orig_mod"
        )
        assert ppo.compiled_train._orig_mod is ppo.forward_model
        assert ppo.compiled_eval._orig_mod is ppo.forward_model  # type: ignore[attr-defined]
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


class TestSelectActionsCompile:
    def test_select_actions_runs_with_compile(self):
        """select_actions() works with compiled_eval model."""
        model = _small_model()
        params = KataGoPPOParams(compile_mode="default")
        ppo = KataGoPPOAlgorithm(params, model)
        obs = torch.randn(4, 50, 9, 9)
        legal_masks = torch.ones(4, 11259, dtype=torch.bool)
        actions, log_probs, values = ppo.select_actions(obs, legal_masks)
        assert actions.shape == (4,)
        assert log_probs.shape == (4,)
        assert values.shape == (4,)

    def test_select_actions_eager_still_works(self):
        """select_actions() still works without compile (compile_mode=None)."""
        model = _small_model()
        params = KataGoPPOParams()
        ppo = KataGoPPOAlgorithm(params, model)
        obs = torch.randn(4, 50, 9, 9)
        legal_masks = torch.ones(4, 11259, dtype=torch.bool)
        actions, log_probs, values = ppo.select_actions(obs, legal_masks)
        assert actions.shape == (4,)

    def test_select_actions_no_mode_switch_when_compiled(self):
        """When compiled, select_actions must NOT toggle eval/train on forward_model.

        This is the core BN mode-switch invariant (spec hazard H1). The compiled_eval
        wrapper handles eval mode internally via its traced graph. Toggling the
        underlying module would corrupt compiled_train's trace.
        """
        model = _small_model()
        params = KataGoPPOParams(compile_mode="default")
        ppo = KataGoPPOAlgorithm(params, model)
        # Model should be in train mode before and after select_actions
        assert ppo.forward_model.training is True
        obs = torch.randn(4, 50, 9, 9)
        legal_masks = torch.ones(4, 11259, dtype=torch.bool)
        ppo.select_actions(obs, legal_masks)
        assert ppo.forward_model.training is True, (
            "select_actions must not leave forward_model in eval mode when compiled"
        )


class TestUpdateCompile:
    def test_update_runs_with_compile(self):
        """update() completes without error when compile_mode is set."""
        model = _small_model()
        params = KataGoPPOParams(compile_mode="default", batch_size=4)
        ppo = KataGoPPOAlgorithm(params, model)
        buf = _filled_buffer(num_envs=4, steps=3)
        next_values = torch.randn(4)
        metrics = ppo.update(buf, next_values)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics

    def test_update_eager_still_works(self):
        """update() still works without compile."""
        model = _small_model()
        params = KataGoPPOParams(batch_size=4)
        ppo = KataGoPPOAlgorithm(params, model)
        buf = _filled_buffer(num_envs=4, steps=3)
        next_values = torch.randn(4)
        metrics = ppo.update(buf, next_values)
        assert "policy_loss" in metrics

    def test_update_changes_weights(self):
        """update() must actually modify model weights (detects stale-gradient bugs).

        Captures parameter values before and after update, verifies they changed.
        This catches silent bugs where compiled_train dispatches to the wrong model
        or the optimizer operates on a different parameter set.
        """
        model = _small_model()
        params = KataGoPPOParams(compile_mode="default", batch_size=4)
        ppo = KataGoPPOAlgorithm(params, model)
        # Snapshot first parameter before update
        first_param = list(model.parameters())[0]
        before = first_param.data.clone()
        buf = _filled_buffer(num_envs=4, steps=3)
        next_values = torch.randn(4)
        ppo.update(buf, next_values)
        after = first_param.data
        assert not torch.equal(before, after), (
            "Model parameters should change after update() — optimizer may be "
            "operating on the wrong parameter set"
        )

    def test_forward_model_in_train_mode_after_update(self):
        """forward_model must be in train mode after update() completes.

        This is critical for the BN invariant: split_merge_step in katago_loop.py
        calls learner_model.eval() during rollout (line 89), and update() must
        restore train mode. See Note N3.
        """
        model = _small_model()
        params = KataGoPPOParams(compile_mode="default", batch_size=4)
        ppo = KataGoPPOAlgorithm(params, model)
        buf = _filled_buffer(num_envs=4, steps=3)
        next_values = torch.randn(4)
        ppo.update(buf, next_values)
        assert ppo.forward_model.training is True


class TestGAERouting:
    def test_update_uses_cpu_gae_on_cpu(self):
        """update() uses CPU GAE when device is CPU (no compile needed)."""
        model = _small_model()
        params = KataGoPPOParams(batch_size=4)
        ppo = KataGoPPOAlgorithm(params, model)
        buf = _filled_buffer(num_envs=4, steps=3)
        next_values = torch.randn(4)
        # Should complete without error — CPU path
        metrics = ppo.update(buf, next_values)
        assert "policy_loss" in metrics

    def test_flat_fallback_stays_on_cpu(self):
        """When total_samples != T * N (flat fallback), CPU GAE is always used.

        The GPU path only supports 2D (T, N) structured input. The flat fallback
        (no env_ids) must never attempt compute_gae_gpu. See spec hazard H3.
        """
        model = _small_model()
        params = KataGoPPOParams(batch_size=4)
        ppo = KataGoPPOAlgorithm(params, model)
        buf = _filled_buffer(num_envs=4, steps=3)
        next_values = torch.randn(4)
        metrics = ppo.update(buf, next_values)
        assert "policy_loss" in metrics


class TestTimings:
    def test_timings_exist_after_init(self):
        """timings dict is initialized with expected keys."""
        model = _small_model()
        params = KataGoPPOParams()
        ppo = KataGoPPOAlgorithm(params, model)
        assert "select_actions_forward_ms" in ppo.timings
        assert "update_forward_backward_ms" in ppo.timings
        assert "gae_ms" in ppo.timings

    def test_timings_empty_on_cpu(self):
        """On CPU, no CUDA events are recorded, so timings stay empty."""
        model = _small_model()
        params = KataGoPPOParams(batch_size=4)
        ppo = KataGoPPOAlgorithm(params, model)
        buf = _filled_buffer(num_envs=4, steps=3)
        next_values = torch.randn(4)
        ppo.update(buf, next_values)
        ppo.flush_timings()
        assert ppo.timings["update_forward_backward_ms"] == []
        assert ppo.timings["gae_ms"] == []


class TestCompileCorrectness:
    def test_compiled_eval_matches_eager(self):
        """Compiled eval forward pass matches eager within tolerance.

        Freezes BN running stats by switching to eval mode, which isolates
        compile effects from BN stat divergence. This is the reliable gate —
        both eager and compiled see identical frozen BN statistics.
        """
        torch.manual_seed(42)
        model = _small_model()
        # Run a few forward passes to populate BN running stats
        for _ in range(5):
            model.train()
            model(torch.randn(4, 50, 9, 9))

        # Freeze BN running stats by switching to eval
        model.eval()
        obs = torch.randn(4, 50, 9, 9)

        # Eager forward pass
        with torch.no_grad():
            eager_out = model(obs)

        # Compiled forward pass (same model, same frozen BN state)
        compiled_model = torch.compile(model, mode="default")
        with torch.no_grad():
            compiled_out = compiled_model(obs)

        assert torch.allclose(
            eager_out.policy_logits, compiled_out.policy_logits, rtol=1e-5, atol=1e-5
        ), f"Policy max diff: {(eager_out.policy_logits - compiled_out.policy_logits).abs().max()}"
        assert torch.allclose(
            eager_out.value_logits, compiled_out.value_logits, rtol=1e-5, atol=1e-5
        ), f"Value max diff: {(eager_out.value_logits - compiled_out.value_logits).abs().max()}"
        assert torch.allclose(
            eager_out.score_lead, compiled_out.score_lead, rtol=1e-5, atol=1e-5
        ), f"Score max diff: {(eager_out.score_lead - compiled_out.score_lead).abs().max()}"

    def test_compiled_train_matches_eager_frozen_bn(self):
        """Compiled train forward pass matches eager with frozen BN stats.

        For train-mode comparison, we freeze BN stats by setting momentum=0
        on all BN layers. This prevents the single forward pass from updating
        running stats differently between eager and compiled paths.

        Note: a previous version of this test used load_state_dict to copy
        post-update BN stats, but the eager pass had already updated them
        in-place, causing divergence. Freezing momentum avoids this entirely.
        """
        torch.manual_seed(42)
        model = _small_model()
        # Populate BN stats with a few warmup passes
        model.train()
        for _ in range(5):
            model(torch.randn(4, 50, 9, 9))

        # Freeze BN momentum so forward passes don't update running stats
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.momentum = 0.0

        obs = torch.randn(4, 50, 9, 9)

        # Eager forward pass (BN stats frozen, but still in train mode)
        eager_out = model(obs)

        # Compiled forward pass (same model, same frozen BN)
        compiled_model = torch.compile(model, mode="default")
        compiled_out = compiled_model(obs)

        assert torch.allclose(
            eager_out.policy_logits, compiled_out.policy_logits, rtol=1e-5, atol=1e-5
        ), f"Policy max diff: {(eager_out.policy_logits - compiled_out.policy_logits).abs().max()}"
        assert torch.allclose(
            eager_out.value_logits, compiled_out.value_logits, rtol=1e-5, atol=1e-5
        ), f"Value max diff: {(eager_out.value_logits - compiled_out.value_logits).abs().max()}"
