"""Unit tests for CallbackManager: setup, execution, add/remove, async, and clearing."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from keisei.training.callback_manager import CallbackManager
from keisei.training.callbacks import (
    AsyncCallback,
    AsyncEvaluationCallback,
    Callback,
    CheckpointCallback,
    EvaluationCallback,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    checkpoint_interval=10000,
    steps_per_epoch=2048,
    evaluation_interval=50000,
    enable_periodic_evaluation=True,
):
    """Build a minimal config namespace matching what CallbackManager reads."""
    return SimpleNamespace(
        training=SimpleNamespace(
            checkpoint_interval_timesteps=checkpoint_interval,
            steps_per_epoch=steps_per_epoch,
            evaluation_interval_timesteps=evaluation_interval,
        ),
        evaluation=SimpleNamespace(
            evaluation_interval_timesteps=evaluation_interval,
            enable_periodic_evaluation=enable_periodic_evaluation,
        ),
    )


def _make_config_no_evaluation(
    checkpoint_interval=10000,
    steps_per_epoch=2048,
):
    """Build a config without an evaluation section."""
    config = SimpleNamespace(
        training=SimpleNamespace(
            checkpoint_interval_timesteps=checkpoint_interval,
            steps_per_epoch=steps_per_epoch,
            evaluation_interval_timesteps=1000,
        ),
    )
    # Explicitly no evaluation attribute -- getattr(config, "evaluation", None)
    # will return None
    return config


class _DummyCallback(Callback):
    """Callback that records on_step_end calls."""

    def __init__(self):
        self.call_count = 0

    def on_step_end(self, trainer):
        self.call_count += 1


class _FailingCallback(Callback):
    """Callback that always raises on on_step_end."""

    def on_step_end(self, trainer):
        raise RuntimeError("boom")


class _DummyAsyncCallback(AsyncCallback):
    """Async callback that records calls."""

    def __init__(self):
        self.call_count = 0

    async def on_step_end_async(self, trainer):
        self.call_count += 1
        return {"dummy_metric": 1.0}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    """Default config with aligned intervals."""
    return _make_config(
        checkpoint_interval=2048,
        steps_per_epoch=2048,
        evaluation_interval=4096,
    )


@pytest.fixture
def config_no_eval():
    """Config missing the evaluation section."""
    return _make_config_no_evaluation(
        checkpoint_interval=2048,
        steps_per_epoch=2048,
    )


@pytest.fixture
def manager(config, tmp_path):
    """CallbackManager with default config."""
    return CallbackManager(config, str(tmp_path / "models"))


@pytest.fixture
def manager_no_eval(config_no_eval, tmp_path):
    """CallbackManager with config missing evaluation section."""
    return CallbackManager(config_no_eval, str(tmp_path / "models"))


@pytest.fixture
def mock_trainer():
    """Minimal MagicMock Trainer for execute_step_callbacks."""
    trainer = MagicMock()
    trainer.log_both = MagicMock()
    return trainer


# ---------------------------------------------------------------------------
# 1. setup_default_callbacks
# ---------------------------------------------------------------------------


class TestSetupDefaultCallbacks:
    """Tests for CallbackManager.setup_default_callbacks()."""

    def test_setup_default_callbacks_creates_checkpoint_and_evaluation(self, manager):
        cbs = manager.setup_default_callbacks()
        types = [type(cb) for cb in cbs]
        assert CheckpointCallback in types
        assert EvaluationCallback in types

    def test_setup_default_callbacks_returns_two_callbacks(self, manager):
        cbs = manager.setup_default_callbacks()
        assert len(cbs) == 2

    def test_setup_default_callbacks_stores_callbacks_internally(self, manager):
        cbs = manager.setup_default_callbacks()
        assert manager.callbacks is cbs

    def test_setup_default_callbacks_checkpoint_interval_matches_config(self, config, tmp_path):
        mgr = CallbackManager(config, str(tmp_path))
        cbs = mgr.setup_default_callbacks()
        ckpt_cb = [cb for cb in cbs if isinstance(cb, CheckpointCallback)][0]
        # 2048 is already aligned with steps_per_epoch=2048
        assert ckpt_cb.interval == 2048

    def test_setup_default_callbacks_checkpoint_model_dir_matches(self, tmp_path):
        config = _make_config(checkpoint_interval=2048, steps_per_epoch=2048)
        model_dir = str(tmp_path / "my_models")
        mgr = CallbackManager(config, model_dir)
        cbs = mgr.setup_default_callbacks()
        ckpt_cb = [cb for cb in cbs if isinstance(cb, CheckpointCallback)][0]
        assert ckpt_cb.model_dir == model_dir

    def test_setup_default_callbacks_evaluation_interval_matches_config(self, config, tmp_path):
        mgr = CallbackManager(config, str(tmp_path))
        cbs = mgr.setup_default_callbacks()
        eval_cb = [cb for cb in cbs if isinstance(cb, EvaluationCallback)][0]
        assert eval_cb.interval == 4096

    def test_setup_default_callbacks_evaluation_passes_eval_cfg(self, config, tmp_path):
        mgr = CallbackManager(config, str(tmp_path))
        cbs = mgr.setup_default_callbacks()
        eval_cb = [cb for cb in cbs if isinstance(cb, EvaluationCallback)][0]
        assert eval_cb.eval_cfg is config.evaluation

    def test_setup_default_callbacks_aligns_checkpoint_interval(self, tmp_path):
        # checkpoint_interval=5000 not divisible by steps_per_epoch=2048
        config = _make_config(
            checkpoint_interval=5000,
            steps_per_epoch=2048,
            evaluation_interval=4096,
        )
        mgr = CallbackManager(config, str(tmp_path))
        cbs = mgr.setup_default_callbacks()
        ckpt_cb = [cb for cb in cbs if isinstance(cb, CheckpointCallback)][0]
        # (5000 // 2048 + 1) * 2048 = 3 * 2048 = 6144
        assert ckpt_cb.interval == 6144

    def test_setup_default_callbacks_aligns_evaluation_interval(self, tmp_path):
        # evaluation_interval=5000 not divisible by steps_per_epoch=2048
        config = _make_config(
            checkpoint_interval=2048,
            steps_per_epoch=2048,
            evaluation_interval=5000,
        )
        mgr = CallbackManager(config, str(tmp_path))
        cbs = mgr.setup_default_callbacks()
        eval_cb = [cb for cb in cbs if isinstance(cb, EvaluationCallback)][0]
        assert eval_cb.interval == 6144

    def test_setup_default_callbacks_no_eval_section_uses_training_fallback(
        self, manager_no_eval
    ):
        cbs = manager_no_eval.setup_default_callbacks()
        eval_cb = [cb for cb in cbs if isinstance(cb, EvaluationCallback)][0]
        # Falls back to config.training.evaluation_interval_timesteps=1000
        # Not divisible by 2048 so aligns to 2048
        assert eval_cb.interval == 2048

    def test_setup_default_callbacks_no_eval_section_passes_none_eval_cfg(
        self, manager_no_eval
    ):
        cbs = manager_no_eval.setup_default_callbacks()
        eval_cb = [cb for cb in cbs if isinstance(cb, EvaluationCallback)][0]
        assert eval_cb.eval_cfg is None

    def test_setup_default_callbacks_zero_steps_per_epoch_skips_alignment(self, tmp_path):
        config = _make_config(
            checkpoint_interval=5000,
            steps_per_epoch=0,
            evaluation_interval=3000,
        )
        mgr = CallbackManager(config, str(tmp_path))
        cbs = mgr.setup_default_callbacks()
        ckpt_cb = [cb for cb in cbs if isinstance(cb, CheckpointCallback)][0]
        eval_cb = [cb for cb in cbs if isinstance(cb, EvaluationCallback)][0]
        # When steps_per_epoch is 0, no alignment happens
        assert ckpt_cb.interval == 5000
        assert eval_cb.interval == 3000


# ---------------------------------------------------------------------------
# 2. execute_step_callbacks
# ---------------------------------------------------------------------------


class TestExecuteStepCallbacks:
    """Tests for CallbackManager.execute_step_callbacks()."""

    def test_execute_step_callbacks_calls_each_callback(self, manager, mock_trainer):
        cb1 = _DummyCallback()
        cb2 = _DummyCallback()
        manager.callbacks = [cb1, cb2]
        manager.execute_step_callbacks(mock_trainer)
        assert cb1.call_count == 1
        assert cb2.call_count == 1

    def test_execute_step_callbacks_handles_exception_gracefully(
        self, manager, mock_trainer
    ):
        failing = _FailingCallback()
        good = _DummyCallback()
        manager.callbacks = [failing, good]
        # Should NOT raise
        manager.execute_step_callbacks(mock_trainer)
        # The good callback after the failing one should still execute
        assert good.call_count == 1

    def test_execute_step_callbacks_logs_error_on_exception(self, manager, mock_trainer):
        failing = _FailingCallback()
        manager.callbacks = [failing]
        manager.execute_step_callbacks(mock_trainer)
        mock_trainer.log_both.assert_called_once()
        call_args = mock_trainer.log_both.call_args
        assert "_FailingCallback" in call_args[0][0]
        assert "boom" in call_args[0][0]

    def test_execute_step_callbacks_no_callbacks_does_nothing(self, manager, mock_trainer):
        manager.callbacks = []
        # Should not raise
        manager.execute_step_callbacks(mock_trainer)

    def test_execute_step_callbacks_no_log_both_still_survives_exception(self, manager):
        trainer = MagicMock(spec=[])  # Empty spec so hasattr returns False
        failing = _FailingCallback()
        manager.callbacks = [failing]
        # Should not raise even without log_both
        manager.execute_step_callbacks(trainer)


# ---------------------------------------------------------------------------
# 3. add_callback
# ---------------------------------------------------------------------------


class TestAddCallback:
    """Tests for CallbackManager.add_callback()."""

    def test_add_callback_appends_to_list(self, manager):
        cb = _DummyCallback()
        manager.add_callback(cb)
        assert cb in manager.callbacks

    def test_add_callback_increases_count(self, manager):
        initial = len(manager.callbacks)
        manager.add_callback(_DummyCallback())
        assert len(manager.callbacks) == initial + 1

    def test_add_multiple_callbacks_preserves_order(self, manager):
        cb1 = _DummyCallback()
        cb2 = _DummyCallback()
        manager.add_callback(cb1)
        manager.add_callback(cb2)
        assert manager.callbacks[-2] is cb1
        assert manager.callbacks[-1] is cb2


# ---------------------------------------------------------------------------
# 4. remove_callback
# ---------------------------------------------------------------------------


class TestRemoveCallback:
    """Tests for CallbackManager.remove_callback()."""

    def test_remove_callback_returns_true_when_found(self, manager):
        manager.callbacks = [_DummyCallback()]
        result = manager.remove_callback(_DummyCallback)
        assert result is True

    def test_remove_callback_removes_matching_type(self, manager):
        manager.callbacks = [_DummyCallback()]
        manager.remove_callback(_DummyCallback)
        assert len(manager.callbacks) == 0

    def test_remove_callback_returns_false_when_not_found(self, manager):
        manager.callbacks = [_DummyCallback()]
        result = manager.remove_callback(CheckpointCallback)
        assert result is False

    def test_remove_callback_leaves_other_types(self, manager):
        dummy = _DummyCallback()
        manager.setup_default_callbacks()
        manager.add_callback(dummy)
        manager.remove_callback(CheckpointCallback)
        assert dummy in manager.callbacks
        assert not any(isinstance(cb, CheckpointCallback) for cb in manager.callbacks)

    def test_remove_callback_removes_all_instances_of_type(self, manager):
        manager.callbacks = [_DummyCallback(), _DummyCallback(), _FailingCallback()]
        manager.remove_callback(_DummyCallback)
        assert len(manager.callbacks) == 1
        assert isinstance(manager.callbacks[0], _FailingCallback)


# ---------------------------------------------------------------------------
# 5. setup_async_callbacks
# ---------------------------------------------------------------------------


class TestSetupAsyncCallbacks:
    """Tests for CallbackManager.setup_async_callbacks()."""

    def test_setup_async_callbacks_creates_async_evaluation_callback(self, manager):
        async_cbs = manager.setup_async_callbacks()
        assert len(async_cbs) == 1
        assert isinstance(async_cbs[0], AsyncEvaluationCallback)

    def test_setup_async_callbacks_stores_internally(self, manager):
        async_cbs = manager.setup_async_callbacks()
        assert manager.async_callbacks is async_cbs

    def test_setup_async_callbacks_uses_eval_cfg_interval(self, config, tmp_path):
        mgr = CallbackManager(config, str(tmp_path))
        async_cbs = mgr.setup_async_callbacks()
        assert async_cbs[0].interval == 4096

    def test_setup_async_callbacks_aligns_interval(self, tmp_path):
        config = _make_config(
            checkpoint_interval=2048,
            steps_per_epoch=2048,
            evaluation_interval=5000,
        )
        mgr = CallbackManager(config, str(tmp_path))
        async_cbs = mgr.setup_async_callbacks()
        assert async_cbs[0].interval == 6144

    def test_setup_async_callbacks_no_eval_section_uses_training_fallback(self, tmp_path):
        config = _make_config_no_evaluation(
            checkpoint_interval=2048, steps_per_epoch=2048
        )
        mgr = CallbackManager(config, str(tmp_path))
        async_cbs = mgr.setup_async_callbacks()
        # Falls back to training.evaluation_interval_timesteps=1000, aligned to 2048
        assert async_cbs[0].interval == 2048

    def test_setup_async_callbacks_passes_eval_cfg(self, config, tmp_path):
        mgr = CallbackManager(config, str(tmp_path))
        async_cbs = mgr.setup_async_callbacks()
        assert async_cbs[0].eval_cfg is config.evaluation


# ---------------------------------------------------------------------------
# 6. has_async_callbacks
# ---------------------------------------------------------------------------


class TestHasAsyncCallbacks:
    """Tests for CallbackManager.has_async_callbacks()."""

    def test_has_async_callbacks_false_initially(self, manager):
        assert manager.has_async_callbacks() is False

    def test_has_async_callbacks_true_after_setup(self, manager):
        manager.setup_async_callbacks()
        assert manager.has_async_callbacks() is True

    def test_has_async_callbacks_true_after_manual_add(self, manager):
        manager.add_async_callback(_DummyAsyncCallback())
        assert manager.has_async_callbacks() is True

    def test_has_async_callbacks_false_after_clear(self, manager):
        manager.setup_async_callbacks()
        manager.clear_async_callbacks()
        assert manager.has_async_callbacks() is False


# ---------------------------------------------------------------------------
# 7. get_callbacks / get_async_callbacks
# ---------------------------------------------------------------------------


class TestGetCallbacks:
    """Tests for get_callbacks() and get_async_callbacks()."""

    def test_get_callbacks_returns_copy(self, manager):
        manager.add_callback(_DummyCallback())
        returned = manager.get_callbacks()
        assert returned == manager.callbacks
        assert returned is not manager.callbacks

    def test_get_callbacks_mutation_does_not_affect_internal(self, manager):
        manager.add_callback(_DummyCallback())
        returned = manager.get_callbacks()
        returned.clear()
        assert len(manager.callbacks) == 1

    def test_get_async_callbacks_returns_copy(self, manager):
        manager.add_async_callback(_DummyAsyncCallback())
        returned = manager.get_async_callbacks()
        assert returned == manager.async_callbacks
        assert returned is not manager.async_callbacks

    def test_get_async_callbacks_mutation_does_not_affect_internal(self, manager):
        manager.add_async_callback(_DummyAsyncCallback())
        returned = manager.get_async_callbacks()
        returned.clear()
        assert len(manager.async_callbacks) == 1


# ---------------------------------------------------------------------------
# 8. clear_callbacks / clear_async_callbacks
# ---------------------------------------------------------------------------


class TestClearCallbacks:
    """Tests for clear_callbacks() and clear_async_callbacks()."""

    def test_clear_callbacks_empties_list(self, manager):
        manager.add_callback(_DummyCallback())
        manager.add_callback(_DummyCallback())
        manager.clear_callbacks()
        assert len(manager.callbacks) == 0

    def test_clear_async_callbacks_empties_list(self, manager):
        manager.add_async_callback(_DummyAsyncCallback())
        manager.clear_async_callbacks()
        assert len(manager.async_callbacks) == 0

    def test_clear_callbacks_is_idempotent(self, manager):
        manager.clear_callbacks()
        manager.clear_callbacks()
        assert len(manager.callbacks) == 0


# ---------------------------------------------------------------------------
# 9. use_async_evaluation
# ---------------------------------------------------------------------------


class TestUseAsyncEvaluation:
    """Tests for CallbackManager.use_async_evaluation()."""

    def test_use_async_evaluation_removes_sync_eval_callback(self, manager):
        manager.setup_default_callbacks()
        assert any(isinstance(cb, EvaluationCallback) for cb in manager.callbacks)
        manager.use_async_evaluation()
        assert not any(isinstance(cb, EvaluationCallback) for cb in manager.callbacks)

    def test_use_async_evaluation_adds_async_eval_callback(self, manager):
        manager.setup_default_callbacks()
        manager.use_async_evaluation()
        assert any(
            isinstance(cb, AsyncEvaluationCallback) for cb in manager.async_callbacks
        )

    def test_use_async_evaluation_keeps_checkpoint_callback(self, manager):
        manager.setup_default_callbacks()
        manager.use_async_evaluation()
        assert any(isinstance(cb, CheckpointCallback) for cb in manager.callbacks)

    def test_use_async_evaluation_does_not_duplicate_async_callback(self, manager):
        manager.setup_default_callbacks()
        manager.use_async_evaluation()
        manager.use_async_evaluation()  # Call again
        async_eval_count = sum(
            1
            for cb in manager.async_callbacks
            if isinstance(cb, AsyncEvaluationCallback)
        )
        assert async_eval_count == 1

    def test_use_async_evaluation_sets_correct_interval(self, tmp_path):
        config = _make_config(
            checkpoint_interval=2048,
            steps_per_epoch=2048,
            evaluation_interval=4096,
        )
        mgr = CallbackManager(config, str(tmp_path))
        mgr.setup_default_callbacks()
        mgr.use_async_evaluation()
        async_eval_cb = [
            cb
            for cb in mgr.async_callbacks
            if isinstance(cb, AsyncEvaluationCallback)
        ][0]
        assert async_eval_cb.interval == 4096


# ---------------------------------------------------------------------------
# 10. execute_step_callbacks_async
# ---------------------------------------------------------------------------


class TestExecuteStepCallbacksAsync:
    """Tests for CallbackManager.execute_step_callbacks_async()."""

    def test_execute_step_callbacks_async_calls_each_callback(self, manager, mock_trainer):
        cb1 = _DummyAsyncCallback()
        cb2 = _DummyAsyncCallback()
        manager.async_callbacks = [cb1, cb2]
        asyncio.run(manager.execute_step_callbacks_async(mock_trainer))
        assert cb1.call_count == 1
        assert cb2.call_count == 1

    def test_execute_step_callbacks_async_combines_metrics(self, manager, mock_trainer):
        cb = _DummyAsyncCallback()
        manager.async_callbacks = [cb]
        result = asyncio.run(manager.execute_step_callbacks_async(mock_trainer))
        assert result == {"dummy_metric": 1.0}

    def test_execute_step_callbacks_async_returns_none_when_empty(
        self, manager, mock_trainer
    ):
        manager.async_callbacks = []
        result = asyncio.run(manager.execute_step_callbacks_async(mock_trainer))
        assert result is None

    def test_execute_step_callbacks_async_handles_exception_gracefully(
        self, manager, mock_trainer
    ):
        class _FailingAsyncCallback(AsyncCallback):
            async def on_step_end_async(self, trainer):
                raise RuntimeError("async boom")

        good = _DummyAsyncCallback()
        manager.async_callbacks = [_FailingAsyncCallback(), good]
        asyncio.run(manager.execute_step_callbacks_async(mock_trainer))
        # Good callback still ran
        assert good.call_count == 1


# ---------------------------------------------------------------------------
# 11. add_async_callback / remove_async_callback
# ---------------------------------------------------------------------------


class TestAsyncCallbackManagement:
    """Tests for add_async_callback() and remove_async_callback()."""

    def test_add_async_callback_appends(self, manager):
        cb = _DummyAsyncCallback()
        manager.add_async_callback(cb)
        assert cb in manager.async_callbacks

    def test_remove_async_callback_returns_true_when_found(self, manager):
        manager.async_callbacks = [_DummyAsyncCallback()]
        result = manager.remove_async_callback(_DummyAsyncCallback)
        assert result is True
        assert len(manager.async_callbacks) == 0

    def test_remove_async_callback_returns_false_when_not_found(self, manager):
        manager.async_callbacks = [_DummyAsyncCallback()]
        result = manager.remove_async_callback(AsyncEvaluationCallback)
        assert result is False
        assert len(manager.async_callbacks) == 1


# ---------------------------------------------------------------------------
# 12. __init__
# ---------------------------------------------------------------------------


class TestInit:
    """Tests for CallbackManager.__init__()."""

    def test_init_stores_config(self, config, tmp_path):
        mgr = CallbackManager(config, str(tmp_path))
        assert mgr.config is config

    def test_init_stores_model_dir(self, config, tmp_path):
        model_dir = str(tmp_path / "my_models")
        mgr = CallbackManager(config, model_dir)
        assert mgr.model_dir == model_dir

    def test_init_starts_with_empty_callback_lists(self, config, tmp_path):
        mgr = CallbackManager(config, str(tmp_path))
        assert mgr.callbacks == []
        assert mgr.async_callbacks == []
