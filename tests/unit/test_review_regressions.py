"""Unit tests for regressions identified during code review.

Covers:
- P1: Async callbacks executed in the synchronous training loop path
- P2: WebUIConfig.update_rate_hz wired into StreamlitManager throttle
- P2: Streamlit subprocess cleaned up on Trainer.__init__ failure
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from keisei.config_schema import WebUIConfig
from keisei.webui.streamlit_manager import StreamlitManager


# ---------------------------------------------------------------------------
# P2: update_rate_hz wired into StreamlitManager write throttle
# ---------------------------------------------------------------------------


class TestUpdateRateHzConfig:
    """StreamlitManager._min_write_interval derives from config.update_rate_hz."""

    def test_default_2hz_gives_half_second_interval(self):
        cfg = WebUIConfig(update_rate_hz=2.0)
        mgr = StreamlitManager(cfg)
        assert mgr._min_write_interval == pytest.approx(0.5)

    def test_10hz_gives_100ms_interval(self):
        cfg = WebUIConfig(update_rate_hz=10.0)
        mgr = StreamlitManager(cfg)
        assert mgr._min_write_interval == pytest.approx(0.1)

    def test_1hz_gives_1_second_interval(self):
        cfg = WebUIConfig(update_rate_hz=1.0)
        mgr = StreamlitManager(cfg)
        assert mgr._min_write_interval == pytest.approx(1.0)

    def test_zero_hz_clamped_to_floor(self):
        """Zero Hz doesn't cause ZeroDivisionError; clamps to 0.1 Hz floor."""
        cfg = WebUIConfig(update_rate_hz=0.0)
        mgr = StreamlitManager(cfg)
        assert mgr._min_write_interval == pytest.approx(10.0)

    def test_negative_hz_clamped_to_floor(self):
        """Negative Hz is treated like zero — clamped to floor."""
        cfg = WebUIConfig(update_rate_hz=-5.0)
        mgr = StreamlitManager(cfg)
        assert mgr._min_write_interval == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Helper: build a minimal TrainingLoopManager for testing run()
# ---------------------------------------------------------------------------


def _build_tlm_with_mocks(total_timesteps=10, has_async=False, async_execute_fn=None):
    """Build a TrainingLoopManager wired to mocks that terminate after one full epoch.

    The _run_epoch side_effect advances global_timestep on the SECOND call so the
    first loop iteration fully executes (including the post-epoch callback bridge),
    and the second iteration's _run_epoch triggers termination.
    """
    from keisei.training.training_loop_manager import TrainingLoopManager

    mock_trainer = MagicMock()
    mock_trainer.config.training.total_timesteps = total_timesteps
    mock_trainer.config.training.steps_per_epoch = 5
    mock_trainer.config.training.render_every_steps = 100
    mock_trainer.config.training.rich_display_update_interval_seconds = 999
    mock_trainer.config.parallel.enabled = False
    mock_trainer.metrics_manager.global_timestep = 0
    mock_trainer.log_both = MagicMock()
    mock_trainer.webui_manager = None

    # Async callback configuration
    mock_trainer.callback_manager.has_async_callbacks.return_value = has_async
    if async_execute_fn is not None:
        mock_trainer.callback_manager.execute_step_callbacks_async = async_execute_fn

    # Construct TLM without running __init__ (avoids needing real Trainer)
    tlm = TrainingLoopManager.__new__(TrainingLoopManager)
    tlm.trainer = mock_trainer
    tlm.config = mock_trainer.config
    tlm.agent = mock_trainer.agent
    tlm.buffer = mock_trainer.experience_buffer
    tlm.step_manager = mock_trainer.step_manager
    tlm.display = mock_trainer.display
    tlm.callbacks = mock_trainer.callbacks
    tlm.current_epoch = 0
    tlm.last_time_for_sps = 0.0
    tlm.steps_since_last_time_for_sps = 0
    tlm.last_display_update_time = 0.0
    tlm.parallel_manager = None
    tlm.episode_state = SimpleNamespace(current_obs="dummy_obs")

    # _run_epoch: first call does nothing; second call bumps timestep to terminate
    epoch_calls = {"n": 0}

    def run_epoch_side_effect(log_both):
        epoch_calls["n"] += 1
        if epoch_calls["n"] >= 2:
            mock_trainer.metrics_manager.global_timestep = total_timesteps

    tlm._run_epoch = MagicMock(side_effect=run_epoch_side_effect)

    return tlm, mock_trainer


# ---------------------------------------------------------------------------
# P1: Async callbacks bridged in synchronous run() path
# ---------------------------------------------------------------------------


class TestAsyncCallbackBridgeInSyncPath:
    """Async callbacks must be executed even when using the sync run() path."""

    def test_sync_run_executes_async_callbacks(self):
        """When async callbacks are registered, sync run() bridges them."""
        async_called = {"count": 0}

        async def fake_async_execute(trainer):
            async_called["count"] += 1
            return {"async_metric": 1.0}

        tlm, mock_trainer = _build_tlm_with_mocks(
            total_timesteps=10,
            has_async=True,
            async_execute_fn=fake_async_execute,
        )

        tlm.run()

        assert async_called["count"] >= 1
        # Verify the metrics were integrated
        mock_trainer.metrics_manager.pending_progress_updates.update.assert_called()

    def test_sync_run_skips_async_bridge_when_no_async_callbacks(self):
        """When no async callbacks are registered, asyncio.run is never called."""
        tlm, _ = _build_tlm_with_mocks(total_timesteps=10, has_async=False)

        with patch("keisei.training.training_loop_manager.asyncio") as mock_asyncio:
            tlm.run()
            mock_asyncio.run.assert_not_called()


# ---------------------------------------------------------------------------
# P2: Streamlit subprocess cleaned up on Trainer.__init__ failure
# ---------------------------------------------------------------------------


class TestStreamlitCleanupOnInitFailure:
    """Streamlit subprocess is stopped if Trainer.__init__ fails after launch."""

    @patch("keisei.training.trainer.TrainingLoopManager")
    @patch("keisei.training.trainer.SetupManager")
    @patch("keisei.training.trainer.CallbackManager")
    @patch("keisei.training.trainer.EnhancedEvaluationManager")
    @patch("keisei.training.trainer.MetricsManager")
    @patch("keisei.training.trainer.EnvManager")
    @patch("keisei.training.trainer.ModelManager")
    @patch("keisei.training.trainer.DisplayManager")
    @patch("keisei.training.trainer.SessionManager")
    @patch("keisei.training.trainer.TrainingLogger")
    def test_webui_stopped_on_init_component_failure(
        self,
        MockLogger,
        MockSession,
        MockDisplay,
        MockModel,
        MockEnv,
        MockMetrics,
        MockEvalMgr,
        MockCallback,
        MockSetup,
        MockTLM,
    ):
        """If post-launch init raises, webui_manager.stop() is called."""
        from keisei.training.trainer import Trainer

        # Setup mock session manager
        mock_session = MockSession.return_value
        mock_session.run_name = "test_run"
        mock_session.run_artifact_dir = "/tmp/test"
        mock_session.model_dir = "/tmp/test/models"
        mock_session.log_file_path = "/tmp/test/log.txt"
        mock_session.eval_log_file_path = "/tmp/test/eval.txt"
        mock_session.is_wandb_active = False

        # Setup mock logger
        MockLogger.return_value.log = MagicMock()

        # Make ModelManager fail — this is the first call inside the try block
        MockModel.side_effect = RuntimeError("Simulated init failure")

        # Create a mock StreamlitManager that will be injected
        mock_webui = MagicMock()
        mock_webui.start.return_value = True

        config = MagicMock()
        config.webui.enabled = True
        config.webui.host = "localhost"
        config.webui.port = 8501
        config.env.device = "cpu"
        config.display.trend_history_length = 100
        config.display.elo_initial_rating = 1500
        config.display.elo_k_factor = 32

        args = SimpleNamespace(resume=None)

        with patch.dict(
            "sys.modules",
            {
                "keisei.webui.streamlit_manager": MagicMock(
                    STREAMLIT_AVAILABLE=True,
                    StreamlitManager=MagicMock(return_value=mock_webui),
                )
            },
        ):
            with pytest.raises(RuntimeError, match="Simulated init failure"):
                Trainer(config=config, args=args)

        # The critical assertion: stop() was called on the webui manager
        mock_webui.stop.assert_called_once()
