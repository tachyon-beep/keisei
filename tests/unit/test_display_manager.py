"""Unit tests for keisei.training.display_manager: DisplayManager."""

import time
from types import SimpleNamespace
from unittest.mock import patch

from keisei.training.display_manager import DisplayManager


def _make_config():
    """Build a minimal config namespace for DisplayManager."""
    return SimpleNamespace(
        display=SimpleNamespace(display_moves=False, turn_tick=0.0),
        training=SimpleNamespace(total_timesteps=1000),
    )


def _make_trainer(global_timestep=100):
    """Build a minimal trainer namespace for update_progress."""
    return SimpleNamespace(
        config=SimpleNamespace(training=SimpleNamespace(total_timesteps=1000)),
        metrics_manager=SimpleNamespace(global_timestep=global_timestep),
    )


class TestDisplayManagerSetup:
    """Tests for DisplayManager initialization and setup."""

    def test_setup_display_returns_self(self):
        dm = DisplayManager(_make_config(), "/tmp/test.log")
        result = dm.setup_display(trainer=None)
        assert result is dm

    def test_initial_log_messages_empty(self):
        dm = DisplayManager(_make_config(), "/tmp/test.log")
        assert dm.get_log_messages() == []


class TestUpdateProgressThrottling:
    """Tests for update_progress throttling behavior."""

    def test_first_call_prints(self):
        dm = DisplayManager(_make_config(), "/tmp/test.log")
        trainer = _make_trainer()
        with patch("sys.stderr") as mock_stderr:
            dm.update_progress(trainer, speed=10.0, pending_updates={})
            mock_stderr.write.assert_called()

    def test_second_call_within_two_seconds_suppressed(self):
        dm = DisplayManager(_make_config(), "/tmp/test.log")
        trainer = _make_trainer()
        with patch("sys.stderr"):
            dm.update_progress(trainer, speed=10.0, pending_updates={})

        # Second call immediately after should be suppressed
        with patch("sys.stderr") as mock_stderr:
            dm.update_progress(trainer, speed=10.0, pending_updates={})
            mock_stderr.write.assert_not_called()


class TestContextManager:
    """Tests for the start() context manager."""

    def test_start_context_manager_yields(self):
        dm = DisplayManager(_make_config(), "/tmp/test.log")
        with dm.start():
            pass  # Should not raise


class TestSaveConsoleOutput:
    """Tests for save_console_output no-op."""

    def test_returns_false(self):
        dm = DisplayManager(_make_config(), "/tmp/test.log")
        assert dm.save_console_output("/tmp/output") is False


class TestPrintRule:
    """Tests for print_rule separator output."""

    def test_print_rule_writes_to_stderr(self):
        dm = DisplayManager(_make_config(), "/tmp/test.log")
        with patch("sys.stderr") as mock_stderr:
            dm.print_rule("Test Section")
            # Should have written separator lines and title
            assert mock_stderr.write.call_count >= 1


class TestLogMessages:
    """Tests for log message accumulation."""

    def test_add_and_get_log_messages(self):
        dm = DisplayManager(_make_config(), "/tmp/test.log")
        dm.add_log_message("msg1")
        dm.add_log_message("msg2")
        assert dm.get_log_messages() == ["msg1", "msg2"]


class TestFinalizeDisplay:
    """Tests for finalize_display output."""

    def test_finalize_display_writes_to_stderr(self):
        dm = DisplayManager(_make_config(), "/tmp/test.log")
        with patch("sys.stderr") as mock_stderr:
            dm.finalize_display("test_run", "/tmp/artifacts")
            assert mock_stderr.write.call_count >= 1
