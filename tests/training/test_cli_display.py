"""Tests for the barebones CLI DisplayManager (stderr output, no Rich)."""

import io
import sys
from unittest.mock import MagicMock

from keisei.training.display_manager import DisplayManager


def _make_display():
    """Create a DisplayManager with a minimal config stub."""
    config = MagicMock()
    return DisplayManager(config, log_file_path="/dev/null")


def _make_trainer(step=5000, total=500000):
    """Create a minimal trainer mock for display tests."""
    trainer = MagicMock()
    trainer.config.training.total_timesteps = total
    trainer.metrics_manager.global_timestep = step
    return trainer


class TestDisplayManagerInit:
    def test_no_rich_dependencies(self):
        dm = _make_display()
        assert not hasattr(dm, "rich_console")
        assert not hasattr(dm, "rich_log_messages")

    def test_setup_display_returns_self(self):
        dm = _make_display()
        result = dm.setup_display(MagicMock())
        assert result is dm


class TestUpdateProgress:
    def test_writes_to_stderr(self, capsys):
        dm = _make_display()
        dm._last_progress_time = 0  # force output
        trainer = _make_trainer()
        pending = {
            "ep_metrics": "L:87 R:0.34",
            "black_win_rate": 0.52,
            "white_win_rate": 0.38,
            "draw_rate": 0.10,
        }
        dm.update_progress(trainer, 42.3, pending)
        captured = capsys.readouterr()
        assert "Step 5000/500000" in captured.err
        assert "42.3 it/s" in captured.err

    def test_throttled(self, capsys):
        dm = _make_display()
        trainer = _make_trainer()
        pending = {}
        # First call should print
        dm._last_progress_time = 0
        dm.update_progress(trainer, 10.0, pending)
        # Second call within 2s should be suppressed
        dm.update_progress(trainer, 10.0, pending)
        captured = capsys.readouterr()
        assert captured.err.count("Step 5000") == 1


class TestNoOps:
    def test_refresh_dashboard_panels_noop(self):
        dm = _make_display()
        dm.refresh_dashboard_panels(MagicMock())  # should not raise

    def test_save_console_output_returns_false(self):
        dm = _make_display()
        assert dm.save_console_output("/tmp") is False


class TestContextManager:
    def test_start_context_manager(self):
        dm = _make_display()
        with dm.start():
            pass  # should not raise


class TestFinalizeDisplay:
    def test_prints_messages(self, capsys):
        dm = _make_display()
        dm.finalize_display("test_run", "/tmp/artifacts")
        captured = capsys.readouterr()
        assert "Run Finished" in captured.err
        assert "test_run" in captured.err
        assert "/tmp/artifacts" in captured.err


class TestLogMessages:
    def test_accumulates_plain_strings(self):
        dm = _make_display()
        dm.add_log_message("hello")
        dm.add_log_message("world")
        msgs = dm.get_log_messages()
        assert msgs == ["hello", "world"]
        assert all(isinstance(m, str) for m in msgs)
