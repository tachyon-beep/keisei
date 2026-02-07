"""
training/display_manager.py: Barebones CLI display — prints throttled one-line
summaries to stderr. All rich visualization lives in the Streamlit dashboard.
"""

import sys
import time
from contextlib import contextmanager
from typing import Any, List


class DisplayManager:
    """Plain stderr display manager replacing the former Rich TUI."""

    def __init__(self, config: Any, log_file_path: str):
        self.config = config
        self.log_file_path = log_file_path
        self.log_messages: List[str] = []
        self._last_progress_time: float = 0.0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup_display(self, trainer: Any) -> "DisplayManager":
        """Return *self* — no separate display object needed."""
        return self

    # ------------------------------------------------------------------
    # Progress / dashboard
    # ------------------------------------------------------------------

    def update_progress(self, trainer: Any, speed: float, pending_updates: dict) -> None:
        """Print a throttled one-line progress summary to stderr."""
        now = time.time()
        if now - self._last_progress_time < 2.0:
            return
        self._last_progress_time = now

        mm = trainer.metrics_manager
        total = trainer.config.training.total_timesteps
        step = mm.global_timestep
        pct = (step / total * 100) if total > 0 else 0.0

        ep_metrics = pending_updates.get("ep_metrics", "")
        bw = pending_updates.get("black_win_rate", 0.0)
        ww = pending_updates.get("white_win_rate", 0.0)
        dr = pending_updates.get("draw_rate", 0.0)
        ppo = pending_updates.get("ppo_metrics", "")

        parts = [f"Step {step}/{total} ({pct:.1f}%)", f"{speed:.1f} it/s"]
        if ep_metrics:
            parts.append(f"Ep {ep_metrics}")
        parts.append(f"B:{bw:.0%} W:{ww:.0%} D:{dr:.0%}")
        if ppo:
            parts.append(ppo)

        print(" | ".join(parts), file=sys.stderr)

    def refresh_dashboard_panels(self, trainer: Any) -> None:
        """No-op — Streamlit handles visualization."""

    # ------------------------------------------------------------------
    # Context manager (replaces Rich Live)
    # ------------------------------------------------------------------

    @contextmanager
    def start(self):
        """Yield a no-op context manager (replaces Rich Live)."""
        yield

    # ------------------------------------------------------------------
    # Console output helpers
    # ------------------------------------------------------------------

    def save_console_output(self, output_dir: str) -> bool:
        """No HTML export — return False."""
        return False

    def print_rule(self, title: str, style: str = "") -> None:
        print(f"{'=' * 60}", file=sys.stderr)
        print(f"  {title}", file=sys.stderr)
        print(f"{'=' * 60}", file=sys.stderr)

    def print_message(self, message: str, style: str = "") -> None:
        print(message, file=sys.stderr)

    def finalize_display(self, run_name: str, run_artifact_dir: str) -> None:
        self.print_rule("Run Finished")
        self.print_message(f"Run '{run_name}' processing finished.")
        self.print_message(f"Output and logs are in: {run_artifact_dir}")

    # ------------------------------------------------------------------
    # Log message accumulator (used by TrainingLogger)
    # ------------------------------------------------------------------

    def get_log_messages(self) -> List[str]:
        return self.log_messages

    def add_log_message(self, message: str) -> None:
        self.log_messages.append(message)
