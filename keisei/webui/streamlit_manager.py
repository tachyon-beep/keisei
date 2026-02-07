"""
streamlit_manager.py: Drop-in replacement for WebUIManager.

Launches a Streamlit subprocess and bridges training data to it via an
atomically-written JSON state file.
"""

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from keisei.config_schema import WebUIConfig

try:
    import streamlit  # noqa: F401

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from .state_snapshot import build_snapshot, write_snapshot_atomic

_APP_PATH = str(Path(__file__).parent / "streamlit_app.py")


class StreamlitManager:
    """Manages a Streamlit dashboard subprocess for training visualization.

    Same 2-method contract as the old WebUIManager so the training loop
    doesn't need changes:
        - update_progress(trainer, speed, pending_updates)
        - refresh_dashboard_panels(trainer)
    """

    def __init__(
        self,
        config: WebUIConfig,
        state_dir: Optional[Path] = None,
    ) -> None:
        self.config = config
        self._logger = logging.getLogger(__name__)

        # State file location
        if state_dir is not None:
            self._state_dir = Path(state_dir)
        else:
            self._state_dir = Path.cwd() / ".keisei_webui"
        self._state_path = self._state_dir / "state.json"

        # Subprocess
        self._process: Optional[subprocess.Popen] = None

        # Rate limiting — minimum interval between writes
        self._min_write_interval = 0.5  # seconds
        self._last_write_time = 0.0

    def start(self, timeout: float = 5.0) -> bool:
        """Launch the Streamlit subprocess.

        Returns True if the process was launched (not necessarily serving yet),
        False if streamlit is unavailable.
        """
        if not STREAMLIT_AVAILABLE:
            self._logger.warning(
                "Streamlit not available. Install 'streamlit' to enable the dashboard."
            )
            return False

        if self._process is not None and self._process.poll() is None:
            return True  # Already running

        self._state_dir.mkdir(parents=True, exist_ok=True)
        port = str(self.config.port)

        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            _APP_PATH,
            "--server.port",
            port,
            "--server.headless",
            "true",
            "--server.address",
            self.config.host,
            "--",
            "--state-file",
            str(self._state_path),
        ]

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._logger.info(
                "Streamlit dashboard launched on http://%s:%s",
                self.config.host,
                port,
            )
            return True
        except (OSError, FileNotFoundError) as e:
            self._logger.error("Failed to launch Streamlit: %s", e)
            return False

    def stop(self, timeout: float = 5.0) -> None:
        """Terminate the Streamlit subprocess and clean up the state file."""
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    self._logger.warning("Streamlit process did not terminate, killing")
                    self._process.kill()
                    self._process.wait(timeout=2)
            self._process = None

        # Clean up state file
        try:
            if self._state_path.exists():
                self._state_path.unlink()
        except OSError:
            pass

    def update_progress(
        self,
        trainer: Any,
        speed: float,
        pending_updates: Dict[str, Any],
    ) -> None:
        """Write a training state snapshot (rate-limited)."""
        self._write_if_due(trainer, speed, pending_updates)

    def refresh_dashboard_panels(self, trainer: Any) -> None:
        """Write a training state snapshot (rate-limited).

        Same underlying operation as update_progress — both paths converge
        on building a snapshot.
        """
        self._write_if_due(trainer, speed=0.0, pending_updates={})

    def _write_if_due(
        self,
        trainer: Any,
        speed: float,
        pending_updates: Dict[str, Any],
    ) -> None:
        """Build and write a snapshot if enough time has elapsed."""
        now = time.time()
        if now - self._last_write_time < self._min_write_interval:
            return

        try:
            snapshot = build_snapshot(trainer, speed, pending_updates)
            write_snapshot_atomic(snapshot, self._state_path)
            self._last_write_time = now
        except Exception as e:
            self._logger.warning("Failed to write state snapshot: %s", e)
