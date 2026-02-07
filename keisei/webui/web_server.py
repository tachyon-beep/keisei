"""
web_server.py: HTTP server for serving WebUI static files.
"""

import os
import socket
import threading
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import logging
import atexit


class WebUIHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Custom HTTP request handler for WebUI static files."""
    
    def __init__(self, *args, static_dir=None, **kwargs):
        self.static_dir = static_dir
        super().__init__(*args, **kwargs)
    
    def translate_path(self, path):
        """Translate URL path to static file path."""
        if self.static_dir:
            # Strip query string and fragment
            path = path.split('?', 1)[0].split('#', 1)[0]
            # Remove leading slash and join with static directory
            path = path.lstrip('/')
            if not path:  # Root path
                path = 'index.html'
            full_path = os.path.normpath(os.path.join(self.static_dir, path))
            # Prevent directory traversal — resolved path must stay within static_dir
            if not full_path.startswith(os.path.normpath(self.static_dir) + os.sep) and full_path != os.path.normpath(self.static_dir):
                return os.path.join(self.static_dir, 'index.html')
            return full_path
        return super().translate_path(path)
    
    def log_message(self, format, *args):
        """Override to use Python logging instead of stderr."""
        logging.getLogger(__name__).info(format % args)


class WebUIHTTPServer:
    """HTTP server for serving WebUI static files."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8766):
        self.host = host
        self.port = port
        self.server = None
        self.thread = None
        self._running = False
        self._startup_event = threading.Event()
        self._startup_success = False
        self._logger = logging.getLogger(__name__)

        # Find static directory
        self.static_dir = Path(__file__).parent / "static"
        if not self.static_dir.exists():
            self._logger.error(f"Static directory not found: {self.static_dir}")

        # Register cleanup handler
        atexit.register(self.stop)

    def start(self, timeout: float = 5.0):
        """Start the HTTP server in a background thread.

        Blocks until the server socket is bound or startup fails.

        Returns:
            True if the server started and is listening, False otherwise.
        """
        if self._running:
            return True

        if not self.static_dir.exists():
            self._logger.error("Cannot start HTTP server: static directory not found")
            return False

        self._startup_event.clear()
        self._startup_success = False
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()

        self._startup_event.wait(timeout=timeout)
        if not self._startup_success:
            self._logger.error(f"WebUI HTTP server failed to start on {self.host}:{self.port}")
            self._running = False
            return False

        self._logger.info(f"WebUI HTTP server listening on http://{self.host}:{self.port}")
        return True
    
    def stop(self):
        """Stop the HTTP server."""
        self._running = False
        # Don't call server.shutdown() — it blocks waiting for serve_forever()
        # which we don't use. Setting _running=False causes the handle_request()
        # loop to exit, and server_close() is called in _run_server()'s finally block.

        # Wait for thread to finish (handle_request timeout ensures prompt exit)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)
            if not self.thread.is_alive():
                self._logger.info(f"WebUI HTTP server stopped on port {self.port}")
    
    def _run_server(self):
        """Run the HTTP server."""
        try:
            # Create handler with static directory
            def handler(*args, **kwargs):
                return WebUIHTTPRequestHandler(*args, static_dir=str(self.static_dir), **kwargs)

            self.server = HTTPServer((self.host, self.port), handler)
            self.server.timeout = 0.5  # Short timeout so _running flag is checked promptly

            # Enable socket reuse to prevent "Address already in use" errors
            self.server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # Signal successful startup now that the socket is bound
            self._running = True
            self._startup_success = True
            self._startup_event.set()

            while self._running:
                self.server.handle_request()

        except OSError as e:
            if e.errno == 98:  # Address already in use
                self._logger.error(f"Port {self.port} already in use. Try a different port or kill the process using it.")
            else:
                self._logger.error(f"HTTP server socket error: {e}")
            self._startup_success = False
            self._startup_event.set()
        except Exception as e:
            self._logger.error(f"HTTP server error: {e}")
            self._startup_success = False
            self._startup_event.set()
        finally:
            self._running = False
            if self.server:
                try:
                    self.server.server_close()
                except Exception:
                    pass