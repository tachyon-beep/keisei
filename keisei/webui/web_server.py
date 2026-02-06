"""
web_server.py: HTTP server for serving WebUI static files.
"""

import os
import signal
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
            # Remove leading slash and join with static directory
            path = path.lstrip('/')
            if not path:  # Root path
                path = 'index.html'
            return os.path.join(self.static_dir, path)
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
        self._logger = logging.getLogger(__name__)
        
        # Find static directory
        self.static_dir = Path(__file__).parent / "static"
        if not self.static_dir.exists():
            self._logger.error(f"Static directory not found: {self.static_dir}")
        
        # Register cleanup handler
        atexit.register(self.stop)
    
    def start(self):
        """Start the HTTP server in a background thread."""
        if self._running:
            return True
            
        if not self.static_dir.exists():
            self._logger.error("Cannot start HTTP server: static directory not found")
            return False
            
        self._running = True
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        
        self._logger.info(f"WebUI HTTP server starting on http://{self.host}:{self.port}")
        return True
    
    def stop(self):
        """Stop the HTTP server."""
        self._running = False
        if self.server:
            try:
                self.server.shutdown()
                self.server.server_close()
                self._logger.info(f"WebUI HTTP server stopped on port {self.port}")
            except Exception as e:
                self._logger.error(f"Error stopping HTTP server: {e}")
        
        # Wait for thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
    
    def _run_server(self):
        """Run the HTTP server."""
        try:
            # Create handler with static directory
            def handler(*args, **kwargs):
                return WebUIHTTPRequestHandler(*args, static_dir=str(self.static_dir), **kwargs)
            
            self.server = HTTPServer((self.host, self.port), handler)
            
            # Enable socket reuse to prevent "Address already in use" errors
            self.server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            self._logger.info(f"WebUI HTTP server started on http://{self.host}:{self.port}")
            
            while self._running:
                self.server.handle_request()
                
        except OSError as e:
            if e.errno == 98:  # Address already in use
                self._logger.error(f"Port {self.port} already in use. Try a different port or kill the process using it.")
            else:
                self._logger.error(f"HTTP server socket error: {e}")
        except Exception as e:
            self._logger.error(f"HTTP server error: {e}")
        finally:
            self._running = False
            if self.server:
                try:
                    self.server.server_close()
                except Exception:
                    pass