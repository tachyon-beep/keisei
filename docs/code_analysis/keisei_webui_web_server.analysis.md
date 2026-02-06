# Code Analysis: `keisei/webui/web_server.py`

## 1. Purpose & Role

This module provides an HTTP server (`WebUIHTTPServer`) for serving static frontend files (HTML, CSS, JS, SVG images) that compose the WebUI dashboard. It runs in a background daemon thread and serves files from the `keisei/webui/static/` directory. It is the companion to the WebSocket server in `webui_manager.py`, providing the HTTP endpoint for the browser to load the dashboard page.

## 2. Interface Contracts

### `WebUIHTTPRequestHandler` (lines 15-34)
- **Inherits**: `http.server.SimpleHTTPRequestHandler`
- **Constructor**: Accepts `static_dir` keyword argument to override the file serving root.
- **`translate_path(path)`**: Maps URL paths to filesystem paths within `static_dir`. Root path `/` maps to `index.html`.
- **`log_message(format, *args)`**: Redirects HTTP access logs to Python's `logging` module instead of stderr.

### `WebUIHTTPServer` (lines 37-117)
- **Constructor**: `__init__(host="0.0.0.0", port=8766)` -- binds to all interfaces by default on port 8766.
- **`start() -> bool`**: Starts the HTTP server in a background daemon thread. Returns `True` if started or already running, `False` if static directory is missing.
- **`stop()`**: Shuts down the server and joins the background thread with a 2-second timeout.
- **`_run_server()`**: Internal method that creates the `HTTPServer` and enters the request-handling loop.

## 3. Correctness Analysis

### Path Traversal Vulnerability (line 29)
The `translate_path` method strips the leading `/` and joins directly with `static_dir` using `os.path.join`. However, it does **not** sanitize path components such as `..`. A request for `/../../../etc/passwd` would have the leading `/` stripped to become `../../../etc/passwd`, and `os.path.join(static_dir, "../../../etc/passwd")` would resolve outside the static directory. While `SimpleHTTPRequestHandler.translate_path` in the parent class does include path normalization and CWD-rooting, this override **bypasses that logic entirely** when `static_dir` is set (which is always the case in practice). This is a directory traversal vulnerability.

### `SO_REUSEADDR` Set After Bind (lines 94-97)
The `HTTPServer` constructor on line 94 calls `server_bind()` and `server_activate()` internally, which binds the socket and starts listening. The `SO_REUSEADDR` option is set on line 97, **after** the socket is already bound. This means the option has no effect on preventing "Address already in use" errors. The comment on line 96 is misleading.

### `handle_request()` Loop Without Timeout (lines 101-102)
The server loop uses `self.server.handle_request()` which blocks until a request arrives or the default socket timeout expires. The default timeout for `HTTPServer` is `None` (blocking indefinitely). When `self._running` is set to `False`, the loop will not exit until the next HTTP request arrives, causing a potential hang on shutdown. The `stop()` method calls `server.shutdown()` (line 77), but `shutdown()` is designed for `serve_forever()`, not for a manual `handle_request()` loop. The `shutdown()` call sets an internal threading event that `_BaseServer.serve_forever()` checks, but `handle_request()` does not check this event.

### `_running` Flag Not Thread-Safe (line 45, 65, 74, 101)
The `_running` boolean is read and written from multiple threads (main thread sets it in `stop()`, background thread reads it in `_run_server()`) without any synchronization primitive (no `threading.Event`, no lock). On CPython this is practically safe due to the GIL, but it is technically a data race.

## 4. Robustness & Error Handling

- **Static directory missing**: Handled correctly -- `start()` returns `False` and logs an error (lines 50-51, 61-63).
- **Port in use**: The `OSError` with errno 98 is caught and logged with a helpful message (lines 104-106). However, due to the `SO_REUSEADDR` ordering issue noted above, this error will still occur in practice when restarting quickly.
- **Server close in finally block**: The `_run_server` method has a `finally` block (lines 111-117) that calls `server_close()`, ensuring socket cleanup even on unexpected errors.
- **Thread join timeout**: The `stop()` method joins the background thread with a 2-second timeout (line 85), preventing indefinite blocking. However, if the thread is stuck in `handle_request()`, it will not terminate within 2 seconds, and the thread will leak (it is a daemon thread, so it will not prevent process exit).
- **atexit handler**: An atexit handler is registered in the constructor (line 54) to call `stop()`, providing cleanup on interpreter shutdown.

## 5. Performance & Scalability

- The server uses `handle_request()` in a loop rather than `serve_forever()`, which means it handles one request at a time and cannot overlap request processing with the loop condition check. This is adequate for a low-traffic development/streaming dashboard but would not scale to many concurrent users.
- The `SimpleHTTPRequestHandler` reads files synchronously from disk for each request. For the static files involved (HTML, JS, CSS, SVG images), this is acceptable.
- No caching headers are set, so browsers will re-request static assets on each page load.

## 6. Security & Safety

### Critical: Directory Traversal (line 22-29)
As detailed in the correctness section, the `translate_path` override allows path traversal attacks. An attacker with network access to the HTTP port can read arbitrary files readable by the process. The server binds to `0.0.0.0` by default, making this accessible from any network interface.

### Binding to All Interfaces (line 40)
The default host is `0.0.0.0`, which exposes the HTTP server on all network interfaces. This is intentional (for Twitch streaming use cases where remote access is needed), but it increases the attack surface. Combined with the path traversal issue, this is particularly concerning.

### No Authentication
There is no authentication or authorization mechanism. Anyone with network access to the port can access the dashboard and, due to the traversal issue, any file on the system.

### No HTTPS
The server uses plain HTTP with no TLS support. Data transmitted between the browser and server is unencrypted.

## 7. Maintainability

- The code is well-structured with clear separation between the handler class and the server class.
- Logging is properly routed through Python's logging module (line 34).
- The `handler` factory closure on lines 91-92 is a clean pattern for passing `static_dir` to the handler constructor.
- The module is 117 lines and focused on a single responsibility.
- The `signal` import on line 6 is unused.
- The `Path` import on line 9 is used (line 49).

## 8. Verdict

**CRITICAL**

The directory traversal vulnerability in `translate_path` (line 22-29) is a security-critical issue. The override bypasses `SimpleHTTPRequestHandler`'s built-in path sanitization, allowing arbitrary file reads from the filesystem. Combined with the default binding to `0.0.0.0`, this exposes the host to remote file disclosure attacks. Additionally, the `SO_REUSEADDR` is set after the socket is already bound (line 97), rendering it ineffective, and the `handle_request()` loop can cause hangs during shutdown.
