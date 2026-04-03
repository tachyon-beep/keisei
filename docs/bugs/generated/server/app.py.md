## Summary

`/ws` connections are not protected by the configured host allowlist, so untrusted hosts can still open the WebSocket stream.

## Severity

- Severity: minor
- Priority: P2 (downgraded: minimal evidence)
## Evidence

- [app.py](/home/john/keisei/keisei/server/app.py):106 defines `HostFilterMiddleware` as `BaseHTTPMiddleware`.
- [app.py](/home/john/keisei/keisei/server/app.py):132 installs that middleware globally, and [app.py](/home/john/keisei/keisei/server/app.py):144 exposes `@app.websocket("/ws")`.
- Starlette’s `BaseHTTPMiddleware` explicitly bypasses non-HTTP scopes: [base.py](/home/john/keisei/.venv/lib/python3.13/site-packages/starlette/middleware/base.py):101-104 checks `if scope["type"] != "http": await self.app(...)`.
- Therefore the host check at [app.py](/home/john/keisei/keisei/server/app.py):113-120 is never applied to WebSocket handshakes.

## Root Cause Hypothesis

Host filtering was implemented with an HTTP-only middleware class; WebSocket connections use ASGI `scope["type"] == "websocket"`, so they skip the allowlist logic and go straight to `websocket.accept()`.

## Suggested Fix

Replace `HostFilterMiddleware(BaseHTTPMiddleware)` with ASGI middleware that validates `Host` for both `"http"` and `"websocket"` scopes (or add an explicit host check in `ws_endpoint` before `accept()` and close with policy-violation if disallowed). Keep one shared allowlist check helper to avoid drift between HTTP and WebSocket paths.
