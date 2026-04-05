## Summary

Host allowlist validation in `/home/john/keisei/keisei/server/app.py` mis-parses IPv6 `Host` headers, causing valid IPv6 requests/WebSocket connections to be rejected with 403/1008.

## Severity

- Severity: major
- Priority: P1

## Evidence

- In [`/home/john/keisei/keisei/server/app.py:118`](\/home/john/keisei/keisei/server/app.py:118), HTTP host parsing is `hostname = host.split(":")[0]`.
- In [`/home/john/keisei/keisei/server/app.py:157`](\/home/john/keisei/keisei/server/app.py:157), WebSocket host parsing uses the same logic.
- For RFC-compliant IPv6 host headers like `[::1]:8741`, `split(":")[0]` yields `"["`, not `"::1"` or `"[::1]"`, so membership check at [`/home/john/keisei/keisei/server/app.py:119`](\/home/john/keisei/keisei/server/app.py:119) / [`/home/john/keisei/keisei/server/app.py:158`](\/home/john/keisei/keisei/server/app.py:158) always fails.

## Root Cause Hypothesis

The code assumes `Host` is always `name:port` with a single colon delimiter, which is false for bracketed IPv6 literals containing multiple colons. Any IPv6-based access path triggers false rejection.

## Suggested Fix

Replace ad-hoc `split(":")[0]` parsing with bracket-aware host extraction (or Starlette/FastAPI URL parsing utilities), then normalize before allowlist comparison. For example, handle:

- `[::1]:8741` -> `::1` (or `[::1]`, consistently with allowlist format)
- `localhost:8741` -> `localhost`
- `localhost` -> `localhost`

Also apply the same parser in both `HostFilterMiddleware.dispatch` and `_check_ws_host` to keep HTTP/WS behavior consistent.
