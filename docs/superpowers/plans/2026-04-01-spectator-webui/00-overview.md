# Spectator WebUI — Implementation Plans

**Goal:** Build a read-only spectator dashboard that connects to the training harness's SQLite database and renders live game boards + training metrics in the browser.

**Architecture:** FastAPI backend (Python) polls SQLite and pushes updates via WebSocket. Svelte SPA (JS) renders the dashboard. The build output goes into `keisei/server/static/` and is served by FastAPI.

**Spec:** `docs/superpowers/specs/2026-04-01-training-harness-webui-design.md` (sections: FastAPI Server, Svelte Dashboard)

**Depends on:** Training harness (Plan 1) — specifically `keisei/db.py` for read functions and `keisei/config.py` for shared config loading.

---

## Plan Index

| # | Plan | What it builds | Depends on |
|---|------|---------------|------------|
| 1 | [FastAPI Server](./01-fastapi-server.md) | `keisei/server/app.py` — HTTP + WebSocket server | db.py, config.py |
| 2 | [Svelte Scaffolding](./02-svelte-scaffold.md) | `webui/` project with Vite + Svelte + uPlot | None |
| 3 | [WebSocket Client + Stores](./03-ws-stores.md) | `ws.js`, `games.js`, `metrics.js` Svelte stores | Plans 1, 2 |
| 4 | [Board + Piece Trays](./04-board-pieces.md) | `Board.svelte`, `PieceTray.svelte` components | Plan 3 |
| 5 | [Game Thumbnails + Move Log](./05-thumbnails-movelog.md) | `GameThumbnail.svelte`, `MoveLog.svelte`, selection logic | Plan 4 |
| 6 | [Training Metrics Charts](./06-metrics-charts.md) | `MetricsChart.svelte` with uPlot, 2x2 grid | Plan 3 |
| 7 | [App Shell + Status Indicator](./07-app-shell.md) | `App.svelte` layout, header, training status | Plans 5, 6 |
| 8 | [Build Pipeline + Integration](./08-build-integration.md) | Vite build → `keisei/server/static/`, end-to-end test | Plan 7 |

Plans 4 and 6 are independent (board vs charts) and can run in parallel after Plan 3.
