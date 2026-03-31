# Plan 8: Build Pipeline + Integration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the Vite build into the FastAPI static serving, verify the full end-to-end flow (training → SQLite → server → browser), and add `.gitignore` entries.

**Architecture:** `npm run build` outputs to `keisei/server/static/`. FastAPI serves those files at `/`. The dev workflow uses Vite's proxy to forward `/ws` and `/healthz` to the FastAPI server running on port 8000.

**Tech Stack:** Vite, FastAPI, npm scripts

---

### Task 1: Build and Serve

**Files:**
- Modify: `webui/package.json` (add build script if not present)
- Create: `webui/.gitignore`
- Modify: `.gitignore` (add `keisei/server/static/` — build output)

- [ ] **Step 1: Add gitignore entries**

`webui/.gitignore`:
```
node_modules/
dist/
```

Append to root `.gitignore`:
```
# Svelte build output (served by FastAPI, rebuilt from webui/)
keisei/server/static/
```

- [ ] **Step 2: Build the Svelte app**

```bash
cd webui && npm run build
```

Expected: `keisei/server/static/` contains `index.html` and `assets/` directory.

- [ ] **Step 3: Verify FastAPI serves the built SPA**

```bash
# In one terminal: start training to populate the DB
uv run keisei-train --config keisei.toml --epochs 3 --steps-per-epoch 32

# In another terminal: start the server
uv run keisei-serve --config keisei.toml --port 8000

# Open http://localhost:8000 in browser
# Should see the dashboard with live data
```

- [ ] **Step 4: Commit**

```bash
git add webui/.gitignore .gitignore
git commit -m "chore: gitignore for webui node_modules and build output"
```

---

### Task 2: End-to-End Integration Test

**Files:**
- Modify: `tests/test_server.py`

- [ ] **Step 1: Add static file serving test**

Append to `tests/test_server.py`:
```python
@pytest.mark.asyncio
async def test_serves_index_html(db_path: str, tmp_path: Path) -> None:
    """If static/ dir exists with index.html, GET / returns it."""
    import os
    static_dir = Path(__file__).parent.parent / "keisei" / "server" / "static"
    if not static_dir.is_dir():
        pytest.skip("No built SPA in keisei/server/static/")

    app = create_app(db_path)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    assert "html" in resp.headers.get("content-type", "").lower()
```

- [ ] **Step 2: Add development workflow documentation**

Create `webui/README.md`:
```markdown
# Keisei Spectator Dashboard

Svelte SPA for watching Shogi training in real time.

## Development

```bash
# Terminal 1: Start training (populates SQLite)
uv run keisei-train --config keisei.toml --epochs 100 --steps-per-epoch 64

# Terminal 2: Start API server
uv run keisei-serve --config keisei.toml

# Terminal 3: Start Svelte dev server (proxies /ws to FastAPI)
cd webui && npm run dev
# Open http://localhost:5173
```

## Production Build

```bash
cd webui && npm run build
# Output: keisei/server/static/
# Then just run: uv run keisei-serve --config keisei.toml
# Open http://localhost:8000
```
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_server.py webui/README.md
git commit -m "feat: end-to-end integration test and dev workflow docs"
```

---

### Task 3: Final Verification

- [ ] **Step 1: Run full Python test suite**

```bash
uv run pytest -v --ignore=shogi-engine
```

Expected: All tests pass (50+ from Plan 1 + server tests from this plan).

- [ ] **Step 2: Run ruff lint on all Python**

```bash
uv run ruff check keisei/ tests/
```

Expected: Clean.

- [ ] **Step 3: Verify both entry points**

```bash
uv run keisei-train --help
uv run keisei-serve --help
```

Expected: Both show argparse help.

- [ ] **Step 4: Final commit if any fixes needed**

```bash
git add -A
git commit -m "chore: final integration verification for spectator webui"
```
