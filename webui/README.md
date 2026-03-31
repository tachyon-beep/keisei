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
