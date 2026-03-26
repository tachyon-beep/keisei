# Changelog

## 1.0.0rc2 (2026-03-27)

### Runtime Safety Fixes
- **Checkpoint load**: `load_model()` raises `FileNotFoundError` instead of returning ambiguous error dict
- **PPO grad mode**: `select_action()` wraps both training and eval paths in `torch.no_grad()` — rollout collection doesn't need gradients, saves GPU memory
- **Async entrypoint**: Safe coroutine cleanup prevents "coroutine never awaited" warnings on early `asyncio.run()` failure
- **Evaluation mode restore**: All evaluation paths (`sync`, `async`, `in_memory`) use `try/finally` to restore `model.train()` after `model.eval()`
- **Event loop safety**: `ParallelGameExecutor` uses thread pool instead of `run_coroutine_threadsafe` when called from loop-owning thread
- **Process liveness**: `StreamlitManager` detects crashed dashboard subprocess; state file writes continue so a restarted dashboard picks up fresh data
- **Non-writable arrays**: `ModelSynchronizer` and `self_play_worker` copy non-writable NumPy arrays before `torch.from_numpy()`
- **Queue portability**: `WorkerCommunicator` handles unsupported `qsize()`/`empty()` on macOS
- **Callback guards**: Checkpoint and evaluation callbacks skip timestep 0 and deduplicate on resume boundaries via `_last_fired_timestep`
- **Optimizer init**: Removed silent lr=1e-3 fallback — bad config now fails fast

### Streamlit Dashboard v2
- **Fragment-based refresh**: Replaced `time.sleep(2) + st.rerun()` with `@st.fragment(run_every=2s)` — users can interact without interruption
- **Tab navigation**: `st.tabs(["Metrics", "Game", "Lineage", "Overview"])` inside the fragment; Lineage conditional on data availability
- **Piece differentiation**: Black pieces `#f0d9a0` (warm wheat), white pieces `#b5ab99` (cool grey-tan)
- **Board accessibility**: `role="table"`, `<th scope>` headers, `aria-label` on every cell, `alt` on piece images, numeric row labels (1-9)
- **Policy insight panel**: `torch.no_grad()` forward pass extracts 9x9 action heatmap (log-scaled), top-K actions in USI notation, V(s), entropy-derived confidence
- **Legal move masking**: Heatmap and top-actions reflect only legal moves, matching `PPOAgent.select_action` behavior
- **Board interactivity (v2.1)**: Custom `st.components.v2` board with click-to-inspect, roving tabindex keyboard navigation, focus-pause, selection invalidation
- **Per-square actions**: `square_actions` field in `PolicyInsight` with top-3 actions per destination square
- **Overview tab**: Single-screen dashboard with board, heatmap, charts, and auto-selected hottest square
- **Dark theme**: CSS media query detection and variable adaptation
- **Export state**: Download button inside fragment for fresh state snapshots

### Code Quality
- **XSS hardening**: `html.escape()` on all state-derived text in `unsafe_allow_html` contexts; JS board uses `escHtml()` + `isSafeDataUri()`
- **Exception narrowing**: Replaced bare `except Exception` with specific types across callbacks, evaluation, snapshot extraction, board fallback
- **Exception chaining**: `from e` on all re-raised exceptions in evaluation checkpoint loading
- **Logging**: Silent catches now log at appropriate levels (Elo registry, pre-eval rating, queue drain, policy insight, W&B init with traceback)
- **Type consolidation**: `TopAction`/`SquareAction` merged into single `ActionProb` TypedDict; `PolicyInsight` uses `total=True`
- **Config validators**: `port` (1-65535), `update_rate_hz` (>0), `policy_insight_top_k` (1-100)
- **Named constants**: `_PROB_FLOOR`, `_ENTROPY_EPSILON`, `_SQUARE_DETAIL_THRESHOLD`, `_SQUARE_DETAIL_TOP_N`; `max_entropy` computed via `math.log()`
- **`send_model_weights`**: Returns `bool` so callers detect total failure; `restore_model_from_sync` raises instead of silent `False`

### Dependencies
- Bumped all dependencies to latest stable (numpy 2.4.3, scipy 1.17.1, streamlit 1.55.0, isort 8.0.1, pylint 4.0.5, etc.)
- Minimum version floors updated in `pyproject.toml`

### Housekeeping
- Removed legacy WebUI config fields (`max_connections`, `board_update_rate_hz`, `metrics_update_rate_hz`)
- Removed W&B import-state debug logging
- Removed `CRITICAL FIX:` changeset annotations from `core_manager.py` and `performance_manager.py`
- Migrated issue tracking from beads to filigree

## 1.0.0rc1

Initial release candidate.
