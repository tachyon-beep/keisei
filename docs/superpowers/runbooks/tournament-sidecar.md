# Tournament Sidecar Runbook

## What is it?

The tournament sidecar extracts the league round-robin tournament from the training process into a supervised subprocess. Training and tournament run as separate processes with their own CUDA contexts, communicating only through SQLite WAL.

- **Training** (cuda:0): rollout + PPO update + dispatcher (enqueues pairings)
- **Tournament worker** (cuda:1): claims pairings from the queue, plays matches, writes results

## Configuration

Set `tournament_mode = "sidecar"` in your config's `[league]` section:

```toml
[league]
tournament_enabled = true
tournament_mode = "sidecar"      # "in_process" (legacy) or "sidecar"
tournament_device = "cuda:1"
dispatcher_max_queue_depth = 400  # adaptive dispatch cap
max_staleness_epochs = 50         # expire pairings older than this
```

`run.sh` reads these fields and launches the worker automatically. No separate command needed.

## Launching

```bash
./run.sh keisei-500k-league.toml
```

When `tournament_mode = "sidecar"`, `run.sh` starts four processes:
1. Training (cuda:0)
2. Dashboard (web server)
3. Showcase sidecar (CPU)
4. **Tournament worker** (cuda:1)

Opt out with `--no-tournament`:
```bash
./run.sh keisei-500k-league.toml --no-tournament
```

## Verifying health

```bash
# Worker heartbeat (should update every ~10s)
sqlite3 data/keisei-500k-league.db "SELECT * FROM tournament_worker_heartbeat;"

# Queue throughput
sqlite3 data/keisei-500k-league.db \
  "SELECT status, COUNT(*) FROM tournament_pairing_queue GROUP BY status;"

# Recent match results
sqlite3 data/keisei-500k-league.db \
  "SELECT * FROM league_results ORDER BY id DESC LIMIT 5;"
```

## Diagnosing problems

**Queue depth growing:**
Check `get_worker_health`. If no alive workers, check the tournament log file for crash tracebacks.

**Worker crashed:**
`run.sh`'s monitor loop auto-restarts it within 30s. Check tournament log for the restart entry.

**Pairings stuck in 'playing':**
Worker startup sweep resets its own stale pairings. If the worker crashed and a different worker_id restarts, run:
```sql
UPDATE tournament_pairing_queue 
SET status='pending', worker_id=NULL 
WHERE status='playing';
```

**DynamicTrainer single-writer conflict:**
Check `dynamic_update_worker` column:
```sql
SELECT id, display_name, dynamic_update_worker 
FROM league_entries WHERE dynamic_update_worker IS NOT NULL;
```
Stale values clear on the next successful claim cycle.

## Scaling to N>1 workers

Not yet implemented in `run.sh`. For manual scaling:
```bash
uv run python -m keisei.training.tournament_runner \
    --db-path data/keisei-500k-league.db \
    --league-dir checkpoints/500k-league/ \
    --worker-id worker-1 \
    --device cuda:2
```

## Pre-merge test gate

Before merging changes to `tournament_runner.py` or `tournament_queue.py`:
```bash
uv run pytest -m integration
```
