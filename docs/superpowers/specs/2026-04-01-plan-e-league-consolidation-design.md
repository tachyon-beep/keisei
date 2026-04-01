# Plan E: Pipeline Consolidation, Opponent League & Demonstrator Games

## Purpose

Unify the two training pipelines (old scalar-value and new KataGo multi-head) into a single loop, add a two-tier opponent league with cross-architecture support and rotating training seat, and provide demonstrator games for web UI spectators. This plan assumes Plans A-D are complete.

## Context

### Problem

The current training pipeline (`TrainingLoop` + `PPOAlgorithm`) collapsed at epoch 220 of a 273-epoch run due to self-play entropy collapse. The agent found a degenerate strategy exploiting the fact that both sides share the same policy. Meanwhile, Plans A-D build a parallel KataGo-inspired pipeline (`KataGoTrainingLoop` + `KataGoPPOAlgorithm` + `SEResNetModel`) that would suffer the same collapse.

### Goals

1. **Consolidate** the two pipelines into one unified loop that supports both model contracts (scalar value and multi-head W/D/L+score).
2. **Prevent entropy collapse** by introducing opponent diversity via a league of past checkpoints.
3. **Enable cross-architecture comparison** — different model architectures (ResNet, SE-ResNet, Transformer, MLP) can play against each other and be ranked by Elo.
4. **Provide a demonstrator experience** — a live training game plus frozen-vs-frozen exhibition games for the web UI.
5. **Design for extension** — the two-tier league (historical snapshots + current best) is the foundation for a full Elo-weighted league with exploiter agents in the future.

### Non-Goals

- Full population-based training (concurrent co-training of multiple models). The rotating seat gives league movement at single-model GPU cost.
- Exploiter agents (tier 3 of the league). Designed for but not implemented.
- Web UI changes. The existing spectator UI already shows game snapshots from `keisei.db`; this plan adds data it can consume but does not modify the UI itself.

---

## Architecture

### Dual-Contract Model System

Two abstract base classes coexist, representing different value head designs:

**`BaseModel`** (scalar value) — existing contract used by ResNet, MLP, Transformer:
- Input: `(batch, obs_channels, 9, 9)`
- Output: `(policy_logits: (batch, action_space), value: (batch, 1))`

**`KataGoBaseModel`** (multi-head) — new contract used by SE-ResNet:
- Input: `(batch, obs_channels, 9, 9)`
- Output: `KataGoOutput(policy_logits, value_logits: (batch, 3), score_lead: (batch, 1))`

The model registry stores which contract each architecture implements. The unified training loop detects the contract at init:

```python
self.multi_head = isinstance(unwrap_model(self.model), KataGoBaseModel)
```

Loss computation dispatches on this flag:
- Scalar: MSE loss on tanh-activated value vs returns (existing behavior).
- Multi-head: cross-entropy on W/D/L classification + MSE on score lead (KataGo behavior).

Action selection is identical for both contracts — only `policy_logits` is needed, and both contracts provide it.

### Pipeline Consolidation

The `KataGoTrainingLoop` (from Plan C) becomes the sole training loop. The `KataGoPPOAlgorithm` becomes the sole PPO implementation. Old classes are removed.

#### What Gets Deleted

| Component | File | Reason |
|-----------|------|--------|
| `TrainingLoop` | `keisei/training/loop.py` | Replaced by unified loop |
| `PPOAlgorithm` | `keisei/training/ppo.py` | Replaced by `KataGoPPOAlgorithm` |
| `RolloutBuffer` | `keisei/training/ppo.py` | Replaced by `KataGoRolloutBuffer` |
| `PPOParams` | `keisei/training/algorithm_registry.py` | Replaced by `KataGoPPOParams` |

#### What Gets Kept

| Component | Action | Detail |
|-----------|--------|--------|
| `compute_gae()` | Move to `keisei/training/gae.py` | Shared utility, extracted from deleted `ppo.py` |
| `BaseModel` | Keep | Scalar-value contract for old architectures |
| `KataGoBaseModel` | Keep | Multi-head contract for SE-ResNet |
| `ResNetModel` | Keep | Registered in model registry under `BaseModel` contract |
| `MLPModel` | Keep | Same |
| `TransformerModel` | Keep | Same |
| `SEResNetModel` | Keep | Registered under `KataGoBaseModel` contract |

#### What Gets Modified

| Component | Change |
|-----------|--------|
| `model_registry.py` | Stores contract type (`"scalar"` or `"multi_head"`) per architecture |
| `algorithm_registry.py` | Only `KataGoPPOParams` (can be renamed to `PPOParams`) |
| `keisei-train` entrypoint | Rewired to unified loop |
| `config.py` | New `[league]` and `[demonstrator]` sections |
| `db.py` | New tables for league tracking |

### Opponent League

#### Data Model

```
OpponentEntry:
    id: int
    architecture: str              # e.g., "resnet", "se_resnet"
    model_params: dict             # constructor params for model_registry.build_model()
    checkpoint_path: Path          # path to saved weights
    elo_rating: float              # current Elo (initial 1000)
    created_epoch: int             # training epoch when snapshot was taken
    games_played: int              # total games as opponent
    created_at: datetime
```

#### OpponentPool

Manages the collection of checkpoint snapshots available as opponents.

- **Snapshot trigger:** every `snapshot_interval` epochs (default 10), the current learner's weights are saved as a new `OpponentEntry` in the pool.
- **Rolling window:** keeps the most recent `max_pool_size` entries (default 20). When the pool exceeds this size, the oldest entry is evicted (checkpoint file deleted).
- **Storage:** league checkpoints live in `checkpoints/league/` to avoid confusion with training checkpoints in `checkpoints/`.
- **Cross-architecture loading:** each entry stores its architecture name and model params. To instantiate an opponent, the pool calls `model_registry.build_model(entry.architecture, entry.model_params)` and loads the checkpoint weights. This means a `ResNetModel` checkpoint can be loaded as an opponent even when the current learner is an `SEResNetModel`.
- **Persistence:** pool metadata is stored in the `league_entries` table in `keisei.db`, not in memory. The pool reconstructs its state from the DB on restart.

#### OpponentSampler

Selects which opponent the learner faces each epoch. Two-tier sampling:

| Tier | Source | Default Ratio | Description |
|------|--------|---------------|-------------|
| Historical | Random entry from pool | 80% | Prevents degenerate equilibria by exposing the learner to diverse past strategies |
| Current best | Most recent snapshot | 20% | Ensures the learner faces a strong sparring partner |

Ratios are configurable via TOML (`historical_ratio`, `current_best_ratio`). The sampler returns an `OpponentEntry`; the loop loads the corresponding model once per epoch.

The `OpponentSampler` is an abstract interface so that future implementations (Elo-weighted, exploiter-aware) can drop in without changing the loop.

#### Split-Merge Step Logic

Within the inner training loop, each step processes all 128 environments simultaneously. The environments have a mix of "whose turn is it" (Black or White). One side is the learner, the other is the opponent.

Per step:

1. Read `current_players` array to determine which envs have the learner's turn vs the opponent's turn.
2. **Learner subset:** forward-pass the training model with gradients enabled. Store transitions in the rollout buffer.
3. **Opponent subset:** forward-pass the frozen opponent model with `torch.no_grad()`. Do NOT store these transitions.
4. Merge the two action arrays into a single action list.
5. Call `vecenv.step(merged_actions)`.

The learner always plays Black (side 0). This simplifies the split logic and avoids sign-flipping reward complications. When the seat rotates to a new model, it still plays Black.

#### Elo Tracking

After each epoch, the system records the match result in the `league_results` table:

```
LeagueResult:
    epoch: int
    learner_id: int               # FK to league_entries
    opponent_id: int              # FK to league_entries
    wins: int
    losses: int
    draws: int
    recorded_at: datetime
```

Elo updates use the standard formula with K=32:

```
expected = 1 / (1 + 10^((opponent_elo - learner_elo) / 400))
new_elo = old_elo + K * (actual - expected)
```

Both the learner's and opponent's Elo are updated after each epoch (the opponent's rating adjusts even though it's frozen — this reflects the pool's evolving difficulty landscape).

### Rotating Training Seat

The league contains N models (each a different architecture or a different training snapshot of the same architecture). One model holds the "training seat" at a time.

**Rotation schedule:**

1. The learner trains for `epochs_per_seat` epochs (default 50).
2. After the rotation, the learner's updated weights are saved back to its `OpponentEntry` in the pool.
3. The next model in rotation order is loaded from its checkpoint into the training seat with a fresh optimizer.
4. All other models serve as frozen opponents during that rotation.

**Rotation order:** round-robin through registered architectures. Future extension: Elo-weighted (lowest-rated gets more training time).

**Optimizer reset:** each rotation creates a fresh Adam optimizer for the new learner. The SL optimizer's momentum from the previous model would fight the new model's gradient signal. The elevated entropy warmup (Plan D Task 3) applies for the first N epochs of each seat to compensate.

### Live Game + Demonstrator Games

#### Live Game (1)

The actual training loop, running at full speed with `num_games` envs (e.g., 128). This is the rotating-seat learner vs league opponents. The web UI shows one representative game from this batch, identified by `game_type = "live"` in the game snapshots table.

#### Demonstrator Games (up to 3)

Inference-only, frozen vs frozen. Run in a separate thread. Matchups are auto-selected:

| Slot | Matchup Rule | Purpose |
|------|-------------|---------|
| Demo 1 | #1 Elo vs #2 Elo | The championship match |
| Demo 2 | Cross-architecture (different architecture types) | Shows architectural differences |
| Demo 3 | Random pairing from pool | Variety |

When a demonstrator game ends, a new matchup is selected from the current pool state and a fresh game starts.

Demonstrator games write to `keisei.db` game snapshots with `game_type = "demo"` and a `demo_slot` field (1, 2, or 3) so the web UI can display them in fixed positions.

**Implementation:** a `DemonstratorRunner` class manages the demonstrator games. It holds up to 3 `VecEnv(num_envs=1)` instances (one per demo slot) and two frozen models per slot. It runs in its own thread, polling at a configurable rate (default: one move per second for watchability). The runner checks the pool for new entries periodically and refreshes matchups when the pool changes.

### Evaluation Entrypoint

`keisei-evaluate` CLI for head-to-head evaluation outside of training:

```
keisei-evaluate \
    --checkpoint-a checkpoints/league/resnet_ep100.pt \
    --arch-a resnet \
    --checkpoint-b checkpoints/league/se_resnet_ep200.pt \
    --arch-b se_resnet \
    --games 100 \
    --max-ply 500
```

Output: win/loss/draw counts, win rate with 95% confidence interval, Elo delta.

This reuses the same game-playing loop as the demonstrator (inference-only, two frozen models) but runs synchronously for a fixed number of games and reports results to stdout.

---

## Configuration

### TOML Additions

```toml
[league]
max_pool_size = 20           # maximum entries in the opponent pool
snapshot_interval = 10       # epochs between pool snapshots
epochs_per_seat = 50         # epochs before rotating the training seat
historical_ratio = 0.8       # fraction of epochs vs historical opponents
current_best_ratio = 0.2     # fraction of epochs vs current best
initial_elo = 1000           # starting Elo for new entries
elo_k_factor = 32            # K-factor for Elo updates

[demonstrator]
num_games = 3                # concurrent demonstrator games (0 to disable)
auto_matchup = true          # auto-select pairings based on Elo/architecture
moves_per_minute = 60        # playback speed for demonstrator games
```

### DB Schema Additions

```sql
-- League opponent pool
CREATE TABLE league_entries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    architecture    TEXT NOT NULL,
    model_params    TEXT NOT NULL,          -- JSON
    checkpoint_path TEXT NOT NULL,
    elo_rating      REAL NOT NULL DEFAULT 1000.0,
    created_epoch   INTEGER NOT NULL,
    games_played    INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- Match results for Elo tracking
CREATE TABLE league_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    epoch           INTEGER NOT NULL,
    learner_id      INTEGER NOT NULL REFERENCES league_entries(id),
    opponent_id     INTEGER NOT NULL REFERENCES league_entries(id),
    wins            INTEGER NOT NULL,
    losses          INTEGER NOT NULL,
    draws           INTEGER NOT NULL,
    recorded_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX idx_league_results_epoch ON league_results(epoch);
CREATE INDEX idx_league_entries_elo ON league_entries(elo_rating);
```

The existing `game_snapshots` table gains a `game_type` column (`"live"` or `"demo"`, default `"live"`) and a `demo_slot` column (`INTEGER`, nullable).

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `keisei/training/league.py` | `OpponentPool`, `OpponentEntry`, `OpponentSampler`, Elo calculations |
| Create | `keisei/training/demonstrator.py` | `DemonstratorRunner` — inference-only exhibition matches |
| Create | `keisei/training/evaluate.py` | `keisei-evaluate` CLI entrypoint |
| Create | `keisei/training/gae.py` | `compute_gae()` extracted from deleted `ppo.py` |
| Modify | `keisei/training/katago_loop.py` | Split-merge step logic, league integration, seat rotation, dual-contract dispatch |
| Modify | `keisei/training/katago_ppo.py` | Import `compute_gae` from new location |
| Modify | `keisei/training/model_registry.py` | Add contract type per architecture, cross-architecture loading helper |
| Modify | `keisei/training/algorithm_registry.py` | Remove old `PPOParams`, keep only `KataGoPPOParams` |
| Modify | `keisei/config.py` | `[league]` and `[demonstrator]` config sections, `LeagueConfig`, `DemonstratorConfig` |
| Modify | `keisei/db.py` | `league_entries`, `league_results` tables, `game_type`/`demo_slot` columns on `game_snapshots` |
| Modify | `pyproject.toml` | Add `keisei-evaluate` entrypoint, rewire `keisei-train` |
| Delete | `keisei/training/loop.py` | Old training loop (replaced by unified `katago_loop.py`) |
| Delete | `keisei/training/ppo.py` | Old PPO (replaced; `compute_gae` extracted to `gae.py`) |
| Create | `tests/test_league.py` | OpponentPool, OpponentSampler, Elo calculation tests |
| Create | `tests/test_demonstrator.py` | DemonstratorRunner tests |
| Create | `tests/test_evaluate.py` | Evaluation CLI tests |
| Create | `tests/test_pipeline_consolidation.py` | Both model contracts work in unified loop |

---

## Dependencies

- **Requires Plans A-D complete.** Plan E modifies `katago_loop.py`, `katago_ppo.py`, and `model_registry.py` as they exist after Plans A-D.
- **No external dependencies.** Elo math is ~20 lines; no need for a library.
- **No Rust changes.** The `VecEnv` API is sufficient — demonstrator games just use `VecEnv(num_envs=1)`.

## Future Extensions (Not in Plan E)

- **Elo-weighted sampling:** replace uniform historical sampling with Elo-proximity weighting.
- **Exploiter agents (tier 3):** a model trained to exploit the main agent's weaknesses, added as a new `OpponentSampler` strategy.
- **Concurrent co-training:** multiple models train simultaneously (population-based training).
- **Web UI league dashboard:** Elo leaderboard, rating-over-time charts, matchup history. The DB schema supports this; the UI work is separate.
