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
- Input: `(batch, obs_channels, 9, 9)` where `obs_channels = 46`
- Output: `(policy_logits: (batch, action_space), value: (batch, 1))`

**`KataGoBaseModel`** (multi-head) — new contract used by SE-ResNet:
- Input: `(batch, obs_channels, 9, 9)` where `obs_channels = 50`
- Output: `KataGoOutput(policy_logits, value_logits: (batch, 3), score_lead: (batch, 1))`

#### Observation Channel Reconciliation

Plan A expands observations from 46 to 50 channels (adding repetition history and KataGo-specific planes). Cross-architecture opponents require all models to accept the same observation tensor from VecEnv. **Resolution: all models accept 50 channels post-Plan-A.** The old `BaseModel` architectures (ResNet, MLP, Transformer) are updated to accept 50 input channels by changing their `input_conv` / first layer. Existing 46-channel checkpoints are loaded with `strict=False` and the new input weights are zero-initialized — the model sees the extra 4 channels as zero-contribution until it trains on them. This is a one-time migration applied when Plan E registers old architectures in the 50-channel VecEnv environment.

The model registry stores `obs_channels` per architecture. `OpponentPool.load_opponent()` asserts `model.obs_channels == vecenv.observation_channels` at load time, converting a silent shape mismatch into a loud startup failure.

#### Value-Head Adapter (avoids dual-contract branching)

Rather than `if self.multi_head:` branches throughout the loop (a God Object anti-pattern), use a **value-head adapter** — a thin interface that encapsulates loss computation:

```python
class ValueHeadAdapter(ABC):
    @abstractmethod
    def compute_value_loss(self, output, value_targets, score_targets) -> torch.Tensor: ...
    @abstractmethod
    def scalar_value(self, output) -> torch.Tensor: ...

class ScalarValueAdapter(ValueHeadAdapter):
    """For BaseModel: MSE loss on tanh-activated value vs returns."""
    ...

class MultiHeadValueAdapter(ValueHeadAdapter):
    """For KataGoBaseModel: cross-entropy W/D/L + MSE score lead."""
    ...
```

The model registry returns the appropriate adapter alongside the model. The unified loop calls `self.value_adapter.compute_value_loss(...)` and `self.value_adapter.scalar_value(...)` without ever branching on model type. This keeps the clean separation that Plans B and C established.

Detection at init uses the inline DataParallel unwrap idiom (no `unwrap_model` helper — there is no such function in the codebase):

```python
base_model = self.model.module if hasattr(self.model, "module") else self.model
self.value_adapter = get_value_adapter(base_model)  # returns ScalarValueAdapter or MultiHeadValueAdapter
```

#### BatchNorm Strategy

The split-merge step runs the learner model in `eval()` mode during rollout collection (standard PPO pattern from Plans B/C). This means BatchNorm uses running statistics, not batch statistics, during data collection. The variable batch size from the split (learner-subset ≈ 64 of 128 envs) does not affect BN in eval mode. During `update()`, the model switches to `train()` mode and BN updates from full mini-batches. This is the same BN strategy used in Plans B/C and does not require changing to GroupNorm/LayerNorm.

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

**Deletion Sequencing (CRITICAL — wrong order breaks CI):**

These deletions must be phased across 3 PRs to avoid breaking the test suite:

1. **PR 1 — Add:** Create `gae.py`, `league.py`, `demonstrator.py`, `evaluate.py`. Extend `model_registry.py`, `config.py`, `db.py`. Write all new test files. All existing tests still pass.
2. **PR 2 — Migrate:** Update `tests/conftest.py` to import from `katago_ppo` instead of `ppo`. Update `tests/test_ppo.py` GAE tests to import from `gae.py`. Update `tests/test_registries.py` and `tests/test_registry_gaps.py` to use `KataGoPPOParams`. Rewire `keisei-train` entrypoint. Verify all tests pass.
3. **PR 3 — Delete:** Remove `loop.py`, `ppo.py`, and their now-migrated test files (`test_loop.py`, `test_loop_gaps.py`, `test_ppo.py`, `test_ppo_gaps.py`). Verify all tests pass.

**Test files requiring disposition (not in original file map):**

| File | Action | Detail |
|------|--------|--------|
| `tests/conftest.py` | Modify | Update `PPOAlgorithm`/`RolloutBuffer` imports → `KataGoPPOAlgorithm`/`KataGoRolloutBuffer` |
| `tests/test_loop.py` | Delete (PR 3) | Coverage transfers to `test_pipeline_consolidation.py` |
| `tests/test_loop_gaps.py` | Delete (PR 3) | Same |
| `tests/test_ppo.py` | Delete (PR 3) | GAE tests migrate to `test_gae.py`; PPO tests transfer to consolidation tests |
| `tests/test_ppo_gaps.py` | Delete (PR 3) | Same |
| `tests/test_registries.py` | Modify | Update `PPOParams` → `KataGoPPOParams` |
| `tests/test_registry_gaps.py` | Modify | Same |

#### What Gets Kept

| Component | Action | Detail |
|-----------|--------|--------|
| `compute_gae()` | Move to `keisei/training/gae.py` | Extracted from deleted `ppo.py`. Own file justified: used by `KataGoPPOAlgorithm` and future exploiter agents. Also imported by `test_gae.py` regression tests. |
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

- **Bootstrap (epoch 0):** before the first training step, the learner's initial (random) weights are saved as the first `OpponentEntry` in the pool. This ensures the pool is never empty — the learner always has at least one opponent. The first `snapshot_interval` epochs face this initial-weights opponent (effectively self-play against a frozen copy of the initialization).
- **Snapshot trigger:** every `snapshot_interval` epochs (default 10), the current learner's weights are saved as a new `OpponentEntry` in the pool.
- **Rolling window:** keeps the most recent `max_pool_size` entries (default 20). When the pool exceeds this size, the oldest entry is evicted — but only if it is not **pinned** by a `DemonstratorRunner` slot (see eviction safety below).
- **Eviction safety:** the `DemonstratorRunner` registers a set of "pinned" entry IDs before loading models. `OpponentPool.evict()` skips pinned entries and evicts the next-oldest instead. Pinned entries are released after the demonstrator finishes loading the model into memory. This prevents `FileNotFoundError` when eviction races with demonstrator model loading.
- **Storage:** league checkpoints live in `checkpoints/league/` to avoid confusion with training checkpoints in `checkpoints/`. The directory is created by `OpponentPool.__init__` if it does not exist.
- **Cross-architecture loading:** each entry stores its architecture name and model params. To instantiate an opponent, the pool calls `model_registry.build_model(entry.architecture, entry.model_params)` and loads the checkpoint weights with `weights_only=True`. This means a `ResNetModel` checkpoint can be loaded as an opponent even when the current learner is an `SEResNetModel` (all models accept 50-channel observations post-Plan-A; see Observation Channel Reconciliation above).
- **Persistence:** pool metadata is stored in the `league_entries` table in `keisei.db`, not in memory. The pool reconstructs its state from the DB on restart.
- **Elo floor and pool health:** entries with Elo below `elo_floor` (default 500) are excluded from the historical sampling tier but retained in the pool. A pool health metric (fraction of entries above the floor) is logged each epoch. If all entries fall below the floor, a fresh snapshot of the current learner is injected to provide a non-trivial sparring partner.

#### OpponentSampler

Selects which opponent the learner faces each epoch. Two-tier sampling:

| Tier | Source | Default Ratio | Description |
|------|--------|---------------|-------------|
| Historical | Random entry from pool | 80% | Prevents degenerate equilibria by exposing the learner to diverse past strategies |
| Current best | Most recent snapshot | 20% | Ensures the learner faces a strong sparring partner |

Ratios are configurable via TOML (`historical_ratio`, `current_best_ratio`). Validated at config load: `historical_ratio + current_best_ratio == 1.0`. If the pool has only one entry, both tiers sample the same model (this is expected during bootstrap). The sampler returns an `OpponentEntry`; the loop loads the corresponding model once per epoch.

The `OpponentSampler` is a concrete class with a `strategy: str` field (default `"two_tier"`). Future strategies (Elo-weighted, exploiter-aware) extend this class or replace it via config. An ABC would be premature — there is currently one implementation.

#### Split-Merge Step Logic

Within the inner training loop, each step processes all 128 environments simultaneously. The environments have a mix of "whose turn is it" (Black or White). One side is the learner, the other is the opponent.

Per step:

1. Read `current_players` array to determine which envs have the learner's turn vs the opponent's turn.
2. **Learner subset:** forward-pass the training model with gradients enabled. Store transitions in the rollout buffer.
3. **Opponent subset:** forward-pass the frozen opponent model with `torch.no_grad()`. Do NOT store these transitions.
4. Merge the two action arrays into a single action list.
5. Call `vecenv.step(merged_actions)`.

The learner always plays Black (side 0). This simplifies the split logic and avoids sign-flipping reward complications. When the seat rotates to a new model, it still plays Black.

**Known limitation — side bias:** Shogi has a meaningful first-player advantage (~52-55% Black win rate in professional play). Training exclusively as Black biases the value function and policy toward Black-optimal play. When the model is used as an opponent (playing White) or for evaluation (playing both sides), it may be systematically weaker as White. **Tracked for follow-up:** a future league tier should alternate the learner's side each epoch or each rotation to train a side-balanced policy.

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

**Optimizer reset:** each rotation creates a fresh Adam optimizer for the new learner. The previous model's momentum would fight the new model's gradient signal. The elevated entropy warmup (Plan D Task 3) applies for the first N epochs of each seat to compensate for the Adam cold-start period (~10 epochs for moment estimates to stabilize). The LR plateau scheduler's patience counter is also reset at each rotation (`scheduler.num_bad_epochs = 0`) so that value_loss spikes from the cold-start don't consume patience intended for the real training phase.

**Cold-start budget:** with `epochs_per_seat = 50`, the cold-start period (warmup entropy + Adam stabilization) is ~10-15 epochs or ~20-30% of the seat. If this proves too expensive, `epochs_per_seat` should be increased to 100+ to amortize the fixed cost.

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

**Slot fallbacks for small pools:**
- Demo slot 1 (#1 vs #2 Elo) requires ≥ 2 pool entries. If the pool has < 2 entries, slot 1 is inactive.
- Demo slot 2 (cross-architecture) requires ≥ 2 entries of different architecture types. If unavailable, falls back to a random pairing.
- Demo slot 3 (random) requires ≥ 2 entries. If unavailable, inactive.

**Error handling:** demonstrator thread crashes are non-fatal. The runner wraps its main loop in `try/except`, logs the exception with `logger.exception(...)`, and continues. The training loop checks `DemonstratorRunner.is_alive()` at each epoch boundary and logs a WARNING if the thread has died, but does not restart it (restart is a future extension). This policy is explicit: demonstrator failure must never block or crash training.

**GPU isolation:** the demonstrator creates a dedicated `torch.cuda.Stream()` and runs all inference inside `with torch.cuda.stream(self.stream):`. This isolates demonstrator CUDA kernels from the training loop's backward pass, preventing nondeterministic kernel interleaving. Alternatively, if GPU memory is constrained, the demonstrator can run on CPU by setting `demonstrator.device = "cpu"` in config (inference-only workloads are CPU-viable at 1 move/second).

### Evaluation Entrypoint

`keisei-evaluate` CLI for head-to-head evaluation outside of training:

```
keisei-evaluate \
    --checkpoint-a checkpoints/league/resnet_ep100.pt \
    --arch-a resnet \
    --checkpoint-b checkpoints/league/se_resnet_ep200.pt \
    --arch-b se_resnet \
    --games 400 \
    --max-ply 500
```

Output: win/loss/draw counts, win rate with 95% confidence interval, Elo delta.

**Statistical note:** 400 games provides ~±50 Elo precision at 95% confidence (binomial CI on win rate). The `--games` default is 400. At 100 games, the CI is ~±70 Elo — the CLI warns when `--games < 200` that results may not be statistically significant.

This reuses the same game-playing loop as the demonstrator (inference-only, two frozen models) but runs synchronously for a fixed number of games and reports results to stdout.

---

## Configuration

### TOML Additions

```toml
[league]
max_pool_size = 20           # maximum entries in the opponent pool
snapshot_interval = 10       # epochs between pool snapshots
epochs_per_seat = 50         # epochs before rotating the training seat
historical_ratio = 0.8       # fraction of epochs vs historical opponents (must sum to 1.0 with current_best_ratio)
current_best_ratio = 0.2     # fraction of epochs vs current best
initial_elo = 1000           # starting Elo for new entries
elo_k_factor = 32            # K-factor for Elo updates
elo_floor = 500              # entries below this are excluded from historical sampling

[demonstrator]
num_games = 3                # concurrent demonstrator games (0 to disable)
auto_matchup = true          # auto-select pairings based on Elo/architecture
moves_per_minute = 60        # playback speed for demonstrator games
device = "cuda"              # "cuda" (with stream isolation) or "cpu" (for constrained VRAM)
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

**Schema migration for existing databases:**

`SCHEMA_VERSION` bumps from 1 to 2. `init_db()` checks the current version and applies migrations:

```sql
-- Migration v1 → v2: add league columns to game_snapshots
ALTER TABLE game_snapshots ADD COLUMN game_type TEXT NOT NULL DEFAULT 'live';
ALTER TABLE game_snapshots ADD COLUMN demo_slot INTEGER;
```

`write_game_snapshots()` must be updated to accept and pass `game_type` and `demo_slot` fields. Existing callers that don't pass these fields get the defaults (`"live"`, `NULL`).

**Security:** all `torch.load()` calls for opponent checkpoint loading MUST use `weights_only=True`. League opponents are loaded from paths stored in the DB — if a path is ever compromised, `weights_only=False` would execute arbitrary Python.

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
| Delete | `keisei/training/loop.py` | Old training loop — **PR 3 only** (after test migration) |
| Delete | `keisei/training/ppo.py` | Old PPO — **PR 3 only** (after `compute_gae` extracted to `gae.py`) |
| Modify | `tests/conftest.py` | Update imports: `PPOAlgorithm` → `KataGoPPOAlgorithm`, `RolloutBuffer` → `KataGoRolloutBuffer` |
| Modify | `tests/test_registries.py` | Update `PPOParams` → `KataGoPPOParams` |
| Modify | `tests/test_registry_gaps.py` | Same |
| Delete | `tests/test_loop.py` | **PR 3 only** — coverage transfers to `test_pipeline_consolidation.py` |
| Delete | `tests/test_loop_gaps.py` | **PR 3 only** |
| Delete | `tests/test_ppo.py` | **PR 3 only** — GAE tests migrate to `test_gae.py` first |
| Delete | `tests/test_ppo_gaps.py` | **PR 3 only** |
| Create | `tests/test_gae.py` | Regression tests for extracted `compute_gae()` (migrated from `test_ppo.py`) |
| Create | `tests/test_league.py` | OpponentPool, OpponentSampler, Elo calculation tests |
| Create | `tests/test_demonstrator.py` | DemonstratorRunner tests |
| Create | `tests/test_evaluate.py` | Evaluation CLI tests |
| Create | `tests/test_pipeline_consolidation.py` | Both model contracts work in unified loop |
| Modify | `scripts/run-500k.sh` | Update `keisei-train` invocation if args changed |
| Modify | `scripts/pod-setup.sh` | Same |

---

## Dependencies

- **Requires Plans A-D complete.** Plan E modifies `katago_loop.py`, `katago_ppo.py`, and `model_registry.py` as they exist after Plans A-D. **Prerequisite gate:** before starting any Plan E task, verify:
  ```bash
  uv run python -c "from keisei.training.katago_loop import KataGoTrainingLoop; from keisei.training.katago_ppo import KataGoPPOAlgorithm; from keisei.training.models.se_resnet import SEResNetModel; print('Plans A-D ready')"
  ```
- **No external dependencies.** Elo math is ~20 lines; no need for a library.
- **No Rust changes.** The `VecEnv` API is sufficient — demonstrator games just use `VecEnv(num_envs=1)`.
- **Migration note:** existing training runs using the old `TrainingLoop` cannot be resumed with the unified loop. Checkpoints from the old pipeline can be loaded into `BaseModel` architectures for the league, but optimizer state is discarded (fresh Adam at each seat).

## Future Extensions (Not in Plan E)

- **Elo-weighted sampling:** replace uniform historical sampling with Elo-proximity weighting.
- **Exploiter agents (tier 3):** a model trained to exploit the main agent's weaknesses, added as a new `OpponentSampler` strategy.
- **Concurrent co-training:** multiple models train simultaneously (population-based training).
- **Web UI league dashboard:** Elo leaderboard, rating-over-time charts, matchup history. The DB schema supports this; the UI work is separate.
