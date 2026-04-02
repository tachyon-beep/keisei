# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-02

Second release. Adds the KataGo-inspired SE-ResNet architecture, multi-head PPO,
opponent league play, supervised learning warmup, and significant infrastructure
improvements.

### Added

#### KataGo-Style Neural Network (Plans A–B)
- **SE-ResNet** architecture with GlobalPoolBiasBlock (global pooling bias +
  Squeeze-and-Excitation with scale+shift), adapted from KataGo. (`5898460`,
  `1d45608`)
- `KataGoBaseModel` ABC and `KataGoOutput` dataclass for multi-head models.
  (`b729d75`)
- Three output heads: spatial policy `(B, 9, 9, 139)`, W/D/L value
  classification `(B, 3)`, and score prediction `(B, 1)`. (`1d45608`)
- `KataGoPPOAlgorithm` with multi-head loss (policy + W/D/L cross-entropy +
  score MSE). (`54c2944`, `f4b5848`)
- `KataGoRolloutBuffer` with `value_categories` and `score_targets` storage,
  including NaN/unnormalized guards. (`5369152`)
- `compute_value_metrics` for W/D/L prediction monitoring (degeneracy
  detection). (`e1467f9`)
- `se_resnet` and `katago_ppo` registered in model and algorithm registries.
  (`6ecb5ed`, `a19e6a6`, `7417e6b`)

#### Rust Engine Extensions (Plan A)
- 50-channel KataGo observation generator with repetition count planes (1x, 2x,
  3x, 4+) and check indicator. (`d653c8c`)
- `SpatialActionMapper` with 11,259-action spatial encoding (81 squares x 139
  move types). (`f49fde9`)
- `VecEnv` mode dispatch for observation and action encoding formats. (`2709e70`)
- `material_balance()` and `piece_value()` in shogi-core for score head
  targets. (`256fa4f`)
- Per-step `material_balance` buffer in `VecEnv` and `StepMetadata`. (`3edb6d3`)

#### Supervised Learning Warmup (Plans C–D)
- `CSAParser` for Floodgate game record parsing with Shift-JIS detection.
  (`00d51a2`, `463e60a`)
- `SLDataset` with memory-mapped binary shards for efficient data loading.
  (`af4edcb`)
- `SLTrainer` for supervised learning warmup before RL. (`b2c75bf`)
- `keisei-prepare-sl` CLI for SL data preparation. (`50595e1`)
- `ReduceLROnPlateau` scheduler in `KataGoTrainingLoop`. (`0420d49`)
- RL warmup with elevated entropy coefficient (`get_entropy_coeff` method) to
  soften overconfident SL policies. (`19bb0b5`)

#### Training Infrastructure (Plan E)
- `ValueHeadAdapter` pattern with `ScalarValueAdapter` and
  `MultiHeadValueAdapter` for dual-contract model support. (`89efabf`)
- Contract type and `obs_channels` added to model registry. (`9bd3be8`)
- `KataGoTrainingLoop` wiring SE-ResNet + KataGoPPO + VecEnv. (`f6f96ef`)
- Architecture metadata saved in checkpoints for safe resume. (`6e00422`)
- `compute_gae` extracted to shared `gae.py` module. (`efaf366`)

#### Opponent League (Plan E-1)
- `OpponentPool` with SQLite-backed model storage and Elo tracking. (`d12f2bd`)
- `OpponentSampler` for weighted opponent selection during training.
- `LeagueConfig` and `DemonstratorConfig` added to config system. (`33219c5`)
- DB schema v2 with league tables and `game_type` column. (`e001fbd`)

#### Unified Training Loop (Plan E-2)
- `split_merge_step` for learner vs. opponent forward passes. (`bf47a8c`)
- League integration, split-merge, Elo updates, and seat rotation in training
  loop. (`623a1a8`)

#### Demonstrator and Evaluation (Plan E-3)
- `DemonstratorRunner` for inference-only exhibition matches. (`47ee5b1`)
- `keisei-evaluate` CLI for head-to-head checkpoint comparison. (`84c6a3d`)
- GPU device support in `keisei-evaluate`. (`bcfe0aa`)
- Old `TrainingLoop` and `PPOAlgorithm` removed; `keisei-train` rewired to
  unified loop. (`c9e8505`, `f878759`)

#### Score Head Fix
- Per-step material balance replaces reward/76 for score targets (denser, more
  accurate signal). (`57ceff8`)
- NaN masking removed from score loss — per-step material provides targets for
  every position. (`bf0c257`)

#### Project Infrastructure
- `CONTRIBUTING.md`, `SECURITY.md`, `.editorconfig` added.
- GitHub Actions CI workflow (Python 3.12/3.13 + Rust).
- PR template.
- `py.typed` marker for PEP 561 compliance.
- pyproject.toml: added authors, URLs, classifiers, keywords.
- README: badges, architecture detail, prior art attribution, collapsible
  sections.

### Fixed
- Score targets used reward/76 instead of material difference. (`57ceff8`)
- NaN sentinel in score targets caused loss corruption. (`bf0c257`)
- Eval/train mode not toggled correctly in PPO action selection. (`5cb68cf`)
- Optimizer state tensors not moved to model device after checkpoint load.
  (`16f4a27`)
- White knight decode produced wrong target square in shogi-gym. (`5ab51d1`)
- Multiple review findings addressed across Plans A–E. (`e484dd7`, `30bee78`,
  `beaf645`, `1031123`, `f1b4452`, `089d977`, `2a13856`, `d1ac185`, `f5f1471`,
  `74670a0`, `f2141c0`)

### Changed
- Model registry extended from 3 to 4 architectures (added SE-ResNet).
- Test suite grew from 74 to 255 Python tests and 188 to 111 Rust tests (Rust
  count decreased due to test consolidation and removal of redundant cases).
- Observation encoding expanded from 46 to 50 channels for KataGo mode.
- Action space: added 11,259 spatial encoding alongside 13,527 flat encoding.

## [0.1.0] - 2026-04-01

First release of the rebuilt Keisei, powered by a Rust game engine replacing
the original pure-Python implementation.

### Added

#### Rust Engine (`shogi-engine/`)
- **shogi-core** crate — full Shogi implementation: board representation, legal
  move generation, rule enforcement (check, checkmate, repetition, impasse),
  SFEN parsing/serialization, and Zobrist hashing. (`55f1666`)
- **shogi-gym** crate — RL environment layer: `VecEnv` vectorized environment,
  46-plane board observations, 13,527-action mapping, and step/reset API
  exposed to Python via PyO3. (`55f1666`)
- `VecEnv.get_spectator_data()` for live game visualization. (`5f74bc3`)
- `SpectatorEnv.from_sfen()` static constructor for replaying positions.
  (`5fa589a`)
- `VecEnv.get_sfen()` and `get_sfens()` for SFEN extraction from running
  games. (`7754599`)
- `VecEnv.mean_episode_length` and `truncation_rate` properties for episode
  statistics. (`794232d`)
- Shared `build_spectator_dict()` helper extracted into `spectator_data.rs`.
  (`8b2ed18`)
- 188 Rust tests across both crates covering move generation, rules, SFEN,
  position logic, observations, action mapping, VecEnv, and spectator data.
  (`c92722a`)

#### Python Training Harness (`keisei/`)
- Package scaffold with `keisei-train` and `keisei-serve` entry points.
  (`c1af445`)
- TOML config loading with frozen dataclass validation (`AppConfig`,
  `TrainingConfig`, `ModelConfig`, `DisplayConfig`). (`813c9d0`)
- `BaseModel` ABC and ResNet policy+value network (policy head: 13,527 actions,
  value head: scalar). (`593a3a0`)
- MLP and Transformer model architectures. (`3e6b50e`)
- SQLite database layer (WAL mode) for training metrics, game snapshots, and
  training state. (`bea432a`)
- Model checkpointing with save/load of model weights, optimizer state, epoch,
  and step. (`a8c33db`)
- PPO algorithm with rollout buffer, Generalized Advantage Estimation (GAE),
  clipped surrogate objective, and mini-batch updates. (`55c39ea`)
- Model and algorithm registries with validation. (`dc5b223`)
- Training loop orchestrator tying config, models, PPO, database, and VecEnv
  together, with resume-from-checkpoint support. (`2cd9bb5`)
- 74 Python tests covering config, database, checkpointing, models, PPO, and
  the training loop. (`73c537a`)

### Fixed
- Entropy NaN caused by `0 * -inf` in masked softmax — replaced with safe
  log-prob masking. (`cd2e9e6`)
- BatchNorm train/eval mode bug — model now correctly switches to eval mode
  during action selection and back to train mode during updates. (`73c537a`)
- Property access for Rust PyO3 getters (use attribute access, not method
  calls). (`cd2e9e6`)

### Changed
- Stripped the old pure-Python Shogi engine; all game logic now lives in Rust.
  (`3cd44ae`)
- Cleaned up stale docs, plans, and specs from the pre-Rust era. (`ae8d7d7`)
- Fixed all Clippy warnings across shogi-core and shogi-gym. (`5baabd6`)
- Fixed lint issues in Python (import sorting, line length, unused imports).
  (`b99f403`)

[0.2.0]: https://github.com/tachyon-beep/keisei/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/tachyon-beep/keisei/releases/tag/v0.1.0
