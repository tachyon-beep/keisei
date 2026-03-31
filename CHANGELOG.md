# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/tachyon-beep/keisei/releases/tag/v0.1.0
