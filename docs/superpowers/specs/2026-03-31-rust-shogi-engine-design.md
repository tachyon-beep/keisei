# Rust Shogi Engine Design Spec

**Date:** 2026-03-31
**Status:** Draft
**Scope:** Standalone Rust Shogi engine with PyTorch/Gym bindings for Keisei integration

## Overview

Rewrite the Shogi game engine from Python (~3,200 lines, 0.5 it/s) to Rust, structured as two crates:

1. **`shogi-core`** — general-purpose Shogi game engine. No ML dependencies. Suitable for mobile apps, USI engines, analysis tools, WASM builds.
2. **`shogi-gym`** — RL training environment built on `shogi-core`. Vectorized batch stepping, observation tensor generation, legal mask computation, PyO3/maturin Python bindings.

**Goal:** GPU-saturating throughput — the engine should never be the bottleneck. Sensible optimizations, not diminishing-returns heroics.

**Non-goal:** Policy, model, or agent logic. The Rust side is a pure state machine. All ML stays in Python.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Python (Keisei)                                     │
│  ┌──────────┐  ┌───────────┐  ┌──────────────────┐  │
│  │ PPO Agent│  │ Evaluator │  │ Streamlit WebUI  │  │
│  └────┬─────┘  └─────┬─────┘  └────────┬─────────┘  │
│       │              │                  │             │
│  ┌────▼──────────────▼──┐    ┌─────────▼──────────┐  │
│  │  shogi_gym.VecEnv    │    │ shogi_gym.Spectator │  │
│  │  (batch training)    │    │ (display games)     │  │
│  └────────┬─────────────┘    └──────────┬──────────┘  │
├───────────┼─────────────────────────────┼─────── FFI ─┤
│  Rust     │                             │             │
│  ┌────────▼─────────────────────────────▼──────────┐  │
│  │              shogi-gym crate                     │  │
│  │  VecEnv, SpectatorEnv, ActionMapper, ObsGen     │  │
│  └────────────────────┬────────────────────────────┘  │
│                       │                               │
│  ┌────────────────────▼────────────────────────────┐  │
│  │              shogi-core crate                    │  │
│  │  Position, GameState, AttackMap, MoveGen, Rules  │  │
│  └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## `shogi-core` — Game Engine Crate

### Rule Profile

v1 implements CSA computer shogi rules with RL adaptations. This is **not** a literal JSA tournament rules engine — several outcomes that JSA rules handle as replays are mapped to terminal states for RL training purposes. The core crate exposes the raw game outcomes; the gym crate maps them to RL reward signals.

**Divergences from JSA tournament rules:**
- JSA fourfold repetition is a replay (rematch with swapped sides). We treat it as a terminal draw for RL.
- JSA perpetual check is a foul requiring the offending player to retract. We treat it as a terminal loss for the checking side.
- Impasse follows CSA computer shogi convention (point-based resolution), not the JSA agreed-draw protocol.

**Rules implemented:**

- **Checkmate** (tsumi): no legal moves while in check — opponent wins
- **Impasse / jishogi** (entering king, 27-point rule): when both kings have entered the opponent's promotion zone and the position is stalemated, the player with >= 24 points (Rook/Bishop = 5 pts, other pieces = 1 pt) wins; otherwise draw. Exposed as `GameResult::Impasse(Option<Color>)`.
- **Sennichite** (fourfold repetition): draw, *unless* the repeating position arises from one side giving continuous check, in which case the checking side loses (`GameResult::PerpetualCheck(Color)`)
- **Uchi-fu-zume** (pawn drop checkmate): illegal — a pawn drop that delivers immediate checkmate is not a legal move
- **Nifu** (double pawn): illegal — cannot drop a pawn on a file that already has an unpromoted pawn of the same color
- **Dead drops** (行き所のない駒): illegal — cannot drop a piece to a square from which it has no legal future move (pawn/lance on last rank, knight on last two ranks)
- **Max ply limit**: configurable per-game, enforced as `GameResult::MaxMoves` (truncation, not a rule-based terminal state — this distinction matters for RL bootstrapping)

No `Stalemate` variant — orthodox Shogi does not produce stalemate positions under normal play. If encountered (e.g., in contrived test positions), it is treated as a loss for the side with no legal moves, consistent with JSA rules.

### Board Representation

- **Board**: `[u8; 81]` flat array, row-major (index = `row * 9 + col`). Empty square = `0x00`. Piece encoding uses `NonZeroU8` semantics (0 is reserved for empty), so the board is compact without needing `Option` wrappers.
- **Piece encoding** (packed `NonZeroU8`): 3 bits piece type (1-8), 1 bit color, 1 bit promoted. 3 bits spare. The `0` value is reserved for empty, so `Option<Piece>` where `Piece` wraps `NonZeroU8` is 1 byte via niche optimization.
- **Square** newtype: `struct Square(u8)` — wraps board index (0-80). All coordinate conversions (row/col, SFEN notation, perspective flip) go through this type. Prevents silent coordinate bugs across SFEN parsing, perspective flips, legal masks, and FFI.
- **Hands**: `[[u8; 7]; 2]` — piece counts per `HandPieceType` per color
- **AttackMap**: `[[u8; 81]; 2]` — per-color attack count per square (how many pieces of that color attack each square), incrementally updated on make/unmake. Count > 0 means attacked. Count (not just bool) enables X-ray/battery detection.
- **ZobristHash**: `u64` — incrementally updated position hash for sennichite detection
- **PawnColumns**: `[[bool; 9]; 2]` — per-color pawn presence per file for O(1) nifu checks

### Core Types

```rust
#[derive(Copy, Clone)]
enum Color { Black, White }

#[derive(Copy, Clone)]
enum PieceType { Pawn, Lance, Knight, Silver, Gold, Bishop, Rook, King }

/// Pieces that can be held in hand and dropped. King is excluded at the type level.
#[derive(Copy, Clone)]
enum HandPieceType { Pawn, Lance, Knight, Silver, Gold, Bishop, Rook }

/// Board square index (0-80). Coordinate convention: row-major, row 0 = rank 1 (Black's promotion zone).
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct Square(u8);

enum Move {
    Board { from: Square, to: Square, promote: bool },
    Drop { to: Square, piece_type: HandPieceType },  // King drops are unrepresentable
}

/// All decisive results use `winner: Color` for consistency.
/// This prevents reward-sign bugs from inconsistent semantics
/// (e.g., Checkmate storing winner but PerpetualCheck storing loser).
enum GameResult {
    InProgress,
    Checkmate { winner: Color },           // opponent has no legal moves while in check
    Impasse { winner: Option<Color> },     // Some(winner) by point advantage, None for draw
    Repetition,                            // fourfold repetition (draw)
    PerpetualCheck { winner: Color },      // winner is the side that was being checked
    MaxMoves,                              // configurable ply limit (truncation, not terminal)
}
```

### Position vs GameState

- **`Position`**: board + hands + current player + Zobrist hash. Pure positional state with no history. SFEN serialization operates on `Position`. Round-trip fidelity: `Position::from_sfen(pos.to_sfen()) == pos`.
- **`GameState`**: wraps `Position` and adds: move history, repetition counter (`HashMap<u64, u8>` pre-sized to expected game length), attack map, pawn columns, ply count, max ply limit. `GameState` owns the full game lifecycle.

### Make/Unmake

- `make_move(&mut self, mv: Move) -> UndoInfo` — applies move, returns undo state
- `unmake_move(&mut self, mv: Move, undo: UndoInfo)` — restores state exactly
- Both incrementally update: board array, hands, attack map, Zobrist hash, pawn columns, repetition counter
- No heap allocation after construction — integer writes only on the hot path
- Repetition `HashMap` is pre-sized with `HashMap::with_capacity(max_ply)` at `GameState` construction to avoid reallocation during play

### Dual API: Ergonomic and Hot-Path

**Ergonomic API** (for tools, tests, USI engines):
```rust
let mut game = GameState::new();
let moves = game.legal_moves();            // -> Vec<Move> (allocates)
let in_check = game.is_in_check();         // O(1)
let undo = game.make_move(moves[0]);
game.unmake_move(moves[0], undo);
println!("{}", game.position().to_sfen());
```

**Hot-path API** (for VecEnv, zero allocation after warmup):
```rust
let mut move_buf = MoveList::new();         // fixed-capacity, reusable
game.generate_legal_moves_into(&mut move_buf);  // writes into caller's buffer
let mask_buf: &mut [bool] = ...;              // length = mapper.action_space_size()
game.write_legal_mask_into(mask_buf, &mapper);  // writes mask in-place, asserts length
```

`MoveList` is a fixed-capacity stack-allocated array. The theoretical maximum legal moves in Shogi is debated in the literature (~593 is commonly cited, but some constructed positions may push higher). Capacity is set to 1024 — a few hundred extra bytes on the stack is negligible cost for eliminating any risk of overflow in pathological positions.

### Move Generation

- **Pseudo-legal generation**: iterate pieces, emit candidate moves using piece movement rules. Sliding pieces (Rook, Bishop, Lance) use pre-computed direction offset arrays for the flat board: orthogonal `{-9, +9, -1, +1}`, diagonal `{-10, -8, +8, +10}`. File-wrap prevention via pre-computed per-square valid direction tables (a piece on file 0 cannot move with offset -1).
- **Legality filter**: make move, check own king's square in attack map (O(1) lookup), unmake
- **Check detection**: `attack_map[opponent][king_square] != 0` — O(1)

### Special Rules

**Uchi-fu-zume (pawn drop checkmate) — bounded local search:**

A pawn only attacks one square (directly forward), so a pawn-drop check cannot be interposed. The defender's escapes are limited to:

1. **Capture the pawn** — check which pieces attack the pawn's square (O(1) attack map lookup). For each potential capturer, verify it is not pinned to the king by simulating the capture and checking king safety (make/unmake + O(1) attack map check). Worst case: ~3-4 pieces to check.
2. **Move the king** — check each of the king's <= 8 adjacent squares for safety (O(1) attack map lookup each), plus verify the king is not moving into a square attacked by the dropped pawn itself.

Total complexity: bounded local search, O(1) per candidate with a small constant (at most ~12 candidates). Much cheaper than full legal move generation, but not constant-time — pin detection requires local simulation.

**Sennichite (fourfold repetition + perpetual check):**

Two data structures work together:

1. **Position repetition counter**: `HashMap<u64, u8>` keyed on Zobrist hash, pre-sized at construction. O(1) check for "has this position occurred 4 times?"
2. **Per-ply check history**: `Vec<bool>` indexed by ply number, tracking whether the side to move was in check at each ply. Pre-sized alongside the HashMap.

When the repetition counter reaches 4 for the current position, walk back through the per-ply check history to find the repeating plies (those with matching Zobrist hashes). If one side was giving check on all repeating plies, that side loses (`PerpetualCheck`). Otherwise, draw (`Repetition`). The two structures are kept separate because position-indexed repetition counting and ply-indexed check history serve different purposes and would be a subtle data model error to conflate.

**Nifu (double pawn):** `pawn_columns[color][file]` maintained incrementally. O(1) lookup.

**Dead drops:** conditional on piece type + destination rank. O(1). Pawn/Lance cannot drop on last rank; Knight cannot drop on last two ranks.

**Forced promotion:** conditional on piece type + destination rank. O(1). Pawn/Lance must promote on last rank; Knight must promote on last two ranks.

**Impasse / jishogi (27-point rule):**

v1 follows the CSA computer shogi convention (the standard for computer shogi tournaments):

- **Trigger condition**: a player's king is in the opponent's promotion zone (ranks 1-3 for Black, ranks 7-9 for White)
- **Point counting**: `compute_impasse_score(color) -> u8` sums piece values for that player: Rook/Bishop = 5 pts, all other pieces (excluding King) = 1 pt. Counts pieces both on the board and in hand.
- **Declaration**: when a player's king is in the opponent's camp and that player has >= 10 pieces (including king) in the promotion zone:
  - If the player has >= 24 points: they win (`GameResult::Impasse(Some(color))`)
  - If both players meet the entry conditions but neither reaches 24 points: draw (`GameResult::Impasse(None)`)
- Impasse is checked after each move when the moving side's king is in the opponent's promotion zone. It is not auto-declared — the check only fires when the positional trigger is met.
- Note: the exact impasse rules vary between JSA, CSA, and informal play. The CSA convention is chosen because this engine targets computer shogi. The rule set could be made configurable in a future version if needed.

### Attack Map: Incremental Update Strategy

The attack map is the foundation of the engine's performance. Incremental updates for sliding pieces (Rook, Bishop, Lance, and their promoted forms) are the hardest part to get right.

**Implementation strategy:**
1. First implement `recompute_attack_map_from_scratch(&mut self)` — iterates all pieces, populates the full attack map. This is the ground truth oracle.
2. Then implement incremental updates in make/unmake: when a piece moves from square A to square B, decrement counts for all squares A attacked, increment counts for all squares B attacks. For sliding pieces, also recalculate rays that were blocked by the piece at A (now unblocked) and rays that are now blocked by the piece at B.
3. Property-based tests assert `incremental == from_scratch` after every make/unmake pair.

### Serialization

- `Position::to_sfen(&self) -> String` — standard SFEN format
- `Position::from_sfen(s: &str) -> Result<Position>` — parse SFEN with validation
- Round-trip fidelity: `Position::from_sfen(pos.to_sfen()) == pos` for all valid positions
- `GameState` is not SFEN-serializable (history-dependent state has no SFEN representation)

### Error Handling

```rust
enum ShogiError {
    InvalidSfen(String),       // malformed SFEN string
    IllegalMove(Move),         // move violates rules
    InvalidSquare(u8),         // square index out of range
    GameOver(GameResult),      // attempted move on finished game
}
```

All fallible operations return `Result<T, ShogiError>`. Across the FFI boundary, PyO3 converts these to Python exceptions (`ValueError` for invalid input, `RuntimeError` for illegal state).

### Invariants

The following properties must hold at all times. These serve as the definitive correctness checklist and the basis for property-based tests:

1. **Make/unmake round-trip**: after every `make_move(mv)` / `unmake_move(mv, undo)` pair, `GameState` is identical to its state before the pair (byte-level equality of all fields).
2. **Attack map consistency**: `attack_map` always equals the result of `recompute_attack_map_from_scratch()`.
3. **Pawn column consistency**: `pawn_columns` always equals a fresh scan of the board for unpromoted pawns.
4. **Zobrist hash consistency**: `zobrist_hash` always equals a fresh computation from the current position.
5. **Legal move soundness**: `legal_moves()` never includes an illegal move (no false positives).
6. **Legal move completeness**: `legal_moves()` never excludes a legal move (no false negatives).
7. **Repetition counter consistency**: the `HashMap<u64, u8>` position counter always reflects the exact count of times the current position has occurred in the game history.
8. **Buffer strides**: all pre-allocated NumPy buffers have correct C-contiguous strides after construction, and these strides never change across `step()` calls.

### Dependencies

`std` only. The sole `std` dependency is `HashMap` for sennichite tracking. A future `no_std` feature flag could replace this with a fixed-size hash table for embedded/WASM targets, but this is not a v1 goal.

## `shogi-gym` — RL Environment Crate

### VecEnv (batch training)

Manages N game environments stepped in a single FFI call. Pre-allocated output buffers, no per-game allocation after construction.

**Thread safety and parallelism:** Each game is independent. The N-game step loop uses `rayon` for parallel iteration when N is large enough to amortize threading overhead (configurable threshold, default N >= 64). For small N, sequential iteration avoids overhead. Parallelism is a runtime toggle: `VecEnv(num_envs=512, parallel=True)`. The Rust stepping section releases the Python GIL via `py.allow_threads()` during both validation and application phases — without this, rayon's thread pool would be serialised by the GIL and parallelism would be illusory.

**Step cycle:**
```
Python:  actions = model(obs_batch)                          # GPU inference
Python:  result = env.step(actions)                          # one FFI call
         └─ Rust (rayon parallel over N games):
             ├─ decode action index → Move (via ActionMapper)
             ├─ validate action is legal (debug_assert, or error in release)
             ├─ make_move()
             ├─ if terminated/truncated: auto-reset, write initial obs
             ├─ write observation into pre-allocated batch buffer
             ├─ write legal mask into pre-allocated batch buffer
             └─ write reward, terminated, truncated flags
Python:  obs_gpu = result.observations.to(device)            # explicit GPU transfer
```

**Illegal action semantics — two-phase step contract:**

Because `step()` may run in parallel via rayon, action validation must happen *before* any environment is mutated. Otherwise, one bad action could leave the batch half-advanced.

1. **Phase 1 — Decode and validate all N actions** (parallel, read-only): decode each action index via `ActionMapper::decode()`, verify it is in the legal mask for that environment. If any action is invalid, raise `RuntimeError` with the env index and action index *before mutating any game state*. No partial stepping.
2. **Phase 2 — Apply all N moves** (parallel, mutating): all actions are known-valid. Apply moves, generate observations, write outputs.

This is a fail-loud design — silent corruption of game state is never acceptable.

**Tensor outputs (pre-allocated, C-contiguous, written in-place):**

| Output | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `observations` | `(N, C, 9, 9)` | float32 | Board state tensors (C = channels from ObsGen) |
| `legal_masks` | `(N, A)` | bool | Valid action masks (A = action_space_size from ActionMapper) |
| `rewards` | `(N,)` | float32 | Terminal rewards |
| `terminated` | `(N,)` | bool | Rule-based episode end (checkmate, repetition, impasse, perpetual check) |
| `truncated` | `(N,)` | bool | Non-rule episode end (max ply limit) |
| `step_metadata` | `(N,)` | structured | Packed per-env metadata: captured piece type (u8), termination reason (u8), ply count (u16). No Python dict allocation on the hot path. |

All buffers are C-contiguous (row-major) to match PyTorch's default memory layout. Strides are set correctly on the `PyArray` to ensure `torch.from_numpy()` shares memory without a copy. Note: `torch.from_numpy()` produces a CPU tensor sharing memory with the NumPy array; transfer to GPU requires an explicit `.to(device)` call (which performs a DMA copy). The total per-step transfer at N=512 is ~14 MB, well within PCIe bandwidth.

**Auto-reset:** When a game terminates or is truncated, VecEnv automatically resets it. The main `observations` buffer contains the *new* game's initial observation. The terminal observation is stored in a separate pre-allocated `terminal_observations` buffer (same shape as a single env's obs), accessible via `result.terminal_observations[env_idx]`. This avoids the stale-view problem where the terminal obs is overwritten by the reset.

**Constructor contract:**
```python
env = shogi_gym.VecEnv(
    num_envs=512,
    max_ply=500,                       # per-game ply limit (truncation)
    parallel=True,                     # rayon parallelism
    action_mapper="default",           # or a custom Python class
    observation_channels=46,           # from ObservationGenerator
)
# Properties available after construction:
env.action_space_size    # -> 13527 (from ActionMapper)
env.observation_channels # -> 46 (from ObservationGenerator)
env.num_envs             # -> 512
```

**Seeding and initial position variance:**
- `reset()` resets all N games to the standard opening position (`startpos` SFEN)
- `reset_from_sfen(sfens: List[str])` is a future extension point (v2) for starting games from arbitrary positions — useful for curriculum learning, evaluation from specific positions, or opening book diversity. Not implemented in v1 but the `Position::from_sfen()` foundation is there.

### SpectatorEnv (display games)

Wraps a single `GameState` for human-viewable games. Same Rust engine, different interface:

- `step(action)` returns rich game state (board, move notation, capture info) as a Python dict — acceptable here since SpectatorEnv is not on the hot path
- No auto-reset — game ends and stays ended until explicitly reset
- `to_dict()` for serialization to JSON (Streamlit dashboard, websockets, etc.)
- Pacing is the caller's responsibility — Python calls `step()` on a timer

```python
# Training — max throughput
train_env = shogi_gym.VecEnv(num_envs=512)
obs, masks = train_env.reset()
result = train_env.step(actions)

# Spectator — display games
spectator = shogi_gym.SpectatorEnv()
state = spectator.step(action)
board_json = spectator.to_dict()
```

### ActionMapper (trait-based, configurable)

```rust
trait ActionMapper: Send + Sync {
    fn encode(&self, mv: Move) -> usize;
    fn decode(&self, idx: usize) -> Result<Move, ShogiError>;  // out-of-range → error, not nonsense
    fn action_space_size(&self) -> usize;
}
```

**Customisation boundary:** `ActionMapper` and `ObservationGenerator` are Rust-side traits only. Custom implementations must be written in Rust and compiled into the crate. Python cannot provide a callback as a custom mapper — a Python callback in the stepping hot loop would drag the interpreter back into every iteration, negating the performance gains of the Rust rewrite. If Python-side customisation is needed in the future, it should be driven by a static descriptor/config that Rust consumes once at construction, not a per-step callback.

- **DefaultActionMapper**: ships with the 13,527-action encoding:
  - Board moves: 81 source squares x 80 destination squares (excluding same-square) x 2 (promote flag) = **12,960**
  - Drop moves: 81 destination squares x 7 hand piece types = **567**
  - Total: **12,960 + 567 = 13,527**
  - Index layout: board moves first (sorted by from, then to, then promote), drops second (sorted by to, then piece type)
- **Intentionally sparse encoding.** The vast majority of action indices map to moves that are never legal for any piece (e.g., a king on square 40 moving to square 0). Typical legal mask density is well under 1% (~30-100 legal moves out of 13,527 slots). This is the same design used by AlphaZero for Chess. Do not attempt to "compress" the action space to be denser — this would break the `ActionMapper` trait contract, the pre-allocated buffer sizes, and any trained model checkpoints.
- Perspective flipping (White sees board rotated 180 degrees) handled inside the mapper
- Users can implement custom `ActionMapper` for alternative encodings
- `action_space_size()` is queried at `VecEnv` construction to allocate the legal mask buffer

### ObservationGenerator (trait-based, configurable)

```rust
trait ObservationGenerator: Send + Sync {
    fn generate(&self, state: &GameState, buffer: &mut [f32]);
    fn channels(&self) -> usize;
}
```

- **DefaultObservationGenerator**: 46-channel layout matching current Keisei implementation
- Writes directly into batch buffer slice — no intermediate allocation
- Perspective handling (board flip for White) during generation
- `channels()` is queried at `VecEnv` construction to allocate the observation buffer

### Zero-copy FFI

- Observation and mask buffers are Rust-owned, exposed to Python as NumPy arrays via `PyArray` (no copy on the CPU side)
- Action arrays received as NumPy views — Rust reads directly
- **Buffer lifetime and safety:** By default, `step()` returns **copies** of the observation and mask buffers. This is safe — Python can store them freely without worrying about overwrites. For users who understand the lifetime constraint and want maximum performance, `step(zero_copy=True)` returns views into the Rust-owned buffers directly. In zero-copy mode, buffers are overwritten on the next `step()` call — holding a reference across steps produces stale data. The default-copy cost at N=512 is ~14 MB of memcpy, negligible compared to GPU inference time. Documentation includes examples of both the correct and incorrect usage patterns.
- PyO3 + maturin for binding generation

### Dependencies

- `shogi-core` (workspace dependency)
- `pyo3` — Python bindings
- `numpy` (Rust crate) — zero-copy NumPy array interop
- `rayon` — parallel iteration for VecEnv stepping
- `criterion` (dev dependency) — Rust benchmarking

## Integration with Keisei

### What Changes

- **`EnvManager`**: swaps `ShogiGame()` + `PolicyOutputMapper()` for `shogi_gym.VecEnv(num_envs=N)`. Config gains `env.num_envs`, `env.backend`, and `env.parallel` parameters.
- **`StepManager`**: becomes a batch stepper. `execute_step()` becomes `execute_batch_step()`. Feeds N actions, gets N results. Experience buffer receives N experiences per call.
- **`PolicyOutputMapper`**: deleted from Python. Logic moves to `shogi-gym`'s `DefaultActionMapper`.
- **Evaluation system**: uses `SpectatorEnv` for human-viewable games, small `VecEnv` for batch evaluation (e.g., 50 games in parallel).
- **Spectator dashboard**: uses `SpectatorEnv.to_dict()` instead of reaching into `ShogiGame` internals.

### What Stays the Same

- PPO agent, neural networks, model architecture
- Training loop structure, callback system, metrics
- Config system (just new `env.*` fields)
- Checkpoint format (model weights are engine-independent)

### Migration Path

1. Python `keisei/shogi/` package stays in place during development
2. Config flag `env.backend: "python" | "rust"` switches between engines
3. Cross-validation test harness runs both engines on identical positions, asserts identical legal move sets (order-independent) and identical observation tensors (within float tolerance)
4. **Record and replay**: during training with the Python engine, log action index sequences for a set of games. Replay those exact sequences through the Rust engine and assert identical game trajectories (same captures, same terminal states, same ply counts). This catches subtle differences in action encoding or edge-case rule handling that position-by-position comparison might miss.
5. Once verified, Python engine becomes a reference/test oracle
6. Eventually removable once Rust engine is proven stable

## Showcase: Tournament Display

Since the engine is policy-agnostic, tournaments can pit different agents against each other using `SpectatorEnv`:

- Random agent vs. trained PPO model
- Small CNN vs. large ResNet
- Different training checkpoints (early vs. late training)
- Future: MCTS-augmented agents

The engine treats them all identically — it just takes moves. Agent selection and tournament orchestration stays in Python.

## Directory Structure

```
shogi-engine/                    # top-level in keisei repo (future: own repo)
├── Cargo.toml                   # workspace root
├── crates/
│   ├── shogi-core/
│   │   ├── Cargo.toml           # pure Rust, std only, zero external deps
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── types.rs         # Color, PieceType, HandPieceType, Square, Move, GameResult
│   │       ├── board.rs         # Board, Piece (NonZeroU8), AttackMap, PawnColumns
│   │       ├── position.rs      # Position (board + hands + player + hash)
│   │       ├── game.rs          # GameState (Position + history + rules)
│   │       ├── movegen.rs       # move generation (pseudo-legal + legality filter)
│   │       ├── rules.rs         # special rules (uchi-fu-zume, nifu, sennichite, impasse)
│   │       ├── zobrist.rs       # Zobrist hash tables and incremental updates
│   │       ├── sfen.rs          # SFEN parse/serialize
│   │       └── error.rs         # ShogiError enum
│   └── shogi-gym/
│       ├── Cargo.toml           # depends on shogi-core, pyo3, numpy, rayon
│       ├── pyproject.toml       # maturin build config
│       └── src/
│           ├── lib.rs           # PyO3 module definition
│           ├── vec_env.rs       # VecEnv batch environment
│           ├── spectator.rs     # SpectatorEnv for display games
│           ├── action_mapper.rs # ActionMapper trait + DefaultActionMapper
│           └── observation.rs   # ObservationGenerator trait + default 46-channel
└── python/
    └── shogi_gym/
        ├── __init__.py          # re-exports from native module
        └── py.typed             # PEP 561 type stub marker
```

## Build & Tooling

- **Local dev**: `maturin develop` — builds and installs into active venv
- **Release**: `maturin build --release` — production wheels
- **Rust tests**: `cargo test` in workspace root
- **Rust benchmarks**: `cargo bench` (uses `criterion`)
- **Python tests**: `pytest` against the built module

## Testing Strategy

### `shogi-core` (Rust-native)

- Legal move correctness for known positions (opening, midgame, endgame)
- SFEN round-trip fidelity on `Position`
- Attack map correctness: property-based tests asserting incremental update == from-scratch recomputation after every make/unmake
- Zobrist hash: verify incremental updates match full recomputation; statistical collision rate checks
- Special rule edge cases: uchi-fu-zume with pinned defenders, nifu attempts, sennichite sequences, perpetual check detection, impasse point counting, dead drop prevention, forced promotion scenarios

### Cross-validation (Python, migration safety net)

- Run both Python and Rust engines on identical positions
- Assert identical legal move sets (order-independent)
- Assert identical observation tensors (within float tolerance)
- Record-and-replay: log action sequences from Python engine games, replay through Rust engine, assert identical trajectories
- Run on thousands of random game positions for confidence

### `shogi-gym` (Python integration)

- VecEnv: observation shape/dtype/strides, legal mask correctness, auto-reset behavior, terminal observation buffer
- SpectatorEnv: step/reset lifecycle, to_dict() completeness
- ActionMapper: encode/decode round-trip for all 13,527 actions, perspective flip correctness
- Illegal action handling: verify panic in debug, exception in release
- Performance: end-to-end step rate benchmarks

### Performance Benchmarks

**Rust-native (criterion):**
- Move generation throughput: positions/second for opening, midgame, endgame positions
- Make/unmake cycle time (nanoseconds)
- Attack map incremental update vs from-scratch (nanoseconds)
- Observation generation throughput per position

**End-to-end (Python):**
- VecEnv batch step throughput at N = 64, 256, 512, 1024 (env-steps/second)
- Target: engine step time < GPU inference time at all tested N values (i.e., GPU is always the bottleneck)
- Maximum allocations after warmup: zero on the hot path
- Full PPO training loop it/s comparison: Python engine vs Rust engine on identical config

**Baseline:** current Python engine at ~0.5 it/s. Target: sufficient throughput that the 2x RTX 4060 GPUs are saturated during training. Concrete env-steps/sec targets will be established after initial benchmarking of the Rust engine on target hardware — the goal is "GPU never waits for the engine," not an arbitrary multiplier.
