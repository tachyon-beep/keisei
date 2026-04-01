# KataGo-Style SE-ResNet Architecture Design

**Date:** 2026-04-01
**Status:** Approved
**Scope:** Full-stack architecture change — Rust observation/action encoding, Python model, multi-head PPO, supervised learning pipeline

## Overview

Replace the current vanilla ResNet (46-channel input, scalar value, flat 13527 policy) with a KataGo-inspired SE-ResNet architecture featuring expanded observations, spatial policy encoding, and multi-head output (W/D/L value + score lead). Includes a supervised learning warmup pipeline for bootstrapping from human game records.

The new architecture lives alongside the existing one (new base class, new PPO variant, new training loop) — existing models remain functional as baselines for ablation.

## Implementation Strategy

Vertical slices, each delivering a testable increment:

1. **Rust observation expansion** (46 → 50 channels)
2. **Rust spatial action mapper** (13,527 → 11,259 spatial actions)
3. **Python SE-ResNet model** (new `KataGoBaseModel` + `SEResNetModel`)
4. **Python multi-head PPO** (W/D/L value, score lead, adapted GAE)
5. **End-to-end RL integration** (config, registry, training loop)
6. **Supervised learning pipeline** (game parsing, data preparation, SL trainer)

Slices 1-2 (Rust) and 3-4 (Python) are parallelizable. Slice 5 integrates them. Slice 6 is independent once the model interface exists.

---

## Slice 1: Observation Space Expansion (Rust)

### Channel Layout (50 channels)

| Channels | Content | Encoding |
|---|---|---|
| 0-7 | Current player's unpromoted pieces | Binary (0/1), 8 types |
| 8-13 | Current player's promoted pieces | Binary, 6 types |
| 14-21 | Opponent's unpromoted pieces | Binary, 8 types |
| 22-27 | Opponent's promoted pieces | Binary, 6 types |
| 28-34 | Current player's hand counts | Normalized float, 7 types |
| 35-41 | Opponent's hand counts | Normalized float, 7 types |
| 42 | Player indicator | 1.0=Black, 0.0=White |
| 43 | Move count | ply / max_ply |
| 44-47 | Repetition count | 4 binary planes: 1x, 2x, 3x, 4x |
| 48 | Check indicator | 1.0 if current player is in check |
| 49 | Reserved | All zeros (earmarked: last move destination) |

### Design Decisions

- **Repetition as binary planes, not scalar.** The thresholds matter discretely: 3 repetitions is categorically different from 2 (next repetition = draw). A scalar `rep_count/4` obscures this boundary.
- **Check indicator from current player's perspective.** "Am I in check?" — since observations are always from the current player's viewpoint, this falls out naturally.
- **Channel 49 reserved for last-move-destination.** A plane with 1.0 at the square the previous move landed on. Cheap, gives recency information, useful for detecting drop threats and piece sacrifices. Deferred to avoid scope creep.

### Implementation

- New `KataGoObservationGenerator` implementing the existing `ObservationGenerator` trait.
- `VecEnv` constructor gains `observation_mode: str` parameter (`"default"` | `"katago"`).
- `observation_channels` property reflects the chosen mode.
- Repetition tracking requires the `GameState` to maintain a position history hash table. The Rust `GameState` must track Zobrist hashes of prior positions to count repetitions.

### Test Cases

- All existing observation tests pass unchanged for `"default"` mode.
- Channel 44 is 1.0 only after the same position occurs exactly once before.
- Channel 47 is 1.0 only at the 4th repetition (sennichite).
- Channels 44-47 are mutually exclusive (exactly one is set per repetition count, or all zero).
- Channel 48 is 1.0 when the current player is in check (construct a position with check, verify).
- Channel 48 is 0.0 when not in check.
- Channel 49 is all zeros.
- No NaN anywhere in the buffer for any valid position.

---

## Slice 2: Spatial Action Mapper (Rust)

### Encoding: (9 x 9 x 139) = 11,259 actions

Moves indexed by source square (board moves) or destination square (drops), with 139 move types per square.

| Slots | Content | Count |
|---|---|---|
| 0-63 | Sliding moves: 8 directions x 8 distances | 64 |
| 64-127 | Sliding moves with promotion: 8 directions x 8 distances | 64 |
| 128 | Knight jump left (non-promote) | 1 |
| 129 | Knight jump left (promote) | 1 |
| 130 | Knight jump right (non-promote) | 1 |
| 131 | Knight jump right (promote) | 1 |
| 132-138 | Drop to this square: 7 piece types | 7 |

**Total: 139 move types x 81 squares = 11,259**

### Direction Encoding

Compass directions relative to current player's perspective (N = toward opponent):

| Index | Direction | (dr, dc) |
|---|---|---|
| 0 | N | (-1, 0) |
| 1 | NE | (-1, +1) |
| 2 | E | (0, +1) |
| 3 | SE | (+1, +1) |
| 4 | S | (+1, 0) |
| 5 | SW | (+1, -1) |
| 6 | W | (0, -1) |
| 7 | NW | (-1, -1) |

Non-sliding pieces (Gold, Silver, King, promoted pieces) encoded as direction + distance=1.

### Flat Index Contract

**`flat_index = square * 139 + move_type`**

This ordering is load-bearing — the Python model outputs `(B, 9, 9, 139)` which flattens to `(B, 11259)` using this convention. The Rust legal mask is generated in this order. Documented and tested at the boundary.

### Knight Encoding

Knights get dedicated slots (128-131) rather than direction+distance because the L-shaped jump doesn't decompose into compass directions. Two forward targets (left/right), each with promote/non-promote variants.

### Drop Encoding

Slots 132-138 indexed by **destination** square (drops have no source). A drop to e5 sets `mask[e5_idx][132 + piece_type] = true`. Board moves *to* e5 are encoded in the *source* square's direction slots — no collision.

### Perspective Flipping

White's moves flip squares via `80 - idx` before encoding. Applies to both source and destination for board moves, and to destination for drops. Explicit code path for drops to prevent accidentally skipping the flip due to "no source square."

### Implementation

- `SpatialActionMapper` implements the existing `ActionMapper` trait with `action_space_size() = 11259`.
- `VecEnv` constructor gains `action_mode: str` parameter (`"default"` | `"spatial"`).
- Legal masks shape: `(num_envs, 11259)` when spatial, reshaped to `(num_envs, 9, 9, 139)` Python-side.

### Test Matrix

- Rook on e5, unobstructed: 4 directions x distances up to board edge.
- Lance on 9a (promotion zone boundary): forced promotion slot only, not both.
- Knight on 8a: only right jump legal (left off-board), 2 slots not 4.
- Drop pawn: illegal on opponent's back rank, illegal if own pawn on file.
- White piece moves: verify flipped indices produce mirrored board positions.
- Round-trip: `move -> action_idx -> move` reconstruction for all move types.
- Exhaustive: all legal moves in starting position encode/decode correctly.
- Cross-position: round-trip over thousands of positions from test corpus.

---

## Slice 3: SE-ResNet Model (Python)

### New Base Class

```python
@dataclass
class KataGoOutput:
    policy_logits: Tensor  # (B, 9, 9, 139) spatial, raw, unmasked
    value_logits: Tensor   # (B, 3) W/D/L logits (pre-softmax)
    score_lead: Tensor     # (B, 1) predicted point advantage

class KataGoBaseModel(ABC, nn.Module):
    """Contract:
    - Input: (batch, obs_channels, 9, 9)
    - Output: KataGoOutput
    """
```

Lives alongside existing `BaseModel`. Existing models unchanged.

### SE-ResNet Architecture

```
Input (B, 50, 9, 9)
    |
Input Conv: 50 -> C, 3x3, BN, ReLU
    |
SE-ResBlock x N
    |   conv1: C->C, 3x3, BN, ReLU
    |   global_pool(block_input): mean+max+std -> FC -> C  [pools from INPUT x, not post-conv1]
    |   add global bias to conv1 output
    |   conv2: C->C, 3x3, BN
    |   SE: pool(block_output) -> FC(C/r) -> ReLU -> FC(2C) -> split
    |       -> scale (sigmoid) + shift
    |   residual add + ReLU
    |
    +------------------+------------------+
    |                  |                  |
Policy Head       Value Head         Score Head
Conv C->32, 1x1   Global pool        Global pool
BN, ReLU          (mean+max+std)     (mean+max+std)
Conv 32->139,1x1  FC(3C -> 256)      FC(3C -> 128)
permute to         ReLU              ReLU
(B,9,9,139)       FC(256 -> 3)      FC(128 -> 1)
                  raw logits         raw scalar
```

### Key Design Decisions

1. **Policy head: two conv layers (C->32->139).** Intermediate 32-channel conv gives the head its own feature space for spatial move patterns before projecting to 139 move types.

2. **Value head: global pool, not spatial flatten.** Value is a global property — no reason to preserve spatial structure. Uses mean+max+std pooling of full C-channel trunk, matching the score head's pattern. FC input is `3C` (mean, max, std each contribute C features).

3. **Score head: same global pool pattern.** Point advantage is global. FC(3C -> 128 -> 1).

4. **Global pool bias from block input.** The SE blocks pool from `x` (the residual input), not post-conv1 `out`. Global context reflects the stable state entering the block, more robust during early training when conv1 weights are noisy.

5. **SE produces scale AND shift.** `FC -> 2C -> split` into scale (sigmoid) and shift (additive). Scale alone can suppress/amplify channels; shift can inject new signal based on global state (e.g., "rooks in hand -> bias attack channels").

### Configurable Parameters

```python
@dataclass(frozen=True)
class SEResNetParams:
    num_blocks: int             # 20, 40, 60, 80
    channels: int               # 128, 256, 320, 384
    se_reduction: int           # SE squeeze ratio, default 16
    global_pool_channels: int   # global pool FC width in SE blocks, default 128
    policy_channels: int        # intermediate policy conv channels, default 32
    value_fc_size: int          # value head hidden dim, default 256
    score_fc_size: int          # score head hidden dim, default 128
    obs_channels: int           # input channels, default 50
```

### Invariant Assertions

At model `__init__` time:
```python
assert obs_channels > 0
assert num_blocks > 0
assert channels > 0
assert se_reduction > 0 and channels % se_reduction == 0
```

At forward time (first call):
```python
assert obs.shape[1] == self.params.obs_channels, (
    f"Expected {self.params.obs_channels} input channels, got {obs.shape[1]}"
)
```

### Default Configurations

| Config | Blocks | Channels | Params (approx) |
|---|---|---|---|
| Validation (b20c128) | 20 | 128 | ~5M |
| Baseline (b40c256) | 40 | 256 | ~25M |
| Scale-up (b60c320) | 60 | 320 | ~55M |
| Maximum (b80c384) | 80 | 384 | ~100M |

---

## Slice 4: Multi-Head PPO (Python)

### KataGoRolloutBuffer

Extends storage for multi-head targets:

```python
class KataGoRolloutBuffer:
    # Same as RolloutBuffer:
    observations, actions, log_probs, rewards, dones, legal_masks

    # New:
    value_categories: list[Tensor]  # (num_envs,) int {0=W, 1=D, 2=L}
    score_targets: list[Tensor]     # (num_envs,) float, normalized [-1, 1]
```

Value categories are backfilled at episode end from game outcome. Score targets are final material difference normalized by max material (76 points).

### Scalar Value Projection for GAE

The 3-way value maps to a scalar via `P(W) - P(L)`:

```python
value_probs = softmax(output.value_logits)  # (B, 3)
scalar_value = value_probs[:, 0] - value_probs[:, 2]  # bounded [-1, 1]
```

This projection is applied consistently:
- During action selection (for buffer storage)
- At bootstrap (next_values computation at rollout end)
- Nowhere else — GAE operates on these scalars, reusing existing `compute_gae` unchanged

### Loss Function

```
L = lambda_policy  * clipped_surrogate_loss(pi)
  + lambda_value   * cross_entropy(value_logits, value_target)
  + lambda_score   * mse(score_pred, score_target)
  - lambda_entropy * entropy(pi)
```

Default weights: `lambda_policy=1.0, lambda_value=1.5, lambda_score=0.02, lambda_entropy=0.01`

All weights configurable via TOML `algorithm_params`.

### Key Design Decisions

1. **No PPO value clipping.** Standard PPO clips value updates via `v_clipped = old_v + clip(v_new - old_v, -eps, eps)`. With cross-entropy value loss (not MSE), value clipping is ill-motivated — it was designed for regression. Documented as intentional.

2. **Score targets: outcome-conditioned.** Final material difference backfilled uniformly through all positions in an episode. This means the score head learns "given this position, what's the expected final material margin" — a reasonable auxiliary signal. Not position-conditional (early positions don't have independent score estimates during RL). Documented for future improvement during SL warmup.

3. **Score normalization at buffer level.** Raw material diff divided by 76 (max possible material) to normalize to [-1, 1]. Applied when adding to buffer, not inside the model, so the raw score head output remains interpretable.

4. **Piece values for material scoring:** P=1, L=3, N=3, S=4, G=5, B=6, R=8. Standard Shogi piece values.

5. **NaN guard on entropy.** Assert no NaN in policy logits before entropy computation. Prevents silent NaN propagation from degenerate all-masked positions.

### Metrics

Existing metrics (`policy_loss`, `value_loss`, `entropy`, `gradient_norm`) plus:

| Metric | Description |
|---|---|
| `score_loss` | MSE on normalized score predictions |
| `value_accuracy` | Fraction where argmax(W/D/L) matches outcome |
| `frac_predicted_win` | Fraction of predictions with argmax=W |
| `frac_predicted_draw` | Fraction with argmax=D |
| `frac_predicted_loss` | Fraction with argmax=L |
| `mean_score_error` | Mean absolute error of score predictions |

The prediction breakdown (`frac_predicted_*`) detects degenerate optimization — if `value_accuracy` plateaus at ~65% with 99% Win predictions, the model isn't learning.

### KataGoPPOParams

```python
@dataclass(frozen=True)
class KataGoPPOParams:
    learning_rate: float = 2e-4
    gamma: float = 0.99          # revisit: may raise to 1.0 for undiscounted
    clip_epsilon: float = 0.2
    epochs_per_batch: int = 4
    batch_size: int = 256
    lambda_policy: float = 1.0
    lambda_value: float = 1.5
    lambda_score: float = 0.02
    lambda_entropy: float = 0.01
    score_normalization: float = 76.0
    grad_clip: float = 1.0           # looser than SL (0.5) — policy needs room to move
```

Note on `gamma = 0.99`: at this discount, a reward 100 steps away is worth ~37% of face value. Shogi games average 80-120 moves, so terminal rewards are significantly discounted at the halfway point. This favors shorter wins over sound positional play. Starting at 0.99 for training stability; raise toward 1.0 if the agent becomes overly aggressive.

---

## Slice 5: End-to-End RL Integration

### Config Schema

```toml
[model]
display_name = "KataGo-SE-b40c256"
architecture = "se_resnet"

[model.params]
num_blocks = 40
channels = 256
se_reduction = 16
global_pool_channels = 128
policy_channels = 32
value_fc_size = 256
score_fc_size = 128
obs_channels = 50

[training]
algorithm = "katago_ppo"
num_games = 128
max_ply = 512

[training.algorithm_params]
learning_rate = 0.0002
gamma = 0.99
clip_epsilon = 0.2
epochs_per_batch = 4
batch_size = 256
lambda_policy = 1.0
lambda_value = 1.5
lambda_score = 0.02
lambda_entropy = 0.01
score_normalization = 76.0
grad_clip = 1.0              # looser than SL (0.5) — policy needs room to move

[training.algorithm_params.lr_schedule]
type = "plateau"
monitor = "value_loss"       # explicitly: reduce when value_loss plateaus
factor = 0.5
patience = 50
min_lr = 0.00001

[training.algorithm_params.rl_warmup]
epochs = 5                   # epochs with elevated entropy after SL->RL transition
entropy_bonus = 0.05         # elevated from default 0.01
reset_lr_schedule = true     # reset plateau patience after warmup ends
```

Note: The plateau scheduler must be reset when RL warmup ends. During warmup, elevated entropy causes value loss to spike as the policy softens — these epochs would consume plateau patience and trigger premature LR reduction at the transition. `reset_lr_schedule = true` excludes warmup epochs from plateau tracking.

### Registry Additions

- `VALID_ARCHITECTURES` gains `"se_resnet"`
- `VALID_ALGORITHMS` gains `"katago_ppo"`
- `model_registry.build_model()` routes `"se_resnet"` -> `SEResNetModel(SEResNetParams(...))`
- `algorithm_registry.validate_algorithm_params()` routes `"katago_ppo"` -> `KataGoPPOParams(...)`

### KataGoTrainingLoop

New training loop class (does not modify existing `TrainingLoop`):

```python
class KataGoTrainingLoop:
    def __init__(self, config: AppConfig):
        # VecEnv with new modes
        self.vecenv = VecEnv(
            num_envs=config.training.num_games,
            max_ply=config.training.max_ply,
            observation_mode="katago",
            action_mode="spatial",
        )

        # Startup assertions — fail fast on config mismatch
        model = build_model(config.model.architecture, config.model.params)
        assert self.vecenv.observation_channels == model.params.obs_channels, (
            f"VecEnv produces {self.vecenv.observation_channels} channels "
            f"but model expects {model.params.obs_channels}"
        )
        assert self.vecenv.action_space_size == 11259, (
            f"Expected spatial action space 11259, got {self.vecenv.action_space_size}"
        )
```

### VecEnv Interface

```python
env = VecEnv(
    num_envs=128,
    max_ply=512,
    observation_mode="katago",   # "default" | "katago"
    action_mode="spatial",       # "default" | "spatial"
)
```

Mode strings act as factory selectors on the Rust side. Properties `observation_channels` and `action_space_size` reflect the chosen modes.

### Checkpoint Metadata

Architecture and model params stored in checkpoint for compatibility detection:

```python
torch.save({
    'architecture': config.model.architecture,
    'model_params': dataclasses.asdict(model_params),
    'state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'step': global_step,
}, path)
```

At load time: `assert ckpt['architecture'] == config.model.architecture` before `load_state_dict`. Old checkpoints (without `architecture` field) are incompatible — clean break.

### What Stays Unchanged

- Database schema (metrics are key-value, new keys just appear)
- Spectator dashboard (game visualization independent of model architecture)
- Move notation helpers (decode via spatial mapper instead of default)
- Checkpoint save/load mechanism (same `torch.save`/`torch.load`, different state dict shape)

---

## Slice 6: Supervised Learning Pipeline

### Pipeline Architecture

```
Game Records (CSA/KIF/SFEN)
    |
GameParser (pluggable)
    |
Parsed Game: list of (position, move, outcome)
    |
PositionExtractor (parallelized via Rust rayon or Python multiprocessing)
    |  For each position:
    |  - Observation via KataGoObservationGenerator (50 channels)
    |  - Policy target: played move -> flat spatial action index
    |  - Value target: W/D/L from game outcome relative to side-to-move
    |  - Score target: final material difference, normalized by 76
    |
SLDataset (memory-mapped shards)
    |
DataLoader -> SLTrainer -> Model
```

### Game Parser Interface

```python
class GameOutcome(Enum):
    WIN_BLACK = "win_black"
    WIN_WHITE = "win_white"
    DRAW = "draw"

class ParsedMove:
    sfen_before: str        # position in SFEN notation
    move_usi: str           # move in USI notation

class GameRecord:
    moves: list[ParsedMove]
    outcome: GameOutcome
    metadata: dict[str, str]  # player names, ratings, date, source_format

class GameParser(ABC):
    @abstractmethod
    def parse(self, path: Path) -> Iterator[GameRecord]: ...

    @abstractmethod
    def supported_extensions(self) -> set[str]: ...
```

Concrete implementations: `SFENParser` (baseline), `CSAParser` (**required for Slice 6 completion** — Floodgate data is CSA format and is the primary SL data source), `KIFParser` (added incrementally post-Slice 6).

### Game Filtering

```python
class GameFilter:
    min_ply: int = 40
    min_rating: int | None = None

    def accepts(self, record: GameRecord) -> bool:
        if len(record.moves) < self.min_ply:
            return False
        if self.min_rating is not None:
            return self._check_rating(record)
        return True

    def _check_rating(self, record: GameRecord) -> bool:
        # Dispatch to source-specific rating comparison
        # Floodgate 2500 != 81dojo 2500 — normalize per source
        source = record.metadata.get("source_format", "unknown")
        rating = self._extract_rating(record)
        threshold = self._normalize_threshold(source, self.min_rating)
        return rating >= threshold
```

Rating filtering is source-aware. Different rating systems (Floodgate, 81dojo, CSA) have different scales. The filter normalizes thresholds per source rather than doing bare integer comparison.

### SL Dataset (Memory-Mapped Shards)

Per-position schema:
- `observation: float32[50 * 81]` = 16,200 bytes
- `policy_target: int64` = 8 bytes (flat spatial action index)
- `value_target: int64` = 8 bytes (0=W, 1=D, 2=L)
- `score_target: float32` = 4 bytes

**Total per position: ~16KB.** At 10M positions: ~160GB on disk.

**Legal masks are NOT stored.** During SL training, cross-entropy against the played move doesn't require masking (the human move is by definition legal). For SL eval (top-1 accuracy over legal moves), masks are generated on-the-fly from SFEN positions. This cuts dataset size by ~40% compared to storing masks.

Note: SL cross-entropy normalizes over all 11,259 logits (including illegal ones). This provides free regularization toward the legal-move distribution. The loss magnitude differs slightly from RL (which masks before softmax). Loss weight `lambda_policy` may need different tuning for SL vs RL.

### Data Preparation

CLI entrypoint: `keisei-prepare-sl`

1. Scans `game_sources` directories for game record files
2. Dispatches to appropriate `GameParser` by file extension
3. Filters by `GameFilter` criteria
4. For each position: generates observation via Rust engine, encodes targets
5. Writes memory-mapped `.bin` shards plus index file to `data_dir`

**Parallelization is explicit.** Preparing 10M positions single-threaded would take hours. The prepare script parallelizes via:
- Rust `rayon` for position encoding (preferred — stays in Rust for the hot loop)
- Python `multiprocessing.Pool` as fallback for game-level parallelism

### SL Trainer

```python
class SLTrainer:
    def __init__(self, model: KataGoBaseModel, config: SLConfig):
        self.optimizer = Adam(model.parameters(), lr=config.learning_rate)
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config.total_epochs
        )

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        for batch in dataloader:
            output = self.model(batch.observations)

            policy_loss = F.cross_entropy(
                output.policy_logits.reshape(B, -1), batch.policy_targets
            )
            value_loss = F.cross_entropy(
                output.value_logits, batch.value_targets
            )
            score_loss = F.mse_loss(
                output.score_lead.squeeze(-1), batch.score_targets
            )

            loss = (policy_loss
                    + 1.5 * value_loss
                    + 0.02 * score_loss)

            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
```

### SL Config

```toml
[sl_training]
data_dir = "data/processed/"
game_sources = ["data/raw/floodgate/"]
batch_size = 4096
learning_rate = 0.001
lr_schedule = "cosine"
warmup_epochs = 5
total_epochs = 30
num_workers = 8
min_rating = 2500
min_ply = 40
lambda_policy = 1.0
lambda_value = 1.5
lambda_score = 0.02
grad_clip = 0.5
```

### SL -> RL Transition

1. SL trainer saves checkpoint via standard checkpoint system (includes `architecture` metadata)
2. `KataGoTrainingLoop` loads checkpoint as starting point for RL self-play
3. First `rl_warmup.epochs` (default 5) of RL use elevated entropy bonus (`rl_warmup.entropy_bonus = 0.05`) to soften the overconfident SL policy before settling to default `lambda_entropy = 0.01`

SL produces one-hot policy targets (confident about one move). RL uses exploration (soft distribution). The elevated entropy during transition encourages the policy to broaden before RL fine-tuning narrows it to the MCTS-refined distribution.

### Smoke Test (Required Before Full SL Run)

Before committing to 160GB of preprocessed data, run a small-scale validation:

1. Prepare script on **1,000 games** (not 4M) — verify shards are well-formed
2. Train for **1 epoch** — verify all loss terms are finite and decreasing
3. Save checkpoint — verify `architecture` metadata is present
4. Load checkpoint into `KataGoTrainingLoop` — verify `obs_channels` and `action_space_size` assertions pass
5. Run **1 RL epoch** — verify the SL→RL handoff works end-to-end

This catches interface mismatches (especially the checkpoint architecture assertion and obs_channels guard) before investing compute in full preprocessing.

---

## Cross-Cutting Concerns

### Interface Contracts (Boundary Tests Required)

1. **Flat index ordering:** `flat_index = square * 139 + move_type`. Rust legal mask generation and Python model output reshaping must agree on this convention. Tested at the boundary.
2. **Observation channel count:** `VecEnv.observation_channels == model.params.obs_channels`. Asserted at startup.
3. **Action space size:** `VecEnv.action_space_size == 11259` when spatial. Asserted at startup.
4. **Score normalization:** Applied at buffer level (divide by `score_normalization`), not in model. Raw model output is interpretable as unnormalized material advantage.

### What Is NOT Changing

- `BaseModel` and existing ResNet/MLP/Transformer models
- `PPOAlgorithm` and `RolloutBuffer`
- `TrainingLoop` (existing)
- Database schema
- Spectator dashboard
- WebUI

### Naming Convention

| Existing | New |
|---|---|
| `BaseModel` | `KataGoBaseModel` |
| `ResNetModel` | `SEResNetModel` |
| `PPOAlgorithm` | `KataGoPPOAlgorithm` |
| `RolloutBuffer` | `KataGoRolloutBuffer` |
| `TrainingLoop` | `KataGoTrainingLoop` |
| `DefaultObservationGenerator` | `KataGoObservationGenerator` |
| `DefaultActionMapper` | `SpatialActionMapper` |
