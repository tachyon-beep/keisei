Absolutely. Here’s a practical spec for a **Keisei Style and Commentary Stats System** that turns league checkpoints from “ranked snapshots” into recognisable opponents with personality.

# Keisei Style and Commentary Stats System

Version 0.1
Status: Draft specification

## 1. Purpose

The system computes **play-style descriptors**, **signature tendencies**, and **rotating colour-commentary facts** for each league checkpoint.

It exists to answer questions like:

* How strong is this model?
* What kind of Shogi does it like to play?
* What makes it different from the others?
* Which checkpoint should I use for a given difficulty tier or vibe?

This is explicitly **not** an attempt to interpret the neural network internals. It is a **behavioural profiling system** built from game logs and league match data.

## 2. Product goals

The system should:

1. Give each checkpoint a recognisable identity beyond Elo.
2. Surface style in a way that is legible to normal players.
3. Support “pick an opponent by flavour” as well as “pick an opponent by rank”.
4. Generate lightweight, rotating commentary for the UI.
5. Avoid fake anthropomorphism by grounding everything in observed match data.
6. Remain robust when new checkpoints enter the pool and league distributions shift.

## 3. Non-goals

The system will not:

* claim formal engine interpretability
* claim absolute human-style opening classification unless explicitly implemented
* claim stable personality from tiny sample sizes
* replace Elo or head-to-head data
* present league-relative traits as universal truths

## 4. Core concepts

Each checkpoint gets four layers of profile data:

### 4.1 Identity

Simple factual descriptors:

* name
* epoch
* Elo
* games played
* age in league
* current rank
* movement in rank since previous snapshot

### 4.2 Style profile

Behavioural tendencies computed from games:

* opening preferences
* aggression/tempo markers
* drop/promotion tendencies
* game-length tendency
* rook/king development behaviour
* tactical volatility

### 4.3 Signature labels

Human-readable summaries derived from the metrics:

* “Patient attacker”
* “Drop-heavy scrapper”
* “Slow builder”
* “Sharp tactical opener”
* “Long-game grinder”

These must be generated from rules over measurable features, not written by hand.

### 4.4 Colour commentary

Short rotating facts or blurbs:

* “Usually opens with P-7f as Black”
* “Shorter games than 82% of the league”
* “Promotes aggressively once ahead”
* “Rarely repeats positions”
* “Most likely to swing the rook early”

These are UI-facing, small, and disposable.

## 5. Data sources

The system consumes:

* league match logs
* game records with per-ply move history
* checkpoint metadata
* head-to-head results
* optional per-move policy metadata if logged later

### Required minimum inputs per game

For both sides:

* checkpoint ID
* opponent checkpoint ID
* side to move
* result
* total plies
* complete move list in machine-readable form
* timestamps or epoch context

### Optional future inputs

* legal move count per position
* policy entropy
* value estimate per move
* evaluation swing
* time taken per move
* repetition flags
* material balance trace

## 6. Profiling windows

Each checkpoint profile should be computed over one or more windows:

### 6.1 Lifetime window

All league games for that checkpoint.

Used for:

* stable identity
* long-term style
* public-facing summary

### 6.2 Recent window

Most recent N games, default 100.

Used for:

* trend detection
* “currently playing more aggressively”
* drift warnings

### 6.3 Side-specific windows

Separate Black and White profiles where useful.

Used for:

* preferred opening move as Black
* preferred response as White

## 7. Minimum sample thresholds

To avoid nonsense:

* Under 25 games: no style labels, only raw stats
* 25 to 74 games: provisional style profile
* 75+ games: full style labels and commentary
* 200+ games: eligible for trend and confidence annotations

UI should visibly distinguish:

* provisional
* established

## 8. Feature extraction

## 8.1 Opening features

Derived from first 6 to 12 plies.

Metrics:

* most common first move as Black
* most common reply as White
* top 3 opening sequences by frequency
* first rook displacement ply
* king displacement by ply 20
* early static-vs-ranging proxy

### Suggested derived fields

* `preferred_first_move_black`
* `preferred_first_move_white_response`
* `rook_moved_early_rate`
* `king_development_speed`
* `opening_diversity_index`

## 8.2 Tempo and aggression features

Metrics:

* average ply of first capture
* average ply of first check
* average ply of first drop
* checks per 100 plies
* captures per 100 plies
* proportion of short games
* proportion of decisive games

### Derived fields

* `first_capture_ply_mean`
* `first_check_ply_mean`
* `checks_per_100_plies`
* `capture_rate`
* `short_game_rate`

## 8.3 Drop and promotion behaviour

Metrics:

* drops per game
* drop rate by phase
* promotions per game
* promotion rate when available
* proportion of games with early drops

### Derived fields

* `drops_per_game`
* `promotion_rate`
* `early_drop_rate`

## 8.4 Positional style proxies

These are approximate, not deep engine truths.

Metrics:

* rook mobility in first 20 plies
* king displacement in first 30 plies
* piece-drop concentration by file
* piece activity spread across board
* repetition tendency
* average game length

### Derived fields

* `rook_mobility_score`
* `king_safety_bias`
* `board_spread_score`
* `repetition_rate`
* `avg_game_length`

## 8.5 Volatility and consistency

Metrics:

* variance in game length
* variance in result by opponent band
* frequency of extreme short wins/losses
* head-to-head upset rate

### Derived fields

* `style_volatility`
* `consistency_score`
* `upset_rate`

## 9. Normalisation model

All style metrics should be normalised **relative to the current league population**, not treated as absolute truths.

For each numeric metric:

* compute league mean
* compute league standard deviation
* compute percentile rank
* store both raw and normalised values

Public labels should usually be driven by:

* percentile bands
* z-scores
* minimum-sample rules

Example:

* “High drop usage” = drops per game above 75th percentile
* “Long-game player” = average game length above 70th percentile
* “Fast starter” = first capture ply below 30th percentile

## 10. Style labels

Each checkpoint may have:

* 1 primary style label
* up to 2 secondary traits

### 10.1 Label generation model

Use a rule-based classifier in v1.

Example rules:

**Patient attacker**

* long-game percentile >= 65
* checks per 100 plies >= 55
* first capture ply not especially early
* consistency score moderate or high

**Chaotic brawler**

* short-game percentile >= 70
* capture rate high
* volatility high
* drop rate high or check rate high

**Slow builder**

* king development early but low early capture rate
* long games
* low volatility

**Drop-heavy scrapper**

* drops per game >= 80th percentile
* early drop rate >= 65th percentile

**Flexible opener**

* opening diversity >= 75th percentile

### 10.2 Label constraints

* never assign contradictory labels
* never show more than 3 labels
* prefer stability over novelty
* do not churn labels every recompute unless thresholds are clearly crossed

## 11. Commentary fact generation

Each checkpoint should expose a rotating pool of short commentary items.

### 11.1 Commentary types

* opening preference
* tempo tendency
* tactical tendency
* game-length tendency
* rank movement
* head-to-head quirk
* signature stat

### 11.2 Examples

* Usually opens with P-7f as Black
* Shorter games than most league rivals
* Drops pieces early more often than average
* Rarely shifts the rook in the opening
* More draw-prone than nearby rivals
* Has climbed 4 places in the last 3 epochs
* Performs unusually well against higher-ranked opponents

### 11.3 Rotation rules

* show 1 to 3 commentary items at once
* rotate from a candidate pool
* avoid repeating the same category twice in a row
* prefer higher-confidence facts
* suppress stale or weak facts

## 12. Confidence and eligibility

Each generated fact or label gets:

* confidence tier: low / medium / high
* sample size
* recency basis: lifetime / recent

Public UI should only show:

* medium or high confidence items by default

Low-confidence items may still be stored for debugging.

## 13. Data model

### 13.1 CheckpointProfile

```json
{
  "checkpoint_id": "epoch_305",
  "name": "Arashi",
  "elo": 1232,
  "games_played": 65,
  "rank": 1,
  "profile_status": "provisional",
  "primary_style": "Sharp tactical opener",
  "secondary_traits": ["High drop usage", "Short games"],
  "stats": {
    "avg_game_length": 81.4,
    "drops_per_game": 2.8,
    "promotion_rate": 0.62,
    "first_capture_ply_mean": 18.1,
    "rook_mobility_score": 0.71,
    "opening_diversity_index": 0.33
  },
  "percentiles": {
    "avg_game_length": 22,
    "drops_per_game": 81,
    "promotion_rate": 64,
    "first_capture_ply_mean": 19
  },
  "commentary": [
    {
      "text": "Usually opens with P-7f as Black",
      "category": "opening",
      "confidence": "high"
    },
    {
      "text": "Starts exchanging early and often",
      "category": "tempo",
      "confidence": "medium"
    }
  ]
}
```

## 14. Processing pipeline

### Stage 1: ingest

Read league games and checkpoint metadata.

### Stage 2: feature extraction

Compute per-game feature rows.

### Stage 3: aggregate

Group by checkpoint and by side, compute lifetime and recent aggregates.

### Stage 4: normalise

Compute percentiles and z-scores relative to the current league pool.

### Stage 5: classify

Assign style labels using rule-based thresholds.

### Stage 6: generate commentary

Create ranked candidate facts, filter by confidence, rotate for UI.

### Stage 7: publish

Store to DB or JSON blob used by the site.

## 15. Storage

Two persisted layers are recommended.

### 15.1 Raw per-game features

Append-only table for reproducibility.

Suggested table:
`checkpoint_game_features`

Fields:

* checkpoint_id
* game_id
* side
* opponent_id
* result
* total_plies
* first_capture_ply
* first_check_ply
* first_drop_ply
* num_drops
* num_promotions
* rook_moved_early
* king_displacement_20
* opening_token
* opening_sequence_token
* repetition_flag
* created_at

### 15.2 Aggregated checkpoint profiles

Materialised table:
`checkpoint_style_profiles`

Fields:

* checkpoint_id
* recomputed_at
* profile_status
* raw_metrics_json
* percentile_metrics_json
* primary_style
* secondary_traits_json
* commentary_json

## 16. UI specification

## 16.1 Opponent card

Extend the current card with:

* Primary style
* Secondary traits
* Opening tendency
* One rotating commentary line

Example:

* Style: Sharp tactical opener
* Traits: High drop usage, Short games
* Opens: Usually P-7f
* Commentary: Exchanges begin earlier than most rivals

## 16.2 League table

Add optional compact columns or tooltips for:

* style
* opening
* commentary snippet

### Good minimal version

Hover on checkpoint name shows:

* primary style
* preferred opening
* average game length
* drop rate percentile

## 16.3 Dedicated checkpoint detail panel

When selected, show:

* rank and Elo
* style labels
* radar or bars for 5 to 6 core traits
* top commentary facts
* common opening moves
* recent-vs-lifetime change

## 16.4 Commentary rotation behaviour

The card should rotate commentary on:

* page refresh
* model selection change
* timed interval, default 15 to 30 seconds

Do not rotate so fast that it feels twitchy.

## 17. Public wording rules

The UI language should be grounded and league-relative.

Prefer:

* “More drop-heavy than most league rivals”
* “Usually opens with...”
* “Tends towards shorter games”

Avoid:

* “Loves chaos”
* “Is aggressive” with no basis
* “Prefers X opening” if sample size is weak
* “Human-like” unless deliberately and separately measured

Fun flavour text can still exist, but it should sit beside the real profile, not replace it.

## 18. v1 implementation scope

For a first working release, implement only these metrics:

1. preferred first move as Black
2. preferred response as White
3. average game length
4. first capture ply
5. drops per game
6. promotions per game
7. rook moved early rate
8. opening diversity index

And only these outputs:

* 1 primary style label
* up to 2 secondary traits
* 3 to 5 commentary facts per checkpoint
* compact card integration
* JSON export for frontend

That is enough to make the system feel real.

## 19. v2 extensions

Later additions:

* side-specific style labels
* recent-trend commentary
* head-to-head speciality tags
* opening family clustering
* cluster-based archetypes
* policy entropy based “creativity” or “narrowness”
* comeback/conversion metrics
* “best against” and “struggles against” opponent notes

## 20. Acceptance criteria

The system is successful when:

1. Every sufficiently sampled checkpoint has a stable, data-grounded style profile.
2. The same checkpoint usually keeps the same broad identity between recomputes unless its behaviour genuinely shifts.
3. Commentary lines feel informative rather than random.
4. Two nearby Elo checkpoints can still feel distinct.
5. A player can choose an opponent by both strength and style.
6. Developers can inspect why a label was assigned.

## 21. Example UI output

### Arashi

* Elo 1232
* Style: Sharp tactical opener
* Traits: High drop usage, Short games
* Opens: Usually P-7f as Black
* Commentary:

  * Starts exchanging earlier than most of the league
  * Uses drops more heavily than nearby rivals
  * Keeps games shorter than average

### Hana

* Elo 1229
* Style: Patient attacker
* Traits: Long games, Stable opening choices
* Opens: Usually develops conservatively
* Commentary:

  * Wins tend to come in longer games
  * Promotes steadily rather than early
  * Less volatile than nearby rivals

## 22. Design principle

This system should make a checkpoint feel like:

* a real opponent
* a reusable game-AI difficulty tier
* a member of a living league

Not just a frozen tensor blob with a number next to it.

If you want, I can turn this into a tighter engineering spec with schema definitions and label rules written as explicit pseudocode.
