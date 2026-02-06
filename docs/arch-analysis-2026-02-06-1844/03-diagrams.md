# Keisei Architecture Diagrams

**Date**: 2026-02-06
**Notation**: C4 Model (Context, Container, Component, Code)

---

## C4 Level 1: System Context

```
┌─────────────────────────────────────────────────────────────────┐
│                      External Systems                            │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Weights &   │  │   Browser    │  │  CUDA GPU             │  │
│  │  Biases      │  │  (Dashboard) │  │  (Training Accel.)    │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                 │                      │               │
└─────────┼─────────────────┼──────────────────────┼───────────────┘
          │ HTTPS           │ WebSocket/HTTP       │ CUDA API
          │                 │                      │
┌─────────▼─────────────────▼──────────────────────▼───────────────┐
│                                                                   │
│                      KEISEI DRL SYSTEM                            │
│                                                                   │
│   Deep Reinforcement Learning system that trains a Shogi-playing  │
│   agent via self-play using PPO. Provides real-time visualization │
│   via WebUI dashboard and experiment tracking via W&B.            │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
          │
          │ Reads/Writes
          ▼
┌──────────────────┐
│   File System    │
│  (Checkpoints,   │
│   Configs, Logs) │
└──────────────────┘
```

**Users**: ML researchers, Shogi AI enthusiasts, Twitch stream viewers
**External Dependencies**: W&B (optional), CUDA GPU (optional), Browser (optional)

---

## C4 Level 2: Container Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        KEISEI SYSTEM                                 │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    CLI Application                           │    │
│  │                    (train.py)                                │    │
│  │                                                              │    │
│  │  ┌──────────────────────────────────────────────────────┐   │    │
│  │  │              Training Pipeline                        │   │    │
│  │  │                                                       │   │    │
│  │  │  ┌─────────┐ ┌──────────┐ ┌───────────┐             │   │    │
│  │  │  │ Session  │ │  Model   │ │    Env    │             │   │    │
│  │  │  │ Manager  │ │  Manager │ │  Manager  │             │   │    │
│  │  │  └─────────┘ └──────────┘ └───────────┘             │   │    │
│  │  │  ┌─────────┐ ┌──────────┐ ┌───────────┐             │   │    │
│  │  │  │  Step   │ │ Training │ │  Metrics  │             │   │    │
│  │  │  │ Manager │ │   Loop   │ │  Manager  │             │   │    │
│  │  │  └─────────┘ └──────────┘ └───────────┘             │   │    │
│  │  │  ┌─────────┐ ┌──────────┐ ┌───────────┐             │   │    │
│  │  │  │ Display │ │ Callback │ │   Setup   │             │   │    │
│  │  │  │ Manager │ │  Manager │ │  Manager  │             │   │    │
│  │  │  └─────────┘ └──────────┘ └───────────┘             │   │    │
│  │  └──────────────────────────────────────────────────────┘   │    │
│  │                          │                                   │    │
│  │           ┌──────────────┼──────────────┐                   │    │
│  │           ▼              ▼              ▼                    │    │
│  │  ┌──────────────┐ ┌──────────┐ ┌─────────────┐             │    │
│  │  │  Core RL     │ │  Shogi   │ │ Evaluation  │             │    │
│  │  │  Engine      │ │  Engine  │ │   System    │             │    │
│  │  │  (PPO+Buf)   │ │  (Rules) │ │ (Strategies)│             │    │
│  │  └──────────────┘ └──────────┘ └─────────────┘             │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌────────────────────────┐  ┌────────────────────────────────┐     │
│  │   WebUI Server         │  │   Parallel Workers             │     │
│  │   (WebSocket + HTTP)   │  │   (Multi-process self-play)    │     │
│  │   Port 8765 / 8766     │  │   (Optional)                   │     │
│  └────────────────────────┘  └────────────────────────────────┘     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## C4 Level 3: Component Diagram - Training Pipeline

```
┌────────────────────────────────────────────────────────────────────────┐
│                         Trainer (Orchestrator)                          │
│                         trainer.py                                      │
└───────┬────────┬────────┬────────┬────────┬────────┬────────┬─────────┘
        │        │        │        │        │        │        │
        ▼        ▼        ▼        ▼        ▼        ▼        ▼
  ┌──────────┐ ┌────────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
  │ Session  │ │ Model  │ │ Env  │ │ Step │ │Train │ │Metric│ │Displa│
  │ Manager  │ │Manager │ │Mngr  │ │Mngr  │ │Loop  │ │Mngr  │ │Mngr  │
  └────┬─────┘ └───┬────┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘
       │           │         │        │        │        │        │
       │           │         │        │        │        │        │
       ▼           ▼         ▼        ▼        ▼        ▼        ▼
  ┌─────────┐ ┌─────────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
  │ W&B API │ │ Model   │ │Shogi │ │Exper.│ │ PPO  │ │Stats │ │ Rich │
  │ Dirs    │ │ Factory │ │ Game │ │Buffer│ │Agent │ │ Dict │ │ Live │
  │ Config  │ │torch.co │ │Policy│ │      │ │      │ │      │ │      │
  │ Seeding │ │Checkpts │ │Mapper│ │      │ │      │ │      │ │      │
  └─────────┘ └─────────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘

  ┌──────────┐  ┌──────────┐
  │ Callback │  │  Setup   │
  │ Manager  │  │ Manager  │
  └────┬─────┘  └────┬─────┘
       │              │
       ▼              ▼
  ┌──────────┐  ┌──────────┐
  │Evaluation│  │Component │
  │ Triggers │  │Validation│
  │Checkpoint│  │Dependency│
  │ Events   │  │ Wiring   │
  └──────────┘  └──────────┘
```

---

## C4 Level 3: Component Diagram - Evaluation System

```
┌─────────────────────────────────────────────────────────────────┐
│                    Evaluation System                              │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              EnhancedEvaluationManager                   │    │
│  │              (Strategy Dispatch)                          │    │
│  └─────────────────────┬───────────────────────────────────┘    │
│                        │                                         │
│           ┌────────────┼────────────┬────────────┐              │
│           ▼            ▼            ▼            ▼              │
│  ┌──────────────┐ ┌─────────┐ ┌─────────┐ ┌──────────┐        │
│  │   Single     │ │Tourna-  │ │ Ladder  │ │Benchmark │        │
│  │  Opponent    │ │  ment   │ │  (ELO)  │ │  Suite   │        │
│  │  Strategy    │ │Strategy │ │Strategy │ │ Strategy │        │
│  └──────────────┘ └─────────┘ └─────────┘ └──────────┘        │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   Support Layer                          │    │
│  │                                                          │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │    │
│  │  │   Opponent   │  │     ELO      │  │  Background  │  │    │
│  │  │     Pool     │  │   Registry   │  │  Tournament  │  │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  Analytics Layer                         │    │
│  │                                                          │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │    │
│  │  │ Performance  │  │     ELO      │  │   Report     │  │    │
│  │  │  Analyzer    │  │   Tracker    │  │  Generator   │  │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## C4 Level 3: Data Flow - Training Loop

```
                    ┌──────────────┐
                    │ Config (YAML │
                    │  + Pydantic) │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   Trainer    │
                    │ (init all    │
                    │  managers)   │
                    └──────┬───────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │   Training Loop         │
              │   (TrainingLoopManager) │
              └─────────┬───────────────┘
                        │
           ┌────────────┼────────────────┐
           │            │                │
           ▼            ▼                ▼
     ┌──────────┐ ┌──────────┐    ┌──────────┐
     │  Step    │ │  PPO     │    │ Callback │
     │ Manager  │ │  Update  │    │ Manager  │
     └────┬─────┘ └────┬─────┘    └────┬─────┘
          │            │               │
          ▼            ▼               ▼
     ┌──────────┐ ┌──────────┐    ┌──────────┐
     │  Game    │ │Experience│    │ Evaluate │
     │ self-play│ │  Buffer  │    │ + Save   │
     │  step    │ │  (GAE)   │    │ Checkpt  │
     └────┬─────┘ └──────────┘    └──────────┘
          │
     ┌────▼─────────────────────────────────┐
     │  Observation → Model → Action → Env  │
     │                                       │
     │  obs(46,9,9) → ResNet → action_idx   │
     │       → ShogiGame.step(action)        │
     │       → reward, next_obs, done        │
     └──────────────────────────────────────┘
```

---

## C4 Level 3: Component Diagram - Shogi Engine

```
┌─────────────────────────────────────────────────────────────┐
│                     Shogi Engine                             │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                  ShogiGame                            │   │
│  │                  (shogi_game.py)                      │   │
│  │                                                       │   │
│  │  State: board[9][9], hands[2], turn, move_history    │   │
│  │  API: step(), reset(), get_legal_moves(), is_over()  │   │
│  └───────┬──────────────┬──────────────┬────────────────┘   │
│          │              │              │                      │
│          ▼              ▼              ▼                      │
│  ┌──────────────┐ ┌──────────┐ ┌──────────────┐            │
│  │  Rules Logic │ │   Move   │ │   Game I/O   │            │
│  │  (move gen,  │ │Execution │ │  (SFEN, USI, │            │
│  │   check,     │ │(captures,│ │   display)   │            │
│  │   checkmate) │ │ promote, │ │              │            │
│  │              │ │  drops)  │ │              │            │
│  └──────────────┘ └──────────┘ └──────────────┘            │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────────────────┐    │
│  │ Core Definitions │  │  Feature Extraction           │    │
│  │ (Piece, Color,   │  │  (46-channel observation     │    │
│  │  MoveTuple)      │  │   tensor for neural network) │    │
│  └──────────────────┘  └──────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## Dependency Graph (Subsystem Level)

```
                    ┌──────────────┐
                    │    config    │
                    │   (schema,   │
                    │  constants)  │
                    └──────┬───────┘
                           │
              ┌────────────┼──────────────────┐
              │            │                  │
              ▼            ▼                  ▼
       ┌──────────┐  ┌──────────┐     ┌──────────┐
       │   core   │◄─┤  utils   │     │  webui   │
       │(PPO,buf) │──►│(logging, │     │(WebSocket│
       └────┬─────┘   │ mapper)  │     │ server)  │
            │         └────┬─────┘     └──────────┘
            │              │
            │         ┌────▼─────┐
            │         │  shogi   │
            │         │ (engine) │
            │         └────┬─────┘
            │              │
            ▼              ▼
       ┌─────────────────────────┐
       │      evaluation         │
       │   (strategies,          │
       │    analytics)           │
       └────────────┬────────────┘
                    │
                    ▼
       ┌─────────────────────────┐
       │       training          │
       │   (managers, loop,      │
       │    display, CLI)        │
       └─────────────────────────┘

Legend:
  ──► = depends on (imports from)
  ◄── = bidirectional (circular)
  ─── = unidirectional
```

**Note**: The `core <-> utils` bidirectional dependency is the only circular dependency at the subsystem level. All other dependencies flow downward (training depends on everything, evaluation depends on core/shogi/utils, etc.).

---

## Observation Space Diagram

```
Input Tensor: (46, 9, 9)
═══════════════════════════

Channels 0-11:   Current player's pieces on board
                 (12 piece types x 1 binary plane each)

Channels 12-23:  Opponent's pieces on board
                 (12 piece types x 1 binary plane each)

Channels 24-30:  Current player's hand pieces
                 (7 drop-eligible piece types x count-encoded)

Channels 31-37:  Opponent's hand pieces
                 (7 drop-eligible piece types x count-encoded)

Channels 38-43:  Game state features
                 (turn indicator, move count, repetition, etc.)

Channels 44-45:  Additional features
                 (check state, legal move hints)

                    ┌─────────────────────┐
                    │   9x9 Shogi Board   │
                    │  ┌─┬─┬─┬─┬─┬─┬─┬─┬─┐│
                    │  │ │ │ │ │ │ │ │ │ ││
                    │  ├─┼─┼─┼─┼─┼─┼─┼─┼─┤│
                    │  │ │ │ │ │ │ │ │ │ ││     46 channels
                    │  ├─┼─┼─┼─┼─┼─┼─┼─┼─┤│     stacked as
                    │  │ │ │ │ │ │ │ │ │ ││     (C, H, W)
                    │  ├─┼─┼─┼─┼─┼─┼─┼─┼─┤│     tensor
                    │  │ │ │ │ │ │ │ │ │ ││
                    │  └─┴─┴─┴─┴─┴─┴─┴─┴─┘│
                    └─────────────────────┘

                    ↓ Feeds into ↓

          ┌─────────────────────────────────┐
          │   ResNet Tower                   │
          │   (9 residual blocks, 256 ch)   │
          │   with SE blocks (ratio 0.25)   │
          └───────────┬─────────────────────┘
                      │
              ┌───────┴───────┐
              ▼               ▼
     ┌──────────────┐ ┌──────────────┐
     │  Policy Head │ │  Value Head  │
     │  (13,527     │ │  (scalar)    │
     │   actions)   │ │              │
     └──────────────┘ └──────────────┘
```

---

## Action Space Mapping

```
Total Actions: 13,527
═══════════════════════

PolicyOutputMapper decomposes actions into:

┌─────────────────────────────────────────────────┐
│  Board Moves (regular + promotion)               │
│  81 squares x 81 squares x 2 (promote/not)      │
│  = up to 13,122 move actions                     │
│  (filtered by geometry - not all pairs legal)    │
├─────────────────────────────────────────────────┤
│  Drop Moves                                      │
│  7 piece types x 81 squares = 567 drop actions  │
├─────────────────────────────────────────────────┤
│  Total: 13,527 discrete action indices           │
│  (mapped to/from ShogiGame.MoveTuple)            │
└─────────────────────────────────────────────────┘

Action Selection Flow:
  1. Model outputs logits: (13,527,)
  2. Legal mask applied: illegal actions → -inf
  3. Softmax → probability distribution
  4. Sample action index
  5. PolicyOutputMapper.index_to_move(idx) → MoveTuple
  6. ShogiGame.step(MoveTuple) → next state
```
