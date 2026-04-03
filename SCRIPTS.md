# Keisei Training Scripts

## Overview

Training is a two-stage pipeline: **burn-in** (DDP, fast) then **league** (single-GPU, opponent pool). Each script launches both a trainer and a web dashboard reading the same database.

```
run-500k.sh (Hikaru, DDP burn-in)
    │
    ├── checkpoints/500k/epoch_00100.pt
    │
    └──► run-league.sh (Musashi, league self-play)
```

For quick iteration and testing, `run-training.sh` is a lightweight background launcher that works with any config.

---

## Scripts

### `run-500k.sh` — Overnight DDP burn-in

Launches a 500,000-epoch training run on 2 GPUs with foreground monitoring. The script watches both processes, auto-restarts the dashboard if it dies, and shuts down cleanly on Ctrl+C or trainer exit.

**Config:** `keisei-500k.toml` (Hikaru, b10c128 SE-ResNet, 2x RTX 4060 Ti)
**Dashboard:** http://keisei.foundryside.dev:8741
**Logs:** `logs/train_YYYYMMDD_HHMMSS.log`, `logs/server_YYYYMMDD_HHMMSS.log`

```bash
./run-500k.sh              # fresh start — wipes DB + checkpoints
./run-500k.sh resume       # resume from last checkpoint
```

**Environment variables:**
- `CONFIG` — override config file (default: `keisei-500k.toml`)

**What it does on fresh start:**
1. Deletes `keisei.db` and `checkpoints/500k/`
2. Launches `torchrun --nproc_per_node=2` for DDP training
3. Waits for the DB to appear
4. Launches the web dashboard on port 8741
5. Monitors both in a 30-second poll loop

---

### `run-league.sh` — League training from checkpoint

Takes a checkpoint from a prior run and launches single-GPU league training with an opponent pool and Elo tracking. The learner plays Black against sampled historical opponents.

**Config:** `keisei-league.toml` (Musashi, b10c128 SE-ResNet, single GPU)
**Dashboard:** http://keisei.foundryside.dev:8742
**Logs:** `logs/league_train_YYYYMMDD_HHMMSS.log`, `logs/league_server_YYYYMMDD_HHMMSS.log`

```bash
./run-league.sh checkpoints/500k/epoch_00100.pt   # seed from checkpoint
./run-league.sh resume                              # resume existing league run
```

**Environment variables:**
- `CONFIG` — override config file (default: `keisei-league.toml`)
- `EPOCHS` — total epochs (default: `50000`)

**What it does when seeding:**
1. Deletes `keisei-league.db` and `checkpoints/league/`
2. Creates a fresh DB with `training_state` pointing at the checkpoint
3. Launches `python -m keisei.training.katago_loop` (single GPU)
4. Launches the web dashboard on port 8742
5. Monitors both in a 30-second poll loop

**Important:** The league config's model architecture must match the checkpoint. If you trained with b10c128, the league config must also be b10c128.

---

### `run-training.sh` — General-purpose background launcher

Lightweight launcher for quick runs and testing. Both processes run in the background with PID tracking. No foreground monitoring or auto-restart.

**Config:** `configs/ddp_example.toml` (default)
**Dashboard:** http://keisei.foundryside.dev:8741
**Logs:** `training.log`, `webui.log`

```bash
./run-training.sh                                    # defaults: 1000 epochs, 2 GPUs
./run-training.sh --config keisei-ddp.toml           # custom config
./run-training.sh --epochs 5000                      # custom epoch count
./run-training.sh --ngpus 1                          # single GPU
./run-training.sh --port 9000                        # custom dashboard port
./run-training.sh --stop                             # kill running processes
```

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `configs/ddp_example.toml` | Path to TOML config file |
| `--epochs` | `1000` | Number of training epochs |
| `--ngpus` | `2` | Number of GPUs (1 = single GPU, >1 = torchrun DDP) |
| `--port` | `8741` | Web dashboard port |
| `--stop` | — | Kill all running training + dashboard processes |

---

## Configs

| File | Model Name | Architecture | GPU Target | Purpose |
|------|-----------|-------------|------------|---------|
| `keisei-500k.toml` | Hikaru | b10c128 SE-ResNet | 2x RTX 4060 Ti (DDP) | Overnight burn-in |
| `keisei-league.toml` | Musashi | b10c128 SE-ResNet | 1x RTX 4060 Ti | League self-play |
| `keisei-ddp.toml` | Kagami | b10c128 SE-ResNet | 2x RTX 4060 Ti (DDP) | Quick DDP test |
| `keisei-h200.toml` | Raijin | b40c256 SE-ResNet | H200 80GB | Production scale |
| `keisei-katago.toml` | Musashi | b40c256 SE-ResNet | 1x GPU | Single-GPU (no league) |
| `configs/ddp_example.toml` | KataGo-DDP | b20c128 SE-ResNet | Multi-GPU (DDP) | Reference example |

---

## Ports

| Port | Script | Purpose |
|------|--------|---------|
| 8741 | `run-500k.sh`, `run-training.sh` | DDP training dashboard |
| 8742 | `run-league.sh` | League training dashboard |

Both dashboards can run simultaneously — they use separate databases.

---

## Typical Workflow

```bash
# 1. Start overnight burn-in
./run-500k.sh
# Check dashboard at keisei.foundryside.dev:8741
# Ctrl+C when satisfied with training curves

# 2. Pick a checkpoint with good loss curves
ls checkpoints/500k/

# 3. Start league training from that checkpoint
./run-league.sh checkpoints/500k/epoch_00100.pt
# Check dashboard at keisei.foundryside.dev:8742
# Watch Elo ratings develop in the league tab

# 4. Resume league if interrupted
./run-league.sh resume

# 5. When ready for production scale, migrate to H200
./run-training.sh --config keisei-h200.toml --ngpus 4 --epochs 100000
```
