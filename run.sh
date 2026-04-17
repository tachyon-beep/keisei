#!/usr/bin/env bash
# ============================================================================
# run.sh — Unified Keisei training launcher
# ============================================================================
#
# Replaces: run-training.sh, run-500k.sh, run-league.sh, run-500k-league.sh
#
# Usage:
#   ./run.sh <config.toml> [options]
#   ./run.sh --stop                         # kill running processes
#
# Examples:
#   ./run.sh keisei-500k.toml               # fresh start, wipes DB
#   ./run.sh keisei-500k.toml --resume      # resume from checkpoint
#   ./run.sh keisei-league.toml --seed checkpoints/500k/epoch_00100.pt
#   ./run.sh keisei-ddp.toml --epochs 2000
#   ./run.sh keisei-ddp.toml --port 8741    # use TCP instead of unix socket
#   ./run.sh keisei-league.toml --no-showcase  # skip the showcase sidecar
#   ./run.sh keisei-league.toml --no-tournament # skip the tournament worker
#
# Options:
#   --resume              Resume from existing DB/checkpoint (default: fresh)
#   --seed <checkpoint>   Seed league DB from a pre-trained checkpoint
#   --epochs N            Override epoch count (default: from config or 1000)
#   --socket PATH         Unix socket for dashboard (default: /run/keisei/uvicorn.sock)
#   --port N              Use TCP port instead of unix socket
#   --host ADDR           Bind address for TCP mode (default: 0.0.0.0)
#   --no-showcase         Skip the showcase sidecar
#   --no-tournament       Skip the tournament worker sidecar
#   --background          Detach processes (default: foreground with monitor)
#   --stop                Kill processes from a previous --background run
#
# DDP vs single-process is determined by the config: configs with a
# [distributed] section launch via torchrun, others use plain python.
# GPU count for DDP is auto-detected from nvidia-smi.
#
# The script:
#   1. Parses the TOML config to find db_path, checkpoint_dir, and DDP mode
#   2. If the DB already exists (and not --resume), prompts for confirmation
#   3. Launches training (torchrun for DDP configs, python for others)
#   4. Launches the web dashboard on the same DB
#   5. Launches the showcase sidecar (model-vs-model games)
#   6. Launches the tournament worker sidecar (if enabled in config)
#   7. In foreground mode: monitors all, restarts dashboard/sidecar if they die
#   8. Shuts down cleanly on Ctrl+C
# ============================================================================

set -euo pipefail
cd "$(dirname "$0")"

PIDFILE=".training.pids"

# ---- Stop mode (must come first) ----

if [[ "${1:-}" == "--stop" ]]; then
    if [[ -f "$PIDFILE" ]]; then
        echo "Stopping training and web server..."
        while read -r pid; do
            kill "$pid" 2>/dev/null && echo "  Killed PID $pid" || echo "  PID $pid already dead"
        done < "$PIDFILE"
        rm -f "$PIDFILE"
    else
        echo "No PID file found. Checking for running processes..."
        pkill -f 'keisei.training.katago_loop' 2>/dev/null && echo "  Killed training" || true
        pkill -f 'keisei-serve' 2>/dev/null && echo "  Killed web server" || true
        pkill -f 'keisei.showcase.runner' 2>/dev/null && echo "  Killed showcase sidecar" || true
        pkill -f 'keisei.training.tournament_runner' 2>/dev/null && echo "  Killed tournament worker" || true
    fi
    exit 0
fi

# ---- Require config as first positional arg ----

if [[ $# -lt 1 || "$1" == --* ]]; then
    echo "Usage: ./run.sh <config.toml> [options]"
    echo "       ./run.sh --stop"
    echo ""
    echo "Run './run.sh --help' for full options."
    exit 1
fi

if [[ "${1:-}" == "--help" ]]; then
    # Print the header comment as help
    sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
    exit 0
fi

CONFIG="$1"
shift

if [[ ! -f "$CONFIG" ]]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

# ---- Parse options ----

RESUME=false
SEED_CHECKPOINT=""
EPOCHS=""
WEB_HOST="0.0.0.0"
WEB_PORT=""
WEB_SOCKET="/run/keisei/uvicorn.sock"
BACKGROUND=false
NO_SHOWCASE=false
NO_TOURNAMENT=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)       RESUME=true; shift ;;
        --seed)         SEED_CHECKPOINT="$2"; shift 2 ;;
        --epochs)       EPOCHS="$2"; shift 2 ;;
        --port)         WEB_PORT="$2"; WEB_SOCKET=""; shift 2 ;;
        --host)         WEB_HOST="$2"; shift 2 ;;
        --socket)       WEB_SOCKET="$2"; WEB_PORT=""; shift 2 ;;
        --no-showcase)  NO_SHOWCASE=true; shift ;;
        --no-tournament) NO_TOURNAMENT=true; shift ;;
        --background)   BACKGROUND=true; shift ;;
        --help)         sed -n '2,/^$/p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Preflight checks ----

if ! command -v uv &>/dev/null; then
    echo "Error: uv not found in PATH"
    exit 1
fi

if [[ -n "$SEED_CHECKPOINT" && ! -f "$SEED_CHECKPOINT" ]]; then
    echo "Error: Seed checkpoint not found: $SEED_CHECKPOINT"
    exit 1
fi

if [[ -n "$SEED_CHECKPOINT" ]]; then
    SEED_CHECKPOINT="$(realpath "$SEED_CHECKPOINT")"
fi

# ---- Parse config for DB path and checkpoint dir ----

CONFIG_ABS="$(realpath "$CONFIG")"
eval "$(uv run python -c "
import tomllib, os, sys
from pathlib import Path

config_path = Path('$CONFIG_ABS')
config_dir = config_path.parent

with open(config_path, 'rb') as f:
    raw = tomllib.load(f)

d = raw.get('display', {})
db_path = str((config_dir / d.get('db_path', 'keisei.db')).resolve())

t = raw.get('training', {})
ckpt_dir = str((config_dir / t.get('checkpoint_dir', 'checkpoints/')).resolve())
r = raw.get('run', {})
default_epochs = r.get('default_epochs', 1000)

# DDP: configs with [distributed] use torchrun, others use plain python
use_ddp = 'distributed' in raw

# Tournament sidecar settings
league = raw.get('league', {})
tournament_mode = league.get('tournament_mode', 'in_process')
tournament_enabled = str(league.get('tournament_enabled', False)).lower()
tournament_device = league.get('tournament_device', 'cuda:1')
# Also extract league_dir from the checkpoint_dir (it's the same parent)
league_dir = str((config_dir / t.get('checkpoint_dir', 'checkpoints/')).resolve())

print(f'DB_PATH={db_path!r}')
print(f'CKPT_DIR={ckpt_dir!r}')
print(f'DEFAULT_EPOCHS={default_epochs}')
print(f'USE_DDP={str(use_ddp).lower()}')
print(f'TOURNAMENT_MODE={tournament_mode!r}')
print(f'TOURNAMENT_ENABLED={tournament_enabled!r}')
print(f'TOURNAMENT_DEVICE={tournament_device!r}')
print(f'LEAGUE_DIR={league_dir!r}')
")"

EPOCHS="${EPOCHS:-$DEFAULT_EPOCHS}"

# ---- GPU count (only needed for DDP) ----

NGPUS=1
if [[ "$USE_DDP" == "true" ]]; then
    if command -v nvidia-smi &>/dev/null; then
        NGPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    fi
    if [[ "$NGPUS" -lt 1 ]]; then
        echo "Error: [distributed] config requires GPUs but none detected"
        exit 1
    fi
fi

# ---- DB existence check ----

if [[ -f "$DB_PATH" ]]; then
    if [[ "$RESUME" == true || -n "$SEED_CHECKPOINT" && "$SEED_CHECKPOINT" == "resume" ]]; then
        echo "Resuming from existing DB: $DB_PATH"
    else
        echo ""
        echo "  Database already exists: $DB_PATH"
        echo ""
        echo "  [r] Resume from existing state"
        echo "  [w] Wipe and start fresh"
        echo "  [q] Quit"
        echo ""
        read -rp "  Choice [r/w/q]: " choice
        case "$choice" in
            r|R) RESUME=true ;;
            w|W) ;;
            *)   echo "Aborted."; exit 0 ;;
        esac
    fi
fi

# ---- Fresh start: wipe DB and checkpoints ----

if [[ "$RESUME" != true ]]; then
    if [[ -f "$DB_PATH" ]]; then
        echo "Wiping DB: $DB_PATH"
        rm -f "$DB_PATH" "${DB_PATH}-wal" "${DB_PATH}-shm"
    fi
    if [[ -d "$CKPT_DIR" ]]; then
        echo "Wiping checkpoints: $CKPT_DIR"
        rm -rf "$CKPT_DIR"
    fi
fi

# ---- Ensure directories exist ----

mkdir -p "$(dirname "$DB_PATH")" "$CKPT_DIR" logs

# ---- Seed league DB if requested ----

if [[ -n "$SEED_CHECKPOINT" ]]; then
    echo "Seeding league DB from: $SEED_CHECKPOINT"
    uv run python -c "
from keisei.db import init_db, write_training_state
from datetime import datetime, timezone
import json

db = '$DB_PATH'
init_db(db)
write_training_state(db, {
    'config_json': json.dumps({'seeded_from': '$SEED_CHECKPOINT'}),
    'display_name': 'League',
    'model_arch': 'se_resnet',
    'algorithm_name': 'katago_ppo',
    'started_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    'checkpoint_path': '$SEED_CHECKPOINT',
    'current_epoch': 0,
    'current_step': 0,
    'total_epochs': $EPOCHS,
    'status': 'running',
})
print('League DB seeded successfully')
"
fi

# ---- Prepare log files ----

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONFIG_STEM=$(basename "$CONFIG" .toml)
TRAIN_LOG="logs/${CONFIG_STEM}_train_${TIMESTAMP}.log"
SERVER_LOG="logs/${CONFIG_STEM}_server_${TIMESTAMP}.log"
SHOWCASE_LOG="logs/${CONFIG_STEM}_showcase_${TIMESTAMP}.log"
TOURNAMENT_LOG="logs/${CONFIG_STEM}_tournament_${TIMESTAMP}.log"

# ---- Build server command ----

SERVER_CMD=(uv run keisei-serve --config "$CONFIG_ABS")
if [[ -n "$WEB_SOCKET" ]]; then
    mkdir -p "$(dirname "$WEB_SOCKET")"
    SERVER_CMD+=(--socket "$WEB_SOCKET")
    DASHBOARD_URL="unix:$WEB_SOCKET"
elif [[ -n "$WEB_PORT" ]]; then
    SERVER_CMD+=(--host "$WEB_HOST" --port "$WEB_PORT")
    DASHBOARD_URL="http://$WEB_HOST:$WEB_PORT"
else
    # Shouldn't happen — one of socket or port is always set
    echo "Error: no dashboard endpoint configured"
    exit 1
fi

# ---- Build showcase sidecar command ----

SHOWCASE_CMD=(uv run python -m keisei.showcase.runner --db-path "$DB_PATH")

# ---- Build tournament worker sidecar command ----

TOURNAMENT_CMD=(uv run python -m keisei.training.tournament_runner
                --config "$CONFIG_ABS"
                --db-path "$DB_PATH"
                --league-dir "$LEAGUE_DIR"
                --worker-id worker-0
                --device "$TOURNAMENT_DEVICE")

LAUNCH_TOURNAMENT=false
if [[ "$TOURNAMENT_ENABLED" == "true" && "$TOURNAMENT_MODE" == "sidecar" && "$NO_TOURNAMENT" != true ]]; then
    LAUNCH_TOURNAMENT=true
fi

# ---- Build training command ----

if [[ "$USE_DDP" == "true" ]]; then
    TRAIN_CMD=(uv run torchrun --nproc_per_node="$NGPUS"
               -m keisei.training.katago_loop "$CONFIG_ABS" --epochs "$EPOCHS")
    GPU_DESC="DDP ($NGPUS GPUs)"
else
    TRAIN_CMD=(uv run python -m keisei.training.katago_loop "$CONFIG_ABS" --epochs "$EPOCHS")
    GPU_DESC="single process"
fi

# ---- Background mode ----

if [[ "$BACKGROUND" == true ]]; then
    if [[ -f "$PIDFILE" ]]; then
        echo "Error: PID file exists — a run may already be active."
        echo "  Use './run.sh --stop' first, or delete $PIDFILE if stale."
        exit 1
    fi

    echo "Starting dashboard ($DASHBOARD_URL) ..."
    nohup "${SERVER_CMD[@]}" > "$SERVER_LOG" 2>&1 &
    WEB_PID=$!
    echo "$WEB_PID" > "$PIDFILE"
    echo "  Web PID: $WEB_PID (log: $SERVER_LOG)"

    sleep 1

    echo "Starting training ($GPU_DESC, $EPOCHS epochs) ..."
    nohup "${TRAIN_CMD[@]}" > "$TRAIN_LOG" 2>&1 &
    TRAIN_PID=$!
    echo "$TRAIN_PID" >> "$PIDFILE"
    echo "  Training PID: $TRAIN_PID (log: $TRAIN_LOG)"

    if [[ "$NO_SHOWCASE" != true ]]; then
        echo "Starting showcase sidecar ..."
        nohup "${SHOWCASE_CMD[@]}" > "$SHOWCASE_LOG" 2>&1 &
        SHOWCASE_PID=$!
        echo "$SHOWCASE_PID" >> "$PIDFILE"
        echo "  Showcase PID: $SHOWCASE_PID (log: $SHOWCASE_LOG)"
    fi

    if [[ "$LAUNCH_TOURNAMENT" == true ]]; then
        echo "Starting tournament worker sidecar ..."
        nohup "${TOURNAMENT_CMD[@]}" > "$TOURNAMENT_LOG" 2>&1 &
        TOURNAMENT_PID=$!
        echo "$TOURNAMENT_PID" >> "$PIDFILE"
        echo "  Tournament PID: $TOURNAMENT_PID (log: $TOURNAMENT_LOG)"
    fi

    echo ""
    echo "All processes running in background."
    echo "  Dashboard: $DASHBOARD_URL"
    echo "  Stop:      ./run.sh --stop"
    echo "  Logs:      tail -f $TRAIN_LOG"
    exit 0
fi

# ---- Foreground mode with monitor ----

cleanup() {
    echo ""
    echo "Shutting down..."
    [[ -n "${TRAIN_PID:-}" ]] && kill "$TRAIN_PID" 2>/dev/null && echo "  Trainer stopped (PID $TRAIN_PID)"
    [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null && echo "  Server stopped (PID $SERVER_PID)"
    [[ -n "${SHOWCASE_PID:-}" ]] && kill "$SHOWCASE_PID" 2>/dev/null && echo "  Showcase stopped (PID $SHOWCASE_PID)"
    [[ -n "${TOURNAMENT_PID:-}" ]] && kill "$TOURNAMENT_PID" 2>/dev/null && echo "  Tournament stopped (PID $TOURNAMENT_PID)"
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start training
echo "Starting training ($GPU_DESC, $EPOCHS epochs)"
echo "  Config: $CONFIG"
echo "  Log:    $TRAIN_LOG"

"${TRAIN_CMD[@]}" > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
echo "  PID:    $TRAIN_PID"

# Wait for DB to appear
echo -n "Waiting for DB..."
for i in $(seq 1 30); do
    [[ -f "$DB_PATH" ]] && break
    sleep 1
    echo -n "."
done
echo " ready"

# Start dashboard
echo "Starting dashboard: $DASHBOARD_URL"
echo "  Log:    $SERVER_LOG"

"${SERVER_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "  PID:    $SERVER_PID"

# Start showcase sidecar
SHOWCASE_PID=""
if [[ "$NO_SHOWCASE" != true ]]; then
    echo "Starting showcase sidecar"
    echo "  Log:    $SHOWCASE_LOG"
    "${SHOWCASE_CMD[@]}" > "$SHOWCASE_LOG" 2>&1 &
    SHOWCASE_PID=$!
    echo "  PID:    $SHOWCASE_PID"
fi

# Start tournament worker sidecar
TOURNAMENT_PID=""
if [[ "$LAUNCH_TOURNAMENT" == true ]]; then
    echo "Starting tournament worker sidecar"
    echo "  Log:    $TOURNAMENT_LOG"
    "${TOURNAMENT_CMD[@]}" > "$TOURNAMENT_LOG" 2>&1 &
    TOURNAMENT_PID=$!
    echo "  PID:    $TOURNAMENT_PID"
fi

echo ""
echo "=========================================="
echo "  Keisei Training — $(basename "$CONFIG" .toml)"
echo "=========================================="
echo "  Dashboard:  $DASHBOARD_URL"
echo "  Trainer:    PID $TRAIN_PID ($GPU_DESC)"
echo "  Server:     PID $SERVER_PID"
if [[ -n "$SHOWCASE_PID" ]]; then
echo "  Showcase:   PID $SHOWCASE_PID"
fi
if [[ -n "$TOURNAMENT_PID" ]]; then
echo "  Tournament: PID $TOURNAMENT_PID ($TOURNAMENT_DEVICE)"
fi
echo "  Train log:  $TRAIN_LOG"
echo "  Server log: $SERVER_LOG"
if [[ -n "$SHOWCASE_PID" ]]; then
echo "  Showcase:   $SHOWCASE_LOG"
fi
if [[ -n "$TOURNAMENT_PID" ]]; then
echo "  Tournament: $TOURNAMENT_LOG"
fi
if [[ -n "$SEED_CHECKPOINT" ]]; then
echo "  Seeded:     $SEED_CHECKPOINT"
fi
echo ""
echo "  Press Ctrl+C to stop all processes"
echo "=========================================="
echo ""

# ---- Monitor loop ----

while true; do
    if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
        TRAIN_EXIT=$(wait "$TRAIN_PID" 2>/dev/null; echo $?)
        echo ""
        echo "Trainer exited (code $TRAIN_EXIT)"
        tail -5 "$TRAIN_LOG"
        echo ""
        echo "Stopping server and sidecar..."
        kill "$SERVER_PID" 2>/dev/null
        [[ -n "${SHOWCASE_PID:-}" ]] && kill "$SHOWCASE_PID" 2>/dev/null
        [[ -n "${TOURNAMENT_PID:-}" ]] && kill "$TOURNAMENT_PID" 2>/dev/null
        exit "$TRAIN_EXIT"
    fi

    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Server died, restarting..."
        "${SERVER_CMD[@]}" >> "$SERVER_LOG" 2>&1 &
        SERVER_PID=$!
        echo "  Server restarted (PID $SERVER_PID)"
    fi

    if [[ -n "$SHOWCASE_PID" ]] && ! kill -0 "$SHOWCASE_PID" 2>/dev/null; then
        echo "Showcase sidecar died, restarting..."
        "${SHOWCASE_CMD[@]}" >> "$SHOWCASE_LOG" 2>&1 &
        SHOWCASE_PID=$!
        echo "  Showcase restarted (PID $SHOWCASE_PID)"
    fi

    if [[ -n "$TOURNAMENT_PID" ]] && ! kill -0 "$TOURNAMENT_PID" 2>/dev/null; then
        echo "Tournament worker died, restarting..."
        "${TOURNAMENT_CMD[@]}" >> "$TOURNAMENT_LOG" 2>&1 &
        TOURNAMENT_PID=$!
        echo "  Tournament restarted (PID $TOURNAMENT_PID)"
    fi

    sleep 30
done
