#!/usr/bin/env bash
# ============================================================================
# run-league.sh — Launch league training seeded from a pre-trained checkpoint
# ============================================================================
#
# What this does:
#   1. Takes a checkpoint from a prior training run (e.g. run-500k.sh)
#   2. Creates a fresh league DB seeded with that checkpoint
#   3. Launches single-GPU league training with opponent pool + Elo tracking
#   4. Launches the web dashboard on the same DB
#   5. Monitors both processes — restarts dashboard if it dies
#
# The learner starts from the checkpoint's weights and plays against
# snapshots of itself at various training stages. Elo ratings track
# relative strength across the pool. The learner always plays Black.
#
# Usage:
#   ./run-league.sh checkpoints/500k/epoch_00100.pt     # seed from checkpoint
#   ./run-league.sh resume                               # resume existing league run
#
# Dashboard: http://keisei.foundryside.dev:8742
# Logs:      logs/league_train_YYYYMMDD_HHMMSS.log
#            logs/league_server_YYYYMMDD_HHMMSS.log
#
# Config:    keisei-league.toml (single GPU, b10c128 SE-ResNet, league mode)
#
# IMPORTANT: The league config's model architecture must match the checkpoint.
#            If you trained with b10c128, the league config must also be b10c128.
# ============================================================================

set -euo pipefail
cd "$(dirname "$0")"

CONFIG="${CONFIG:-keisei-league.toml}"
EPOCHS="${EPOCHS:-50000}"
WEB_HOST="0.0.0.0"
WEB_PORT=8742           # different port from run-500k.sh (8741) so both can run
LOG_DIR="logs"
DB_PATH="keisei-league.db"

# ---- Parse args ----

if [[ $# -lt 1 ]]; then
    echo "Usage:"
    echo "  ./run-league.sh <checkpoint.pt>    # seed from checkpoint"
    echo "  ./run-league.sh resume             # resume existing league run"
    echo ""
    echo "Example:"
    echo "  ./run-league.sh checkpoints/500k/epoch_00100.pt"
    exit 1
fi

MODE="$1"
CHECKPOINT=""
if [[ "$MODE" != "resume" ]]; then
    CHECKPOINT="$MODE"
    if [[ ! -f "$CHECKPOINT" ]]; then
        echo "Error: Checkpoint not found: $CHECKPOINT"
        exit 1
    fi
    # Resolve to absolute path so the training loop can find it
    CHECKPOINT="$(realpath "$CHECKPOINT")"
fi

# ---- Preflight checks ----

if ! command -v uv &>/dev/null; then
    echo "Error: uv not found in PATH"
    exit 1
fi

if ! [[ -f "$CONFIG" ]]; then
    echo "Error: Config not found: $CONFIG"
    exit 1
fi

mkdir -p "$LOG_DIR" checkpoints/league

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TRAIN_LOG="$LOG_DIR/league_train_${TIMESTAMP}.log"
SERVER_LOG="$LOG_DIR/league_server_${TIMESTAMP}.log"

# ---- Seed or resume ----

if [[ "$MODE" == "resume" ]]; then
    if [[ ! -f "$DB_PATH" ]]; then
        echo "Error: No league DB found at $DB_PATH. Cannot resume."
        echo "  Start a new league run: ./run-league.sh <checkpoint.pt>"
        exit 1
    fi
    echo "Resuming league training from existing DB"
else
    echo "Seeding league from checkpoint: $CHECKPOINT"

    # Wipe old league state for a clean start
    rm -f "$DB_PATH"
    rm -rf checkpoints/league/

    # Initialize the DB and write training_state with checkpoint path.
    # The training loop's _check_resume reads checkpoint_path from the DB
    # and loads weights + optimizer state from there.
    uv run python -c "
from keisei.db import init_db, write_training_state
from datetime import datetime, timezone
import json

db = '$DB_PATH'
init_db(db)
write_training_state(db, {
    'config_json': json.dumps({'seeded_from': '$CHECKPOINT'}),
    'display_name': 'Musashi',
    'model_arch': 'se_resnet',
    'algorithm_name': 'katago_ppo',
    'started_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    'checkpoint_path': '$CHECKPOINT',
    'current_epoch': 0,
    'current_step': 0,
    'total_epochs': $EPOCHS,
    'status': 'running',
})
print('League DB seeded successfully')
"
fi

# ---- Cleanup on exit ----

cleanup() {
    echo ""
    echo "Shutting down..."
    [[ -n "${TRAIN_PID:-}" ]] && kill "$TRAIN_PID" 2>/dev/null && echo "  Trainer stopped (PID $TRAIN_PID)"
    [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null && echo "  Server stopped (PID $SERVER_PID)"
    exit 0
}
trap cleanup SIGINT SIGTERM

# ---- Start training (single GPU — league requires it) ----

echo "Starting league training (single GPU, $EPOCHS epochs)"
echo "  Config: $CONFIG"
echo "  Log:    $TRAIN_LOG"

uv run python -m keisei.training.katago_loop "$CONFIG" --epochs "$EPOCHS" \
    > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
echo "  PID:    $TRAIN_PID"

# Wait for DB to be populated
echo -n "Waiting for training to start..."
for i in $(seq 1 30); do
    if sqlite3 "$DB_PATH" "SELECT current_epoch FROM training_state LIMIT 1" &>/dev/null; then
        break
    fi
    sleep 1
    echo -n "."
done
echo " ready"

# ---- Start web dashboard ----

echo "Starting dashboard on $WEB_HOST:$WEB_PORT"
echo "  Log:    $SERVER_LOG"

uv run keisei-serve --config "$CONFIG" --host "$WEB_HOST" --port "$WEB_PORT" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "  PID:    $SERVER_PID"

echo ""
echo "=========================================="
echo "  Musashi — League Training"
echo "=========================================="
echo "  Dashboard:  http://keisei.foundryside.dev:$WEB_PORT"
echo "  Trainer:    PID $TRAIN_PID (single GPU)"
echo "  Server:     PID $SERVER_PID"
echo "  Train log:  $TRAIN_LOG"
echo "  Server log: $SERVER_LOG"
if [[ -n "$CHECKPOINT" ]]; then
echo "  Seeded:     $CHECKPOINT"
fi
echo ""
echo "  Press Ctrl+C to stop both processes"
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
        echo "Stopping server..."
        kill "$SERVER_PID" 2>/dev/null
        exit "$TRAIN_EXIT"
    fi

    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Server died, restarting..."
        uv run keisei-serve --config "$CONFIG" --host "$WEB_HOST" --port "$WEB_PORT" \
            >> "$SERVER_LOG" 2>&1 &
        SERVER_PID=$!
        echo "  Server restarted (PID $SERVER_PID)"
    fi

    sleep 30
done
