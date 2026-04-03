#!/usr/bin/env bash
# ============================================================================
# run-500k-league.sh — 500K-epoch league training with opponent pool (2x GPU)
# ============================================================================
#
# What this does:
#   1. Optionally wipes DB and checkpoints for a clean start
#   2. Launches single-process league training (learner on cuda:0, opponents
#      on cuda:1 — NOT DDP, which deadlocks with split-merge)
#   3. Launches the web dashboard on the same DB
#   4. Monitors both processes — restarts dashboard if it dies
#   5. Shuts down cleanly on Ctrl+C or if training exits
#
# Usage:
#   ./run-500k-league.sh              # Fresh start
#   ./run-500k-league.sh resume       # Resume from last checkpoint
#
# Dashboard: http://keisei.foundryside.dev (Caddy proxies port 8741)
# Logs:      logs/league500k_train_YYYYMMDD_HHMMSS.log
#            logs/league500k_server_YYYYMMDD_HHMMSS.log
#
# Config:    keisei-500k-league.toml (2x RTX 4060 Ti, b10c128 SE-ResNet, league)
# ============================================================================

set -euo pipefail
cd "$(dirname "$0")"

CONFIG="${CONFIG:-keisei-500k-league.toml}"
EPOCHS=500000
WEB_HOST="0.0.0.0"
WEB_PORT="${WEB_PORT:-8741}"
LOG_DIR="logs"
DB_DIR="data"
DB_PATH="$DB_DIR/keisei-500k-league.db"
CKPT_DIR="checkpoints/500k-league"

# ---- Preflight checks ----

if ! command -v uv &>/dev/null; then
    echo "Error: uv not found in PATH"
    exit 1
fi

if ! [[ -f "$CONFIG" ]]; then
    echo "Error: Config not found: $CONFIG"
    exit 1
fi

GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
if [[ "$GPU_COUNT" -lt 2 ]]; then
    echo "Warning: 2 GPUs requested but only $GPU_COUNT available."
    echo "  Opponent will share GPU with learner (slower but functional)."
fi

mkdir -p "$LOG_DIR" "$CKPT_DIR" "$DB_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TRAIN_LOG="$LOG_DIR/league500k_train_${TIMESTAMP}.log"
SERVER_LOG="$LOG_DIR/league500k_server_${TIMESTAMP}.log"

# ---- Fresh or resume ----

if [[ "${1:-}" != "resume" ]]; then
    echo "Fresh start — wiping DB and checkpoints"
    rm -f "$DB_PATH" "${DB_PATH}-wal" "${DB_PATH}-shm"
    rm -rf "$CKPT_DIR/"
    mkdir -p "$CKPT_DIR"
else
    if [[ ! -f "$DB_PATH" ]]; then
        echo "Error: No DB found at $DB_PATH. Cannot resume."
        echo "  Start fresh: ./run-500k-league.sh"
        exit 1
    fi
    echo "Resuming from last checkpoint"
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

# ---- Start training (single process, 2 GPUs for different roles) ----

echo "Starting league training (learner: cuda:0, opponents: cuda:1)"
echo "  Config: $CONFIG"
echo "  Log:    $TRAIN_LOG"

uv run python -m keisei.training.katago_loop "$CONFIG" --epochs "$EPOCHS" \
    > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
echo "  PID:    $TRAIN_PID"

# ---- Wait for DB to appear ----

echo -n "Waiting for DB..."
for i in $(seq 1 30); do
    [[ -f "$DB_PATH" ]] && break
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
echo "  Kenshi — 500K League Training"
echo "=========================================="
echo "  Dashboard:  https://keisei.foundryside.dev (port $WEB_PORT)"
echo "  Trainer:    PID $TRAIN_PID"
echo "  Server:     PID $SERVER_PID"
echo "  Train log:  $TRAIN_LOG"
echo "  Server log: $SERVER_LOG"
echo ""
echo "  GPU 0: learner (training + gradients)"
echo "  GPU 1: opponent pool (inference only)"
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
