#!/usr/bin/env bash
#
# run-500k.sh — Launch 500,000-epoch Keisei training run with dashboard
#
# Usage:
#   ./run-500k.sh          # Fresh start (wipes DB)
#   ./run-500k.sh resume   # Resume from last checkpoint
#
set -euo pipefail
cd "$(dirname "$0")"

CONFIG="${CONFIG:-keisei-500k.toml}"
EPOCHS=500000
STEPS=256
PORT=8001
LOG_DIR="logs"

mkdir -p "$LOG_DIR" checkpoints

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TRAIN_LOG="$LOG_DIR/train_${TIMESTAMP}.log"
SERVER_LOG="$LOG_DIR/server_${TIMESTAMP}.log"

# --- Cleanup ---
cleanup() {
    echo ""
    echo "Shutting down..."
    [[ -n "${TRAIN_PID:-}" ]] && kill "$TRAIN_PID" 2>/dev/null && echo "Trainer stopped (PID $TRAIN_PID)"
    [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null && echo "Server stopped (PID $SERVER_PID)"
    exit 0
}
trap cleanup SIGINT SIGTERM

# --- Fresh or resume ---
if [[ "${1:-}" != "resume" ]]; then
    echo "Fresh start — wiping DB"
    rm -f keisei.db
else
    echo "Resuming from last checkpoint"
fi

# --- Start trainer ---
echo "Starting trainer: $EPOCHS epochs, $STEPS steps/epoch"
echo "Config: $CONFIG"
echo "Log: $TRAIN_LOG"

uv run keisei-train \
    --config "$CONFIG" \
    --epochs "$EPOCHS" \
    --steps-per-epoch "$STEPS" \
    > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
echo "Trainer PID: $TRAIN_PID"

# Wait for DB to be created
echo -n "Waiting for DB..."
for i in $(seq 1 30); do
    [[ -f keisei.db ]] && break
    sleep 1
    echo -n "."
done
echo " ready"

# --- Start server ---
echo "Starting dashboard server on port $PORT"
echo "Log: $SERVER_LOG"

uv run keisei-serve \
    --config "$CONFIG" \
    --port "$PORT" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

echo ""
echo "=========================================="
echo "  Keisei 500K Training Run"
echo "=========================================="
echo "  Dashboard: http://localhost:$PORT"
echo "  Trainer:   PID $TRAIN_PID"
echo "  Server:    PID $SERVER_PID"
echo "  Train log: $TRAIN_LOG"
echo "  Server log: $SERVER_LOG"
echo ""
echo "  Press Ctrl+C to stop both processes"
echo "=========================================="
echo ""

# --- Monitor ---
while true; do
    # Check trainer is alive
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

    # Check server is alive, restart if needed
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Server died, restarting..."
        uv run keisei-serve \
            --config "$CONFIG" \
            --port "$PORT" \
            >> "$SERVER_LOG" 2>&1 &
        SERVER_PID=$!
        echo "Server restarted (PID $SERVER_PID)"
    fi

    sleep 30
done
