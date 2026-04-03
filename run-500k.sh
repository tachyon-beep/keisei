#!/usr/bin/env bash
# ============================================================================
# run-500k.sh — Launch a 500,000-epoch KataGo DDP training run with dashboard
# ============================================================================
#
# What this does:
#   1. Optionally wipes the DB and checkpoints for a clean start
#   2. Launches DDP training across all available GPUs (torchrun)
#   3. Launches the web dashboard on the same DB
#   4. Monitors both processes — restarts the dashboard if it dies
#   5. Shuts down cleanly on Ctrl+C or if training exits
#
# Usage:
#   ./run-500k.sh              # Fresh start (wipes DB + checkpoints)
#   ./run-500k.sh resume       # Resume from last checkpoint
#
# Dashboard: http://keisei.foundryside.dev:8741
# Logs:      logs/train_YYYYMMDD_HHMMSS.log
#            logs/server_YYYYMMDD_HHMMSS.log
#
# Config:    keisei-500k.toml (2x RTX 4060 Ti, b10c128 SE-ResNet, ~30s/epoch)
# Duration:  ~170 days at 30s/epoch — designed for overnight burn-in runs,
#            stop with Ctrl+C when satisfied with the training curves.
# ============================================================================

set -euo pipefail
cd "$(dirname "$0")"

CONFIG="${CONFIG:-keisei-500k.toml}"
EPOCHS=500000
NGPUS=2
WEB_HOST="0.0.0.0"
WEB_PORT=8741
LOG_DIR="logs"

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
if [[ "$GPU_COUNT" -lt "$NGPUS" ]]; then
    echo "Warning: Requested $NGPUS GPUs but only $GPU_COUNT available. Using $GPU_COUNT."
    NGPUS=$GPU_COUNT
fi

mkdir -p "$LOG_DIR" checkpoints

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TRAIN_LOG="$LOG_DIR/train_${TIMESTAMP}.log"
SERVER_LOG="$LOG_DIR/server_${TIMESTAMP}.log"

# ---- Fresh or resume ----

if [[ "${1:-}" != "resume" ]]; then
    echo "Fresh start — wiping DB and checkpoints"
    rm -f keisei.db
    rm -rf checkpoints/500k/
    mkdir -p checkpoints/500k
else
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

# ---- Start training ----

echo "Starting DDP training ($NGPUS GPUs, $EPOCHS epochs)"
echo "  Config: $CONFIG"
echo "  Log:    $TRAIN_LOG"

if [[ "$NGPUS" -gt 1 ]]; then
    uv run torchrun --nproc_per_node="$NGPUS" \
        -m keisei.training.katago_loop "$CONFIG" --epochs "$EPOCHS" \
        > "$TRAIN_LOG" 2>&1 &
else
    uv run python -m keisei.training.katago_loop "$CONFIG" --epochs "$EPOCHS" \
        > "$TRAIN_LOG" 2>&1 &
fi
TRAIN_PID=$!
echo "  PID:    $TRAIN_PID"

# ---- Wait for DB to appear ----

echo -n "Waiting for DB..."
for i in $(seq 1 30); do
    [[ -f keisei.db ]] && break
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
echo "  Keisei 500K Training Run"
echo "=========================================="
echo "  Dashboard:  http://keisei.foundryside.dev:$WEB_PORT"
echo "  Trainer:    PID $TRAIN_PID ($NGPUS GPUs)"
echo "  Server:     PID $SERVER_PID"
echo "  Train log:  $TRAIN_LOG"
echo "  Server log: $SERVER_LOG"
echo ""
echo "  Press Ctrl+C to stop both processes"
echo "=========================================="
echo ""

# ---- Monitor loop ----
# Checks every 30s. Restarts server if it dies. Exits if training dies.

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
