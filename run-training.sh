#!/usr/bin/env bash
# run-training.sh — Launch DDP training + web dashboard as one command.
#
# Usage:
#   ./run-training.sh                    # 500k epochs, default config
#   ./run-training.sh --epochs 1000      # custom epoch count
#   ./run-training.sh --config my.toml   # custom config
#
# Logs:
#   training.log  — training output (both ranks)
#   webui.log     — web dashboard output
#
# Stop:
#   ./run-training.sh --stop             # kills both processes

set -euo pipefail
cd "$(dirname "$0")"

CONFIG="configs/ddp_example.toml"
EPOCHS=500000
NGPUS=2
WEB_HOST="0.0.0.0"
WEB_PORT=8741
PIDFILE=".training.pids"

# --- Parse args ---
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
    fi
    exit 0
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --ngpus)  NGPUS="$2"; shift 2 ;;
        --port)   WEB_PORT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# --- Preflight checks ---
if ! command -v uv &>/dev/null; then
    echo "Error: uv not found in PATH"
    exit 1
fi

if ! [[ -f "$CONFIG" ]]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
if [[ "$GPU_COUNT" -lt "$NGPUS" ]]; then
    echo "Warning: Requested $NGPUS GPUs but only $GPU_COUNT available. Using $GPU_COUNT."
    NGPUS=$GPU_COUNT
fi

# --- Check for existing runs ---
if [[ -f "$PIDFILE" ]]; then
    echo "Error: PID file exists — a run may already be active."
    echo "  Use './run-training.sh --stop' first, or delete $PIDFILE if stale."
    exit 1
fi

# --- Launch web dashboard ---
echo "Starting web dashboard on $WEB_HOST:$WEB_PORT ..."
nohup uv run keisei-serve --config "$CONFIG" --host "$WEB_HOST" --port "$WEB_PORT" \
    > webui.log 2>&1 &
WEB_PID=$!
echo "$WEB_PID" > "$PIDFILE"
echo "  Web PID: $WEB_PID (log: webui.log)"

# Brief pause so the DB is initialized before training starts
sleep 1

# --- Launch training ---
echo "Starting DDP training ($NGPUS GPUs, $EPOCHS epochs) ..."
if [[ "$NGPUS" -gt 1 ]]; then
    nohup uv run torchrun --nproc_per_node="$NGPUS" \
        -m keisei.training.katago_loop "$CONFIG" --epochs "$EPOCHS" \
        > training.log 2>&1 &
else
    nohup uv run python -m keisei.training.katago_loop "$CONFIG" --epochs "$EPOCHS" \
        > training.log 2>&1 &
fi
TRAIN_PID=$!
echo "$TRAIN_PID" >> "$PIDFILE"
echo "  Training PID: $TRAIN_PID (log: training.log)"

echo ""
echo "Both processes running in background."
echo "  Dashboard: http://$WEB_HOST:$WEB_PORT"
echo "  Stop:      ./run-training.sh --stop"
echo "  Logs:      tail -f training.log"
