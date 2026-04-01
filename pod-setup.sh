#!/usr/bin/env bash
#
# pod-setup.sh — One-shot setup for a fresh cloud GPU pod
#
# Usage:
#   git clone <repo> keisei && cd keisei && ./pod-setup.sh
#
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Keisei Pod Setup ==="

# --- System deps ---
echo "Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq curl build-essential pkg-config libssl-dev > /dev/null 2>&1 || true

# --- Rust (for shogi-gym) ---
if ! command -v cargo &>/dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y > /dev/null 2>&1
    source "$HOME/.cargo/env"
fi

# --- uv (Python package manager) ---
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
    export PATH="$HOME/.local/bin:$PATH"
fi

# --- Python deps ---
echo "Installing Python dependencies..."
uv pip install -e ".[dev]" > /dev/null 2>&1
uv pip install psutil > /dev/null 2>&1

# --- Build Rust engine ---
echo "Building shogi-gym (Rust)..."
uv pip install -e shogi-engine/crates/shogi-gym/ 2>&1 | tail -1

# --- Verify CUDA ---
echo ""
echo "=== Environment ==="
uv run python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} ({props.total_mem // (1024**3)}GB)')
"

# --- Verify shogi-gym ---
uv run python -c "from shogi_gym import VecEnv; v = VecEnv(4); v.reset(); print('shogi-gym: OK')"

# --- Quick smoke test ---
echo ""
echo "Running smoke test (2 epochs)..."
rm -f keisei.db
uv run keisei-train --config keisei-h200.toml --epochs 2 --steps-per-epoch 16 2>&1

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start training:"
echo "  ./run-500k.sh                    # Fresh 500K run (local config)"
echo "  CONFIG=keisei-h200.toml ./run-500k.sh   # Fresh 500K run (H200 config)"
echo "  CONFIG=keisei-h200.toml ./run-500k.sh resume  # Resume"
echo ""
