#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHOGI_ENGINE_DIR="$ROOT_DIR/shogi-engine"
SHOGI_GYM_DIR="$SHOGI_ENGINE_DIR/crates/shogi-gym"

echo "==> Rebuilding Rust workspace"
(
  cd "$SHOGI_ENGINE_DIR"
  cargo build --release
)

MATURIN_BIN=""
if command -v maturin >/dev/null 2>&1; then
  MATURIN_BIN="$(command -v maturin)"
elif [[ -x "$ROOT_DIR/.venv/bin/maturin" ]]; then
  MATURIN_BIN="$ROOT_DIR/.venv/bin/maturin"
elif [[ -x "$SHOGI_GYM_DIR/.venv/bin/maturin" ]]; then
  MATURIN_BIN="$SHOGI_GYM_DIR/.venv/bin/maturin"
else
  echo "error: maturin not found." >&2
  echo "Install it with: uv pip install maturin  (or pip install maturin)" >&2
  exit 1
fi

echo "==> Rebuilding Python bindings with maturin"
(
  cd "$SHOGI_GYM_DIR"
  "$MATURIN_BIN" develop --release
)

echo "==> Done"
echo "Rust and Python bindings have been rebuilt."
