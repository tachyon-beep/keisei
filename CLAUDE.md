# CLAUDE.md

## Project Overview

Keisei — DRL system for Shogi. Rust core (`shogi-core` + `shogi-gym`) with Python training harness.

Repo is in early rebuild after migrating from a pure Python engine to Rust.

## Commands

```bash
uv pip install -e ".[dev]"
uv run pytest
```

## Key Paths

- Rust engine: developed on `feature/shogi-core-rust-engine` branch
- Python training: TBD
- WebUI: TBD
