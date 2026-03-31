# Session Kickoff: shogi-gym Implementation (Plan 2 of 3)

Use this as your opening prompt in a new Claude Code session.

---

## Prompt

I'm building a Rust Shogi engine in two crates. Plan 1 (shogi-core — pure game engine) is complete on branch `feature/shogi-core-rust-engine` in `.worktrees/shogi-core`. 82 tests passing, all special rules implemented, benchmarks working.

Now I need Plan 2: **shogi-gym** — the RL training environment crate with PyO3 bindings. This sits on top of shogi-core and exposes:

1. **VecEnv** — batch-steps N games in one FFI call, pre-allocated buffers, rayon parallelism, GIL release via `py.allow_threads()`, two-phase step contract (validate all actions, then apply all)
2. **SpectatorEnv** — single-game wrapper for display/demo games, returns rich Python dicts, no auto-reset
3. **DefaultActionMapper** — 13,527-action encoding (81×80×2 board + 81×7 drops), perspective flipping, trait-based with `Result` return on decode
4. **DefaultObservationGenerator** — 46-channel (9×9) float32 tensors matching current Keisei layout, trait-based

Key design decisions already made (see spec):

- `terminated` + `truncated` (Gymnasium v1 API), not single `done`
- `step_metadata` as packed structured array, not Python dicts on hot path  
- Default `step()` returns copies (safe), opt-in `zero_copy=True` for perf
- Terminal observations stored in separate pre-allocated buffer
- Custom ActionMapper/ObsGen are Rust-side only (no Python callbacks in hot loop)
- Illegal action = panic debug / RuntimeError release, with two-phase validation

**Read these files first:**

- `docs/superpowers/specs/2026-03-31-rust-shogi-engine-design.md` — full design spec (focus on the `shogi-gym` sections)
- `docs/superpowers/plans/2026-03-31-shogi-core-rust-engine.md` — Plan 1 for format reference
- `.worktrees/shogi-core/shogi-engine/crates/shogi-core/src/lib.rs` — public API surface of the core crate

**Then write an implementation plan** for shogi-gym following the same format as Plan 1. Save it to `docs/superpowers/plans/2026-03-31-shogi-gym-python-bindings.md`. Use the `superpowers:writing-plans` skill.

The crate goes at `shogi-engine/crates/shogi-gym/` in the same Cargo workspace. It depends on shogi-core, pyo3, numpy (Rust crate), and rayon. Build via maturin. Python package at `shogi-engine/python/shogi_gym/`.

After writing the plan, execute it using `superpowers:subagent-driven-development` in the existing `.worktrees/shogi-core` worktree on the same feature branch.
