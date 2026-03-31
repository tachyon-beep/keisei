# Prompt: Archive Python Shogi Engine

---

The `keisei/shogi/` directory contains the original Python Shogi game engine (~3,200 lines). We're replacing it with a Rust engine (in `shogi-engine/`), but we need the Python version as a reference/test oracle during migration.

Move the Python engine into an archive location so it's preserved but out of the active development path:

1. Create `keisei/shogi_python_reference/` 
2. Move all files from `keisei/shogi/` into `keisei/shogi_python_reference/`
3. Update the `__init__.py` in the new location so its internal imports still work
4. Find all imports of `keisei.shogi` in the codebase (grep for `from keisei.shogi` and `import keisei.shogi`) and update them to `keisei.shogi_python_reference`
5. Run `uv run pytest tests/unit/ -x -q` to verify nothing is broken
6. Commit with message: `refactor: archive Python shogi engine to shogi_python_reference/`

**Do NOT touch anything in `shogi-engine/` or `.worktrees/`.**
