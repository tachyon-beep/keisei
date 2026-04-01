# Codex Bug Hunt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the Elspeth codex bug hunt infrastructure to Keisei for a one-shot static analysis sweep using OpenAI Codex.

**Architecture:** Two scripts in `scripts/` — a trimmed `codex_audit_common.py` with shared infrastructure (subprocess management, retry, rate limiting, evidence gating, logging) and an adapted `codex_bug_hunt.py` with Keisei-domain bug categories. No tests — these are one-shot operational scripts.

**Tech Stack:** Python 3.12+, `codex` CLI, `pyrate-limiter`, `tenacity`, optional `tqdm`

**Spec:** `docs/superpowers/specs/2026-04-02-codex-bug-hunt-design.md`

---

### Task 1: Create `scripts/codex_audit_common.py`

**Files:**
- Create: `scripts/codex_audit_common.py`
- Source: `/home/john/elspeth/scripts/codex_audit_common.py`

Copy from Elspeth, then remove unused functions and the `SKILL_FILES` constant.

- [ ] **Step 1: Create scripts directory and copy the file**

```bash
mkdir -p scripts
cp /home/john/elspeth/scripts/codex_audit_common.py scripts/codex_audit_common.py
```

- [ ] **Step 2: Remove unused exports**

Remove these functions/constants that are only used by other Elspeth audit scripts or by removed bug hunt features:

- `SKILL_FILES` constant (Elspeth-specific skill paths)
- `escape_cell` function (only used by `append_log` internally — keep if so, check)
- `get_git_commit` function (used by `write_run_metadata` which is removed)
- `replace_section` function (used by deduplication which is removed)
- `write_run_metadata` function (used by other audit scripts)
- `write_summary_file` function (used by other audit scripts)
- `write_findings_index` function (used by other audit scripts)

**Keep everything else** — `resolve_path`, `is_cache_path`, `utc_now`, `AsyncTqdm`, `chunked`, `load_context`, `extract_section`, `has_file_line_evidence`, `evidence_quality_score`, `apply_evidence_gate`, `run_codex_once`, `run_codex_with_retry_and_logging`, `append_log`, `ensure_log_file`, `generate_summary`, `print_summary`, `priority_from_report`, and all constants (`MAX_RETRIES`, `RETRY_*`, `STDERR_TRUNCATE_CHARS`, `EXCLUDE_DIRS`, `EXCLUDE_SUFFIXES`).

Note: `escape_cell` IS used by `append_log` which we keep — so keep `escape_cell` too.

Update the module docstring to reference Keisei instead of Elspeth.

- [ ] **Step 3: Update `load_context` to remove skill file loading**

Remove the `include_skills` parameter and the `SKILL_FILES` iteration from `load_context`. The function should just load `CLAUDE.md` and any extra files:

```python
def load_context(
    repo_root: Path,
    extra_files: list[str] | None = None,
) -> str:
    """Load context from CLAUDE.md and optional extra files for agent prompts."""
    parts = []

    claude_md = repo_root / "CLAUDE.md"
    if claude_md.exists():
        parts.append(f"--- CLAUDE.md ---\n{claude_md.read_text(encoding='utf-8')}")

    if extra_files:
        for filename in extra_files:
            path = repo_root / filename
            if path.exists():
                parts.append(f"--- {filename} ---\n{path.read_text(encoding='utf-8')}")
            else:
                print(f"Warning: Context file not found: {filename}", file=sys.stderr)

    return "\n\n".join(parts)
```

- [ ] **Step 4: Verify the file parses**

```bash
uv run python -c "import ast; ast.parse(open('scripts/codex_audit_common.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/codex_audit_common.py
git commit -m "feat: add codex audit common infrastructure (ported from elspeth)"
```

---

### Task 2: Create `scripts/codex_bug_hunt.py`

**Files:**
- Create: `scripts/codex_bug_hunt.py`
- Source: `/home/john/elspeth/scripts/codex_bug_hunt.py`

Copy from Elspeth, strip Elspeth-specific features, rewrite the prompt.

- [ ] **Step 1: Copy the file**

```bash
cp /home/john/elspeth/scripts/codex_bug_hunt.py scripts/codex_bug_hunt.py
```

- [ ] **Step 2: Remove Elspeth-specific imports and functions**

Remove these imports from the top:
- `import yaml`
- `from codex_audit_common import ... generate_summary` — update to remove `replace_section` if it was imported

Remove these functions entirely:
- `_load_tier_allowlist`
- `_allowlist_entries_for_file`
- `_extract_file_references`
- `_calculate_bug_similarity`
- `_find_similar_bug`
- `_merge_bug_reports`
- `_deduplicate_and_merge`
- `_organize_by_priority`

- [ ] **Step 3: Rewrite `_build_prompt` with Keisei-domain bug categories**

Replace the entire `_build_prompt` function with:

```python
_BUG_TEMPLATE = """\
## Summary

[One-sentence description of the bug]

## Severity

- Severity: [critical | major | minor | trivial]
- Priority: [P0 | P1 | P2 | P3]

## Evidence

[File paths with line numbers, code snippets showing the issue]

## Root Cause Hypothesis

[Why this bug exists and what conditions trigger it]

## Suggested Fix

[Concrete code change to fix the issue]
"""


def _build_prompt(
    file_path: Path,
    context: str,
    extra_message: str | None = None,
) -> str:
    return (
        "You are a static analysis agent doing a deep bug audit of a Deep RL\n"
        "Shogi training system (Python + PyTorch).\n"
        f"Target file: {file_path}\n\n"
        "Instructions:\n"
        "- Use the bug report template below verbatim.\n"
        "- Fill in every section. If unknown, write 'Unknown'.\n"
        "- You may read any repo file to confirm integration behavior. Prefer\n"
        "  verification over speculation.\n"
        "- Report bugs only if the primary fix belongs in the target file.\n"
        "- If you find multiple distinct bugs, output one full template per bug,\n"
        "  separated by a line with only '---'.\n"
        "- If you find no credible bug, output one template with Summary set to\n"
        f"  'No concrete bug found in {file_path}', Severity 'trivial', Priority 'P3',\n"
        "  and Root Cause Hypothesis 'No bug identified'.\n"
        "- Evidence should cite file paths and line numbers when possible.\n"
        + (f"\n⚠️  IMPORTANT CONTEXT:\n{extra_message}\n" if extra_message else "")
        + "\n"
        "Bug Categories to Check:\n"
        "1. **PyTorch / Tensor Issues**:\n"
        "   - Device mismatches (CPU vs CUDA tensors in same operation)\n"
        "   - Gradient leaks through missing .detach() calls\n"
        "   - In-place operations breaking autograd graph\n"
        "   - dtype mismatches (float32 vs float64, int vs float)\n"
        "   - Missing torch.no_grad() in inference/evaluation paths\n"
        "   - Incorrect tensor shapes passed to operations\n"
        "\n"
        "2. **Checkpoint / State Management**:\n"
        "   - Incomplete save/restore (model, optimizer, scheduler, epoch, step)\n"
        "   - Optimizer state not matching model parameters after load\n"
        "   - Missing model.eval()/model.train() mode switches\n"
        "   - LR scheduler state not persisted or restored\n"
        "   - Random state not saved for reproducibility\n"
        "\n"
        "3. **RL Training Loop**:\n"
        "   - Reward shaping or scaling errors\n"
        "   - Advantage/GAE computation bugs (wrong discount, lambda)\n"
        "   - PPO clipping applied incorrectly\n"
        "   - Episode boundary handling (terminal vs truncated)\n"
        "   - Action masking errors (illegal moves not masked)\n"
        "   - Value function target computation errors\n"
        "\n"
        "4. **Resource Management**:\n"
        "   - GPU memory not freed (tensors accumulating without release)\n"
        "   - Unclosed files or database connections\n"
        "   - Missing cleanup in error/exception paths\n"
        "   - Context managers not used where needed\n"
        "\n"
        "5. **Error Handling Gaps**:\n"
        "   - Silent failures (empty except blocks, catch-all handlers)\n"
        "   - Missing validation at config boundaries\n"
        "   - Unchecked tensor shapes or dimensions\n"
        "   - Missing bounds checking on hyperparameters\n"
        "\n"
        "6. **Concurrency / Async Issues**:\n"
        "   - Race conditions in training loop state\n"
        "   - Shared mutable state across async boundaries\n"
        "   - Missing synchronization on concurrent access\n"
        "\n"
        "7. **Data Pipeline Issues**:\n"
        "   - Shape mismatches between observation/action spaces\n"
        "   - Incorrect normalization or denormalization\n"
        "   - Off-by-one errors in batch slicing or indexing\n"
        "   - Shogi-specific: illegal move encoding, board state representation\n"
        "\n"
        "Repository context (read-only):\n"
        f"{context}\n\n"
        "Bug report template:\n"
        f"{_BUG_TEMPLATE}\n"
    )
```

- [ ] **Step 4: Simplify `_run_batches`**

Remove these parameters and their usage from `_run_batches`:
- `organize_by_priority` (and the `_organize_by_priority` call)
- `bugs_open_dir` (and all deduplication logic)
- `deduplicate` (and the `_deduplicate_and_merge` call)
- `tier_allowlist` (and the `_allowlist_entries_for_file` call)

The `_build_prompt` call inside the loop becomes:
```python
prompt = _build_prompt(file_path, context, extra_message)
```

Remove `total_merged` tracking and the `stats["merged"]` assignment. The function signature becomes:

```python
async def _run_batches(
    *,
    files: list[Path],
    output_dir: Path,
    model: str | None,
    repo_root: Path,
    skip_existing: bool,
    batch_size: int,
    root_dir: Path,
    log_path: Path,
    context: str,
    rate_limit: int | None,
    extra_message: str | None = None,
) -> dict[str, int]:
```

- [ ] **Step 5: Update `main()` — change defaults and remove Elspeth args**

Change defaults:
- `--root` default: `"keisei"`
- `--output-dir` default: `"docs/bugs/generated"`

Remove these arguments:
- `--template` (template is now inline as `_BUG_TEMPLATE`)
- `--deduplicate`
- `--bugs-dir`
- `--organize-by-priority`

Remove from the body:
- `template_path` / `template_text` loading
- `tier_allowlist` loading
- `bugs_open_dir` resolution

Update the `load_context` call to remove `include_skills=True`:
```python
context_text = load_context(repo_root, extra_files=args.context_files)
```

Update the `_run_batches` call to match the simplified signature.

Update the log path:
```python
log_path = resolve_path(repo_root, "docs/bugs/CODEX_LOG.md")
```

Update the epilog examples to reference Keisei paths.

- [ ] **Step 6: Verify the file parses**

```bash
uv run python -c "import ast; ast.parse(open('scripts/codex_bug_hunt.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add scripts/codex_bug_hunt.py
git commit -m "feat: add codex bug hunt script (ported from elspeth, keisei-domain categories)"
```

---

### Task 3: Smoke test with dry run

**Files:**
- None (verification only)

- [ ] **Step 1: Run dry run**

```bash
uv run python scripts/codex_bug_hunt.py --dry-run
```

Expected: Lists Python files found under `keisei/` directory with count. Should find ~25 non-test Python files.

- [ ] **Step 2: Verify file filtering**

The dry run output should NOT include:
- Test files (`test_*.py`)
- `__pycache__` files
- Files outside `keisei/`

- [ ] **Step 3: Commit any fixes if needed**

Only if the dry run revealed issues that needed fixing.
