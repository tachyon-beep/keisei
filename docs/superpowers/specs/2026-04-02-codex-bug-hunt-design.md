# Codex Bug Hunt for Keisei

**Date:** 2026-04-02
**Purpose:** One-shot static analysis sweep of the Keisei codebase using OpenAI Codex

## Overview

Port the Elspeth `codex_bug_hunt.py` and its `codex_audit_common.py` infrastructure
to Keisei. Strip Elspeth-specific features (tier allowlist, deduplication, skill files,
priority organization). Rewrite the analysis prompt with Keisei-domain bug categories.

## Files

- `scripts/codex_audit_common.py` — shared infrastructure (subprocess, retry, rate
  limiting, evidence gating, logging, reporting)
- `scripts/codex_bug_hunt.py` — file discovery, prompt construction, batch orchestration

## What's Kept from Elspeth

### `codex_audit_common.py`

- Path/time utilities: `resolve_path`, `is_cache_path`, `utc_now`
- Progress: `AsyncTqdm` (tqdm with fallback)
- Batching: `chunked`
- Context loading: `load_context` (loads `CLAUDE.md`, no skill files)
- Evidence processing: `extract_section`, `has_file_line_evidence`,
  `evidence_quality_score`, `apply_evidence_gate`
- Subprocess: `run_codex_once`, `run_codex_with_retry_and_logging`
- Logging: `append_log`, `ensure_log_file`
- Reporting: `generate_summary`, `print_summary`, `priority_from_report`

### `codex_bug_hunt.py`

- File discovery: `_list_files` with `--root`, `--file-type`, `--paths-from`
- Git filters: `--changed-since`, `--branch`, `--commit-range`
- Batch execution: `_run_batches` with async concurrency
- CLI args: `--batch-size`, `--rate-limit`, `--skip-existing`, `--dry-run`,
  `--model`, `--extra-message`, `--context-files`

## What's Removed

- Tier model allowlist loading and injection (`_load_tier_allowlist`,
  `_allowlist_entries_for_file`)
- Deduplication against existing bugs (`_find_similar_bug`, `_merge_bug_reports`,
  `_deduplicate_and_merge`, `_calculate_bug_similarity`)
- Priority organization (`_organize_by_priority`)
- CLI flags: `--deduplicate`, `--bugs-dir`, `--organize-by-priority`
- From common: `replace_section`, `write_run_metadata`, `write_summary_file`,
  `write_findings_index`, `SKILL_FILES`, `escape_cell`, `get_git_commit`

## What's Changed

### Default Paths

- `--root`: `keisei` (source package directory)
- `--output-dir`: `docs/bugs/generated`
- `--template`: inline in the script (no separate template file)
- Log: `docs/bugs/CODEX_LOG.md`

### Bug Report Template

Inline markdown template with sections: Summary, Severity, Evidence,
Root Cause Hypothesis, Suggested Fix.

### Bug Categories (Keisei-domain)

1. **PyTorch / Tensor Issues** — device mismatches (CPU vs CUDA), gradient leaks
   through missing `.detach()`, in-place ops breaking autograd, dtype mismatches,
   missing `torch.no_grad()` in inference paths
2. **Checkpoint / State Management** — incomplete save/restore cycles, optimizer
   state not matching model, missing `model.eval()`/`model.train()` mode switches,
   LR scheduler state not persisted
3. **RL Training Loop** — reward shaping errors, advantage/GAE computation bugs,
   PPO clipping issues, episode boundary handling, incorrect discount factor
   application
4. **Resource Management** — GPU memory not freed, unclosed files, tensors
   accumulating on GPU without release, missing cleanup in error paths
5. **Error Handling Gaps** — silent failures, missing validation at config
   boundaries, unchecked tensor shapes, bare except blocks
6. **Concurrency / Async** — race conditions in training loop state, shared
   mutable state across async boundaries
7. **Data Pipeline** — shape mismatches between observation/action spaces,
   incorrect normalization, off-by-one in batch slicing, action masking errors

## Dependencies

Not added to `pyproject.toml` (one-shot tool). Install before running:

```bash
uv pip install pyrate-limiter tenacity
```

Optional: `tqdm` for nicer progress bars (falls back to stderr printing).

## Output Structure

```
docs/bugs/
├── CODEX_LOG.md                          # Execution log
└── generated/
    └── <mirrored source tree>/
        └── <file>.py.md                  # One report per source file
```

## Usage

```bash
# Dry run to see what would be scanned
uv run python scripts/codex_bug_hunt.py --dry-run

# Full scan with rate limiting
uv run python scripts/codex_bug_hunt.py --rate-limit 20

# Scan only files changed on current branch
uv run python scripts/codex_bug_hunt.py --branch main
```
