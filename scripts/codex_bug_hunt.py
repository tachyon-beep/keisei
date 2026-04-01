#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

from codex_audit_common import (  # type: ignore[import-not-found]
    AsyncTqdm,
    chunked,
    ensure_log_file,
    generate_summary,
    is_cache_path,
    load_context,
    print_summary,
    resolve_path,
    run_codex_with_retry_and_logging,
)
from pyrate_limiter import Duration, Limiter, Rate


def _is_python_file(path: Path) -> bool:
    """Check if path is a Python source file (not test)."""
    return path.suffix == ".py" and not path.name.startswith("test_")


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
    """Run analysis in batches. Returns statistics."""
    log_lock = asyncio.Lock()
    failed_files: list[tuple[Path, Exception]] = []
    total_gated = 0

    # Use pyrate-limiter for rate limiting
    rate_limiter = Limiter(Rate(rate_limit, Duration.MINUTE)) if rate_limit else None

    # Progress bar
    pbar = AsyncTqdm(total=len(files), desc="Analyzing files", unit="file")

    for batch in chunked(files, batch_size):
        tasks: list[asyncio.Task[dict[str, int]]] = []
        batch_files: list[Path] = []

        for file_path in batch:
            relative = file_path.relative_to(root_dir)
            output_path = output_dir / relative
            output_path = output_path.with_suffix(output_path.suffix + ".md")

            if skip_existing and output_path.exists():
                pbar.update(1)
                continue

            prompt = _build_prompt(file_path, context, extra_message)
            batch_files.append(file_path)

            task = asyncio.create_task(
                run_codex_with_retry_and_logging(
                    file_path=file_path,
                    output_path=output_path,
                    model=model,
                    prompt=prompt,
                    repo_root=repo_root,
                    log_path=log_path,
                    log_lock=log_lock,
                    file_display=str(file_path.relative_to(repo_root).as_posix()),
                    output_display=str(output_path.relative_to(repo_root).as_posix()),
                    rate_limiter=rate_limiter,
                    evidence_gate_summary_prefix="",
                )
            )
            tasks.append(task)

        # Wait for all tasks in batch to complete
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for file_path, result in zip(batch_files, results, strict=False):
                if isinstance(result, Exception):
                    failed_files.append((file_path, result))
                elif isinstance(result, dict):
                    total_gated += result["gated"]

                pbar.update(1)

    pbar.close()

    print(f"\n{'─' * 60}", file=sys.stderr)

    # Report failures
    if failed_files:
        print(f"\n⚠️  {len(failed_files)} files failed:", file=sys.stderr)
        for path, exc in failed_files[:10]:
            print(f"  {path.relative_to(repo_root)}: {exc}", file=sys.stderr)
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more (see {log_path})", file=sys.stderr)

    # Generate summary statistics
    summary: dict[str, int] = generate_summary(output_dir, no_defect_marker="No concrete bug found")
    summary["gated"] = total_gated
    return summary


def _paths_from_file(path_file: Path, repo_root: Path, root_dir: Path) -> list[Path]:
    selected: list[Path] = []
    lines = path_file.read_text(encoding="utf-8").splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        raw_path = Path(stripped)
        path = raw_path if raw_path.is_absolute() else (repo_root / raw_path).resolve()
        if not path.exists():
            raise RuntimeError(f"paths-from entry does not exist: {raw_path}")
        if path.is_dir():
            selected.extend([p for p in path.rglob("*") if p.is_file() and not is_cache_path(p)])
        else:
            if not is_cache_path(path):
                selected.append(path)
    return [path for path in selected if _is_under_root(path, root_dir)]


def _is_under_root(path: Path, root_dir: Path) -> bool:
    try:
        path.relative_to(root_dir)
        return True
    except ValueError:
        return False


def _changed_files_since(repo_root: Path, root_dir: Path, git_ref: str) -> list[Path]:
    try:
        root_rel = root_dir.relative_to(repo_root)
    except ValueError:
        root_rel = root_dir
    cmd = ["git", "diff", "--name-only", git_ref, "--", str(root_rel)]
    result = subprocess.run(cmd, cwd=repo_root, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git diff failed")
    selected = []
    for line in result.stdout.splitlines():
        rel = line.strip()
        if not rel:
            continue
        path = (repo_root / rel).resolve()
        if path.is_file() and _is_under_root(path, root_dir) and not is_cache_path(path):
            selected.append(path)
    return selected


def _changed_files_on_branch(repo_root: Path, root_dir: Path, base_branch: str) -> list[Path]:
    """Get files changed on current branch vs base branch using merge-base."""
    # Find merge base
    merge_base_cmd = ["git", "merge-base", base_branch, "HEAD"]
    result = subprocess.run(merge_base_cmd, cwd=repo_root, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git merge-base failed for {base_branch}")
    merge_base = result.stdout.strip()

    # Get diff from merge base to HEAD
    try:
        root_rel = root_dir.relative_to(repo_root)
    except ValueError:
        root_rel = root_dir
    cmd = ["git", "diff", "--name-only", f"{merge_base}..HEAD", "--", str(root_rel)]
    result = subprocess.run(cmd, cwd=repo_root, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git diff failed")

    selected = []
    for line in result.stdout.splitlines():
        rel = line.strip()
        if not rel:
            continue
        path = (repo_root / rel).resolve()
        if path.is_file() and _is_under_root(path, root_dir) and not is_cache_path(path):
            selected.append(path)
    return selected


def _changed_files_in_range(repo_root: Path, root_dir: Path, commit_range: str) -> list[Path]:
    """Get files changed in commit range (e.g., 'abc123..def456')."""
    # Validate range format
    if ".." not in commit_range:
        raise ValueError(f"Invalid commit range format: {commit_range}. Expected format: 'START..END'")

    # Get diff for the range
    try:
        root_rel = root_dir.relative_to(repo_root)
    except ValueError:
        root_rel = root_dir
    cmd = ["git", "diff", "--name-only", commit_range, "--", str(root_rel)]
    result = subprocess.run(cmd, cwd=repo_root, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git diff failed for range {commit_range}")

    selected = []
    for line in result.stdout.splitlines():
        rel = line.strip()
        if not rel:
            continue
        path = (repo_root / rel).resolve()
        if path.is_file() and _is_under_root(path, root_dir) and not is_cache_path(path):
            selected.append(path)
    return selected


def _list_files(
    *,
    root_dir: Path,
    repo_root: Path,
    changed_since: str | None,
    branch: str | None,
    commit_range: str | None,
    paths_from: Path | None,
    file_type: str,
) -> list[Path]:
    # Check mutual exclusivity of git filters
    git_filters = [changed_since, branch, commit_range]
    active_filters = [f for f in git_filters if f is not None]
    if len(active_filters) > 1:
        raise ValueError("Only one of --changed-since, --branch, or --commit-range can be used at a time")

    selected: set[Path] | None = None

    if changed_since:
        changed = set(_changed_files_since(repo_root, root_dir, changed_since))
        selected = changed if selected is None else selected & changed

    if branch:
        changed = set(_changed_files_on_branch(repo_root, root_dir, branch))
        selected = changed if selected is None else selected & changed

    if commit_range:
        changed = set(_changed_files_in_range(repo_root, root_dir, commit_range))
        selected = changed if selected is None else selected & changed

    if paths_from:
        listed = set(_paths_from_file(paths_from, repo_root, root_dir))
        selected = listed if selected is None else selected & listed

    if selected is None:
        selected = {path for path in root_dir.rglob("*") if path.is_file() and not is_cache_path(path)}

    # Apply file type filter
    if file_type == "python":
        selected = {p for p in selected if _is_python_file(p)}

    return sorted(selected)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Codex bug audits per file in batches.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan all Python files in keisei
  %(prog)s

  # Scan only changed files since HEAD~5
  %(prog)s --changed-since HEAD~5

  # Scan files changed on current branch vs main
  %(prog)s --branch main

  # Scan files changed in a specific commit range
  %(prog)s --commit-range abc123..def456

  # Dry run to see what would be scanned
  %(prog)s --dry-run

  # Use rate limiting for API quota management
  %(prog)s --rate-limit 30

  # Add extra context message (e.g., migration notes)
  %(prog)s --extra-message "Please note recent PPO refactor - see docs/plans/..."
        """,
    )
    parser.add_argument(
        "--root",
        default="keisei",
        help="Root directory to scan for files (default: keisei).",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/bugs/generated",
        help="Directory to write bug reports (default: docs/bugs/generated).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Maximum concurrent Codex runs per batch (default: 10).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override Codex model (passes --model to codex exec).",
    )
    parser.add_argument(
        "--changed-since",
        default=None,
        help="Only scan files changed since this git ref (e.g. HEAD~1).",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Compare against base branch to get files changed on current branch (e.g. 'main').",
    )
    parser.add_argument(
        "--commit-range",
        default=None,
        help="Only scan files changed in commit range (e.g. 'abc123..def456').",
    )
    parser.add_argument(
        "--paths-from",
        default=None,
        help="Path to a file containing newline-separated paths to scan.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already have an output report.",
    )
    parser.add_argument(
        "--file-type",
        default="python",
        choices=["python", "all"],
        help="Filter by file type (default: python, excludes tests).",
    )
    parser.add_argument(
        "--context-files",
        nargs="+",
        default=None,
        help="Additional context files beyond CLAUDE.md/ARCHITECTURE.md.",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=None,
        help="Max requests per minute (e.g., 30 for API quota management).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which files would be scanned without running analysis.",
    )
    parser.add_argument(
        "--extra-message",
        default=None,
        help="Additional context message to include in the analysis prompt (e.g., migration notes).",
    )

    args = parser.parse_args()

    # Validation
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.rate_limit is not None and args.rate_limit < 1:
        raise ValueError("--rate-limit must be >= 1")

    if shutil.which("codex") is None:
        raise RuntimeError("codex CLI not found on PATH")

    repo_root = Path(__file__).resolve().parents[1]
    root_dir = resolve_path(repo_root, args.root)
    output_dir = resolve_path(repo_root, args.output_dir)
    log_path = resolve_path(repo_root, "docs/bugs/CODEX_LOG.md")

    # List files to scan
    paths_from = resolve_path(repo_root, args.paths_from) if args.paths_from else None
    files = _list_files(
        root_dir=root_dir,
        repo_root=repo_root,
        changed_since=args.changed_since,
        branch=args.branch,
        commit_range=args.commit_range,
        paths_from=paths_from,
        file_type=args.file_type,
    )

    if not files:
        print(f"No files found under {root_dir}", file=sys.stderr)
        return 1

    # Dry run mode
    if args.dry_run:
        print(f"Would analyze {len(files)} files:")
        for f in files[:20]:
            print(f"  {f.relative_to(repo_root)}")
        if len(files) > 20:
            print(f"  ... and {len(files) - 20} more")
        return 0

    # Load context
    context_text = load_context(repo_root, extra_files=args.context_files)
    ensure_log_file(log_path, header_title="Codex Bug Hunt Log")

    # Run analysis
    stats = asyncio.run(
        _run_batches(
            files=files,
            output_dir=output_dir,
            model=args.model,
            repo_root=repo_root,
            skip_existing=args.skip_existing,
            batch_size=args.batch_size,
            root_dir=root_dir,
            log_path=log_path,
            context=context_text,
            rate_limit=args.rate_limit,
            extra_message=args.extra_message,
        )
    )

    # Print summary
    print_summary(stats, icon="🐛", title="Bug Hunt Summary")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
