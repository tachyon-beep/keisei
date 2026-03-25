# Phase 2: Full Infrastructure Cleanup

**Date**: 2026-03-25
**Status**: Design
**Branch**: TBD (off main, after tw5-lineage merge)

## Goal

Bring the project infrastructure to a production-ready state before the first real training run. Fix known bugs, re-enable CI, clean up code debt, and update dependencies.

## Work Items

### 2a. Re-enable CI Pipeline

**Current state**: `.github/workflows/ci.yml.disabled` — 84 lines, has flake8 + mypy + pytest + codecov + integration tests + bandit security scan. Disabled since RC1 cleanup (2026-02-08).

**Changes**:
- Rename `ci.yml.disabled` to `ci.yml`
- Update trigger branches: remove `develop` (doesn't exist), keep `main`
- Wire test markers into CI:
  - PR jobs: run `pytest -m unit` (fast feedback) with `--cov` for coverage
  - Push to main: run `pytest -m "unit or integration"` with `--cov` for full coverage
  - Configure Codecov to accept partial coverage on PRs (unit-only)
- Keep the existing `security-scan` job (bandit) as-is
- Update `codecov-action` from v4 (check latest)
- Verify `.github/actions/setup-project` works (Python 3.13 + uv)

**Acceptance**: CI runs green on a PR to main.

### 2b. Replace print() Calls with unified_logger

**Current state**: 39 print() calls across 6 source files (excluding 6 intentional calls in unified_logger.py itself).

| File | Count | Replacement |
|------|-------|-------------|
| `utils/profiling.py` | 13 | `log_info_to_stderr()` |
| `evaluation/analytics/elo_tracker.py` | 9 | `log_info_to_stderr()` |
| `evaluation/analytics/report_generator.py` | 8 | `log_info_to_stderr()` |
| `training/display_manager.py` | 5 | Already uses `file=sys.stderr`; consolidate to logger |
| `evaluation/analytics/performance_analyzer.py` | 3 | `log_info_to_stderr()` |
| `evaluation/core_manager.py` | 1 | `log_error_to_stderr()` |

Commented-out debug `print()` calls in `shogi/shogi_rules_logic.py` (11) and `shogi/shogi_game.py` (2) should be deleted rather than replaced.

`display_manager.py` already uses `print(..., file=sys.stderr)` — replace these with `log_info_to_stderr()` for consistency.

**Approach**: Mechanical replacement. Each print() becomes the appropriate log level (info for output, warning/error for error cases). Preserve the same output content.

**Acceptance**: `grep -rn "^[^#]*print(" keisei/ --include="*.py"` returns only the 6 intentional calls in unified_logger.py. No commented-out print() calls remain in shogi/ files.

### 2c. Apply Test Markers

**Current state**: Markers defined in pyproject.toml (`unit`, `integration`, `e2e`, `slow`, `performance`) but not applied to any test functions. 66+ test files unmarked.

**Approach**: Add file-level `pytestmark` declarations rather than decorating individual functions:
- `tests/unit/*.py`: `pytestmark = pytest.mark.unit`
- `tests/integration/*.py`: `pytestmark = pytest.mark.integration`
- `tests/e2e/*.py`: `pytestmark = pytest.mark.e2e`
- `tests/evaluation/*.py`: `pytestmark = pytest.mark.integration`
- `tests/webui/*.py`: `pytestmark = pytest.mark.unit`

Delete empty legacy test directories: `tests/core/`, `tests/display/`, `tests/parallel/`, `tests/performance/`, `tests/shogi/`, `tests/training/`, `tests/utils/` (only contain `__pycache__`).

**Acceptance**: `pytest -m unit --co -q` collects the same tests as `pytest tests/unit/ tests/webui/ --co -q`. `pytest -m "unit or integration" --co -q` collects all tests.

### 2d. Break Up Oversized Functions (Top 5)

Target functions (>150 lines):

| File | Function | Lines | Strategy |
|------|----------|-------|----------|
| `core/ppo_agent.py` | `learn()` | 219 | Extract mini-batch loop, advantage computation, logging into private methods |
| `training/callbacks.py` | `on_step_end()` | 195 | Extract evaluation dispatch, Elo tracking, lineage emission into focused methods |
| `utils/agent_loading.py` | `load_evaluation_agent()` | 177 | Extract config construction, model loading, agent creation into helpers |
| `training/trainer.py` | `__init__()` | 167 | Already delegates to SetupManager; verify and extract any remaining inline setup |
| `training/callbacks.py` | `_run_evaluation_async()` | 164 | Mirror the sync refactor structure |

**Constraints**:
- Preserve existing public API — only extract private methods
- Each extracted method must be testable in isolation
- Prefer keeping methods in the same class/module; use a new private module only if the file exceeds 500 lines post-refactor

**Acceptance**: No function in the project exceeds 150 lines. Existing tests still pass. Verified with:
```bash
python -c "
import ast, pathlib, sys
violations = []
for path in pathlib.Path('keisei').rglob('*.py'):
    src = path.read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, 'end_lineno', None)
            if end and end - node.lineno > 150:
                violations.append(f'{path}:{node.lineno} {node.name} ({end - node.lineno} lines)')
for v in violations: print(v)
sys.exit(1 if violations else 0)
"
```

### 2e. Fix Circular Dependency (core <-> utils)

**Current state**: `core → utils` is fine (logging imports). `utils → core` has two problematic direct imports:
- `utils/compilation_validator.py` imports `core.actor_critic_protocol.ActorCriticProtocol`
- `utils/performance_benchmarker.py` imports `core.actor_critic_protocol.ActorCriticProtocol`

**Fix**: Convert these to lazy imports (same pattern as `agent_loading.py`), or accept a `Protocol`-typed parameter instead of importing the concrete protocol.

**Acceptance**: `python -c "from keisei.utils import compilation_validator"` works without importing `keisei.core`.

### 2f. Dependency Updates

**Current state**: GitHub flagged 14 vulnerabilities (2 critical, 8 high, 3 moderate, 1 low). PyTorch currently at 2.7.0.

**Changes**:
- Update PyTorch to 2.11.0 (latest available, user-requested)
- Update protobuf, filelock to latest
- Address remaining security vulnerabilities as flagged by Dependabot
- Run full test suite after each major dependency bump
- Add `pip-audit` to CI security-scan job for dependency vulnerability checking

**Rollback**: If PyTorch 2.11.0 introduces breaking changes, fall back to 2.9.1 (last minor before current).

**Acceptance**: `pip-audit` reports no critical/high CVEs. `pytest tests/unit/ -q` passes with new dependencies.

### 2g. Fix Elo Promotion Baseline Bug

**Current state**: In `callbacks.py`, `old_rating` is captured *after* `evaluate_current_agent()` has already run and updated the Elo registry. Then `new_rating` reads the same (already-updated) value, so `new_rating > old_rating` is never true. Promotion lineage events are silently dropped.

The same pattern exists in both the sync callback (`on_step_end`, ~line 229) and the async callback (`_run_evaluation_async`).

**Fix**: Capture `old_rating` *before* calling `evaluate_current_agent()`. Pass it through to the Elo tracking block.

**Acceptance**: Unit tests cover both the sync (`on_step_end`) and async (`_run_evaluation_async`) paths. Each test confirms that `emit_model_promoted` fires when evaluation improves Elo, and does not fire when rating is unchanged or worsens.

## Ordering

1. **2g** (bug fix) — small, high-value, unblocks lineage correctness
2. **2f** (dependencies) — do early so all subsequent work is on current deps
3. **2e** (circular dep) — low risk, fix before CI catches it as a false positive
4. **2b** (print→logger) — mechanical, low risk
5. **2c** (test markers) — mechanical, low risk
6. **2a** (CI) — depends on markers being applied
7. **2d** (function breakup) — highest risk, do last with full CI backing

## Phase 2 Definition of Done

All 7 items complete, CI green on main, `pip-audit` clean, no function >150 lines, full unit test suite passing. Phase 2 is merged as a single PR (or series of stacked PRs if the diff is too large for review).

## Out of Scope

- Functions 100-150 lines (diminishing returns)
- Test suite rewrite (separate effort)
- Evaluation subsystem simplification (separate effort)
- WebUI board visualization (Phase 3, separate spec)
