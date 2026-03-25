# Phase 2: Infrastructure Cleanup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring Keisei's infrastructure to production-ready state — fix bugs, update dependencies, clean up code debt, and re-enable CI.

**Architecture:** Seven independent work items executed in dependency order. Each produces a self-contained commit. All work happens on a single feature branch off main.

**Tech Stack:** Python 3.13, PyTorch 2.11.0, pytest, GitHub Actions, uv

**Spec:** `docs/superpowers/specs/2026-03-25-phase2-infrastructure-cleanup-design.md`

---

## Setup

- [ ] **Create feature branch**

```bash
git checkout main
git checkout -b phase2-infrastructure-cleanup
```

---

## Task 1: Fix Elo Promotion Baseline Bug (2g)

**Files:**
- Modify: `keisei/training/callbacks.py:82-271` (sync path) and `keisei/training/callbacks.py:291-452` (async path)
- Test: `tests/unit/test_callbacks_lineage.py`

### Bug Summary

`old_rating` is captured at line 229 (sync) and line 391 (async) AFTER `evaluate_current_agent()` has already run and updated the Elo registry on disk. `new_rating` then reads the same already-updated value, so `new_rating > old_rating` is always false. Promotion events never fire.

**Fix:** Capture `old_rating` before `evaluate_current_agent()` is called, pass it into the Elo tracking block.

- [ ] **Step 1: Write failing tests for the sync path**

Add to `tests/unit/test_callbacks_lineage.py`:

```python
def test_emit_model_promoted_fires_on_elo_improvement(self):
    """Verify promotion fires when evaluation improves Elo rating."""
    # Setup: mock EloRegistry to return different ratings before/after eval
    # old_rating=1000 before evaluation, new_rating=1050 after
    # Assert: emit_model_promoted called with from_rating=1000, to_rating=1050

def test_emit_model_promoted_does_not_fire_when_rating_unchanged(self):
    """Verify promotion does NOT fire when rating stays the same."""
    # Setup: mock EloRegistry to return same rating before and after
    # Assert: emit_model_promoted NOT called

def test_emit_model_promoted_does_not_fire_when_rating_worsens(self):
    """Verify promotion does NOT fire when rating drops."""
    # Setup: mock EloRegistry to return lower rating after eval
    # Assert: emit_model_promoted NOT called
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_callbacks_lineage.py -v -k "promoted" --tb=short
```

Expected: FAIL — old_rating equals new_rating because both are read post-evaluation.

- [ ] **Step 3: Fix sync path — capture old_rating before evaluation**

In `keisei/training/callbacks.py`, the sync `on_step_end` method:

1. Before the `evaluate_current_agent()` call (~line 173), add:
```python
# Capture pre-evaluation Elo baseline for promotion detection
pre_eval_rating = None
if getattr(self.eval_cfg, "elo_registry_path", None):
    try:
        pre_eval_registry = EloRegistry(Path(self.eval_cfg.elo_registry_path))
        pre_eval_rating = pre_eval_registry.get_rating(trainer.run_name)
    except (OSError, RuntimeError, ValueError):
        pass
```

2. In the Elo tracking block (~line 256), replace:
```python
new_rating = registry.get_rating(trainer.run_name)
if new_rating > old_rating:
```
with:
```python
new_rating = registry.get_rating(trainer.run_name)
if pre_eval_rating is not None and new_rating > pre_eval_rating:
```

3. Update the `emit_model_promoted` call to use `pre_eval_rating`:
```python
trainer.model_manager.emit_model_promoted(
    from_rating=pre_eval_rating,
    to_rating=new_rating,
    ...
)
```

- [ ] **Step 4: Fix async path — same pattern**

In `keisei/training/callbacks.py`, the `_run_evaluation_async` method:

1. Before `evaluate_current_agent_async()` call (~line 336), capture `pre_eval_rating` with the same pattern.
2. In the Elo tracking block (~line 391), use `pre_eval_rating` for the promotion check.

- [ ] **Step 5: Write and run async path tests**

Add matching tests for the async path:
```python
def test_async_emit_model_promoted_fires_on_elo_improvement(self):
    """Async path: promotion fires when evaluation improves Elo."""

def test_async_emit_model_promoted_does_not_fire_when_unchanged(self):
    """Async path: promotion does NOT fire when rating unchanged."""
```

- [ ] **Step 6: Run all callback lineage tests**

```bash
pytest tests/unit/test_callbacks_lineage.py -v --tb=short
```

Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add keisei/training/callbacks.py tests/unit/test_callbacks_lineage.py
git commit -m "fix(lineage): capture Elo baseline before evaluation for promotion detection"
```

---

## Task 2: Dependency Updates (2f)

**Files:**
- Modify: `pyproject.toml:22-32` (dependencies)

- [ ] **Step 1: Update PyTorch and check CUDA compatibility**

```bash
source .venv/bin/activate
uv pip install torch==2.11.0 --upgrade
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected: `2.11.0 True`

If CUDA breaks, fall back to current: `uv pip install torch==2.10.0 --upgrade`

- [ ] **Step 2: Update pyproject.toml torch constraint**

In `pyproject.toml` line 23, change:
```toml
"torch>=2.0.0",
```
to:
```toml
"torch>=2.11.0",
```

- [ ] **Step 3: Update direct dependencies**

```bash
uv pip install --upgrade numpy scipy wandb pydantic PyYAML Jinja2 requests python-dotenv
```

Note: `protobuf` and `filelock` are transitive dependencies (not in pyproject.toml) — they'll be updated transitively.

- [ ] **Step 4: Update pyproject.toml version constraints to match installed versions**

Check installed versions of direct dependencies with `uv pip list` and update lower bounds in pyproject.toml to match. Only update packages that appear in the `[project.dependencies]` section.

- [ ] **Step 5: Install pip-audit and check for vulnerabilities**

```bash
uv pip install pip-audit
pip-audit 2>&1 | head -40
```

Address any critical/high CVEs by upgrading the affected packages.

- [ ] **Step 6: Run full unit test suite**

```bash
pytest tests/unit/ -q --tb=short
```

Expected: 1322+ tests pass (some count may change with new deps).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml
git commit -m "build: update PyTorch to 2.11.0 and refresh all dependencies"
```

---

## Task 3: Fix Circular Dependency (2e)

**Files:**
- Modify: `keisei/utils/compilation_validator.py:17-21`
- Modify: `keisei/utils/performance_benchmarker.py:22`

- [ ] **Step 1: Convert compilation_validator.py to lazy imports**

Remove only line 17 (the `ActorCriticProtocol` import). Do NOT remove lines 18-21 (the `PerformanceBenchmarker` import — that's a same-package import and must stay).

In each function that uses `ActorCriticProtocol`, add a lazy import:
```python
from keisei.core.actor_critic_protocol import ActorCriticProtocol  # pylint: disable=import-outside-toplevel
```

Follow the same pattern already used in `keisei/utils/agent_loading.py`.

- [ ] **Step 2: Convert performance_benchmarker.py to lazy imports**

Same pattern — remove top-level import at line 22, add lazy import inside each function that uses `ActorCriticProtocol`.

- [ ] **Step 3: Verify circular dep is broken**

```bash
python -c "from keisei.utils import compilation_validator; print('OK')"
python -c "from keisei.utils import performance_benchmarker; print('OK')"
```

Expected: Both print `OK` without importing keisei.core at module level.

- [ ] **Step 4: Run unit tests**

```bash
pytest tests/unit/ -q --tb=short
```

Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/utils/compilation_validator.py keisei/utils/performance_benchmarker.py
git commit -m "refactor: break core<->utils circular dep with lazy imports"
```

---

## Task 4: Replace print() with unified_logger (2b)

**Files:**
- Modify: `keisei/utils/profiling.py` (13 calls — lines 105-113, 127-131, 293-294, 299)
- Modify: `keisei/evaluation/analytics/elo_tracker.py` (9 calls)
- Modify: `keisei/evaluation/analytics/report_generator.py` (8 calls)
- Modify: `keisei/training/display_manager.py` (5 calls — lines 58, 81-83, 86)
- Modify: `keisei/evaluation/analytics/performance_analyzer.py` (3 calls)
- Modify: `keisei/evaluation/core_manager.py` (1 call)
- Modify: `keisei/shogi/shogi_rules_logic.py` (11 commented-out — delete)
- Modify: `keisei/shogi/shogi_game.py` (2 commented-out — delete)

- [ ] **Step 1: Replace profiling.py print() calls**

Add import at top of file:
```python
from keisei.utils.unified_logger import log_info_to_stderr
```

Replace all `print(msg)` calls in `print_summary()` (lines 105-131) with `log_info_to_stderr("profiling", msg)`.

**Important:** `log_info_to_stderr(component, message)` and `log_error_to_stderr(component, message)` both require a `component: str` first argument. Use the module name (e.g., `"profiling"`, `"elo_tracker"`, `"display"`, etc.) as the component for each file.

Lines 293-294, 299 are in dead code (`example_usage()` and `__main__` block) — delete these functions entirely.

- [ ] **Step 2: Replace elo_tracker.py print() calls**

Add import and replace 9 print() calls with `log_info_to_stderr()`.

- [ ] **Step 3: Replace report_generator.py print() calls**

Add import and replace 8 print() calls with `log_info_to_stderr()`.

- [ ] **Step 4: Replace display_manager.py print() calls**

Add import and replace 5 `print(..., file=sys.stderr)` calls with `log_info_to_stderr()`. Remove the `sys` import if no longer used.

- [ ] **Step 5: Replace performance_analyzer.py print() calls**

Add import and replace 3 print() calls with `log_info_to_stderr()`.

- [ ] **Step 6: Replace core_manager.py print() call**

Add import and replace 1 print() call at line 365 with `log_error_to_stderr()`.

- [ ] **Step 7: Delete commented-out prints in shogi files**

In `keisei/shogi/shogi_rules_logic.py`: delete all 11 commented-out `# print(...)` lines.
In `keisei/shogi/shogi_game.py`: delete both commented-out `# print(...)` lines.

- [ ] **Step 8: Verify no stray print() calls remain**

```bash
grep -rn "^[^#]*print(" keisei/ --include="*.py" | grep -v unified_logger.py
```

Expected: No output (only unified_logger.py should have print calls).

- [ ] **Step 9: Run unit tests**

```bash
pytest tests/unit/ -q --tb=short
```

Expected: ALL PASS

- [ ] **Step 10: Commit**

```bash
git add keisei/
git commit -m "refactor: replace all print() calls with unified_logger"
```

---

## Task 5: Apply Test Markers (2c)

**Files:**
- Modify: 52 files in `tests/unit/`
- Modify: 8 files in `tests/integration/`
- Modify: 3 files in `tests/e2e/`
- Modify: 4 files in `tests/evaluation/`
- Modify: 6 files in `tests/webui/`
- Delete: 7 empty directories (`tests/core/`, `tests/display/`, `tests/parallel/`, `tests/performance/`, `tests/shogi/`, `tests/training/`, `tests/utils/`)

- [ ] **Step 1: Add pytestmark to all unit test files**

For every `.py` file in `tests/unit/` (except `conftest.py` and `__init__.py`), add near the top after imports:

```python
import pytest

pytestmark = pytest.mark.unit
```

If `pytest` is already imported, just add the `pytestmark` line.

- [ ] **Step 2: Add pytestmark to integration test files**

Same pattern for `tests/integration/*.py` and `tests/evaluation/*.py`:

```python
pytestmark = pytest.mark.integration
```

- [ ] **Step 3: Add pytestmark to e2e test files**

Same pattern for `tests/e2e/*.py`:

```python
pytestmark = pytest.mark.e2e
```

- [ ] **Step 4: Add pytestmark to webui test files**

Same pattern for `tests/webui/*.py`:

```python
pytestmark = pytest.mark.unit
```

- [ ] **Step 5: Delete empty legacy test directories**

```bash
rm -rf tests/core/ tests/display/ tests/parallel/ tests/performance/ tests/shogi/ tests/training/ tests/utils/
```

- [ ] **Step 6: Verify markers work**

```bash
pytest -m unit --co -q 2>&1 | tail -3
pytest -m integration --co -q 2>&1 | tail -3
pytest -m e2e --co -q 2>&1 | tail -3
```

Expected: unit collects ~1300+ tests, integration collects ~50+, e2e collects a handful.

- [ ] **Step 7: Run full suite to confirm nothing broke**

```bash
pytest tests/ -q --tb=short
```

Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add tests/
git commit -m "test: apply pytest markers to all test files, delete empty test dirs"
```

---

## Task 6: Re-enable CI Pipeline (2a)

**Files:**
- Rename: `.github/workflows/ci.yml.disabled` → `.github/workflows/ci.yml`
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Rename and update CI workflow**

```bash
mv .github/workflows/ci.yml.disabled .github/workflows/ci.yml
```

- [ ] **Step 2: Update trigger branches**

In `.github/workflows/ci.yml`, change:
```yaml
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
```
to:
```yaml
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
```

- [ ] **Step 3: Update test job to use unit markers on PRs**

In the `test` job, change the pytest step:
```yaml
    - name: Test with pytest and collect coverage
      run: |
        pytest tests/ -v --tb=short --cov=keisei --cov-report=xml --cov-report=term-missing
```
to:
```yaml
    - name: Unit tests with coverage
      run: |
        pytest -m unit -v --tb=short --cov=keisei --cov-report=xml --cov-report=term-missing
```

- [ ] **Step 4: Update integration test job to use markers**

In the `integration-test` job, change:
```yaml
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v --tb=short
```
to:
```yaml
    - name: Run integration tests
      run: |
        pytest -m integration -v --tb=short
```

- [ ] **Step 5: Add pip-audit to security-scan job**

In the `security-scan` job, after the bandit step, add:
```yaml
    - name: Dependency audit
      run: |
        pip install pip-audit
        pip-audit
```

- [ ] **Step 6: Verify CI config is valid**

```bash
# Quick syntax check
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))" && echo "Valid YAML"
```

- [ ] **Step 7: Commit**

```bash
git add .github/workflows/
git commit -m "ci: re-enable CI pipeline with test markers and pip-audit"
```

---

## Task 7: Break Up Oversized Functions (2d)

**Files:**
- Modify: `keisei/core/ppo_agent.py` (`learn()` — 219 lines starting at line 241)
- Modify: `keisei/training/callbacks.py` (`on_step_end()` — 195 lines, `_run_evaluation_async()` — 164 lines)
- Modify: `keisei/utils/agent_loading.py` (`load_evaluation_agent()` — 177 lines starting at line 14)
- Modify: `keisei/training/trainer.py` (`__init__()` — 167 lines starting at line 36)

### Approach

For each function: read it, identify natural extraction points, extract private methods, run tests. Do NOT change public API.

- [ ] **Step 1: Refactor ppo_agent.py::learn()**

Read the function and extract into private methods:
- `_compute_advantages_and_returns()` — GAE computation
- `_run_minibatch_updates()` — the inner epoch/minibatch loop
- `_log_learning_metrics()` — post-update logging

- [ ] **Step 2: Run PPO agent tests**

```bash
pytest tests/unit/ -k "ppo" -v --tb=short
```

Note: PPO tests are split across `test_ppo_learning.py`, `test_ppo_action_selection.py`, and `test_ppo_checkpoint.py`.

Expected: ALL PASS

- [ ] **Step 3: Commit ppo_agent refactor**

```bash
git add keisei/core/ppo_agent.py
git commit -m "refactor: break up PPOAgent.learn() into focused private methods"
```

- [ ] **Step 4: Refactor callbacks.py::on_step_end() and _run_evaluation_async()**

Extract into private methods:
- `_dispatch_evaluation()` — evaluation trigger logic
- `_update_elo_and_emit()` — Elo tracking + lineage emission
- `_build_elo_snapshot()` — snapshot construction

Apply same extraction pattern to both sync and async paths.

- [ ] **Step 5: Run callback tests**

```bash
pytest tests/unit/test_callbacks.py tests/unit/test_callbacks_lineage.py -v --tb=short
```

Expected: ALL PASS

- [ ] **Step 6: Commit callbacks refactor**

```bash
git add keisei/training/callbacks.py
git commit -m "refactor: break up callback methods into focused private methods"
```

- [ ] **Step 7: Refactor agent_loading.py::load_evaluation_agent()**

Extract:
- `_build_evaluation_config()` — config construction
- `_load_model_from_checkpoint()` — model loading
- `_create_evaluation_agent()` — agent instantiation

- [ ] **Step 8: Run agent loading tests**

```bash
pytest tests/unit/ -k "agent_loading" -v --tb=short
```

Expected: ALL PASS

- [ ] **Step 9: Commit agent_loading refactor**

```bash
git add keisei/utils/agent_loading.py
git commit -m "refactor: break up load_evaluation_agent() into focused helpers"
```

- [ ] **Step 10: Refactor trainer.py::__init__()**

Read the function — it may already delegate most work to SetupManager. If so, extract only remaining inline setup logic. If it's mostly delegation calls, skip this and document why.

- [ ] **Step 11: Run trainer tests**

```bash
pytest tests/unit/test_trainer.py -v --tb=short
```

Expected: ALL PASS

- [ ] **Step 12: Commit trainer refactor (if changes made)**

```bash
git add keisei/training/trainer.py
git commit -m "refactor: extract remaining inline setup from Trainer.__init__()"
```

- [ ] **Step 13: Verify no function exceeds 150 lines**

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

Expected: Exit code 0 (no violations).

- [ ] **Step 14: Run full unit test suite**

```bash
pytest tests/unit/ -q --tb=short
```

Expected: ALL PASS

---

## Final Verification

- [ ] **Run complete test suite**

```bash
pytest tests/ -q --tb=short
```

Expected: ALL PASS

- [ ] **Push and create PR**

```bash
git push -u origin phase2-infrastructure-cleanup
gh pr create --base main --head phase2-infrastructure-cleanup \
  --title "Phase 2: infrastructure cleanup" \
  --body "$(cat <<'EOF'
## Summary

Full infrastructure cleanup per spec in docs/superpowers/specs/2026-03-25-phase2-infrastructure-cleanup-design.md

- Fix Elo promotion baseline bug (lineage events were silently dropped)
- Update PyTorch to 2.11.0, refresh all dependencies
- Break core<->utils circular dependency with lazy imports
- Replace 39 print() calls with unified_logger
- Apply pytest markers to all test files
- Re-enable CI pipeline with markers and pip-audit
- Break up 5 oversized functions (>150 lines)

## Test plan

- [ ] CI passes (unit tests + linting + type check)
- [ ] `pip-audit` clean
- [ ] No function >150 lines
- [ ] `grep -rn "^[^#]*print(" keisei/` returns only unified_logger.py

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
