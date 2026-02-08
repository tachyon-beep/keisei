# Cross-Cutting Architecture Quality Assessment

**Date**: 2026-02-08
**Analyst**: Claude Opus 4.6
**Scope**: Error handling, security, test health, dependencies, CI/build

---

## Confidence Assessment

| Area | Confidence | Basis |
|------|-----------|-------|
| Error Handling | High | Full grep of codebase, read of key files |
| Security | High | Audited all torch.load calls, config schema, WebUI code |
| Test Suite Health | High | Read all conftest.py files, counted tests and markers |
| Dependency Health | High | Read pyproject.toml, searched for actual imports |
| CI/Build | High | Read CI config, local CI script, pre-commit config |

---

## 1. Error Handling Audit

**Quality Score: 2 / 5**
**Critical Issues: 0**
**High Issues: 3**

### Finding 1.1: Epidemic of Broad Exception Catches -- High

**Evidence:** 82 instances of `except Exception` across the codebase. The worst offenders by file:

| File | Count |
|------|-------|
| `/home/john/keisei/keisei/evaluation/strategies/single_opponent.py` | 13 |
| `/home/john/keisei/keisei/evaluation/core/background_tournament.py` | 6 |
| `/home/john/keisei/keisei/training/env_manager.py` | 6 |
| `/home/john/keisei/keisei/evaluation/enhanced_manager.py` | 7 |
| `/home/john/keisei/keisei/evaluation/strategies/benchmark.py` | 5 |
| `/home/john/keisei/keisei/training/session_manager.py` | 7 |
| `/home/john/keisei/keisei/evaluation/strategies/ladder.py` | 3 |
| `/home/john/keisei/keisei/evaluation/strategies/custom.py` | 3 |

`single_opponent.py` (894 lines) has 13 `except Exception` blocks. That is one catch-all every 69 lines. This is not error handling -- it is error suppression.

**Impact:** Bugs that should crash loudly instead produce confusing downstream failures. When everything is caught, nothing is debuggable. Training could silently produce garbage results because an exception was swallowed during evaluation.

**Recommendation:** Audit each `except Exception` individually. Most should catch specific exception types (RuntimeError, ValueError, FileNotFoundError). Reserve `except Exception` for top-level process boundaries only (e.g., the main training loop entrypoint in `train.py:331`).

### Finding 1.2: Swallowed Exceptions (except + pass/no-action) -- High

**Evidence:** Multiple instances where exceptions are caught and silently discarded:

- `/home/john/keisei/keisei/shogi/shogi_rules_logic.py:48-49` -- `except Exception: pass` in debug code path. If `to_sfen_string()` fails during a check detection, the failure is invisible.
- `/home/john/keisei/keisei/training/session_manager.py:453` -- `except Exception:` followed by `time.sleep(0.1)` in a retry loop for WandB finish. No logging of what failed.
- `/home/john/keisei/keisei/training/session_manager.py:469` -- `except Exception: pass` after `wandb.finish(exit_code=1)`. Complete silence on failure.
- `/home/john/keisei/keisei/evaluation/core/background_tournament.py:442` -- `except Exception:` in summary stats serialization with no logging.
- `/home/john/keisei/keisei/training/parallel/utils.py:27` -- `except Exception:` in `compress_array` falls back silently.
- `/home/john/keisei/keisei/shogi/shogi_game.py:900` -- `except Exception: return False` in `test_move_validity`. Any bug in move validation logic is silently treated as "invalid move".
- `/home/john/keisei/keisei/shogi/shogi_game.py:956` -- Same pattern in another validation method.
- `/home/john/keisei/keisei/evaluation/core/evaluation_result.py:330` -- `except Exception:` silently falls back to default EvaluationConfig when deserialization fails.

**Impact:** Silent failures mean bugs manifest far from their cause. The shogi game engine ones are particularly dangerous: a bug in move validation that throws an unexpected exception would be treated as "move is illegal", silently corrupting training data without any indication.

**Recommendation:** At minimum, add logging to every `except` block. For the game engine (`shogi_game.py:900,956`), log and re-raise to catch validation bugs early rather than silently producing incorrect training data.

### Finding 1.3: 55+ print() Calls Bypassing Unified Logger -- Medium

**Evidence:** The project has a well-designed `unified_logger` (`/home/john/keisei/keisei/utils/unified_logger.py`) but 55+ `print()` calls bypass it:

| Location | Count | Nature |
|----------|-------|--------|
| `keisei/shogi/shogi_rules_logic.py` | 7 active + 11 commented | Debug prints left in production code |
| `keisei/utils/profiling.py` | 8 | Profile output via print() |
| `keisei/evaluation/analytics/performance_analyzer.py` | 3 | Analytics output |
| `keisei/evaluation/analytics/report_generator.py` | 8 | Report output |
| `keisei/evaluation/analytics/elo_tracker.py` | 8 | Demo/example code in production module |
| `keisei/training/display_manager.py` | 4 | Intentional stderr output (acceptable) |
| `keisei/training/callback_manager.py` | 3 | Alignment warnings |
| `keisei/training/setup_manager.py` | 1 | Compile info |
| `keisei/evaluation/core_manager.py` | 1 | Fallback error message |

The `shogi_rules_logic.py` debug prints are gated behind a `debug_recursion` flag, but they use `print()` instead of a debug logger, meaning they cannot be redirected, filtered, or captured in log files.

The `elo_tracker.py` lines 206-234 are a `__main__` example block that uses `print()` -- acceptable but should be removed from a production module.

**Impact:** Log output is inconsistent. Debugging information goes to stdout/stderr unpredictably. Log aggregation tools cannot capture these messages.

**Recommendation:** Replace all non-display `print()` with `unified_logger` calls. Remove or relocate `__main__` example blocks.

### Strengths (Error Handling)

- The unified_logger itself is well-designed with file + stderr output, severity levels, and a clean API.
- Critical path code in `ppo_agent.py` has specific exception types for checkpoint loading errors.
- `training/utils.py:32` catches `(OSError, RuntimeError, EOFError, pickle.UnpicklingError)` -- a good example of targeted exception handling.

---

## 2. Security Assessment

**Quality Score: 3 / 5**
**Critical Issues: 0**
**High Issues: 2**

### Finding 2.1: WebUI Binds to 0.0.0.0 with No Authentication -- High

**Evidence:**
- `/home/john/keisei/keisei/config_schema.py:623` -- `host: str = Field("0.0.0.0", description="Server host (0.0.0.0 for all interfaces)")`
- `/home/john/keisei/keisei/webui/streamlit_manager.py:88-89` -- `"--server.address", self.config.host`
- `/home/john/keisei/keisei/webui/streamlit_app.py` -- No authentication logic anywhere in the file.

The Streamlit dashboard binds to all network interfaces by default and has zero authentication. Anyone on the network can view training state, which includes model configuration, training hyperparameters, and game state.

The previous assessment graded this A- for security and said "WebSocket/HTTP endpoints clearly documented as auth-free (intended for local/demo use)". That is an incorrect characterization. The default configuration binds to `0.0.0.0`, which is explicitly NOT local-only. If the intent is local use, the default should be `127.0.0.1`.

**Impact:** On a shared network (lab, cloud VM, container orchestrator), training metadata is exposed to any network peer. While this does not expose model weights directly (the state file contains metrics, not weights), it leaks configuration details and training progress.

**Recommendation:** Change the default host from `0.0.0.0` to `127.0.0.1`. Users who need network access can override it explicitly.

### Finding 2.2: Checkpoint Path Not Sanitized in Training Utils -- High

**Evidence:**
- `/home/john/keisei/keisei/training/utils.py:41` -- `checkpoints = glob.glob(os.path.join(model_dir_path, "*.pth"))` -- the `model_dir_path` comes from config without path traversal validation.
- `/home/john/keisei/keisei/training/utils.py:82` -- `run_artifact_dir = os.path.join(model_dir, run_name)` -- `run_name` is user-supplied and could contain `../`.
- `/home/john/keisei/keisei/training/utils.py:84` -- `log_file_path = os.path.join(run_artifact_dir, os.path.basename(log_file))` -- `os.path.basename` provides some protection for the log file, but not for the directory.

There is no validation that `model_dir_path` or `run_name` stay within expected boundaries. A crafted config YAML or CLI override with `--override logging.model_dir=../../../../etc/` could cause writes outside the project directory.

**Impact:** In a multi-user or automated pipeline environment, this is a path traversal risk. In single-user local development, the practical risk is low.

**Recommendation:** Validate that resolved paths stay within the project or a designated output directory. Use `pathlib.Path.resolve()` and verify the result starts with an expected prefix.

### Finding 2.3: torch.load() Security -- Low (Properly Mitigated)

**Evidence:** All `torch.load()` calls use `weights_only=True`:
- `/home/john/keisei/keisei/core/ppo_agent.py:501-502` -- `torch.load(file_path, map_location=self.device, weights_only=True)`
- `/home/john/keisei/keisei/evaluation/core/model_manager.py:111-112` -- `torch.load(checkpoint_path, map_location="cpu", weights_only=True)`
- `/home/john/keisei/keisei/evaluation/core_manager.py:93-94` -- `torch.load(agent_checkpoint, map_location="cpu", weights_only=True)`
- `/home/john/keisei/keisei/training/utils.py:30` -- `torch.load(checkpoint_path, map_location="cpu", weights_only=True)`
- All test files also use `weights_only=True`.

This is correct. The pickle deserialization attack vector is mitigated.

### Finding 2.4: No eval()/exec() Calls -- Low (Good)

**Evidence:** All `eval()` matches are `model.eval()` (PyTorch evaluation mode), not Python's `eval()`. No `exec()` calls found.

### Finding 2.5: No Hardcoded Secrets -- Low (Good)

**Evidence:** Searched for API_KEY, SECRET, PASSWORD, TOKEN patterns. Only matches are SFEN token references in game I/O code (game notation, not credentials). W&B keys are loaded via `python-dotenv` from `.env` files.

### Finding 2.6: pickle Import Without Direct Use -- Low

**Evidence:**
- `/home/john/keisei/keisei/training/utils.py:8` -- `import pickle` -- only used for `pickle.UnpicklingError` in exception handling on line 32.
- `/home/john/keisei/keisei/training/parallel/model_sync.py:10` -- `import pickle` -- imported but never called via `pickle.dump/load/dumps/loads`.

No actual pickle serialization/deserialization occurs. The `model_sync.py` import is dead code.

### Strengths (Security)

- Consistent `weights_only=True` on all torch.load calls -- this is better than most ML projects.
- No hardcoded credentials.
- W&B integration uses environment variables.
- Bandit and safety scanners are available in dev dependencies.

---

## 3. Test Suite Health

**Quality Score: 3 / 5**
**Critical Issues: 0**
**High Issues: 2**

### Finding 3.1: Test Markers Defined but Never Applied -- High

**Evidence:**
- `/home/john/keisei/pytest.ini:5-10` -- Defines markers: unit, integration, slow, performance, e2e.
- Grep for `@pytest.mark.(unit|integration|e2e|slow|performance)` across `/home/john/keisei/tests/` returns **0 matches**.

385 test functions exist across 29 test files. Zero of them have marker decorators. Running `pytest -m unit` collects nothing. The markers are purely decorative configuration.

**Impact:** Cannot run subsets of the test suite by category. Running all 385 tests is the only option, which means CI cannot run a fast unit-test tier separately from slow integration tests. This defeats the purpose of having test tiers in the first place.

**Recommendation:** Apply markers. A practical approach: auto-mark by directory using a `conftest.py` hook (e.g., all tests under `tests/unit/` get `@pytest.mark.unit` automatically).

### Finding 3.2: Test Fixture Duplication Across Tiers -- Medium

**Evidence:**
- `/home/john/keisei/tests/conftest.py` defines: `session_policy_mapper`, `app_config`, `shogi_game`, `cnn_model`, `resnet_model`, `ppo_agent`, `experience_buffer`
- `/home/john/keisei/tests/integration/conftest.py` defines: `session_policy_mapper` (duplicate), `integration_config` (near-identical to `app_config`), `cnn_model` (duplicate), `ppo_agent` (duplicate), `shogi_game` (duplicate), `experience_buffer` (duplicate)

The `session_policy_mapper` fixture is defined identically in both root and integration conftest files. `cnn_model`, `ppo_agent`, `shogi_game`, and `experience_buffer` are also duplicated with trivially different configurations.

**Impact:** Changes to fixture behavior must be made in multiple places. A fixture bug fixed in one conftest may not be fixed in the other. The integration conftest has 250 lines that are mostly copy-paste from the root conftest.

**Recommendation:** Define shared fixtures in the root `conftest.py`. Integration-specific fixtures should compose or override root fixtures, not redefine them.

### Finding 3.3: Test Count vs. Coverage Unknown -- Medium

**Evidence:**
- 385 test functions across 29 files. The previous assessment stated 1,218 tests. The discrepancy is likely from parametrized tests that expand at collection time.
- `pytest-cov` is in dev dependencies (`pyproject.toml:38`).
- No `.coveragerc` or coverage threshold configured anywhere.
- No coverage data committed to the repository.

There is no way to know what percentage of the codebase is actually tested. A high test count without measured coverage is vanity metrics.

**Recommendation:** Add `--cov-fail-under=70` (or a realistic threshold) to pytest configuration.

### Finding 3.4: Tests Can Run Without GPU -- Low (Good)

**Evidence:**
- `/home/john/keisei/tests/conftest.py:29` -- `device="cpu"` in EnvConfig
- `/home/john/keisei/tests/e2e/conftest.py:44` -- `env["CUDA_VISIBLE_DEVICES"] = ""` forces CPU in E2E tests
- All fixture configs use `device="cpu"` and `enable_torch_compile=False`.

Tests are properly configured for CPU-only execution. This is good.

### Finding 3.5: E2E Test Infrastructure is Well-Designed -- Low (Strength)

**Evidence:** `/home/john/keisei/tests/e2e/conftest.py` has:
- Subprocess-based test execution (line 65-72)
- Environment isolation with `WANDB_MODE=disabled` (line 41)
- GPU exclusion via `CUDA_VISIBLE_DEVICES=""` (line 44)
- Minimal fast configs for speed (64 timesteps, tiny models)

### Strengths (Testing)

- Tests use real components (not excessive mocking), which catches integration bugs.
- Session-scoped `PolicyOutputMapper` fixture avoids expensive recomputation.
- E2E tests run the actual CLI as a subprocess -- genuine end-to-end coverage.
- All tests run on CPU without GPU dependency.

---

## 4. Dependency Health

**Quality Score: 3 / 5**
**Critical Issues: 0**
**High Issues: 1**

### Finding 4.1: Two Unused Production Dependencies -- High

**Evidence:**
- `pyproject.toml:23` -- `Jinja2>=3.1.0` is a production dependency. Grep for `import jinja2` or `from jinja2` across `/home/john/keisei/keisei/` returns **0 matches**. Jinja2 is not used anywhere in the source code.
- `pyproject.toml:24` -- `requests>=2.31.0` is a production dependency. Grep for `import requests` or `from requests` across `/home/john/keisei/keisei/` returns **0 matches**. Requests is not used anywhere in the source code.

Both are required for installation but never imported. They add attack surface (Jinja2 has had template injection CVEs) and bloat the dependency tree.

**Impact:** Unnecessary dependencies increase install time, increase security exposure, and confuse developers about what the project actually uses.

**Recommendation:** Remove `Jinja2` and `requests` from production dependencies. If they were intended for future features, add them when those features are built, not before.

### Finding 4.2: No Root-Level requirements.txt -- Low (Good)

**Evidence:** No `requirements.txt` at the project root. The only `requirements.txt` files found are inside WandB run artifact directories (`models/*/wandb/*/files/requirements.txt`), which are auto-generated by WandB.

The previous assessment mentioned "stale requirements.txt" but this appears to have been cleaned up. `pyproject.toml` is the single source of truth.

However, `/home/john/keisei/scripts/run_local_ci.sh:39-40` still references `requirements.txt` and `requirements-dev.txt`:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

These files do not exist, so the local CI script will fail on step 1.

### Finding 4.3: Dependency Version Ranges Are Reasonable -- Low (Good)

**Evidence:** Dependencies use minimum version pins (`>=`) which is appropriate for a project not deployed as a library. Examples:
- `torch>=2.0.0` -- allows any PyTorch 2.x
- `pydantic>=2.11.0,<3.0` -- correctly pins major version with upper bound
- `numpy>=1.24.0` -- reasonable minimum

No dependencies are pinned to exact versions, which is correct for a development project (exact pinning belongs in lock files).

### Finding 4.4: scipy is Production-Unused but Declared -- Medium

**Evidence:**
- `pyproject.toml:18` -- `scipy>=1.10.0` is a production dependency.
- Only import: `/home/john/keisei/keisei/evaluation/analytics/advanced_analytics.py:16` -- `from scipy import stats as scipy_stats`
- The `evaluation/analytics/` module appears to be an optional analysis tool, not a core training dependency.

scipy is a heavy dependency (~50MB) used in exactly one file for optional analytics.

**Recommendation:** Move scipy to optional dependencies (e.g., `[analytics]`), or make the import lazy with a try/except like the Streamlit import pattern.

### Strengths (Dependencies)

- Pydantic correctly pinned with upper major version bound.
- Dev/prod dependency split exists with clear categories (testing, linting, type checking, security, profiling).
- WebUI (streamlit) correctly in optional dependencies.

---

## 5. Build & CI

**Quality Score: 2 / 5**
**Critical Issues: 0**
**High Issues: 2**

### Finding 5.1: CI Pipeline is Disabled -- High

**Evidence:**
- `/home/john/keisei/.github/workflows/ci.yml.disabled` -- The CI pipeline is disabled by file rename. It is not temporarily skipped via a workflow toggle; it is fully inoperative.
- The disabled CI was comprehensive: lint, type check, test, coverage upload, integration tests, parallel tests, performance profiling, and security scanning.

There is no automated quality gate. All code merges rely on developer discipline.

**Impact:** Regressions can be merged without detection. The test suite could be broken right now and nobody would know unless they run tests manually. The security scanning (bandit, safety) that was configured in CI is also not running.

**Recommendation:** Re-enable CI. If the full pipeline is too expensive, enable at minimum: lint (flake8 critical errors) + unit tests. This takes <5 minutes on GitHub Actions.

### Finding 5.2: Local CI Script is Broken -- High

**Evidence:** `/home/john/keisei/scripts/run_local_ci.sh`:
- Line 39: `pip install -r requirements.txt` -- file does not exist.
- Line 40: `pip install -r requirements-dev.txt` -- file does not exist.
- Line 83: `pytest tests/test_integration_smoke.py` -- this file likely does not exist in the new test structure (tests were reorganized into `tests/unit/`, `tests/integration/`, `tests/e2e/`).
- Line 59: `mypy keisei/ --ignore-missing-imports --no-strict-optional || true` -- mypy errors are completely ignored (always passes).
- Line 64: `bandit -r keisei/ -f json > bandit_report.json || true` -- security scan results are immediately deleted (line 67: `rm -f bandit_report.json`).
- Line 69: `print_status 0 "Security scan"` -- hardcoded to pass regardless of results.

The local CI script references a dead file structure and ignores every quality check except flake8. It provides false confidence.

**Impact:** A developer running `./scripts/run_local_ci.sh` will either get an immediate failure (requirements.txt missing) or, if they fix that, get a green result regardless of mypy errors, security issues, or missing tests.

**Recommendation:** Rewrite the script to use `uv pip install -e ".[dev]"` and point to the correct test directories.

### Finding 5.3: Pre-commit Hooks Configured and Installed -- Low (Good)

**Evidence:**
- `/home/john/keisei/.pre-commit-config.yaml` -- Comprehensive configuration with: trailing-whitespace, end-of-file, check-yaml, check-large-files, check-merge-conflict, debug-statements, black, isort, flake8, bandit, mypy.
- `/home/john/keisei/.git/hooks/pre-commit` -- Exists, confirming hooks are installed.

This is the actual quality gate. Since CI is disabled, pre-commit hooks are the only automated check running.

However, the bandit hook uses `--exit-zero` (line 41), meaning security findings never block a commit. And mypy excludes all tests (line 51: `exclude: ^(tests/|deprecated/)`).

### Finding 5.4: Disabled CI References Stale File Structure -- Medium

**Evidence:** The disabled CI config (`ci.yml.disabled`) references:
- Line 57: `tests/test_integration_smoke.py` -- likely does not exist after test restructuring.
- Line 72: `tests/test_parallel_smoke.py` -- likely does not exist after test restructuring.
- Line 98: `scripts/profile_training.py` -- needs verification.
- Line 143: `safety check -r requirements.txt` -- `requirements.txt` does not exist.

Even if CI were re-enabled, it would fail due to stale file references.

**Recommendation:** Update the CI config to match the current project structure before re-enabling.

### Strengths (Build/CI)

- Pre-commit hooks are configured and installed with a good set of checks.
- The disabled CI config is well-structured with appropriate job separation.
- GitHub Actions composite action exists for project setup (`.github/actions/setup-project/action.yml`).

---

## Risk Assessment

| Risk | Severity | Likelihood | Area |
|------|----------|-----------|------|
| Silent training corruption from swallowed game engine exceptions | High | Medium | Error Handling |
| Regressions merged without detection (no CI) | High | High | CI/Build |
| WebUI exposing training data on network (0.0.0.0 default) | High | Medium | Security |
| Local CI script giving false confidence | Medium | High | CI/Build |
| Cannot selectively run fast/slow tests | Medium | High | Testing |
| Unused dependencies increasing attack surface | Medium | Low | Dependencies |
| Broad exception catches masking bugs across evaluation subsystem | Medium | High | Error Handling |

---

## Information Gaps

1. **Coverage data**: No measured coverage exists. The actual coverage percentage of the 385 test functions is unknown.
2. **Pre-commit hook enforcement**: Cannot verify if hooks are enforced for all contributors (could be bypassed with `--no-verify`).
3. **Production deployment**: Unclear if this is only used locally or deployed to shared infrastructure, which changes the severity of the WebUI binding issue.
4. **Test collection expansion**: The 385 `def test_` functions may expand to 1,218 via parametrization, but this is not verified in this assessment.

---

## Caveats

1. This assessment evaluates cross-cutting concerns only. Subsystem-level architecture quality (separation of concerns, dependency direction, cohesion) was covered in the prior assessment at `/home/john/keisei/docs/arch-analysis-2026-02-06-1844/05-quality-assessment.md`.
2. The previous assessment graded Security at A-. This assessment downgrades it based on the `0.0.0.0` default binding, which is objectively insecure-by-default regardless of intended use case. Intended use case is a deployment decision, not an architecture quality.
3. Error handling severity is assessed based on impact to training correctness, not code aesthetics. The game engine exception swallowing (`shogi_game.py:900,956`) is rated High because it directly affects training data quality.
4. The "82 broad exception catches" count includes some that are defensible (e.g., top-level entrypoints in `train.py`). Approximately 15-20 are at appropriate boundaries. The remaining 60+ are not.

---

## Overall Cross-Cutting Score: 2.6 / 5

The project has solid foundations (good architecture, correct torch.load security, working pre-commit hooks) undermined by operational gaps (no CI, broken local CI, unused markers) and pervasive error suppression that risks silent training corruption. The evaluation subsystem accounts for a disproportionate share of the broad exception catches (35+ out of 82), suggesting it was built with a "never crash" philosophy that trades correctness for resilience.
