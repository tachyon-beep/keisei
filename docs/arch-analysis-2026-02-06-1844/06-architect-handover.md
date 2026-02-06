# Keisei Architect Handover Document

**Date**: 2026-02-06
**Purpose**: Enable informed architectural decisions and prioritized improvements
**Prepared by**: Claude Opus 4.6

---

## Current State Summary

Keisei is a **production-capable** DRL Shogi training system with:
- 21,586 LOC across 87 source files in 7 subsystems
- Manager-based architecture with 9 specialized components
- Full Shogi rules engine with 13,527-action space
- PPO with GAE, ResNet with SE blocks, mixed precision, torch.compile
- 5 evaluation strategies, WebUI streaming, W&B integration
- 1,218 tests across 113 test files

**Bottom line**: The architecture is sound. The infrastructure works. The priority is now RL training effectiveness.

---

## Architecture Strengths to Preserve

These patterns are working well and should be maintained:

1. **Manager pattern in training/** - Each manager is independently testable and has a clear responsibility. Resist the temptation to merge managers for "simplicity."

2. **ActorCriticProtocol** - The protocol interface enables model swapping. Any new model architecture should implement this protocol, not subclass a base class.

3. **Pydantic configuration** - Type-safe config catches errors at startup. All new configuration should use Pydantic models in `config_schema.py`.

4. **Shogi engine isolation** - `shogi/` has zero outbound dependencies. Keep it that way - game logic should never import from training, evaluation, or utils.

5. **Optional integrations** - W&B, WebUI, CUDA all disable cleanly. New integrations should follow this pattern.

---

## Technical Debt Inventory

### Critical (Fix Before Next Major Feature)

#### TD-1: core <-> utils Circular Dependency
- **Location**: `core/ppo_agent.py` imports from `utils` (PolicyOutputMapper, unified_logger); `utils/agent_loading.py` imports from `core` (PPOAgent)
- **Impact**: Makes testing and refactoring harder, prevents clean dependency layers
- **Fix**: Move `agent_loading.py` into `core/` package, or extract `PolicyOutputMapper` into `core/types.py`
- **Effort**: 1-2 hours
- **Risk**: Low (internal reorganization only)

#### TD-2: Stale requirements.txt Files
- **Location**: `requirements.txt`, `requirements-dev.txt`
- **Impact**: Confusion about dependency source of truth; git self-reference in requirements.txt breaks `pip install -r`
- **Fix**: Delete both files, use `pyproject.toml` exclusively
- **Effort**: 15 minutes
- **Risk**: Very low

### Important (Fix This Quarter)

#### TD-3: Long Functions
- **Locations**: 16 functions > 100 lines (see quality assessment)
- **Top 3 targets**:
  - `display.py` render method (301 lines) - Extract panel rendering helpers
  - `step_manager.py` step/episode method (250 lines) - Extract episode lifecycle stages
  - `ppo_agent.py` PPO update (217 lines) - Extract loss computation, advantage computation, logging
- **Effort**: 2-4 hours per function
- **Risk**: Medium (needs test coverage verification before refactoring)

#### TD-4: Unused Test Markers
- **Location**: `pytest.ini` defines markers; tests don't use them
- **Impact**: Can't run `pytest -m unit` for fast CI feedback
- **Fix**: Apply markers to existing tests (can be done per-conftest.py with `pytestmark`)
- **Effort**: 2-3 hours
- **Risk**: Low

#### TD-5: print() Instead of Logger
- **Locations**: 40 print() calls across evaluation analytics, shogi rules, callbacks, profiling
- **Impact**: Inconsistent output, can't control log levels, breaks Rich display
- **Fix**: Replace with `unified_logger` calls
- **Effort**: 1-2 hours
- **Risk**: Very low

### Deferred (Address When Relevant)

#### TD-6: Evaluation System Proportionality
- The evaluation subsystem is 37% of source LOC (7,965 LOC) vs core RL at 4% (965 LOC)
- Tournament/ladder/benchmark strategies may be premature given current agent strength
- **Action**: Monitor. If the agent starts winning games, this investment pays off. If not, consider simplifying.

#### TD-7: Missing Cross-Config Validators
- `config_schema.py` documents timing alignment constraints but doesn't enforce them
- Example: `steps_per_epoch` should divide `evaluation_interval_timesteps`
- **Action**: Add Pydantic `model_validator` when config bugs are observed

#### TD-8: No CI Pipeline
- `ci.yml.disabled` in .github/workflows/
- Claude-based review workflows exist but don't run tests
- **Action**: Re-enable CI with tiered testing (fast unit tests on PR, full suite on merge)

---

## Improvement Roadmap

### Phase 1: Housekeeping (1-2 days)
**Goal**: Clean up debt that slows daily work

| Task | Effort | Impact |
|------|--------|--------|
| Delete stale requirements.txt | 15 min | Remove confusion |
| Fix core<->utils circular dep | 1-2 hrs | Cleaner imports |
| Replace print() with logger | 1-2 hrs | Consistent output |
| Clean up root directory clutter | 1 hr | Repo hygiene |
| Update .gitignore | 15 min | Stop tracking artifacts |

### Phase 2: Testing Foundation (1-2 days)
**Goal**: Enable confident refactoring

| Task | Effort | Impact |
|------|--------|--------|
| Apply pytest markers to tests | 2-3 hrs | Fast CI tier |
| Measure and set coverage threshold | 1-2 hrs | Coverage visibility |
| Re-enable CI with unit test tier | 2-3 hrs | Automated quality gates |
| Fix 2 scipy collection errors | Done | Clean test runs |

### Phase 3: Core RL Focus (Ongoing)
**Goal**: Train a strong Shogi agent

| Task | Effort | Impact |
|------|--------|--------|
| Establish training benchmarks | 2-3 hrs | Performance baseline |
| Hyperparameter sweep (W&B) | Days | Find good training configs |
| Reward shaping experiments | Days | Faster convergence |
| Model architecture experiments | Days | Better representation |
| Curriculum learning | Days | Progressive difficulty |

### Phase 4: Production Hardening (When Agent is Competitive)
**Goal**: Reliable, monitored training runs

| Task | Effort | Impact |
|------|--------|--------|
| Training health monitoring | 4-8 hrs | Detect divergence early |
| Automatic checkpoint on anomaly | 2-4 hrs | Prevent wasted GPU time |
| Memory profiling / OOM protection | 2-4 hrs | Stable long runs |
| Decompose longest functions | 8-16 hrs | Code maintainability |

---

## Key Architectural Decisions for Future Development

### New Model Architectures
- **Interface**: Must implement `ActorCriticProtocol`
- **Registration**: Add to `model_factory()` in `training/models/__init__.py`
- **Configuration**: Add model-specific fields to `TrainingConfig`
- **Testing**: Add corresponding test fixtures in `tests/conftest.py`

### New Evaluation Strategies
- **Interface**: Follow `strategies/` pattern (see `single_opponent.py` as template)
- **Registration**: Add strategy constant to `config_schema.py` and register in factory
- **Configuration**: Add strategy-specific fields to `EvaluationConfig`

### New Training Features (e.g., Curriculum Learning)
- **Location**: Create a new manager (e.g., `CurriculumManager`) in `training/`
- **Integration**: Initialize in `Trainer.__init__()`, call from `TrainingLoopManager`
- **Configuration**: Add `CurriculumConfig` section in `config_schema.py`
- **Testing**: Mirror structure in `tests/training/`

### Adding External Integrations
- **Pattern**: Follow W&B/WebUI pattern - always optional, disable cleanly
- **Configuration**: Add `XyzConfig` with `enabled: bool = False`
- **Manager**: Create manager class with graceful no-op when disabled
- **Testing**: Mock the integration in test fixtures

---

## Handover Checklist

- [x] Architecture documented with diagrams
- [x] All 10 subsystems cataloged with dependencies
- [x] Technical debt inventoried and prioritized
- [x] Improvement roadmap with phases
- [x] Extension patterns documented
- [x] Code quality metrics established
- [x] Test suite status documented
- [ ] Training benchmarks established (Phase 3)
- [ ] CI pipeline re-enabled (Phase 2)
- [ ] Coverage thresholds set (Phase 2)
