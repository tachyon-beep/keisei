# Keisei Architecture Report

**Date**: 2026-02-06
**Type**: Architect-Ready Analysis
**Analyst**: Claude Opus 4.6
**Overall Health**: Good with targeted improvement opportunities

---

## Executive Summary

Keisei is a well-engineered Deep Reinforcement Learning system for Shogi with a clean manager-based architecture. The codebase demonstrates strong software engineering practices: type-safe configuration via Pydantic, protocol-based interfaces, comprehensive testing (1.8x test-to-source ratio), and clear separation of concerns through 9 specialized managers.

The system is architecturally sound for its purpose. Key strengths include a self-contained Shogi engine, a flexible evaluation system with 5 pluggable strategies, and a WebUI streaming system for demos. The primary technical concerns are limited to a circular dependency between `core` and `utils`, some overly long methods (16 functions > 100 lines), and an evaluation subsystem that may be over-engineered relative to current training maturity.

**The system is ready for focused training runs** - the infrastructure is solid and the priority should shift from architecture to RL experimentation and hyperparameter tuning.

---

## Architecture Assessment

### Strengths

| Strength | Evidence | Impact |
|----------|----------|--------|
| **Manager pattern** | 9 managers with single responsibilities | Testable, maintainable, clear interfaces |
| **Protocol-based typing** | `ActorCriticProtocol` for model interface | Models swappable without inheritance |
| **Pydantic configuration** | 9 typed config sections with validators | Catches config errors at startup, not mid-training |
| **Comprehensive testing** | 1,218 tests, 113 test files | High confidence in refactoring |
| **Self-contained game engine** | `shogi/` has zero outbound deps | Domain isolation, testable independently |
| **Pluggable evaluation** | 5 strategy implementations | Easy to add new evaluation modes |
| **Optional integrations** | W&B, WebUI, CUDA all disable cleanly | Works on any hardware, any environment |

### Concerns

| Concern | Severity | Location | Recommendation |
|---------|----------|----------|----------------|
| `core <-> utils` circular dep | Medium | ppo_agent.py ↔ agent_loading.py | Extract shared types to `core.types` |
| 16 functions > 100 lines | Low-Medium | display.py (301), step_manager.py (250) | Decompose in next refactoring pass |
| Test markers unused | Low | pytest.ini defines `unit` but 0 tests tagged | Tag tests for selective CI |
| Evaluation over-engineered | Low | 7,965 LOC (37% of source) | Acceptable if training matures |
| requirements.txt stale | Low | Git self-reference, pinned transitives | Use pyproject.toml exclusively |
| No cross-config validation | Low | config_schema.py | Add model validators for timing alignment |

### Architecture Maturity

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Modularity | High | Clean subsystem boundaries, manager pattern |
| Testability | High | Comprehensive fixtures, mockable interfaces |
| Configurability | High | 4-layer config hierarchy, CLI overrides |
| Extensibility | Medium-High | Strategy pattern for evaluation, factory for models |
| Documentation | Medium | Good inline docs, YAML comments, but sparse API docs |
| Operational Readiness | Medium | W&B integration, WebUI, but limited monitoring/alerting |
| Performance Engineering | Medium | torch.compile support, mixed precision, but no systematic benchmarks |

---

## Subsystem Summary

### Size Distribution
```
evaluation ████████████████████████████████████████  7,965 LOC (37%)
training   █████████████████████████████████         6,706 LOC (31%)
shogi      ████████████                              2,584 LOC (12%)
utils      █████████                                 1,971 LOC  (9%)
core       ████                                        965 LOC  (4%)
config     ███                                         739 LOC  (3%)
webui      ███                                         656 LOC  (3%)
```

### Dependency Health
- **Clean layers**: shogi (0 outbound), config (0 outbound), webui (1 outbound)
- **Expected high coupling**: training (depends on 6 subsystems - it's the orchestrator)
- **One circular**: core <-> utils (manageable, well-understood)
- **No deep circular chains**: No A->B->C->A patterns

### Key Metrics
| Metric | Value | Assessment |
|--------|-------|------------|
| Total source LOC | 21,586 | Medium-sized project |
| Total test LOC | ~39,000+ | Comprehensive |
| Test-to-source ratio | 1.8x | Above industry average |
| Number of subsystems | 7 (+2 supporting) | Well-decomposed |
| Average file size | 248 LOC | Healthy |
| Largest file | 765 LOC | Reasonable |
| Functions > 100 LOC | 16 | Needs attention |
| Circular dependencies | 1 (subsystem level) | Manageable |
| External integrations | 5 major | All optional/disableable |

---

## Risk Assessment

### Low Risk
- **Game engine correctness**: Well-tested, self-contained, deterministic
- **Configuration errors**: Pydantic catches at startup
- **Dependency conflicts**: pyproject.toml defines ranges, uv resolves

### Medium Risk
- **Training performance**: No systematic benchmarks for training throughput
- **Parallel training**: Less tested than single-process path
- **Long function complexity**: 16 functions could harbor subtle bugs

### Higher Risk
- **RL convergence**: The core challenge - architecture is ready but hyperparameter exploration and training strategies need validation
- **Checkpoint compatibility**: Model checkpoint format could break across code changes (mitigated by config saving in checkpoints)

---

## Recommendations (Prioritized)

### Immediate (Before Next Training Run)
1. **Fix the `core <-> utils` circular dependency** - Extract `PolicyOutputMapper` to a shared types module or move `agent_loading` into the `core` package.
2. **Tag tests with markers** - Apply `@pytest.mark.unit` to fast tests for CI speedup.
3. **Remove stale requirements.txt** - Use `pyproject.toml` as sole dependency source.

### Short-Term (Next Sprint)
4. **Decompose long functions** - Target the top 5 (display.py:301, step_manager.py:250, ppo_agent.py:217, agent_loading.py:174, trainer.py:155).
5. **Add cross-config validators** - Enforce that `steps_per_epoch` divides `evaluation_interval_timesteps`.
6. **Establish training benchmarks** - Measure steps/sec, GPU utilization, memory usage as baselines.

### Medium-Term (This Quarter)
7. **Add model architecture options** - Transformer variant for comparison experiments.
8. **Implement curriculum learning** - Progressive difficulty for faster convergence.
9. **Production monitoring** - Training health alerts, automatic checkpoint on anomaly.

---

## Conclusion

Keisei's architecture is well-suited for its purpose as a DRL Shogi training system. The manager-based design provides excellent separation of concerns, the Pydantic configuration system prevents runtime surprises, and the comprehensive test suite enables confident refactoring.

The codebase is at a maturity point where **the bottleneck is no longer architecture but training methodology**. The infrastructure to train, evaluate, visualize, and iterate on Shogi agents is in place. The focus should now shift to RL experimentation: hyperparameter tuning, reward shaping, training curriculum, and model architecture exploration.

The identified concerns (circular dependency, long functions, unused test markers) are all manageable maintenance items that don't block progress on the core mission of training a strong Shogi agent.
