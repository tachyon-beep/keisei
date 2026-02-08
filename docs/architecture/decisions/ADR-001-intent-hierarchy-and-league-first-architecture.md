# ADR-001: Intent Hierarchy and League-First Evaluation Architecture

**Status:** Accepted  
**Date:** 2026-02-08  
**Deciders:** Project maintainers  
**Context:** Align architecture with declared product intent hierarchy

## Summary

Keisei architecture will prioritize two primary goals: (1) showcasing Shogi through watchable, continuous competition and (2) showcasing DRL as a first-class AI paradigm. A third goal, evaluating LLM coding effectiveness, is explicitly subordinate and must not drive runtime architecture away from the first two goals.

## Context

The codebase already contains multiple evaluation strategies (`single_opponent`, `tournament`, `ladder`, `benchmark`, `custom`) and background tournament capabilities, but current training integration primarily uses periodic single-opponent evaluation from callback hooks. This leaves league and exhibition behavior under-integrated despite being part of product intent.

At the same time, training orchestration and callbacks currently pass broad `Trainer` references, which weakens boundaries and makes a durable league runtime harder to evolve safely.

## Decision

1. Retain and build out all implemented evaluation strategies as product-facing league capabilities.
2. Introduce a league-first runtime path that supports continuously running exhibition matches without blocking core training progression.
3. Separate training orchestration from evaluation orchestration through explicit contracts (typed context/events), replacing unconstrained full-`Trainer` callback access over time.
4. Keep the LLM-effectiveness goal process-level:
   - Measure development throughput/quality and document outcomes.
   - Do not couple runtime architecture to LLM-specific mechanisms.
5. Treat presentation as a first-class architecture concern with three required views:
   - Training self-play learning view.
   - Ranked league match/promotion view.
   - Model lineage and ancestry view ("who learned from who").
   These views must support separate channels, rotating display, and split-screen composition.

## Architecture Direction

1. **League runtime as first-class component**
   - Add a dedicated league orchestrator that schedules exhibition and formal evaluation workloads.
   - Reuse existing evaluation strategies and background tournament manager.
2. **Evented boundary between training and evaluation**
   - Training emits evaluation triggers and run metadata through narrow contracts.
   - Evaluation publishes match results, ratings, and league snapshots for WebUI and analytics.
3. **Exhibition flow as an explicit SLO-backed path**
   - Define uptime/latency goals for exhibition updates.
   - Add health checks and recovery behavior for visualization and background evaluators.

## Consequences

### Positive

- Aligns implementation with declared product identity (Shogi + DRL showcase).
- Converts existing strategy code into maintained capabilities instead of passive inventory.
- Improves testability and maintainability by reducing cross-component reach-through.
- Makes exhibition availability an explicit engineering target.

### Negative

- Increases near-term implementation and testing scope.
- Requires incremental refactoring of callback and manager boundaries.
- Adds operational complexity around background evaluation and state publication.

## Implementation Notes

1. Define explicit evaluation/league contracts (request/result/snapshot types).
2. Add league runtime configuration and cross-field validation for exhibition schedules.
3. Introduce league orchestration component and integrate it through `EvaluationManager`.
4. Refactor callbacks to use narrow context objects.
5. Add lineage metadata capture on checkpoint creation and promotion events.
6. Add tests for each retained strategy, exhibition runtime behavior, and lineage graph correctness.

## Decision Criteria for Future Changes

Any architectural proposal should be accepted only if it:

1. Improves or preserves watchable Shogi league behavior.
2. Improves or preserves DRL system integrity and explainability.
3. Does not prioritize LLM-tooling experiments over product/runtime quality.
