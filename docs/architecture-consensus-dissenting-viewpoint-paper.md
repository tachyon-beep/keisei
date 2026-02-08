# Consensus and Dissenting Viewpoint Paper

**Subject:** Keisei architecture assessment response  
**Date:** 2026-02-08  
**Source:** `docs/architecture-assessment.md`

## Purpose

This paper captures where reviewers are aligned (consensus) and where reasonable disagreement remains (dissent), so architecture decisions can be made with explicit tradeoffs rather than implicit assumptions.

## Program Intent Hierarchy (Authoritative)

1. **Primary:** Showcase Shogi as a game via watchable, continuous competition.
2. **Primary:** Showcase deep reinforcement learning as a first-class AI paradigm.
3. **Subordinate:** Evaluate LLM effectiveness at coding and evolving this system.

Architecture and roadmap choices should optimize for the first two goals. The third goal should inform process choices without overriding product direction.

## Consensus Positions

1. **The architecture is viable but uneven.**  
   Shared view: the project has a solid RL + Shogi core, but maintainability and delivery discipline are materially weaker than core algorithm quality.

2. **CI must be re-enabled as a near-term gate.**  
   Shared view: no automated quality gate creates unacceptable regression risk, especially with broad subsystem surface area and uneven test coverage.

3. **The evaluation subsystem is under-operationalized relative to intended scope.**  
   Shared view: implemented strategies outpace current runtime adoption, so ownership, testing, and production integration now need to catch up to the intended league vision.

4. **Training subsystem coupling is a real design issue.**  
   Shared view: manager separation is partially undermined by deep `Trainer` reach-through and unconstrained callback access.

5. **Configuration validation is incomplete.**  
   Shared view: missing cross-field validation pushes avoidable failures to runtime and increases operator error risk.

6. **No immediate critical security blocker is present.**  
   Shared view: current local-only assumptions are acceptable for research workflows, but not sufficient for broader exposure scenarios.

## Dissenting Viewpoints

### 1) Evaluation Strategy Footprint

**Consensus-leaning view (prior):**  
Deprecate or isolate unused strategies now; keep only actively used paths until test/documentation parity exists.

**Dissenting view (now selected):**  
Keep and build out all implemented strategies as core league infrastructure, including a continuously running exhibition game for observers.

**Adjudication criterion:**  
If no production or near-term experimental owner exists for a strategy within one planning cycle, move it behind explicit feature flags and maintenance ownership.

**Decision update (stakeholder direction):**  
The project will retain and expand tournament, ladder, benchmark, and custom strategies. This supports the primary goals: Shogi as a spectator experience and DRL as a first-class AI system in operation.

### 2) Coupling Refactor Timing

**Consensus-leaning view:**  
Start decoupling now (constructor-injected dependencies, narrow callback context) to unlock testability and clearer module boundaries.

**Dissenting view:**  
Delay refactor until after feature milestones; current coupling is pragmatic and already functional.

**Adjudication criterion:**  
If coupling blocks test isolation or creates repeated regression cost across two releases, refactor is no longer optional backlog work.

### 3) `torch.compile` Default Behavior

**Consensus-leaning view:**  
Default should be predictable (`False`) or explicitly reported when fallback occurs.

**Dissenting view:**  
Keep default `True` to capture upside where compilation works, since fallback avoids hard failure.

**Adjudication criterion:**  
Keep `True` only if mode selection and fallback are surfaced clearly in runtime logs and telemetry.

### 4) Performance Optimization Priority

**Consensus-leaning view:**  
Address correctness/delivery hygiene first (CI, config validation, no-grad fix), then optimize engine hotspots.

**Dissenting view:**  
Engine and buffer optimizations should come first because training throughput is the dominant practical bottleneck.

**Adjudication criterion:**  
Promote performance work ahead of hygiene only with benchmark evidence showing throughput constraints are the top business blocker.

## Reconciliation Plan (Actionable)

1. **Immediate (Week 1):** Re-enable CI and fix confirmed correctness bug (`torch.no_grad()` condition).  
2. **Short term (Weeks 2-3):** Add AppConfig cross-field validators; formalize evaluation strategy ownership and lifecycle states under a league roadmap (not deprecation).  
3. **Medium term (Weeks 4-6):** Decouple callback context and reduce `TrainingLoopManager` reach-through.  
4. **League buildout:** Define exhibition-game runtime path (scheduling, opponent rotation, result publishing, failure recovery) and treat it as a first-class supported flow.
5. **Governance:** Require explicit owner, test expectation, and lifecycle status for each evaluation strategy.

## Decision Record Recommendation

Record two ADRs to close disagreement loops:

1. **ADR: Evaluation Strategy Lifecycle and League Operation Policy** (active, experimental, deprecated, and exhibition-runtime requirements).  
2. **ADR: Training Manager Boundary Contract** (allowed dependency directions and callback interface scope).

## Bottom Line

There is broad agreement on the main risks: delivery discipline gaps (CI/testing), architectural boundary erosion (training/callback coupling), and weak operationalization of the league-oriented evaluation subsystem.  
After stakeholder direction, disagreement is now mostly about **execution order and operating model**, not whether evaluation strategy expansion should happen.
