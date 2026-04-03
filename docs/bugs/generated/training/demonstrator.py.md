## Summary

`DemonstratorRunner` samples from `softmax(masked_logits)` without guarding against all-false legal masks, so a zero-legal-action state can produce NaNs and crash/skip demo games.

## Severity

- Severity: major
- Priority: P1

## Evidence

- [/home/john/keisei/keisei/training/demonstrator.py:176](/home/john/keisei/keisei/training/demonstrator.py:176) masks illegal actions with `-inf`:
  - `masked = flat.masked_fill(~legal_masks, float("-inf"))`
- [/home/john/keisei/keisei/training/demonstrator.py:177](/home/john/keisei/keisei/training/demonstrator.py:177) immediately applies softmax:
  - `probs = F.softmax(masked, dim=-1)`
- [/home/john/keisei/keisei/training/demonstrator.py:178](/home/john/keisei/keisei/training/demonstrator.py:178) samples from that distribution:
  - `action = torch.distributions.Categorical(probs).sample()`
- There is no zero-legal-action guard in this file before sampling.
- By contrast, rollout code explicitly guards this hazard at [/home/john/keisei/keisei/training/katago_ppo.py:383](/home/john/keisei/keisei/training/katago_ppo.py:383)-[/home/john/keisei/keisei/training/katago_ppo.py:390](/home/john/keisei/keisei/training/katago_ppo.py:390), with the comment that all-false masks produce NaNs.

## Root Cause Hypothesis

The demonstrator loop reused masked-softmax sampling but omitted the legal-count validation used in PPO paths. If `legal_masks.sum(dim=-1) == 0` for any step (terminal/truncated edge case or env inconsistency), policy probabilities become invalid and sampling fails.

## Suggested Fix

In `demonstrator.py` inside `_play_game` before `masked_fill`/`softmax`, add a guard:
- compute `legal_counts = legal_masks.sum(dim=-1)`
- if any zero, log and end/skip the game (or choose a deterministic safe fallback action from legal indices if available)
- only run `softmax`/`Categorical` when every row has at least one legal action.
---
## Summary

`DemonstratorRunner` is a background thread but directly uses a shared `OpponentPool` object whose SQLite connection is documented as single-thread-only, causing thread-safety failures when the runner is used as intended.

## Severity

- Severity: major
- Priority: P1

## Evidence

- `DemonstratorRunner` is explicitly threaded at [/home/john/keisei/keisei/training/demonstrator.py:42](/home/john/keisei/keisei/training/demonstrator.py:42) and uses `self.pool` methods from that thread:
  - [/home/john/keisei/keisei/training/demonstrator.py:97](/home/john/keisei/keisei/training/demonstrator.py:97) `self.pool.list_entries()`
  - [/home/john/keisei/keisei/training/demonstrator.py:133](/home/john/keisei/keisei/training/demonstrator.py:133) `self.pool.pin(...)`
  - [/home/john/keisei/keisei/training/demonstrator.py:136](/home/john/keisei/keisei/training/demonstrator.py:136) `self.pool.load_opponent(...)`
- `OpponentPool` explicitly states it must not be shared across threads at [/home/john/keisei/keisei/training/league.py:87](/home/john/keisei/keisei/training/league.py:87)-[/home/john/keisei/keisei/training/league.py:89](/home/john/keisei/keisei/training/league.py:89):
  - “Single-thread-only … Do not share an OpponentPool instance across threads.”
- `sqlite3.connect(...)` is called without `check_same_thread=False` at [/home/john/keisei/keisei/training/league.py:89](/home/john/keisei/keisei/training/league.py:89), reinforcing single-thread use of that connection object.

## Root Cause Hypothesis

`DemonstratorRunner` accepts an `OpponentPool` instance and stores it directly, but its own execution context is a separate thread. That violates the pool’s thread contract and can trigger SQLite cross-thread errors or undefined concurrent behavior.

## Suggested Fix

Make the fix in `demonstrator.py` by avoiding cross-thread use of the caller’s pool object:
- initialize a thread-local pool inside the runner thread (or lazily in `run`) using DB/league metadata, and close it on thread exit
- do not call methods on an `OpponentPool` created in another thread
- if shared pinning semantics are required, coordinate via DB-backed state instead of in-memory `_pinned` across thread-local pool instances.
