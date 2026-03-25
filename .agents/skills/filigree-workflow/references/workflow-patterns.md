# Workflow Patterns

Detailed procedural patterns for common filigree workflows. Load this reference
when facing a specific workflow challenge.

## Triage Pattern

Triage turns an unsorted pile of issues into a prioritised, actionable backlog.

### Process

1. **Gather**: `filigree list --status=open --json` to get all open issues
2. **Categorise by type**: Separate bugs from features from tasks
3. **Set priorities**:
   - P0/P1 for anything blocking users or other work
   - P2 for standard backlog items
   - P3/P4 for nice-to-haves and future ideas
4. **Batch update**: `filigree batch-update <ids...> --priority=N`
5. **Add dependencies**: Wire up blocking relationships so `ready` reflects reality
6. **Verify**: `filigree ready` should now show a clean, prioritised work queue

### Anti-patterns

- Setting everything to P1 — defeats the purpose of priorities
- Skipping dependency wiring — agents pick blocked work and waste time
- Triaging without reading descriptions — priorities should reflect actual impact

## Sprint Planning Pattern

Plan a focused set of work for a bounded time period.

### Using Milestones

```bash
# Create the plan structure
filigree create-plan --file sprint.json
```

See `examples/sprint-plan.json` for a complete template. The key structure:

```json
{
  "milestone": {"title": "Sprint 3", "priority": 1},
  "phases": [
    {
      "title": "Phase name",
      "steps": [
        {"title": "Step A", "priority": 1},
        {"title": "Step B", "deps": [0]}
      ]
    }
  ]
}
```

Dependencies use indices: integer for same-phase (`0` = first step), cross-phase
uses `"phase.step"` format (`"0.0"` = phase 0, step 0).

### Tracking Progress

```bash
filigree plan <milestone-id>    # tree view with progress bars
filigree stats                  # overall project health
filigree metrics --days 14      # velocity for this sprint period
```

## Dependency Management

### When to Add Dependencies

- Task B cannot start until task A's output exists (data dependency)
- Task B would be invalidated by task A's changes (ordering dependency)
- Task B is a sub-task of epic A (parent-child, not a dep — use `--parent`)

### When NOT to Add Dependencies

- Tasks are merely related but can proceed independently
- The ordering is preferred but not required
- One task "should" be done first but the other won't break without it

### Debugging Blocked Work

```bash
filigree blocked                          # all blocked issues with blockers
filigree critical-path                    # longest chain to unblock
filigree show <blocked-id>               # see what blocks this specific issue
```

To unblock: close the blocker, or if the dependency is wrong, remove it:
```bash
filigree remove-dep <blocked> <blocker>
```

## Bug Lifecycle

### Standard Flow

```
create (open) → in_progress → closed
```

### With Verification

For types that support it (check `filigree type-info bug`):

```
open → fixing → verifying → closed
```

### Bug Report Template

```bash
filigree create "Short description" \
  --type=bug \
  --priority=1 \
  -d "Steps to reproduce: ...
Expected: ...
Actual: ...
Impact: ..."
```

### After Fixing

Always add a comment with:
1. Root cause explanation
2. What was changed
3. How it was tested

```bash
filigree add-comment <id> "Root cause: off-by-one in pagination.
Fixed in commit abc123. Tested with 0, 1, and boundary cases."
filigree close <id> --reason="Fixed off-by-one in pagination logic"
```

## Event History and Auditing

### Reviewing What Happened

```bash
filigree events <id>                         # full history for one issue
filigree changes --since 2026-01-15T00:00:00 # everything since a timestamp
```

### Undoing Mistakes

```bash
filigree undo <id>    # reverts last reversible action (status, priority, etc.)
```

Only reversible actions can be undone. Check `filigree events <id>` first to
see what the last action was.

## Archiving and Maintenance

### Cleaning Up Old Issues

```bash
filigree archive --days 30     # archive issues closed >30 days ago
filigree compact --keep 50     # trim event history for archived issues
```

Archive when the active issue count exceeds ~500 and queries start slowing down.
