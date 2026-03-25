# Team Coordination

Multi-agent swarm protocols for filigree. Load this reference when coordinating
work across multiple agents.

## Atomic Claiming

### The Race Condition Problem

When multiple agents call `filigree update <id> --status=in_progress`
simultaneously, both think they own the issue. Filigree solves this with
optimistic-locking claims.

### Claiming Protocol

```bash
# Option A: Claim a specific issue
filigree claim <id> --assignee <agent-name>

# Option B: Claim the highest-priority ready issue
filigree claim-next --assignee <agent-name>
```

If another agent already claimed the issue, the claim fails with an error.
After a successful claim, advance the state:

```bash
filigree update <id> --status=in_progress
```

### Releasing Claims

If an agent cannot finish the work:

```bash
filigree add-comment <id> "Releasing: blocked on X, needs Y to continue"
filigree release <id>
```

Always add a comment before releasing — the next agent needs context.

## Handoff Protocol

When passing work between agents, follow this sequence:

### Outgoing Agent (Finishing)

1. **Document state**: Add a comment with current progress, decisions made,
   and remaining work
2. **Update status**: Leave at `in_progress` if partially done, or close if complete
3. **Flag blockers**: Create blocker issues and add dependencies if needed

```bash
filigree add-comment <id> "Completed: API endpoints for auth.
Remaining: frontend login page needs the /api/token response format.
Decision: used JWT not sessions — see commit abc123.
Blocker: need CORS config before frontend can call API."
```

### Incoming Agent (Picking Up)

1. **Read context**: `filigree show <id>` and `filigree get-comments <id>`
2. **Check dependencies**: Look at `blocked_by` in the show output
3. **Claim**: `filigree claim <id> --assignee <name>`
4. **Continue**: Build on the previous agent's work, don't restart

## Status Update Conventions

### When to Update Status

| Event | Action |
|-------|--------|
| Starting work | `update --status=in_progress` |
| Hit a blocker | Add comment, create blocker issue, add dep |
| Completed the work | `close --reason="..."` |
| Can't finish, releasing | Comment + `release` |
| Found additional work | Create new issues, add deps if needed |

### Comment Conventions

Prefix comments with context markers for quick scanning:

```bash
filigree add-comment <id> "PROGRESS: implemented X and Y, Z remaining"
filigree add-comment <id> "BLOCKED: waiting on <blocker-id> for API schema"
filigree add-comment <id> "DECISION: chose approach A because of B"
filigree add-comment <id> "HANDOFF: releasing, next agent should start at Z"
```

## Swarm Work Distribution

### Leader-Follower Pattern

One agent acts as coordinator:

1. **Leader** runs `filigree ready` and assigns work
2. **Followers** use `filigree claim <id> --assignee <name>` to accept
3. **Followers** report back via comments when done
4. **Leader** monitors `filigree stats` and `filigree list --status=in_progress`

### Self-Organising Pattern

All agents are peers:

1. Each agent runs `filigree claim-next --assignee <name>`
2. Works on the claimed issue independently
3. Closes and immediately claims next
4. No central coordinator needed

This works best when:
- Issues are well-defined and independent
- Dependencies are properly wired (so `claim-next` only returns unblocked work)
- Priority ordering reflects actual importance

### Filtering by Type

Specialised agents can filter claims:

```bash
# Backend agent
filigree claim-next --assignee backend-1 --type task

# Bug-fixing agent
filigree claim-next --assignee bugfix-1 --type bug --priority-max 1
```

## Conflict Resolution

### Two Agents Modified the Same Code

1. The second agent's commit will show merge conflicts
2. Add a comment on the issue explaining the conflict
3. The agent with the simpler change should rebase
4. Use `filigree add-comment` to document the resolution

### Two Agents Claimed Related Work

If agents discover their tasks overlap:

1. One agent adds a dependency between the tasks
2. The agent with the lower-priority task releases their claim
3. The remaining agent completes the prerequisite first

### Stale Claims

If an agent disappears without completing work:

```bash
filigree list --status=in_progress --assignee <missing-agent>
filigree release <id>                    # free the claim
filigree add-comment <id> "Released: previous agent did not complete"
```

## Session Resumption

When an agent starts a new session and needs to resume context:

```bash
# What was I working on?
filigree list --status=in_progress --assignee <name>

# What happened since I last worked?
filigree changes --since <last-session-timestamp>

# What's ready now?
filigree ready
```

The `filigree session-context` hook does this automatically at session start,
but these commands are useful for manual context recovery.
