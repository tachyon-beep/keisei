# WebUI Subsystem Catalog (bucket H)

Scope: `webui/` — Svelte 4 SPA (29 `.svelte` components, 10 production stores, helpers, package.json, public/). Tests live alongside sources (`*.test.js`); coverage exists for every store and most pure helpers but is not catalogued here. The single chart library is `uplot ^1.6.31` (`webui/package.json:18`).

## Data flow: WebSocket message → store(s) updated → consuming view(s)

Source of truth: `webui/src/lib/ws.js:95–225`. Showcase outbound commands are dispatched via `sendShowcaseCommand` (`ws.js:37`) from `MatchControls.svelte` and `MatchQueue.svelte`.

| WS message (in) | Stores written | Views / components consuming |
|---|---|---|
| `init` (`ws.js:97–121`) | `games`, `selectedGameId`, `metrics`, `trainingState`, `leagueEntries` (+ `diffLeagueEntries`), `leagueResults`, `eloHistory`, `historicalLibrary`, `gauntletResults`, `leagueTransitions`, `headToHeadRaw`, `tournamentStats`, `styleProfilesRaw`, `showcaseGame`, `showcaseMoves`, `showcaseQueue`, `sidecarAlive` | All views (cold-start snapshot) |
| `game_update` (`ws.js:123–145`) | `games` (delta-merge by `game_id`), `selectedGameId` (auto-switch off ended games) | Training tab: `App.svelte` board area, `GameThumbnail`, `MoveLog`, `EvalBar`, `PieceTray`, `PlayerCard` (via `selectedGame`/`selectedOpponent`) |
| `metrics_update` (`ws.js:147–149`) | `metrics` (append + cap to MAX_POINTS=10000 in `metrics.js:4`) | Training tab: `MetricsGrid`, `MetricsChart`; Training-status surface: `App.svelte`'s `learnerStats` via `latestMetrics` |
| `training_status` (`ws.js:151–167`) | `trainingState` (merge-update of status, phase, heartbeat, epoch, step, episodes, config_json, display_name, model_arch, total_epochs, system_stats, learner_entry_id) | `StatusIndicator`, `App.svelte` (PlayerCard learner side), `HistoricalLibrary`, `learnerEntry` derived |
| `league_update` (`ws.js:169–180`) | Same league stores as `init` (entries, results, eloHistory, historicalLibrary, gauntletResults, transitions, headToHeadRaw, tournamentStats, styleProfilesRaw) | League tab: `LeagueView`, `LeagueTable`, `LeagueEventLog`, `MatchupMatrix`, `RecentMatches`, `EntryDetail`, `HistoricalLibrary`; cross-tab: `selectedOpponent` derived |
| `showcase_update` (`ws.js:182–205`) | `showcaseGame`, `showcaseMoves` (game-aware reset+append), `sidecarAlive=true`; calls `resetShowcaseSelectionOnGameChange` on game change | Showcase tab: `ShowcaseView`, `MatchScorecard`, `WinProbGraph`, `CommentaryPanel`, `Board`/`PieceTray`/`MoveLog` (showcase mode) |
| `showcase_status` (`ws.js:207–216`) | `showcaseQueue`, `sidecarAlive`; clears `showcaseGame`/`showcaseMoves` when no `active_game_id` | `MatchControls`, `MatchQueue`, `ShowcaseStatsBanner`, `ShowcaseView` offline banner |
| `showcase_error` (`ws.js:218–220`) | None — `console.warn` only | None (silent on UI) |
| `ping` (`ws.js:222–223`) | None (keepalive) | None |
| `connectionState` (internal store, `ws.js:30`) | `connectionState` ∈ {connecting, connected, reconnecting} with 3s grace before flipping to `reconnecting` (`ws.js:27, 67–76`) | `StatusIndicator` banners (`StatusIndicator.svelte:107–115`) |
| Outbound: `request_showcase_match`, `cancel_showcase_match`, `change_showcase_speed` | n/a (sent via `sendShowcaseCommand`) | `MatchControls.svelte:41–53`, `MatchQueue.svelte:23` |

---

## H1. App shell and WebSocket client

**Location:** `webui/src/App.svelte`, `webui/src/main.js`, `webui/src/app.css`, `webui/src/lib/ws.js`, `webui/src/lib/StatusIndicator.svelte`, `webui/src/lib/TabBar.svelte`

**Responsibility:** Bootstraps the SPA, owns the WS connection lifecycle (connect/reconnect/disconnect), routes between four tabs (Training / League / Showcase / About), and renders the Training-tab layout inline (thumbnails, player cards, board, metrics).

**Key Components:**
- `App.svelte` (445 lines) — Root. Mounts `connect()` (`App.svelte:24-27`), renders `StatusIndicator`, dispatches on `$activeTab` to `LeagueView`/`ShowcaseView`/`AboutView` or the inline Training layout, hosts the `<audio>` element for lofi (`App.svelte:162`).
- `main.js` (8 lines) — Standard Svelte 4 mount: `new App({ target: ... })`.
- `lib/ws.js` (240 lines) — WS client. Auto-reconnect with exponential backoff + 50–150% jitter (`ws.js:84–93`); 3 s `DISCONNECT_GRACE_MS` (`ws.js:27`) defers the `'reconnecting'` banner so brief drops are invisible.
- `lib/StatusIndicator.svelte` (236 lines) — Top header showing learner identity, alive/stale phase badge, epoch/step/games counters, wall/train clocks (`StatusIndicator.svelte:34–48`), CPU/GPU stats, plus connection banners.
- `lib/TabBar.svelte` (120 lines) — Four tab buttons (training, league, showcase, about) with arrow/Home/End keyboard nav (`TabBar.svelte:17–29`), audio toggle, theme toggle.
- `app.css` — Global tokens (not read line-by-line).

**Dependencies:**
- Inbound: `main.js` instantiates `App.svelte`; `App.svelte` is the root and is not imported elsewhere.
- Outbound: stores `games`, `training`, `league`, `metrics`, `navigation`, `audio`; helpers `safeParse`. WS messages consumed: ALL (this is the dispatcher entry point).

**Patterns Observed:**
- Tab routing by `{#if $activeTab === '…'}` ladder (`App.svelte:165–253`). No SPA router — there are exactly four mutually exclusive top-level views.
- Reactive `$:` blocks compute derived view-model fields from stores (`App.svelte:49–157`). Heavy use of IIFE-style reactives for multi-step derivations.
- Local `setInterval` for tick-driven freshness (`StatusIndicator.svelte:44–45`, `training.js:8`); cleaned up in `onDestroy`.
- `sendShowcaseCommand` (`ws.js:37`) is the only outbound surface — guarded by `readyState === OPEN`.

**Concerns:**
- `App.svelte` doubles as the Training tab layout (boards, PlayerCards, MoveLog wired inline at `App.svelte:187–246`). Pulling that out into a `TrainingView.svelte` would make it consistent with the other three tabs (`LeagueView`/`ShowcaseView`/`AboutView`). Code-observed: 445 lines vs. the other tab containers each at ≤566.
- Svelte-5 incompatibility surface area is concentrated here: 18 `export let` props across `App.svelte`/`StatusIndicator.svelte`, plus heavy reactive `$:` (will need `$state`/`$derived`/`$effect` rewrite). Filigree task `keisei-a5fe9f710e` (P4) tracks the migration.
- `ws.js` swallows `JSON.parse` failures with `console.warn` only (`ws.js:62–64`); no observability hook for malformed-message rate.
- `connect()` early-returns if `readyState <= OPEN` (`ws.js:44`) — i.e. if `CONNECTING` or `OPEN` — but a stuck `CONNECTING` socket cannot be cancelled by re-calling `connect()`.

**Confidence:** High for routing/lifecycle (read 100% of `App.svelte`, `main.js`, `ws.js`, `StatusIndicator.svelte`, `TabBar.svelte`). High for WS taxonomy.

---

## H2. Live game viewer (Training tab board)

**Location:** `webui/src/lib/Board.svelte`, `PieceTray.svelte`, `MoveLog.svelte`, `EvalBar.svelte`, `WinProbGraph.svelte`, `CommentaryPanel.svelte`, `NotationToggle.svelte`, `ShogiLegend.svelte`, `MoveDots.svelte`, `GameThumbnail.svelte`; helpers `pieces.js`, `handPieces.js`, `moveRows.js`, `usiCoords.js`, `evalCalc.js`, `movePatterns.js`, `gameThumbnail.js`.

**Responsibility:** Render the current learner game position (board grid, hands, eval bar, move log, last-move highlights, optional policy heatmap). Same components are reused by Showcase (H4) — the only delta is showcase passes `lastMoveFromIdx`/`lastMoveToIdx`/`heatmap` props that the Training tab leaves at defaults.

**Key Components:**
- `Board.svelte` (167 lines) — 9×9 grid, props for `board`, `inCheck`, `currentPlayer`, `lastMoveFromIdx/ToIdx`, `heatmap`. Read-only/aria-img.
- `PieceTray.svelte` (108 lines) — Captured-piece tray, uses `getHandPieces()` from `handPieces.js`.
- `MoveLog.svelte` (264 lines) — Notation list, supports interactive scrubbing (`selectedIdx`, `dispatch('select')`); preserves user scroll via `before/afterUpdate` (`MoveLog.svelte:53–72`); embeds `NotationToggle`.
- `EvalBar.svelte` (92 lines) — Vertical W/B percentage bar driven by `computeEval` (`evalCalc.js`).
- `WinProbGraph.svelte` (169 lines) — uplot line chart of value over plies, with vertical scrub marker plugin (`WinProbGraph.svelte:25–39`).
- `CommentaryPanel.svelte` (≥60 lines read) — Top candidates list, win-prob bar, REPLAY badge when scrubbing (`CommentaryPanel.svelte:30`).
- `NotationToggle.svelte` (55 lines) — Cycles `notationStyle` store across western/japanese/usi.
- `ShogiLegend.svelte` (228 lines) — Static piece guide with `MoveDots` for each piece.
- `GameThumbnail.svelte` (134 lines) — Mini board in the Training thumbnail panel; click sets `selectedGameId`.
- Helpers (`pieces.js` 29, `handPieces.js` 16, `moveRows.js` 55, `usiCoords.js` 40, `evalCalc.js` 16, `movePatterns.js` 100, `gameThumbnail.js` 32) — pure functions, all individually unit-tested.

**Dependencies:**
- Inbound: `App.svelte` (Training tab), `ShowcaseView.svelte` (Showcase tab).
- Outbound: stores `games` (`selectedGame`, `selectedOpponent`), `notation` (notationStyle); helpers above. WS messages consumed: `game_update` (Training), `showcase_update` (Showcase).

**Patterns Observed:**
- All board/eval components are pure presentational — no store imports inside `Board`/`PieceTray`/`EvalBar`. Data flows via props from view containers (`App.svelte`/`ShowcaseView.svelte`).
- `moveHistory` is passed as a serialised JSON string (`App.svelte:52`, `ShowcaseView.svelte:29`) and parsed in `MoveLog` via `parseMoves`. Tradeoff documented implicitly: stable prop identity for reactivity.
- Shared `notationStyle` store keeps multiple panels (`MoveLog`, `CommentaryPanel`) in lockstep across tabs (`notation.js:7–11`).
- `safeParse` (`safeParse.js`) used for JSON-typed columns from the backend (`App.svelte:50–52`).

**Concerns:**
- `Board.svelte` accepts board both as `[]` of piece objects but does not validate — a malformed `board_json` payload silently renders an empty grid. Mitigated by `safeParse` falling back, but no sentinel for "parse failed vs. genuinely empty".
- Svelte-5 migration surface: every component uses `export let` props (Board has 5; MoveLog 3; EvalBar 2; CommentaryPanel uses store subscription). All will become `$props()`.
- `MoveLog`'s `dispatch('select')` event pattern (`MoveLog.svelte:21,38`) is `createEventDispatcher` — Svelte 5 prefers callback props.

**Confidence:** High for Board/PieceTray/EvalBar/MoveLog/NotationToggle (read in full or near-full). Medium for `CommentaryPanel` (read 60/≥130 — only the head, but data plumbing is all there). Helpers are tiny and have tests.

---

## H3. League view

**Location:** `webui/src/lib/LeagueView.svelte`, `LeagueTable.svelte`, `LeagueEventLog.svelte`, `MatchupMatrix.svelte`, `RecentMatches.svelte`, `EntryDetail.svelte`, `MatchHistory.svelte`, `HistoricalLibrary.svelte`. Helpers `roleIcons.js`, `collapseEvents.js`, `eloChartData.js`.

**Responsibility:** Render the league leaderboard, head-to-head matrix, event log, recent matches, and a per-entry drill-down. Bound entirely to the league stores; reads from showcase/training only for the `learner_entry_id` linkage.

**Key Components:**
- `LeagueView.svelte` (354 lines) — Layout shell. Stats banner from `leagueStats`/`tournamentStats` (`LeagueView.svelte:46–87`), grid containing `LeagueTable`, optional `EntryDetail` panel (toggled by `focusedEntryId`), `MatchupMatrix`, `LeagueEventLog`, `RecentMatches`. Esc-to-close detail (`:23–28`).
- `LeagueTable.svelte` (474 lines) — Sortable leaderboard with flat/grouped views, role capacity columns; uses `leagueRanked`, `entryWLD`, `eloDelta`, `leagueByRole`, `styleProfiles`, `displayElo`.
- `LeagueEventLog.svelte` (148 lines) — Persistent event log driven by `leagueEvents` (localStorage-backed) plus `transitionCounts`; collapses adjacent same-type events via `collapseEvents.js`.
- `MatchupMatrix.svelte` (398 lines) — 20-slot symmetric grid (active league cap), padded with placeholder slots. Builds an aggregate "trainer" row across all learner snapshots (`:24–38`). Uses `headToHead` derived map.
- `RecentMatches.svelte` (290 lines) — Last 30 results from `leagueResults`, with epoch separators and clash counts.
- `EntryDetail.svelte` (>40 lines read) — Per-entry drill-down: primary/secondary Elo columns, last-round results, style profile, embedded `MetricsChart` for Elo over time.
- `MatchHistory.svelte` (83 lines) — Match list for one entry.
- `HistoricalLibrary.svelte` (114 lines) — Library slot table + gauntlet staleness indicator.

**Dependencies:**
- Inbound: `App.svelte` (loads `LeagueView` when `$activeTab === 'league'`).
- Outbound: stores `league` (most named exports — `leagueEntries`, `leagueResults`, `leagueRanked`, `leagueStats`, `learnerEntry`, `tournamentStats`, `headToHead`, `eloDelta`, `entryWLD`, `styleProfiles`, `leagueByRole`, `eloHistory`, `historicalLibrary`, `gauntletResults`, `leagueTransitions`, `leagueEvents`, `transitionCounts`, `focusedEntryId`); `training` (for `learner_entry_id`); helpers `roleIcons`, `collapseEvents`, `eloChartData`. WS messages consumed: `init`, `league_update`, `training_status` (for learner linkage).

**Patterns Observed:**
- Heavy use of `derived` stores in `league.js` (lines 201, 220, 227, 249, 278, 302, 318, 350, 367) — the view layer is mostly thin and the data shaping is centralised.
- LocalStorage persistence for `leagueEvents` with a (id, display_name) "run marker" fingerprint to invalidate stale events across DB resets (`league.js:30–60, 87–115`).
- `focusedEntryId` is a global writable used as a mini-router for the detail panel (`league.js:14`).
- "Trainer aggregate" row in `MatchupMatrix` is computed from snapshot grouping by `display_name` (`MatchupMatrix.svelte:16–18`).

**Concerns:**
- `league.js` is 383 lines of cross-cutting derived logic — by far the largest store. `diffLeagueEntries` is called explicitly from `ws.js` rather than reactively; this is documented at `league.js:82–86` but creates two paths to keep in sync (init at `ws.js:102`, league_update at `ws.js:171`).
- `LeagueTable.svelte` (474) and `MatchupMatrix.svelte` (398) are the largest components in the bucket; both will be high-effort Svelte 5 migration targets — many `$:` blocks (`LeagueTable.svelte:34–39` and similar) and `export let` props.
- `RecentMatches`'s `clashCounts` (`RecentMatches.svelte:9–17`) recomputes the full map on every reactive run — no memoisation, scaled by full `$leagueResults`.

**Confidence:** High for `LeagueView`, `LeagueEventLog`, `HistoricalLibrary` (read in full or near-full), and the `league.js` store API. Medium for `LeagueTable`/`MatchupMatrix`/`RecentMatches` interior layout (read first ~40 lines of each — data plumbing verified, but did not exhaustively trace all sort/render branches).

---

## H4. Showcase view

**Location:** `webui/src/lib/ShowcaseView.svelte`, `MatchControls.svelte`, `MatchQueue.svelte`, `MatchScorecard.svelte`, `ShowcaseStatsBanner.svelte`, `CommentaryPanel.svelte`, `WinProbGraph.svelte`. Showcase reuses H2 board components.

**Responsibility:** Render queued and live showcase matches with replay scrubbing, optional policy heatmap overlay, commentary, and outbound queue/cancel/speed controls.

**Key Components:**
- `ShowcaseView.svelte` (566 lines — largest in bucket) — Owns scrub keyboard handler (Arrow/Home/End/Space/h, `ShowcaseView.svelte:91–114`), heatmap aggregation across promotion variants (`:48–59`), live-move ARIA announcer (`:139–153`), assembles all child panels.
- `MatchControls.svelte` (242 lines) — Setup form (entry pickers + speed selector) collapsible while a match runs. Outbound: `request_showcase_match`, `change_showcase_speed`.
- `MatchQueue.svelte` (170 lines) — Queue list with two-step cancel confirmation. Outbound: `cancel_showcase_match`.
- `MatchScorecard.svelte` (322 lines) — Black/white player banners with tier badges, role icons, head-to-head stats, progress-bar by ply.
- `ShowcaseStatsBanner.svelte` (135 lines) — 3-card glanceable banner: engine status, live ply, queue depth.

**Dependencies:**
- Inbound: `App.svelte` (when `$activeTab === 'showcase'`).
- Outbound: stores `showcase` (all exports — `showcaseGame`, `showcaseMoves`, `showcaseQueue`, `sidecarAlive`, `showcaseSelectedPly`, `showcaseDisplayedMove`, `isScrubbing`, `winProbHistory`, `queueDepth`, `showcaseHeatmapEnabled`, `showcaseSpeed`, `resetShowcaseSelectionOnGameChange`); `league` (`leagueEntries`, `headToHead`, `displayElo`); helpers `safeParse`, `usiCoords`, `roleIcons`. WS messages consumed: `init` (showcase fields), `showcase_update`, `showcase_status`, `showcase_error`. Outbound WS: `request_showcase_match`, `cancel_showcase_match`, `change_showcase_speed` via `sendShowcaseCommand`.

**Patterns Observed:**
- `showcaseSelectedPly === null` represents "live tail" everywhere (`showcase.js:20–47`); landing on the last ply auto-flips back to live (`ShowcaseView.svelte:71`). This single sentinel keeps the scrub state machine simple.
- Persistent UI prefs in localStorage via store-subscribe pattern: `showcaseHeatmapEnabled` (`showcase.js:60–73`), `showcaseSpeed` (`showcase.js:75–91`), `audioEnabled`, `theme`, `notationStyle`, `aboutLevel`, `activeTab`, `keisei_league_events`, `keisei_league_event_run_marker` — eight distinct localStorage keys total.
- WS `showcase_update` clears moves on game change (`ws.js:188–193`) and only appends moves with `ply > maxPly` (`ws.js:198`) — server can safely re-send overlapping windows.
- Showcase keyboard handler skips form controls (`ShowcaseView.svelte:95–96`) so MatchControls' selects keep arrow keys.
- `disabledReason` pattern in `MatchControls.svelte:30–36` produces a per-state hint instead of a binary `disabled` flag.

**Concerns:**
- `ShowcaseView.svelte` at 566 lines mixes layout, scrub state machine, keyboard, ARIA announcer, and heatmap math. Splitting the scrub controller into a helper module would reduce the future Svelte 5 migration surface.
- `showcase_error` is logged only (`ws.js:218–220`); there's no UI surface for the message — silent on the user experience even when actionable.
- `MatchControls.requestMatch` does not retain optimistic queue state — it sends and waits for the next `showcase_status`. Acceptable, but a slow round-trip will show no feedback in the meantime.

**Confidence:** High for `ShowcaseView` (read 200/566 incl. the entire scrub/keyboard/announcer logic and all data flow), `MatchControls`, `MatchQueue`, `ShowcaseStatsBanner`, the showcase store. Medium for `MatchScorecard` (read first 40/322 — wiring confirmed, full layout not exhaustively traced).

---

## H5. Metrics and charts

**Location:** `webui/src/lib/MetricsGrid.svelte`, `MetricsChart.svelte`. Helpers `chartHelpers.js`, `eloChartData.js`, `metricsColumns.js`. Store `stores/metrics.js`.

**Responsibility:** Render the four training-metrics charts (policy/value loss, win rate, episode length, entropy) in a click-to-expand grid driven by the `metrics` time-series store. Charts are reused by `EntryDetail` (H3) for Elo history.

**Key Components:**
- `MetricsGrid.svelte` (246 lines) — Composes four `MetricsChart` panels (`MetricsGrid.svelte:28–45`); resolves theme colors from CSS variables and re-reads on theme change (`:18–26`); click-to-expand UI.
- `MetricsChart.svelte` (201 lines) — uplot wrapper. Accepts `xData`, `series`, optional side legend mode; rebuilds on prop/theme change; uses `ResizeObserver`.
- `chartHelpers.js` (99 lines) — `buildChartOpts`, `buildChartData`, `resolveThemeColors` (CSS-var driven).
- `eloChartData.js` (42 lines) — Aggregates `eloHistory` rows for `EntryDetail`'s line chart.
- `metricsColumns.js` (49 lines) — `extractColumns`: pivots flat metric rows into per-key arrays (steps, epochs, policyLoss, valueLoss, pvRatio, winRates, etc.).

**Dependencies:**
- Inbound: `MetricsGrid` is rendered by `App.svelte` (Training tab footer). `MetricsChart` is also imported by `EntryDetail.svelte` (`EntryDetail.svelte:4`) and `WinProbGraph.svelte` shares uplot logic via `chartHelpers`.
- Outbound: stores `metrics` (writable + `latestMetrics` derived), `theme`. WS messages consumed: `metrics_update` (append) and `init` (initial load).

**Patterns Observed:**
- The `metrics` store is a custom factory that auto-prunes to 10000 points (`metrics.js:9–16`) — bounds memory growth on long runs.
- Chart colors flow through CSS custom properties so themes (light/dark) re-tint without re-rendering data (`MetricsGrid.svelte:18–26`, `chartHelpers.js`).
- `MetricsChart` is the only uplot consumer outside `WinProbGraph.svelte` — single chart library, two call sites (and `EntryDetail` reuses `MetricsChart`).

**Concerns:**
- `MetricsGrid` re-reads CSS-var colors via `getComputedStyle(document.documentElement).getPropertyValue` on every reactive run (`MetricsGrid.svelte:13`), and `chartColors` is recomputed when either `$theme` or `$metrics` changes (`:18`). The rebuild on `$metrics` change is unnecessary and would cause a getComputedStyle hit per metrics tick.
- `import 'uplot/dist/uPlot.min.css'` happens inside `MetricsChart.svelte:4` only. `WinProbGraph.svelte` imports `uPlot` but not its CSS — relies on `MetricsChart` being mounted first. If a user lands on Showcase before Training, the WinProbGraph might render unstyled.
- Svelte-5 surface: `MetricsChart` exports 8 props with `export let`; lifecycle uses `onMount`/`onDestroy`/`afterUpdate` — `afterUpdate` is being soft-deprecated in Svelte 5.

**Confidence:** High for store + grid + helper plumbing (read in full). Medium for `MetricsChart` interior (read 40/201; uplot lifecycle assumed-correct since tests exist for `chartHelpers`).

---

## H6. About / static views

**Location:** `webui/src/lib/AboutView.svelte`. Store `stores/aboutLevel.js`.

**Responsibility:** Static educational content with a 5-level progressive-disclosure slider ("The Big Idea" → "Research View"); persisted in localStorage.

**Key Components:**
- `AboutView.svelte` (>60 lines read of static content) — Hardcoded data tables for observation planes, model configs, head architectures, training knobs (`AboutView.svelte:5–60`); content gated by `$aboutLevel`.
- `aboutLevel.js` (32 lines) — Five-level enum with localStorage persistence; clamp to 1–5 (`aboutLevel.js:19–24`).

**Dependencies:**
- Inbound: `App.svelte` (when `$activeTab === 'about'`).
- Outbound: store `aboutLevel`. No WS messages consumed.

**Patterns Observed:**
- All content is hardcoded — there is no backend feed for the About tab. This makes it a "pure documentation" surface.
- Same localStorage-clamp pattern as other persistent prefs (`aboutLevel.js:19–24`).

**Concerns:**
- Numeric constants in the About tab (e.g. `lr = 2e-4`, GAE γ/λ, entropy schedule, plane indices at `AboutView.svelte:7–61`) are hardcoded copies of training-side config. If config drifts, About becomes silently wrong. No test asserts agreement with `keisei/config.py` defaults.

**Confidence:** Medium — read first 60/≥? lines of content tables and the full store; did not read the gating/render markup tail.

---

## H7. Cross-cutting infrastructure

**Location:** Stores `theme.js`, `audio.js`, `navigation.js`, `notation.js`, `training.js`, `aboutLevel.js`. Helpers `safeParse.js`, `collapseEvents.js`, `timeFormat.js`, `indicator.js`, `configTooltip.js`, `roleIcons.js`. Components `StatusIndicator.svelte` (banner + clocks; principally H1), audio element (in `App.svelte`).

**Responsibility:** Stateless cross-cutting concerns — persisted UI prefs, time formatting, status iconography, JSON parsing. Each is small and individually unit-tested.

**Key Components:**
- `theme.js` (17 lines) — Dark/light toggle, sets `data-theme` on `<html>`, persists via localStorage.
- `audio.js` (28 lines) — `audioEnabled` flag (default false); App.svelte reconciles the `<audio>` element on store change (`App.svelte:36–47`), gracefully handles `NotAllowedError` autoplay rejection without flipping the flag.
- `navigation.js` (10 lines) — `activeTab` (default 'training'), persisted.
- `notation.js` (35 lines) — Three-style cycle with shared labels.
- `training.js` (19 lines) — `trainingState` plus a derived `trainingAlive` ticking every 10s on heartbeat freshness threshold of 30s (`training.js:13–18`).
- `safeParse.js` (11 lines) — JSON.parse with fallback; pass-through for non-strings.
- `collapseEvents.js` (27 lines) — Adjacent-event run-length compaction for `LeagueEventLog`.
- `timeFormat.js` (35 lines) — UTC parser + elapsed formatter for the wall/train clocks.
- `indicator.js` (14 lines) — Maps `(alive, status)` → display indicator for `StatusIndicator`.
- `configTooltip.js` (24 lines) — Builds tooltip text from the training state's `config_json`.
- `roleIcons.js` (42 lines) — Tier role → emoji/label/colour.

**Dependencies:**
- Inbound: every other bucket consumes one or more of these.
- Outbound: only `localStorage`, `document.documentElement` (theme), `Intl`-style date formatting. No WS coupling except indirectly via the `trainingAlive` derived store.

**Patterns Observed:**
- Universal "pref store" idiom: `loadInitial()` from localStorage with defensive `typeof localStorage !== 'undefined'` guards (SSR-safe even though the app is SPA-only) — repeated in `audio.js:15–22`, `navigation.js:1–10`, `notation.js:14–22`, `theme.js:1–13`, `aboutLevel.js:17–32`, `showcase.js:62–91`, `league.js:35–48`.
- Pure helper modules are exhaustively tested (`pieces.test.js`, `usiCoords.test.js`, `evalCalc.test.js`, `handPieces.test.js`, `safeParse.test.js`, `timeFormat.test.js`, `chartHelpers.test.js`, `configTooltip.test.js`, `metricsColumns.test.js`, `moveRows.test.js`, `eloChartData.test.js`, `gameThumbnail.test.js`, `collapseEvents.test.js`, `movePatterns.test.js`, `indicator.test.js`).

**Concerns:**
- `training.js`'s 10s tick (`training.js:8`) runs unconditionally for the lifetime of the page even when the Training tab is not active — minor, but adds wakeups and CPU work.
- The "pref store + localStorage" idiom is repeated 8× verbatim. Extracting a `createPersistedStore(key, default, validator)` helper would shrink ~40 lines and centralise the SSR guard.

**Confidence:** High — all eight stores and most helpers read in full or are <30 lines. Test coverage is broad.

---

## Cross-cutting Svelte 5 migration risk (informational)

Code patterns observed across this bucket that filigree task `keisei-a5fe9f710e` will need to address:
- `export let` props: every `.svelte` file in `lib/` (29 files). Will become `$props()`.
- Reactive `$:` blocks: most components, often nested IIFEs (`App.svelte:76–123` for `learnerFlavour`/`learnerStats`). Will become `$derived` or `$effect`.
- `createEventDispatcher` + `dispatch('select')`: at least `MoveLog.svelte:21,38`. Will become callback props.
- `<slot/>`-based composition: did not see explicit slot usage in the files read; LeagueView/ShowcaseView/AboutView are routed by `{#if}` from `App.svelte`. Slots are likely used in card/panel patterns not exhaustively read — not flagged with confidence.
- `beforeUpdate`/`afterUpdate`: `MoveLog.svelte:53,59`, `MetricsChart.svelte`, `PlayerCard.svelte:36`. Soft-deprecated in Svelte 5.

## Cross-reference: filigree open issues

`filigree list --label=bug --label=P1 --json` and `--label=P2 --json` both returned `[]` at time of analysis (2026-05-05). Searching `webui` returned only the four-issue Svelte 5 migration cluster: `keisei-9b1171d032` (deps), `keisei-a5fe9f710e` (migrate), `keisei-975949c0b3` (component APIs), `keisei-a1622bc4cf` (verify tests). All P4. No P1/P2 webui-specific bugs are open.

## Files skimmed only superficially

`LeagueTable.svelte` (read 40/474), `MatchupMatrix.svelte` (40/398), `RecentMatches.svelte` (40/290), `MatchScorecard.svelte` (40/322), `MatchControls.svelte` (60/242 — verified outbound surface), `EntryDetail.svelte` (40/≥), `MetricsChart.svelte` (40/201), `ShowcaseView.svelte` (200/566 — read scrub/keyboard/announcer/data plumbing in full, skimmed the layout markup tail), `CommentaryPanel.svelte` (60/≥130), `AboutView.svelte` (60/≥), `ShogiLegend.svelte` (30/228 — content-only). All lighter components (`HistoricalLibrary`, `LeagueEventLog`, `MatchHistory`, `MatchQueue`, `ShowcaseStatsBanner`, `GameThumbnail`, `Board`, `PieceTray`, `MoveLog`, `EvalBar`, `WinProbGraph`, `NotationToggle`, `MoveDots`, `TabBar`, `StatusIndicator`) and ALL stores + ALL non-test JS helpers in `lib/` were read in full or near-full for the data-flow-relevant sections.
