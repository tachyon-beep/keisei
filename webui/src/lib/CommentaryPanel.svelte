<script>
  import { showcaseCurrentMove } from '../stores/showcase.js'

  $: move = $showcaseCurrentMove
  $: topCandidates = (() => {
    if (!move?.top_candidates) return []
    try {
      return typeof move.top_candidates === 'string'
        ? JSON.parse(move.top_candidates) : move.top_candidates
    } catch { return [] }
  })()
  $: winProb = move?.value_estimate ?? 0.5
</script>

<div class="commentary">
  <h3 class="section-label">Commentary</h3>
  <div class="eval-display">
    <span class="label">Win probability</span>
    <div class="eval-bar-container">
      <div class="eval-bar-fill" style="width: {winProb * 100}%"></div>
    </div>
    <span class="eval-value">{(winProb * 100).toFixed(1)}%</span>
  </div>
  {#if move}
    <div class="last-move">
      <span class="label">Last move</span>
      <span class="value">{move.usi_notation}
        {#if topCandidates.length > 0}
          ({((topCandidates.find(c => c.usi === move.usi_notation)?.probability ?? 0) * 100).toFixed(1)}%)
        {/if}
      </span>
    </div>
    <div class="candidates">
      <span class="label">Top candidates</span>
      {#each topCandidates as c, i}
        <div class="candidate" class:chosen={c.usi === move.usi_notation}>
          <span class="rank">{i + 1}.</span>
          <span class="move-name">{c.usi}</span>
          <span class="prob">{(c.probability * 100).toFixed(1)}%</span>
        </div>
      {/each}
    </div>
    {#if move.move_time_ms != null}
      <div class="inference-time">
        <span class="label">Inference</span>
        <span class="value">{move.move_time_ms}ms</span>
      </div>
    {/if}
  {:else}
    <div class="no-data">Waiting for moves...</div>
  {/if}
</div>

<style>
  .commentary { display: flex; flex-direction: column; gap: 10px; padding: 8px; }
  .section-label { font-size: 13px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; margin: 0; }
  .label { font-size: 12px; color: var(--text-secondary); display: block; margin-bottom: 2px; }
  .value { font-size: 13px; color: var(--text-primary); }
  .eval-display { display: flex; flex-direction: column; gap: 4px; }
  .eval-bar-container { height: 8px; background: var(--bg-secondary, #333); border-radius: 4px; overflow: hidden; }
  .eval-bar-fill { height: 100%; background: var(--accent-teal); transition: width 0.3s ease; }
  .eval-value { font-size: 14px; font-weight: 600; color: var(--text-primary); }
  .candidates { display: flex; flex-direction: column; gap: 2px; }
  .candidate { display: flex; gap: 6px; font-size: 13px; padding: 2px 4px; border-radius: 3px; }
  .candidate.chosen { background: var(--tab-active-bg); }
  .rank { color: var(--text-muted); width: 1.5em; }
  .move-name { color: var(--text-primary); flex: 1; }
  .prob { color: var(--text-secondary); }
  .inference-time { font-size: 12px; color: var(--text-muted); }
  .no-data { font-size: 13px; color: var(--text-muted); font-style: italic; }
</style>
