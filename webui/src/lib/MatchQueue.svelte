<script>
  import { showcaseQueue, sidecarAlive } from '../stores/showcase.js'
  import { sendShowcaseCommand } from './ws.js'

  function cancelMatch(queueId) {
    sendShowcaseCommand({ type: 'cancel_showcase_match', queue_id: queueId })
  }
</script>

<div class="match-queue">
  <div class="sidecar-status">
    <span class="dot" class:alive={$sidecarAlive} class:dead={!$sidecarAlive}></span>
    <span class="status-text">{$sidecarAlive ? 'Engine online' : 'Engine offline'}</span>
  </div>
  {#if $showcaseQueue.length > 0}
    <div class="queue-list">
      {#each $showcaseQueue as q}
        <div class="queue-item" class:running={q.status === 'running'}>
          <span class="q-status">{q.status}</span>
          <span class="q-entries">{q.entry_id_1} vs {q.entry_id_2}</span>
          <span class="q-speed">{q.speed}</span>
          {#if q.status === 'pending'}
            <button class="cancel-btn" on:click={() => cancelMatch(q.id)}>Cancel</button>
          {/if}
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .match-queue { padding: 8px; font-size: 12px; }
  .sidecar-status { display: flex; align-items: center; gap: 6px; margin-bottom: 8px; }
  .dot { width: 8px; height: 8px; border-radius: 50%; }
  .dot.alive { background: var(--accent-teal); }
  .dot.dead { background: var(--accent-gold); }
  .status-text { color: var(--text-secondary); }
  .queue-list { display: flex; flex-direction: column; gap: 4px; }
  .queue-item { display: flex; align-items: center; gap: 8px; padding: 4px 6px; border-radius: 4px; border: 1px solid var(--border); }
  .queue-item.running { border-color: var(--accent-teal); }
  .q-status { font-weight: 600; text-transform: uppercase; font-size: 10px; min-width: 60px; }
  .q-entries { flex: 1; }
  .q-speed { color: var(--text-muted); text-transform: capitalize; }
  .cancel-btn { font-size: 11px; padding: 2px 8px; border: 1px solid var(--border); border-radius: 3px; background: transparent; color: var(--text-secondary); cursor: pointer; }
</style>
