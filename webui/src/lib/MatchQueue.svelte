<script>
  import { showcaseQueue } from '../stores/showcase.js'
  import { leagueEntries } from '../stores/league.js'
  import { sendShowcaseCommand } from './ws.js'

  // Resolve entry IDs (TEXT in showcase_queue) to display names. Falls back
  // to the raw ID when an entry has been retired or isn't yet loaded.
  $: entryName = (() => {
    const map = new Map()
    for (const e of $leagueEntries) {
      map.set(String(e.id), e.display_name || e.architecture || `#${e.id}`)
    }
    return (id) => map.get(String(id)) || `#${id}`
  })()

  let pendingCancelId = null

  function requestCancel(queueId) {
    pendingCancelId = queueId
  }
  function confirmCancel(queueId) {
    sendShowcaseCommand({ type: 'cancel_showcase_match', queue_id: queueId })
    pendingCancelId = null
  }
  function dismissCancel() {
    pendingCancelId = null
  }
</script>

{#if $showcaseQueue.length > 0}
  <section class="up-next" aria-label="Up next queue">
    <h3 class="section-label">Up Next</h3>
    <div class="queue-list">
      {#each $showcaseQueue as q (q.id)}
        {@const blackName = entryName(q.entry_id_1 ?? q.entry_id_black)}
        {@const whiteName = entryName(q.entry_id_2 ?? q.entry_id_white)}
        <div class="queue-item" class:running={q.status === 'running'} class:cancel-prompt={pendingCancelId === q.id}>
          <span class="q-status" class:running={q.status === 'running'}>{q.status}</span>
          <span class="q-pairing">
            <span class="player-name">{blackName}</span>
            <span class="vs">vs</span>
            <span class="player-name">{whiteName}</span>
          </span>
          <span class="q-speed">{q.speed}</span>
          {#if q.status === 'pending'}
            {#if pendingCancelId === q.id}
              <span class="confirm-row">
                <span class="confirm-text">Cancel?</span>
                <button class="confirm-btn yes" on:click={() => confirmCancel(q.id)}>Yes</button>
                <button class="confirm-btn no" on:click={dismissCancel}>No</button>
              </span>
            {:else}
              <button
                class="cancel-btn"
                on:click={() => requestCancel(q.id)}
                aria-label={`Cancel queued match ${blackName} vs ${whiteName}`}
              >Cancel</button>
            {/if}
          {/if}
        </div>
      {/each}
    </div>
  </section>
{/if}

<style>
  .up-next {
    padding: 8px 16px 12px;
    font-size: 12px;
    border-top: 1px solid var(--border);
  }

  .section-label {
    font-size: 11px;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 0 0 6px;
  }

  .queue-list { display: flex; flex-direction: column; gap: 4px; }

  .queue-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 10px;
    border-radius: 4px;
    border: 1px solid var(--border);
    background: var(--bg-secondary);
  }
  .queue-item.running {
    border-color: var(--accent-teal);
    background: var(--badge-bg-teal);
  }
  .queue-item.cancel-prompt {
    border-color: var(--accent-gold);
    background: var(--badge-bg-gold);
  }

  .q-status {
    font-weight: 700;
    text-transform: uppercase;
    font-size: 10px;
    min-width: 60px;
    color: var(--text-muted);
    letter-spacing: 0.5px;
  }
  .q-status.running { color: var(--accent-teal); }

  .q-pairing {
    flex: 1;
    display: flex;
    align-items: baseline;
    gap: 6px;
    font-family: monospace;
    color: var(--text-primary);
  }

  .player-name { font-weight: 600; }
  .q-pairing .vs { color: var(--text-muted); font-size: 11px; }

  .q-speed {
    color: var(--text-muted);
    text-transform: capitalize;
    font-family: monospace;
    font-size: 11px;
  }

  .cancel-btn {
    font-size: 11px;
    padding: 3px 10px;
    min-height: 28px;
    border: 1px solid var(--border);
    border-radius: 3px;
    background: transparent;
    color: var(--text-secondary);
    cursor: pointer;
  }
  .cancel-btn:hover { color: var(--danger); border-color: var(--danger); }
  .cancel-btn:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }

  .confirm-row {
    display: flex;
    align-items: center;
    gap: 4px;
  }
  .confirm-text { font-weight: 600; color: var(--accent-gold); font-size: 11px; }

  .confirm-btn {
    font-size: 11px;
    padding: 3px 8px;
    min-height: 28px;
    border-radius: 3px;
    cursor: pointer;
    font-weight: 600;
  }
  .confirm-btn.yes {
    background: var(--danger);
    color: #fff;
    border: 1px solid var(--danger);
  }
  .confirm-btn.no {
    background: transparent;
    color: var(--text-secondary);
    border: 1px solid var(--border);
  }
  .confirm-btn:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
</style>
