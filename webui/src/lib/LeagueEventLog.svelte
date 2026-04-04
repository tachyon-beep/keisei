<script>
  import { leagueEvents } from '../stores/league.js'
</script>

<div class="event-log">
  <h2 class="section-header">Event Log</h2>
  {#if $leagueEvents.length === 0}
    <p class="empty">No league events yet.</p>
  {:else}
    <div class="feed">
      {#each $leagueEvents as event}
        <div class="event" class:arrival={event.type === 'arrival'} class:departure={event.type === 'departure'} class:promotion={event.type === 'promotion'} class:demotion={event.type === 'demotion'}>
          <span class="event-time">{event.time}</span>
          <span class="event-icon" aria-hidden="true">{event.icon}</span>
          <span class="sr-only">{event.type}</span>
          <span class="event-name">{event.name}</span>
          <span class="event-detail">{event.detail}</span>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .event-log {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px;
    display: flex;
    flex-direction: column;
    min-height: 0;
    flex: 1;
    min-height: 120px;
  }

  .feed {
    overflow-y: auto;
    min-height: 0;
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 1px;
  }

  .event {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 3px 6px;
    font-size: 12px;
    border-radius: 3px;
  }

  .event:hover {
    background: var(--bg-card);
  }

  .event-time {
    color: var(--text-muted);
    font-family: monospace;
    font-size: 11px;
    flex-shrink: 0;
    min-width: 60px;
  }

  .event-icon {
    flex-shrink: 0;
    width: 14px;
    text-align: center;
    font-weight: 700;
  }

  .arrival .event-icon { color: var(--accent-teal); }
  .departure .event-icon { color: var(--danger); }
  .promotion .event-icon { color: var(--accent-gold); }
  .demotion .event-icon { color: var(--text-muted); }

  .event-name {
    color: var(--text-primary);
    font-weight: 600;
    flex-shrink: 0;
  }

  .event-detail {
    color: var(--text-muted);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .empty {
    color: var(--text-muted);
    font-size: 12px;
    text-align: center;
    padding: 12px;
  }
</style>
