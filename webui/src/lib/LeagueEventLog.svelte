<script>
  import { leagueEvents, transitionCounts } from '../stores/league.js'
  import { getRoleIcon } from './roleIcons.js'
  import { collapseEvents } from './collapseEvents.js'

  $: counts = $transitionCounts
  $: collapsedEvents = collapseEvents($leagueEvents)
</script>

<div class="event-log">
  <h2 class="section-header">Event Log</h2>
  <div class="transition-summary" aria-live="polite">
    {#if counts.promotions > 0}<span class="summary-item promotion">↑ {counts.promotions} promoted</span>{/if}
    {#if counts.evictions > 0}<span class="summary-item eviction">↓ {counts.evictions} evicted</span>{/if}
    {#if counts.admissions > 0}<span class="summary-item admission">→ {counts.admissions} admitted</span>{/if}
  </div>
  {#if $leagueEvents.length === 0}
    <p class="empty">No league events yet.</p>
  {:else}
    <div class="feed">
      {#each collapsedEvents as event}
        {#if event.collapsed}
          <div class="event {event.type}" tabindex="0" aria-label="{event.count} {event.type === 'arrival' ? 'arrivals' : event.type === 'departure' ? 'departures' : event.type === 'promotion' ? 'promotions' : 'demotions'}: {event.names.join(', ')}">
            <span class="event-time">{event.time}</span>
            <span class="event-icon" aria-hidden="true">{event.icon}</span>
            <span class="sr-only">{event.type}</span>
            <span class="event-name">{event.count} {event.type === 'arrival' ? 'arrivals' : event.type === 'departure' ? 'departures' : event.type === 'promotion' ? 'promotions' : 'demotions'}</span>
            <span class="event-detail" title={event.names.join(', ')}>{event.names.slice(0, 3).join(', ')}{event.names.length > 3 ? ` +${event.names.length - 3}` : ''}</span>
          </div>
        {:else}
          <div class="event" class:arrival={event.type === 'arrival'} class:departure={event.type === 'departure'} class:promotion={event.type === 'promotion'} class:demotion={event.type === 'demotion'}>
            <span class="event-time">{event.time}</span>
            <span class="event-icon" aria-hidden="true">{event.icon}</span>
            <span class="sr-only">{event.type}</span>
            {#if event.role}<span class="role-icon" aria-hidden="true">{getRoleIcon(event.role)}</span>{/if}
            <span class="event-name">{event.name}</span>
            <span class="event-detail">{event.detail}</span>
          </div>
        {/if}
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
    font-size: 12px;
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

  .role-icon {
    font-size: 12px;
    flex-shrink: 0;
  }

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

  .transition-summary {
    display: flex;
    gap: 10px;
    font-size: 12px;
    padding: 4px 6px;
    margin-bottom: 4px;
    border-bottom: 1px solid var(--border-subtle);
  }
  .summary-item { font-weight: 600; }
  .summary-item.promotion { color: var(--accent-gold); }
  .summary-item.eviction { color: var(--danger); }
  .summary-item.admission { color: var(--accent-teal); }
</style>
