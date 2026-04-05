<script>
  import { leagueResults, leagueEntries } from '../stores/league.js'

  /** @type {number} */
  export let entryId

  $: entryMap = new Map($leagueEntries.map(e => [e.id, e]))

  $: matches = $leagueResults.filter(
    r => r.entry_a_id === entryId || r.entry_b_id === entryId
  ).sort((a, b) => b.epoch - a.epoch)

  function opponent(result) {
    const oppId = result.entry_a_id === entryId ? result.entry_b_id : result.entry_a_id
    return entryMap.get(oppId) || null
  }
</script>

<div class="match-history">
  {#if matches.length === 0}
    <p class="empty">No matches recorded</p>
  {:else}
    <table>
      <caption class="sr-only">Match history for selected entry</caption>
      <thead>
        <tr>
          <th>Epoch</th>
          <th>Opponent</th>
          <th class="num">Elo</th>
          <th class="num">W</th>
          <th class="num">L</th>
          <th class="num">D</th>
        </tr>
      </thead>
      <tbody>
        {#each matches as m}
          {#if opponent(m)}
            <tr>
              <td>{m.epoch === -1 ? 'T' : m.epoch}</td>
              <td>{opponent(m).display_name || opponent(m).architecture}</td>
              <td class="num elo">{Math.round(opponent(m).elo_rating)}</td>
              <td class="num win">{m.entry_a_id === entryId ? m.wins_a : m.wins_b}</td>
              <td class="num loss">{m.entry_a_id === entryId ? m.wins_b : m.wins_a}</td>
              <td class="num draw">{m.draws}</td>
            </tr>
          {:else}
            <tr>
              <td>{m.epoch === -1 ? 'T' : m.epoch}</td>
              <td class="unknown">#{m.entry_a_id === entryId ? m.entry_b_id : m.entry_a_id}</td>
              <td class="num elo">—</td>
              <td class="num win">{m.entry_a_id === entryId ? m.wins_a : m.wins_b}</td>
              <td class="num loss">{m.entry_a_id === entryId ? m.wins_b : m.wins_a}</td>
              <td class="num draw">{m.draws}</td>
            </tr>
          {/if}
        {/each}
      </tbody>
    </table>
  {/if}
</div>

<style>
  .match-history {
    max-height: 200px;
    overflow-y: auto;
    padding: 8px;
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    margin: 4px 0;
  }

  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  thead { color: var(--text-muted); }
  th, td { text-align: left; padding: 3px 8px; }
  th.num, td.num { text-align: right; width: 40px; font-family: monospace; }
  .elo { color: var(--text-secondary); }
  .win { color: var(--accent-teal); }
  .loss { color: var(--danger); }
  .draw { color: var(--accent-gold); }
  .unknown { color: var(--text-muted); font-style: italic; }
  .empty { color: var(--text-muted); font-size: 13px; text-align: center; padding: 12px; }
</style>
