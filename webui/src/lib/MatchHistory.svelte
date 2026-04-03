<script>
  import { leagueResults, leagueEntries } from '../stores/league.js'

  /** @type {number} */
  export let entryId

  $: matches = $leagueResults.filter(
    r => r.learner_id === entryId || r.opponent_id === entryId
  ).sort((a, b) => b.epoch - a.epoch)

  function opponentName(result) {
    const oppId = result.learner_id === entryId ? result.opponent_id : result.learner_id
    const entry = $leagueEntries.find(e => e.id === oppId)
    return entry ? entry.architecture : `#${oppId}`
  }
</script>

<div class="match-history">
  {#if matches.length === 0}
    <p class="empty">No matches recorded</p>
  {:else}
    <table>
      <thead>
        <tr>
          <th>Epoch</th>
          <th>Opponent</th>
          <th class="num">W</th>
          <th class="num">L</th>
          <th class="num">D</th>
        </tr>
      </thead>
      <tbody>
        {#each matches as m}
          <tr>
            <td>{m.epoch}</td>
            <td>{opponentName(m)}</td>
            <td class="num win">{m.wins}</td>
            <td class="num loss">{m.losses}</td>
            <td class="num draw">{m.draws}</td>
          </tr>
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
  .win { color: var(--accent-teal); }
  .loss { color: var(--danger); }
  .draw { color: var(--accent-gold); }
  .empty { color: var(--text-muted); font-size: 13px; text-align: center; padding: 12px; }
</style>
