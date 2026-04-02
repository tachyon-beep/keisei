<script>
  import { activeTab } from '../stores/navigation.js'
  import { theme, toggleTheme } from '../stores/theme.js'

  const tabs = [
    { id: 'training', label: 'Training' },
    { id: 'league', label: 'League' },
  ]
</script>

<div class="tab-bar" role="tablist" aria-label="Dashboard views">
  {#each tabs as tab}
    <button
      role="tab"
      aria-selected={$activeTab === tab.id}
      class:active={$activeTab === tab.id}
      on:click={() => activeTab.set(tab.id)}
    >
      {tab.label}
    </button>
  {/each}
  <button
    class="theme-toggle"
    on:click={toggleTheme}
    aria-label="Toggle {$theme === 'dark' ? 'light' : 'dark'} theme"
    title="{$theme === 'dark' ? 'Light' : 'Dark'} mode"
  >
    {$theme === 'dark' ? '☀' : '☾'}
  </button>
</div>

<style>
  .tab-bar {
    display: flex;
    gap: 4px;
  }

  button {
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 600;
    border-radius: 4px;
    border: 1px solid var(--tab-inactive-border);
    background: transparent;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.15s;
  }

  button:hover {
    border-color: var(--text-secondary);
  }

  button:focus-visible {
    outline: 2px solid var(--accent-blue);
    outline-offset: 2px;
  }

  button.active {
    border-color: var(--tab-active-border);
    color: var(--tab-active-border);
    background: var(--tab-active-bg);
  }

  .theme-toggle {
    margin-left: 8px;
    font-size: 14px;
    padding: 4px 8px;
  }

  @media (prefers-reduced-motion: reduce) {
    button { transition: none; }
  }
</style>
