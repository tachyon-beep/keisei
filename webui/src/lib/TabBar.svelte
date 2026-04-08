<script>
  import { activeTab } from '../stores/navigation.js'
  import { theme, toggleTheme } from '../stores/theme.js'

  const tabs = [
    { id: 'training', label: 'Training' },
    { id: 'league', label: 'League' },
    { id: 'showcase', label: 'Showcase' },
  ]

  function handleTabKeydown(e) {
    const idx = tabs.findIndex(t => t.id === $activeTab)
    let next = -1
    if (e.key === 'ArrowRight') next = (idx + 1) % tabs.length
    else if (e.key === 'ArrowLeft') next = (idx - 1 + tabs.length) % tabs.length
    else if (e.key === 'Home') next = 0
    else if (e.key === 'End') next = tabs.length - 1
    if (next >= 0) {
      e.preventDefault()
      activeTab.set(tabs[next].id)
      e.currentTarget.parentElement.querySelectorAll('[role="tab"]')[next]?.focus()
    }
  }
</script>

<nav aria-label="Dashboard views"><div class="tab-bar" role="tablist">
  {#each tabs as tab}
    <button
      role="tab"
      aria-selected={$activeTab === tab.id}
      class:active={$activeTab === tab.id}
      tabindex={$activeTab === tab.id ? 0 : -1}
      on:click={() => activeTab.set(tab.id)}
      on:keydown={handleTabKeydown}
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
</div></nav>

<style>
  .tab-bar {
    display: flex;
    gap: 4px;
  }

  button {
    padding: 8px 16px;
    min-height: 44px;
    font-size: 13px;
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
    outline: 2px solid var(--focus-ring);
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
    padding: 8px 12px;
    min-width: 44px;
    min-height: 44px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  @media (prefers-reduced-motion: reduce) {
    button { transition: none; }
  }
</style>
