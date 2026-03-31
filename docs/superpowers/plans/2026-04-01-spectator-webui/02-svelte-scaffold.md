# Plan 2: Svelte Project Scaffolding

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the Svelte project with Vite, install uPlot, configure the build to output to `keisei/server/static/`.

**Architecture:** Svelte 4 + Vite, dark theme CSS, uPlot for charts. Build output goes to `../keisei/server/static/`.

**Tech Stack:** Svelte 4, Vite, uPlot, npm

---

### Task 1: Create Svelte Project

**Files:**
- Create: `webui/package.json`
- Create: `webui/vite.config.js`
- Create: `webui/svelte.config.js`
- Create: `webui/src/main.js`
- Create: `webui/src/App.svelte`
- Create: `webui/src/app.css`
- Create: `webui/index.html`

- [ ] **Step 1: Initialize Svelte project**

```bash
cd /home/john/keisei
npx create-vite webui --template svelte
cd webui
npm install
npm install uplot
```

- [ ] **Step 2: Configure Vite build output**

`webui/vite.config.js`:
```js
import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

export default defineConfig({
  plugins: [svelte()],
  build: {
    outDir: '../keisei/server/static',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
      '/healthz': 'http://localhost:8000',
    },
  },
})
```

- [ ] **Step 3: Set up dark theme CSS**

`webui/src/app.css`:
```css
:root {
  --bg-primary: #0d1117;
  --bg-secondary: #111;
  --bg-card: #1a1a2e;
  --bg-board: #d4a76a;
  --border: #333;
  --text-primary: #e0e0e0;
  --text-secondary: #888;
  --text-muted: #555;
  --accent-green: #4ade80;
  --accent-blue: #60a5fa;
  --accent-amber: #f59e0b;
  --accent-purple: #a78bfa;
  --accent-pink: #f472b6;
  --danger: #ef4444;
  --warning: #f59e0b;

  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  color: var(--text-primary);
  background-color: var(--bg-primary);
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  min-height: 100vh;
}
```

- [ ] **Step 4: Create placeholder App.svelte**

`webui/src/App.svelte`:
```svelte
<script>
  // Placeholder — will be built out in Plan 7
</script>

<main>
  <h1>Keisei Training Dashboard</h1>
  <p>Connecting...</p>
</main>

<style>
  main {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    color: var(--text-primary);
  }
  h1 { font-size: 1.5rem; margin-bottom: 0.5rem; }
  p { color: var(--text-secondary); }
</style>
```

- [ ] **Step 5: Verify dev server starts**

```bash
cd webui && npm run dev
# Should start on http://localhost:5173
# Ctrl+C to stop
```

- [ ] **Step 6: Verify build outputs to keisei/server/static/**

```bash
cd webui && npm run build
ls ../keisei/server/static/
# Should contain index.html, assets/
```

- [ ] **Step 7: Commit**

```bash
git add webui/ keisei/server/static/
git commit -m "scaffold: Svelte project with Vite, uPlot, dark theme"
```
