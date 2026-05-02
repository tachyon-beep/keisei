<script>
  import { tick, onMount, onDestroy } from 'svelte'
  import { aboutLevel, ABOUT_LEVELS } from '../stores/aboutLevel.js'

  // Observation plane groups (matches keisei/training feature builder).
  const planeGroups = [
    { range: '0–13', label: 'Own pieces (8 unpromoted + 6 promoted)', color: 'teal' },
    { range: '14–27', label: 'Opponent pieces', color: 'ink' },
    { range: '28–34', label: 'Own hand counts (normalized)', color: 'teal' },
    { range: '35–41', label: 'Opponent hand counts', color: 'ink' },
    { range: '42', label: 'Side-to-move plane', color: 'gold' },
    { range: '43', label: 'Ply / max_ply', color: 'gold' },
    { range: '44–47', label: 'Repetition planes (1×, 2×, 3×, 4+)', color: 'moss' },
    { range: '48', label: 'In-check plane', color: 'danger' },
    { range: '49', label: 'Reserved', color: 'muted' },
  ]

  const configRows = [
    {
      name: 'Lite (b10c128)',
      shape: '10 × 128',
      params: '~3.9M',
      use: 'Quick training / DDP / league',
    },
    {
      name: 'Heavy (b40c256)',
      shape: '40 × 256',
      params: '~53M',
      use: 'Primary training target',
    },
  ]

  const headRows = [
    {
      name: 'Policy',
      path: 'Conv 1×1 (C→32) → BN → ReLU → Conv 1×1 (32→139) → permute',
      shape: '(B, 9, 9, 139)',
      loss: 'Legal-masked softmax + clipped PPO surrogate',
    },
    {
      name: 'Value',
      path: 'shared pool (B, 3C) → Linear(3C→256) → ReLU → Linear(256→3)',
      shape: '(B, 3)',
      loss: 'Cross-entropy over W/D/L',
    },
    {
      name: 'Score',
      path: 'shared pool (B, 3C) → Linear(3C→128) → ReLU → Linear(128→1)',
      shape: '(B, 1)',
      loss: 'MSE on material balance / 76',
    },
  ]

  const trainingKnobs = [
    ['Optimizer', 'Adam, lr = 2e-4'],
    ['GAE', 'γ = 0.99, λ = 0.95, value blend α = 0.1 (WDL + normalized score lead)'],
    ['PPO', 'clip ε = 0.2, 4 epochs/batch, mini-batch 1024'],
    ['Entropy schedule', 'λ_entropy = 0.01, linearly decayed over 200 epochs'],
    ['Grad clip', 'global norm 1.0'],
    ['Mixed precision', 'bf16 (AMP) on rollout + update + SL'],
    ['Compile', 'torch.compile(mode="default")'],
    ['Parallelism', '128 envs / rank, DDP-ready'],
    [
      'Engine',
      'Rust (shogi-core + shogi-gym) via PyO3 — 400+ Rust tests guarding move-gen and observation correctness',
    ],
  ]

  const lossWeights = [
    ['λ_policy', '1.0', 'Clipped PPO surrogate, the main objective.'],
    ['λ_value', '1.5', 'Heavy on purpose — good advantages need a good value head, especially early.'],
    ['λ_score', '0.1', 'Auxiliary; small enough not to dominate, large enough to shape features.'],
    ['λ_entropy', '0.01', 'Subtracted (encourages exploration via legal-masked entropy). Linearly decayed to 0 over 200 epochs.'],
  ]

  // TOC entries — keep in render order. minLevel filters per active level.
  const tocItems = [
    { id: 'about-big-idea', label: 'The big idea', minLevel: 1 },
    { id: 'about-self-play', label: 'Self-play loop', minLevel: 2 },
    { id: 'about-observation', label: 'Observation tensor', minLevel: 3 },
    { id: 'about-architecture', label: 'Architecture', minLevel: 3 },
    { id: 'about-block', label: 'Residual block', minLevel: 3 },
    { id: 'about-heads', label: 'Output heads', minLevel: 3 },
    { id: 'about-ppo', label: 'PPO objective', minLevel: 4 },
    { id: 'about-knobs', label: 'Training knobs', minLevel: 4 },
    { id: 'about-framing', label: 'Problem framing', minLevel: 5 },
    { id: 'about-limitations', label: 'Limitations', minLevel: 5 },
  ]

  $: visibleToc = tocItems.filter((item) => item.minLevel <= $aboutLevel)
  $: currentLevelMeta = ABOUT_LEVELS.find((l) => l.id === $aboutLevel)

  // Roving-tabindex + arrow-key navigation for the L1–L5 radio strip.
  let levelGroup
  function handleLevelKeydown(e) {
    const idx = ABOUT_LEVELS.findIndex((l) => l.id === $aboutLevel)
    let next = -1
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') next = (idx + 1) % ABOUT_LEVELS.length
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') next = (idx - 1 + ABOUT_LEVELS.length) % ABOUT_LEVELS.length
    else if (e.key === 'Home') next = 0
    else if (e.key === 'End') next = ABOUT_LEVELS.length - 1
    if (next < 0) return
    e.preventDefault()
    aboutLevel.set(ABOUT_LEVELS[next].id)
    tick().then(() => {
      const radios = levelGroup?.querySelectorAll('[role="radio"]')
      radios?.[next]?.focus()
    })
  }

  // Active-section tracking for the right-rail TOC.
  let activeSection = ''
  let observer

  onMount(() => {
    if (typeof IntersectionObserver === 'undefined') return
    observer = new IntersectionObserver(
      (entries) => {
        // Track the topmost intersecting section in the readable band.
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)
        if (visible.length > 0) activeSection = visible[0].target.id
      },
      { rootMargin: '-25% 0px -55% 0px', threshold: [0, 0.25, 0.5, 1] }
    )
    document.querySelectorAll('#about-main .card[id]').forEach((el) => observer.observe(el))
  })

  onDestroy(() => observer?.disconnect())
</script>

<!-- svelte-ignore a11y-no-noninteractive-element-to-interactive-role -->
<section
  id="about-main"
  class="about-view"
  class:level-1={$aboutLevel === 1}
  class:level-2={$aboutLevel === 2}
  class:level-3={$aboutLevel === 3}
  class:level-4={$aboutLevel === 4}
  class:level-5={$aboutLevel === 5}
  role="tabpanel"
  aria-labelledby="tab-about"
  tabindex="-1"
>
  <div class="about-layout">
    <div class="about-main">
      <header class="about-header">
        <p class="level-mode" aria-live="polite">
          Level {$aboutLevel} of {ABOUT_LEVELS.length} ·
          <span>{currentLevelMeta?.label}</span>
        </p>
        <h1 id="about-title">What is Keisei doing?</h1>
        <p class="lede">
          Keisei is a deep reinforcement-learning system that teaches a neural network to play
          <strong>Shogi</strong> — Japanese chess — by playing millions of games against itself and
          against snapshots of its former selves. Pick the level of detail you want; the page reveals
          progressively deeper material as you move right.
        </p>
      </header>

      <div class="level-bar">
        <!-- svelte-ignore a11y-no-noninteractive-tabindex -->
        <div
          class="level-selector"
          role="radiogroup"
          aria-label="Detail level"
          tabindex="-1"
          bind:this={levelGroup}
          on:keydown={handleLevelKeydown}
        >
          {#each ABOUT_LEVELS as lvl}
            <button
              type="button"
              role="radio"
              aria-checked={$aboutLevel === lvl.id}
              tabindex={$aboutLevel === lvl.id ? 0 : -1}
              class:active={$aboutLevel === lvl.id}
              on:click={() => aboutLevel.set(lvl.id)}
            >
              <span class="level-num">L{lvl.id}</span>
              <span class="level-label">{lvl.label}</span>
            </button>
          {/each}
        </div>
        <div class="level-progress" aria-hidden="true">
          <div class="level-progress-bar" style="--p: {($aboutLevel / ABOUT_LEVELS.length) * 100}%"></div>
        </div>
        <p class="level-blurb">{currentLevelMeta?.blurb}</p>
      </div>

      <!-- ── L1: The big idea ──────────────────────────────────── -->
      <article id="about-big-idea" class="card hero" data-min-level="1">
        <h2>The big idea</h2>
        <p>
          Two copies of the same player sit down at a Shogi board. They make moves. One wins, the
          other loses. We tell each of them which moves seemed to lead toward winning, and they get a
          tiny bit better at choosing those moves next time.
        </p>
        <p>
          Repeat for an enormous number of games. Eventually <em>"a tiny bit better, a tiny bit better,
          a tiny bit better"</em> adds up to a player.
        </p>
        <div class="callout">
          <strong>Why Shogi and not chess?</strong>
          Shogi has <em>drops</em> — captured pieces re-enter your army — which makes it a much harder,
          higher-branching game and a classic stress-test for game-playing AI. AlphaZero famously
          learned it; KataGo's residual-block design (which we borrow) was originally pioneered for Go.
        </div>
      </article>

      <h2 class="level-section" data-min-level="2">
        <span class="level-section-tag">L2</span> Learning loop
      </h2>

      <!-- ── L2: Self-play loop ─────────────────────────────────── -->
      <article id="about-self-play" class="card" data-min-level="2">
        <h2>The self-play loop</h2>
        <p>
          Every step in training is one of these mini-cycles, repeated across 128 games happening in
          parallel:
        </p>

        <div class="diagram-frame">
          <svg viewBox="0 0 720 220" xmlns="http://www.w3.org/2000/svg"
               role="img" aria-labelledby="loop-title loop-desc">
            <title id="loop-title">Self-play loop</title>
            <desc id="loop-desc">
              The neural network proposes an action; the Rust Shogi engine applies it and returns
              the next observation and any reward; the transition is stored in a buffer; after each
              rollout (~512 steps × 128 envs ≈ 65k transitions) the buffer is consumed by a PPO
              gradient update that revises the network weights.
            </desc>
            <defs>
              <marker id="arrow-loop" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
                <path d="M0,0 L0,6 L9,3 z" fill="currentColor" />
              </marker>
            </defs>

            <!-- Network -->
            <rect x="20" y="60" width="170" height="80" rx="8"
                  fill="var(--badge-bg-teal)" stroke="var(--accent-teal)" stroke-width="1.5" />
            <text x="105" y="92" text-anchor="middle" class="svg-title">Neural Network</text>
            <text x="105" y="112" text-anchor="middle" class="svg-sub">SE-ResNet (b40c256)</text>
            <text x="105" y="128" text-anchor="middle" class="svg-sub">3 heads: π · V · score</text>

            <!-- Action -->
            <rect x="240" y="20" width="120" height="56" rx="8"
                  fill="var(--badge-bg-gold)" stroke="var(--accent-gold)" stroke-width="1.5" />
            <text x="300" y="44" text-anchor="middle" class="svg-title">Action</text>
            <text x="300" y="62" text-anchor="middle" class="svg-sub">sample legal move</text>

            <!-- Engine -->
            <rect x="410" y="60" width="180" height="80" rx="8"
                  fill="var(--badge-bg-ink)" stroke="var(--accent-ink)" stroke-width="1.5" />
            <text x="500" y="92" text-anchor="middle" class="svg-title">Shogi Engine (Rust)</text>
            <text x="500" y="112" text-anchor="middle" class="svg-sub">apply move</text>
            <text x="500" y="128" text-anchor="middle" class="svg-sub">return new obs · reward</text>

            <!-- Observation -->
            <rect x="240" y="148" width="120" height="56" rx="8"
                  fill="var(--badge-bg-teal)" stroke="var(--accent-teal)" stroke-width="1.5" />
            <text x="300" y="172" text-anchor="middle" class="svg-title">Observation</text>
            <text x="300" y="190" text-anchor="middle" class="svg-sub">(50, 9, 9) tensor</text>

            <!-- Buffer + PPO -->
            <rect x="620" y="60" width="80" height="80" rx="8"
                  fill="var(--bg-card)" stroke="var(--text-secondary)" stroke-width="1.5"
                  stroke-dasharray="4 3" />
            <text x="660" y="92" text-anchor="middle" class="svg-title">Buffer</text>
            <text x="660" y="112" text-anchor="middle" class="svg-sub">(s,a,r,…)</text>
            <text x="660" y="128" text-anchor="middle" class="svg-sub">→ PPO</text>

            <!-- Arrows: Net→Action→Engine→Obs→Net -->
            <path d="M150 75 Q 220 30 240 48" fill="none" stroke="currentColor" stroke-width="1.4"
                  marker-end="url(#arrow-loop)" />
            <path d="M360 48 Q 400 30 440 75" fill="none" stroke="currentColor" stroke-width="1.4"
                  marker-end="url(#arrow-loop)" />
            <path d="M460 140 Q 420 175 360 176" fill="none" stroke="currentColor" stroke-width="1.4"
                  marker-end="url(#arrow-loop)" />
            <path d="M240 176 Q 180 175 150 140" fill="none" stroke="currentColor" stroke-width="1.4"
                  marker-end="url(#arrow-loop)" />

            <!-- Buffer arrow (engine -> buffer, dashed) -->
            <path d="M590 100 L 615 100" fill="none" stroke="var(--text-secondary)"
                  stroke-width="1.4" stroke-dasharray="3 3" marker-end="url(#arrow-loop)" />
            <!-- PPO update back to network (dashed, long) -->
            <path d="M660 60 Q 660 22 105 22 L 105 56" fill="none" stroke="var(--accent-gold)"
                  stroke-width="1.4" stroke-dasharray="4 3" marker-end="url(#arrow-loop)" />
            <text x="380" y="16" text-anchor="middle" class="svg-sub" fill="var(--accent-gold)">
              gradient update after each rollout
            </text>
          </svg>
        </div>

        <ol class="loop-list">
          <li><strong>Network sees</strong> a 50-channel encoding of the position.</li>
          <li><strong>Network proposes</strong> a probability over every legal move plus a guess at who's winning.</li>
          <li><strong>Engine plays</strong> the sampled move and returns the next position (and the reward, if the game ended).</li>
          <li><strong>Buffer collects</strong> the transition. After each rollout (≈65k transitions: 512 steps across 128 parallel envs) we run PPO, which nudges the weights up on moves that worked out and down on ones that didn't.</li>
        </ol>
      </article>

      <h2 class="level-section" data-min-level="3">
        <span class="level-section-tag">L3</span> Inside the demo
      </h2>

      <!-- ── L3: What the network sees ──────────────────────────── -->
      <article id="about-observation" class="card" data-min-level="3">
        <h2>What the network sees: the observation tensor</h2>
        <p>
          Every position is encoded as a stack of 50 9×9 binary or scalar planes — always from the
          perspective of the player to move, so the network never has to learn "am I black or white".
        </p>

        <div class="planes-grid">
          {#each planeGroups as g}
            <div class="plane-row plane-{g.color}">
              <span class="plane-range">{g.range}</span>
              <span class="plane-label">{g.label}</span>
            </div>
          {/each}
        </div>

        <p class="footnote">
          Total: <strong>50 × 9 × 9 = 4,050 input features per position.</strong> 400+ Rust tests in
          <code>shogi-core</code> / <code>shogi-gym</code> guard this encoding — a regression here
          silently corrupts every gradient update.
        </p>
      </article>

      <!-- ── L3: Architecture overview ─────────────────────────── -->
      <article id="about-architecture" class="card" data-min-level="3">
        <h2>Architecture: stem → trunk → three heads</h2>
        <p>
          Keisei's network is a <strong>KataGo-flavoured SE-ResNet</strong>: a stack of residual blocks
          with squeeze-and-excitation gates and a global-pool bias injection. Two trained configurations
          share topology and differ only in trunk depth/width:
        </p>

        <div class="table-scroll">
          <table class="data-table">
            <thead>
              <tr><th scope="col">Config</th><th scope="col">Blocks × Channels</th><th scope="col">Params</th><th scope="col">Use</th></tr>
            </thead>
            <tbody>
              {#each configRows as r}
                <tr>
                  <td><code>{r.name}</code></td>
                  <td>{r.shape}</td>
                  <td>{r.params}</td>
                  <td>{r.use}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>

        <div class="diagram-frame">
          <svg viewBox="0 0 760 380" xmlns="http://www.w3.org/2000/svg"
               role="img" aria-labelledby="arch-title arch-desc">
            <title id="arch-title">Keisei network architecture</title>
            <desc id="arch-desc">
              The 50×9×9 observation passes through a Conv-BN-ReLU stem, then N residual
              GlobalPoolBiasBlocks (10 in lite, 40 in heavy) producing a (B, C, 9, 9) trunk feature
              map. From the trunk, the policy head produces a (B, 9, 9, 139) action distribution
              via two 1×1 convolutions, while a global pool over the trunk feeds two MLPs: the value
              head emits W/D/L logits and the score head emits a single material-balance scalar.
            </desc>
            <defs>
              <marker id="arrow-arch" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
                <path d="M0,0 L0,6 L9,3 z" fill="currentColor" />
              </marker>
            </defs>

            <!-- Input -->
            <rect x="280" y="10" width="200" height="46" rx="6"
                  fill="var(--badge-bg-ink)" stroke="var(--accent-ink)" stroke-width="1.4" />
            <text x="380" y="32" text-anchor="middle" class="svg-title">Observation (B, 50, 9, 9)</text>
            <text x="380" y="48" text-anchor="middle" class="svg-sub">50 perspective-relative planes</text>

            <!-- Stem -->
            <rect x="280" y="78" width="200" height="38" rx="6"
                  fill="var(--bg-card)" stroke="var(--text-secondary)" stroke-width="1.2" />
            <text x="380" y="103" text-anchor="middle" class="svg-title">Stem: Conv 3×3 → BN → ReLU</text>

            <!-- Trunk -->
            <rect x="240" y="138" width="280" height="84" rx="8"
                  fill="var(--badge-bg-teal)" stroke="var(--accent-teal)" stroke-width="1.6" />
            <text x="380" y="160" text-anchor="middle" class="svg-title">Trunk</text>
            <text x="380" y="180" text-anchor="middle" class="svg-sub">GlobalPoolBiasBlock × N</text>
            <text x="380" y="198" text-anchor="middle" class="svg-sub">N = 10 (lite) · 40 (heavy)</text>
            <text x="380" y="214" text-anchor="middle" class="svg-sub">Output: (B, C, 9, 9)</text>

            <!-- Split: policy on left, pool→V/S on right -->
            <!-- Policy head -->
            <rect x="40" y="256" width="240" height="106" rx="8"
                  fill="var(--badge-bg-gold)" stroke="var(--accent-gold)" stroke-width="1.4" />
            <text x="160" y="278" text-anchor="middle" class="svg-title">Policy head</text>
            <text x="160" y="298" text-anchor="middle" class="svg-sub">Conv 1×1 (C→32) → BN → ReLU</text>
            <text x="160" y="314" text-anchor="middle" class="svg-sub">Conv 1×1 (32→139) → permute</text>
            <text x="160" y="334" text-anchor="middle" class="svg-sub svg-emph">→ (B, 9, 9, 139)</text>
            <text x="160" y="352" text-anchor="middle" class="svg-sub">11,259 spatial actions</text>

            <!-- Global pool -->
            <rect x="320" y="256" width="160" height="42" rx="6"
                  fill="var(--bg-card)" stroke="var(--text-secondary)" stroke-width="1.2" />
            <text x="400" y="280" text-anchor="middle" class="svg-title">Global pool</text>
            <text x="400" y="294" text-anchor="middle" class="svg-sub">mean ‖ max ‖ std → (B, 3C)</text>

            <!-- Value head -->
            <rect x="320" y="316" width="160" height="50" rx="6"
                  fill="var(--badge-bg-teal)" stroke="var(--accent-teal)" stroke-width="1.4" />
            <text x="400" y="336" text-anchor="middle" class="svg-title">Value head</text>
            <text x="400" y="354" text-anchor="middle" class="svg-sub">→ (B, 3) W/D/L logits</text>

            <!-- Score head -->
            <rect x="520" y="316" width="200" height="50" rx="6"
                  fill="var(--badge-bg-ink)" stroke="var(--accent-ink)" stroke-width="1.4" />
            <text x="620" y="336" text-anchor="middle" class="svg-title">Score head (auxiliary)</text>
            <text x="620" y="354" text-anchor="middle" class="svg-sub">→ (B, 1) material balance</text>

            <!-- Arrows -->
            <path d="M380 56 L 380 76" fill="none" stroke="currentColor" stroke-width="1.4" marker-end="url(#arrow-arch)" />
            <path d="M380 116 L 380 136" fill="none" stroke="currentColor" stroke-width="1.4" marker-end="url(#arrow-arch)" />
            <path d="M380 222 L 380 240" fill="none" stroke="currentColor" stroke-width="1.4" />
            <!-- split horizontal -->
            <path d="M160 240 L 600 240" fill="none" stroke="currentColor" stroke-width="1.4" />
            <path d="M160 240 L 160 254" fill="none" stroke="currentColor" stroke-width="1.4" marker-end="url(#arrow-arch)" />
            <path d="M400 240 L 400 254" fill="none" stroke="currentColor" stroke-width="1.4" marker-end="url(#arrow-arch)" />
            <!-- pool to value -->
            <path d="M400 298 L 400 314" fill="none" stroke="currentColor" stroke-width="1.4" marker-end="url(#arrow-arch)" />
            <!-- pool to score -->
            <path d="M460 290 Q 580 290 600 314" fill="none" stroke="currentColor" stroke-width="1.4" marker-end="url(#arrow-arch)" />
          </svg>
        </div>

        <p class="footnote">
          The pooled <code>(B, 3C)</code> vector is computed <em>once</em> and shared by the value and
          score heads — they're the same tensor flowing into two small MLPs.
        </p>
      </article>

      <!-- ── L3: Residual block ─────────────────────────────────── -->
      <article id="about-block" class="card" data-min-level="3">
        <h2>The GlobalPoolBiasBlock</h2>
        <p>
          Each block in the trunk is a KataGo-flavoured residual unit. It diverges from a vanilla
          SE-ResNet block in two specific ways:
        </p>
        <ul class="diff-list">
          <li>
            <strong>Global-pool bias</strong> is injected mid-block, so every spatial position learns
            about board-wide context (king safety, material, repetition pressure) without needing the
            trunk to be deep enough to propagate that signal spatially.
          </li>
          <li>
            <strong>Squeeze-and-excitation</strong> produces both a <em>scale</em> and a
            <em>shift</em>, not just a multiplicative gate. So the SE branch can both attenuate and
            translate channels.
          </li>
        </ul>

        <div class="diagram-frame">
          <svg viewBox="0 0 780 320" xmlns="http://www.w3.org/2000/svg"
               role="img" aria-labelledby="block-title block-desc">
            <title id="block-title">GlobalPoolBiasBlock structure</title>
            <desc id="block-desc">
              Input x flows through Conv 3×3, BN, and ReLU. A parallel branch global-pools the same
              input via mean, max, and standard deviation, projects it through a 128-channel
              bottleneck, and broadcasts the result over the 9×9 board to bias the post-conv1
              activations. A second Conv 3×3 with BN follows. A squeeze-and-excitation gate produces
              both a scale and a shift to modulate the channels. Finally, the original input x is
              added back as a residual skip connection and the sum is passed through ReLU.
            </desc>
            <defs>
              <marker id="arrow-block" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
                <path d="M0,0 L0,6 L9,3 z" fill="currentColor" />
              </marker>
            </defs>

            <!-- Input x -->
            <circle cx="40" cy="160" r="20"
                    fill="var(--badge-bg-ink)" stroke="var(--accent-ink)" stroke-width="1.4" />
            <text x="40" y="166" text-anchor="middle" class="svg-title">x</text>

            <!-- Skip line top -->
            <path d="M60 150 Q 350 60 660 150" fill="none" stroke="var(--accent-ink)"
                  stroke-width="1.4" stroke-dasharray="3 3" />

            <!-- Conv1 -->
            <rect x="100" y="140" width="120" height="40" rx="6"
                  fill="var(--bg-card)" stroke="var(--text-secondary)" stroke-width="1.2" />
            <text x="160" y="164" text-anchor="middle" class="svg-sub">Conv 3×3 → BN → ReLU</text>

            <!-- Pool branch (bottom) -->
            <rect x="100" y="240" width="200" height="50" rx="6"
                  fill="var(--badge-bg-teal)" stroke="var(--accent-teal)" stroke-width="1.2" />
            <text x="200" y="258" text-anchor="middle" class="svg-sub svg-emph">Global pool (mean ‖ max ‖ std)</text>
            <text x="200" y="276" text-anchor="middle" class="svg-sub">Linear(3C → G) → ReLU → Linear(G → C)</text>
            <text x="200" y="290" text-anchor="middle" class="svg-sub">G = 128</text>

            <!-- Sum 1 (broadcast bias add) -->
            <circle cx="350" cy="160" r="14" fill="var(--bg-card)" stroke="var(--accent-gold)" stroke-width="1.4" />
            <text x="350" y="165" text-anchor="middle" class="svg-title">+</text>

            <!-- Conv2 -->
            <rect x="380" y="140" width="100" height="40" rx="6"
                  fill="var(--bg-card)" stroke="var(--text-secondary)" stroke-width="1.2" />
            <text x="430" y="164" text-anchor="middle" class="svg-sub">Conv 3×3 → BN</text>

            <!-- SE -->
            <rect x="500" y="140" width="120" height="40" rx="6"
                  fill="var(--badge-bg-gold)" stroke="var(--accent-gold)" stroke-width="1.2" />
            <text x="560" y="164" text-anchor="middle" class="svg-sub">SE(scale, shift)</text>

            <!-- Sum 2 (skip add) -->
            <circle cx="660" cy="160" r="14" fill="var(--bg-card)" stroke="var(--accent-ink)" stroke-width="1.4" />
            <text x="660" y="165" text-anchor="middle" class="svg-title">+</text>

            <!-- ReLU + out -->
            <rect x="690" y="140" width="70" height="40" rx="6"
                  fill="var(--bg-card)" stroke="var(--text-secondary)" stroke-width="1.2" />
            <text x="725" y="164" text-anchor="middle" class="svg-sub">ReLU</text>

            <!-- Arrows along main -->
            <path d="M62 160 L 98 160" fill="none" stroke="currentColor" stroke-width="1.4" marker-end="url(#arrow-block)" />
            <path d="M222 160 L 334 160" fill="none" stroke="currentColor" stroke-width="1.4" marker-end="url(#arrow-block)" />
            <path d="M366 160 L 378 160" fill="none" stroke="currentColor" stroke-width="1.4" marker-end="url(#arrow-block)" />
            <path d="M482 160 L 498 160" fill="none" stroke="currentColor" stroke-width="1.4" marker-end="url(#arrow-block)" />
            <path d="M622 160 L 644 160" fill="none" stroke="currentColor" stroke-width="1.4" marker-end="url(#arrow-block)" />
            <path d="M676 160 L 688 160" fill="none" stroke="currentColor" stroke-width="1.4" marker-end="url(#arrow-block)" />

            <!-- Pool branch arrows -->
            <path d="M40 180 Q 40 240 98 264" fill="none" stroke="var(--accent-teal)" stroke-width="1.3" marker-end="url(#arrow-block)" />
            <path d="M302 264 Q 340 264 350 174" fill="none" stroke="var(--accent-teal)" stroke-width="1.3" stroke-dasharray="4 3" marker-end="url(#arrow-block)" />
            <text x="320" y="220" text-anchor="middle" class="svg-sub" fill="var(--accent-teal)">broadcast over 9×9</text>

            <!-- Out arrow -->
            <path d="M760 160 L 776 160" fill="none" stroke="currentColor" stroke-width="1.4"
                  marker-end="url(#arrow-block)" />
          </svg>
        </div>

        <p class="footnote">
          The SE excitation is computed as: mean-pool the post-conv2 activations <code>out</code> →
          <code>Linear(C → C/R)</code> → ReLU → <code>Linear(C/R → 2C)</code>; split into
          <code>(scale, shift)</code>; modulated output = <code>out · σ(scale) + shift</code>.
          Reduction <code>R = 16</code>.
        </p>
      </article>

      <!-- ── L3: Output heads ──────────────────────────────────── -->
      <article id="about-heads" class="card" data-min-level="3">
        <h2>The three output heads</h2>
        <div class="table-scroll">
          <table class="data-table">
            <thead>
              <tr><th scope="col">Head</th><th scope="col">Path</th><th scope="col">Output shape</th><th scope="col">Loss</th></tr>
            </thead>
            <tbody>
              {#each headRows as h}
                <tr>
                  <td><strong>{h.name}</strong></td>
                  <td><code class="path">{h.path}</code></td>
                  <td><code>{h.shape}</code></td>
                  <td>{h.loss}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>

        <div class="callout">
          <strong>Action space:</strong> 81 squares × 139 move types =
          <strong>11,259 spatial actions</strong>. Move types cover sliding directions × distance ×
          promotion flag for board moves, plus 7 piece-drop slots. Illegal actions are masked to
          <code>-inf</code> before softmax.
        </div>

        <details>
          <summary>Why W/D/L instead of a scalar value?</summary>
          <p>
            A KataGo trick: a position evaluated as 50% win / 50% draw is very different from
            50% win / 50% loss, but both collapse to the same scalar ~0.0. The categorical head
            preserves the distinction. The scalar used for GAE bootstrapping is recovered as
            <code>P(Win) − P(Loss)</code>.
          </p>
        </details>
        <details>
          <summary>Why a score head?</summary>
          <p>
            Auxiliary task — regularizes trunk features by forcing them to encode material count.
            Loss weight is intentionally small (<code>λ_score = 0.1</code>): enough to shape features,
            small enough not to compete with the policy/value objectives.
          </p>
        </details>
      </article>

      <h2 class="level-section" data-min-level="4">
        <span class="level-section-tag">L4</span> Algorithmic detail
      </h2>

      <!-- ── L4: PPO loss ─────────────────────────────────────── -->
      <article id="about-ppo" class="card" data-min-level="4">
        <h2>The PPO objective</h2>
        <div class="ppo-grid">
          <div class="loss-formula">
            <span class="loss-label">Objective</span>
<pre class="loss-block">L = <span class="loss-coef">λ_policy</span>  · L_policy
  + <span class="loss-coef">λ_value</span>   · L_value      <span class="loss-comment">(W/D/L cross-entropy)</span>
  + <span class="loss-coef">λ_score</span>   · L_score      <span class="loss-comment">(material MSE)</span>
  − <span class="loss-coef">λ_entropy</span> · H(π)         <span class="loss-comment">(legal-masked entropy)</span></pre>
          </div>

          <div class="table-scroll">
            <table class="data-table">
              <thead><tr><th scope="col">Term</th><th scope="col">Default</th><th scope="col">Note</th></tr></thead>
              <tbody>
                {#each lossWeights as [k, v, note]}
                  <tr><td><code>{k}</code></td><td><code>{v}</code></td><td>{note}</td></tr>
                {/each}
              </tbody>
            </table>
          </div>
        </div>

        <p>
          <code>L_policy</code> is the standard clipped-PPO surrogate
          <code>min(rₜ · Âₜ, clip(rₜ, 1±ε) · Âₜ)</code> evaluated over legal-masked log-probs.
          Advantages <code>Âₜ</code> come from GAE on a blended value scalar:
          <code>(1 − α) · (P(Win) − P(Loss)) + α · score_lead / 76</code> with
          <code>α = score_blend_alpha = 0.1</code>. The W/D/L head dominates; the score
          lead nudges the value estimate toward material-aware gradations.
        </p>
      </article>

      <!-- ── L4: Training knobs ───────────────────────────────── -->
      <article id="about-knobs" class="card" data-min-level="4">
        <h2>Training knobs (Heavy config)</h2>
        <p class="footnote">
          Values shown reflect the production Heavy config (<code>keisei-katago.toml</code>). The
          in-code <code>KataGoPPOParams</code> dataclass defaults differ for some fields
          (e.g. <code>batch_size=256</code>, <code>use_amp=False</code>,
          <code>compile_mode=None</code>) — running the trainer with no toml will not reproduce
          the numbers below.
        </p>
        <div class="table-scroll">
          <table class="data-table knobs">
            <tbody>
              {#each trainingKnobs as [k, v]}
                <tr><td class="knob-key">{k}</td><td>{v}</td></tr>
              {/each}
            </tbody>
          </table>
        </div>

        <details>
          <summary>Why heavy value loss weight?</summary>
          <p>
            <code>λ_value = 1.5</code> > <code>λ_policy = 1.0</code> on purpose. Bad value estimates
            produce noisy advantages, which produces noisy policy gradients, which slows down or breaks
            learning early. We invest extra capacity in fitting the value head until the policy has
            something useful to clip against.
          </p>
        </details>
      </article>

      <h2 class="level-section" data-min-level="5">
        <span class="level-section-tag">L5</span> Research view
      </h2>

      <!-- ── L5: Problem framing & evaluation ─────────────────── -->
      <article id="about-framing" class="card research" data-min-level="5">
        <h2>Problem framing & evaluation</h2>
        <p>
          Shogi is an episodic, deterministic, perfect-information, zero-sum, two-player game. The raw
          game has non-Markovian repetition rules; the agent's MDP is recovered by including the
          repetition planes (44–47) in the observation, so the network can detect cycles without
          depending on hidden state.
        </p>

        <h3 class="card-h3">Action-space structure</h3>
        <p>
          11,259 spatial actions decompose as <strong>81 squares × (132 board move-types + 7 drops)</strong>.
          Legal-mask sparsity is extreme: a typical mid-game position has on the order of 80 legal moves
          out of 11,259 — well under 1%. Legal-masked softmax and legal-masked entropy are not
          stylistic choices; they're a precondition for stable learning at this density.
        </p>

        <h3 class="card-h3">Reward & credit assignment</h3>
        <p>
          Reward is terminal only: <code>+1</code> on win, <code>0</code> on draw, <code>-1</code> on
          loss. With <code>γ = 0.99</code> and ~100-ply games, the effective horizon is roughly the
          length of a game. The score auxiliary (<code>λ_score = 0.1</code>) injects a denser feature
          signal but does not change the fundamental sparse-reward structure.
        </p>

        <h3 class="card-h3">What we measure</h3>
        <ul class="diff-list">
          <li><strong>Self-play Elo</strong> via the league pool (see the League tab).</li>
          <li><strong>Head-to-head W/D/L</strong> against archived snapshots of the learner's former selves.</li>
          <li><strong>Value-head log-loss</strong> against eventual game outcomes — measures how well the W/D/L head is calibrated.</li>
        </ul>

        <h3 class="card-h3">What we do not measure</h3>
        <ul class="diff-list">
          <li>No comparison against external engines (YaneuraOu, Apery, etc.).</li>
          <li>No human-game play strength.</li>
          <li>No tsume / puzzle accuracy.</li>
          <li>No opening-diversity or repertoire metrics.</li>
        </ul>

        <details>
          <summary>Lineage: what we borrow, what we don't</summary>
          <p>
            The block design comes from <strong>KataGo</strong> (Wu, 2019), which itself extended
            AlphaZero's residual trunk with global-pool bias and SE gating. The W/D/L value head and
            score auxiliary are also KataGo conventions. <strong>PPO</strong> (Schulman et al., 2017)
            replaces AlphaZero's MCTS-driven supervised target — we are not running MCTS at training
            time, and not at inference either. The policy is trained directly from on-policy rollouts.
            This is a deliberate scope choice (see limitations below), not an oversight.
          </p>
        </details>
      </article>

      <!-- ── L5: Limitations & open questions ─────────────────── -->
      <article id="about-limitations" class="card research" data-min-level="5">
        <h2>Limitations & open questions</h2>
        <ul class="diff-list">
          <li>
            <strong>No search at training or inference.</strong> AlphaZero-style improvement comes from
            the policy chasing MCTS targets; Keisei has only on-policy PPO. This caps achievable
            strength relative to MCTS+NN systems at comparable compute.
          </li>
          <li>
            <strong>Self-play opponent diversity is bounded by the league pool.</strong> Snapshots can
            collude into a narrow strategic distribution. The current league refactor (head-to-head
            bookkeeping, Elo tracking) is what mitigates this — it doesn't eliminate it.
          </li>
          <li>
            <strong>Sparse terminal reward + long horizon.</strong> Even with GAE
            (<code>λ = 0.95</code>), early-game credit assignment is the dominant variance source. The
            score auxiliary is a partial mitigation, not a solution.
          </li>
          <li>
            <strong>Training stability is not formally characterised.</strong> We don't currently
            publish KL between consecutive policies, advantage explained-variance, or value-target
            decorrelation — the diagnostics a reviewer would expect to see.
          </li>
          <li>
            <strong>No ablations published.</strong> We have no published evidence that the W/D/L value
            head, the score auxiliary, or SE-with-shift (vs. plain SE) help — only that the combined
            system trains.
          </li>
          <li>
            <strong>Single-host research prototype.</strong> This is not a distributed AlphaZero
            reproduction. Strength claims should be framed accordingly.
          </li>
        </ul>

        <div class="callout">
          <strong>Frame:</strong>
          Keisei is best read as an <em>interpretable DRL prototype for studying self-play dynamics on
          a hard action-space</em> — not a competitive shogi engine.
        </div>
      </article>

      <footer class="about-footer">
        <p>
          <em>If something here doesn't match what you're seeing in the metrics panel — the metrics
          panel doesn't lie, this page does.</em>
          <a
            class="issue-link"
            href="https://github.com/tachyon-beep/keisei/issues/new"
            target="_blank"
            rel="noopener noreferrer"
          >Open an issue ↗<span class="sr-only"> (opens in new tab)</span></a>
        </p>
      </footer>
    </div>

    <aside class="about-toc" aria-label="On this page">
      <p class="toc-heading">On this page</p>
      <ol class="toc-list">
        {#each visibleToc as item}
          <li class:active={activeSection === item.id}>
            <a href={`#${item.id}`}>{item.label}</a>
          </li>
        {/each}
      </ol>
    </aside>
  </div>
</section>

<style>
  .about-view {
    --hide-l2: none;
    --hide-l3: none;
    --hide-l4: none;
    --hide-l5: none;
    height: 100%;
    overflow-y: auto;
    padding: 24px clamp(16px, 3vw, 40px) 64px;
    color: var(--text-primary);
  }
  .about-view :global([data-min-level="2"]) { display: var(--hide-l2, block); }
  .about-view :global([data-min-level="3"]) { display: var(--hide-l3, block); }
  .about-view :global([data-min-level="4"]) { display: var(--hide-l4, block); }
  .about-view :global([data-min-level="5"]) { display: var(--hide-l5, block); }

  /* Levels: hide deeper sections than the active level. */
  .about-view.level-1 { --hide-l2: none;  --hide-l3: none;  --hide-l4: none;  --hide-l5: none;  }
  .about-view.level-2 { --hide-l2: block; --hide-l3: none;  --hide-l4: none;  --hide-l5: none;  }
  .about-view.level-3 { --hide-l2: block; --hide-l3: block; --hide-l4: none;  --hide-l5: none;  }
  .about-view.level-4 { --hide-l2: block; --hide-l3: block; --hide-l4: block; --hide-l5: none;  }
  .about-view.level-5 { --hide-l2: block; --hide-l3: block; --hide-l4: block; --hide-l5: block; }

  /* ── Layout: content + optional right-rail TOC ─────────── */
  .about-layout {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: 32px;
    max-width: 1180px;
    margin: 0 auto;
  }
  .about-main {
    min-width: 0;
  }
  .about-toc {
    display: none;
  }

  @media (min-width: 1280px) {
    .about-layout {
      grid-template-columns: minmax(0, 1180px) 220px;
      max-width: 1452px;
    }
    .about-toc { display: block; }
  }

  /* ── Header ─────────────────────────────────────────── */
  .about-header { max-width: 760px; margin: 0 auto 24px; }
  .level-mode {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 8px;
  }
  .level-mode span { color: var(--accent-teal); }
  .about-header h1 {
    font-size: clamp(26px, 3.6vw, 38px);
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 12px;
    letter-spacing: -0.01em;
  }
  .lede {
    font-size: 16px;
    line-height: 1.55;
    color: var(--text-primary);
    margin-bottom: 20px;
    max-width: 70ch;
  }
  .lede strong { color: var(--accent-teal); }

  /* ── Sticky level bar with elevation ───────────────── */
  .level-bar {
    position: sticky;
    top: 0;
    z-index: 5;
    max-width: 920px;
    margin: 0 auto 28px;
    padding: 10px 0 8px;
    background: color-mix(in srgb, var(--bg-primary) 88%, transparent);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border-bottom: 1px solid var(--border-subtle);
    box-shadow: 0 6px 14px -10px rgba(0, 0, 0, 0.55);
  }
  .level-selector {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin-bottom: 8px;
  }
  .level-selector button {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 2px;
    padding: 8px 14px;
    min-height: 44px;
    border-radius: 4px;
    border: 1px solid var(--tab-inactive-border);
    background: transparent;
    color: var(--text-secondary);
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s, color 0.15s, transform 0.15s;
    font-family: inherit;
  }
  .level-selector button:hover {
    border-color: var(--text-secondary);
    background: var(--bg-card);
    transform: translateY(-1px);
  }
  .level-selector button:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: 2px;
  }
  .level-selector button.active {
    border-color: var(--tab-active-border);
    color: var(--tab-active-border);
    background: var(--tab-active-bg);
    transform: none;
  }
  .level-num {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    color: var(--text-muted);
  }
  .level-selector button.active .level-num { color: var(--tab-active-border); }
  .level-label { font-size: 14px; font-weight: 600; }

  .level-progress {
    height: 3px;
    background: var(--border-subtle);
    border-radius: 1.5px;
    overflow: hidden;
    margin-bottom: 6px;
  }
  .level-progress-bar {
    height: 100%;
    width: var(--p, 0%);
    background: linear-gradient(90deg, var(--accent-teal), var(--accent-gold));
    transition: width 0.25s ease;
  }

  .level-blurb {
    font-size: 12.5px;
    color: var(--text-muted);
    margin-top: 2px;
  }

  /* ── Level section dividers (group cards by level) ─── */
  .level-section {
    max-width: 1180px;
    margin: 8px auto 14px;
    padding: 0 4px;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .level-section::before,
  .level-section::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border-subtle);
  }
  .level-section-tag {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 28px;
    height: 22px;
    padding: 0 8px;
    border-radius: 11px;
    background: var(--badge-bg-teal);
    color: var(--accent-teal);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
  }

  /* ── Card base ─────────────────────────────────────── */
  .card {
    max-width: 1180px;
    margin: 0 auto 30px;
    padding: 22px 26px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    position: relative;
  }
  .card h2 {
    font-size: 19px;
    font-weight: 700;
    margin-bottom: 12px;
    color: var(--text-primary);
  }
  .card .card-h3 {
    font-size: 15.5px;
    font-weight: 700;
    margin: 16px 0 6px;
    color: var(--text-primary);
    letter-spacing: 0;
    text-transform: none;
  }
  .card p {
    line-height: 1.55;
    color: var(--text-primary);
    margin-bottom: 12px;
    max-width: 70ch;
  }
  .card > ol,
  .card > ul,
  .card > .footnote {
    max-width: 70ch;
  }
  .card p:last-child { margin-bottom: 0; }
  .card em { color: var(--accent-gold); font-style: normal; }
  .card code {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12.5px;
    padding: 1px 6px;
    border-radius: 3px;
    background: color-mix(in srgb, var(--accent-ink) 10%, var(--bg-card));
    color: var(--text-primary);
  }

  /* Hero (L1) variant — accent-teal stripe + slight elevation. */
  .card.hero {
    background: linear-gradient(
      135deg,
      color-mix(in srgb, var(--accent-teal) 7%, var(--bg-secondary)) 0%,
      var(--bg-secondary) 60%
    );
    border-left: 4px solid var(--accent-teal);
    padding-left: 26px;
  }
  .card.hero h2 {
    font-size: 22px;
    color: var(--accent-teal);
  }
  .card.hero > p:first-of-type {
    font-size: 15.5px;
    line-height: 1.6;
  }

  /* Research (L5) variant — gold stripe, matches "open question" tone. */
  .card.research {
    border-left: 4px solid var(--accent-gold);
    padding-left: 26px;
  }
  .card.research h2 { color: var(--accent-gold); }

  /* ── Callout, lists ─────────────────────────────────── */
  .callout {
    margin: 14px 0 4px;
    padding: 12px 14px;
    border-left: 3px solid var(--accent-gold);
    background: var(--badge-bg-gold);
    border-radius: 0 4px 4px 0;
    font-size: 14px;
    line-height: 1.55;
    max-width: 80ch;
  }
  .callout strong { color: var(--accent-gold); margin-right: 4px; }

  .loop-list {
    margin: 10px 0 0 24px;
    line-height: 1.65;
  }
  .loop-list strong { color: var(--accent-teal); }

  .diff-list { margin: 4px 0 8px 22px; line-height: 1.6; }
  .diff-list li { margin-bottom: 6px; }
  .diff-list strong { color: var(--accent-teal); }

  /* ── Plane grid: now multi-column on wider viewports ─ */
  .planes-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 6px;
    margin: 12px 0 14px;
    border-radius: 6px;
  }
  .plane-row {
    display: grid;
    grid-template-columns: 64px 1fr;
    gap: 10px;
    padding: 8px 12px 8px 9px;
    align-items: center;
    font-size: 13.5px;
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-left: 3px solid transparent;
    border-radius: 4px;
  }
  .plane-range {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-weight: 700;
    text-align: right;
  }
  .plane-teal { border-left-color: var(--accent-teal); }
  .plane-ink { border-left-color: var(--accent-ink); }
  .plane-gold { border-left-color: var(--accent-gold); }
  .plane-moss { border-left-color: var(--accent-moss); }
  .plane-danger { border-left-color: var(--danger); }
  .plane-muted { border-left-color: var(--text-muted); }
  .plane-teal .plane-range { color: var(--accent-teal); }
  .plane-ink .plane-range { color: var(--accent-ink); }
  .plane-gold .plane-range { color: var(--accent-gold); }
  .plane-moss .plane-range { color: var(--accent-moss); }
  .plane-danger .plane-range { color: var(--danger); }
  .plane-muted .plane-range { color: var(--text-muted); }

  .footnote {
    font-size: 13px;
    color: var(--text-secondary);
    margin-top: 6px;
    line-height: 1.5;
  }

  /* ── Tables ────────────────────────────────────────── */
  .table-scroll {
    overflow-x: auto;
    margin: 8px 0 4px;
  }
  .data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13.5px;
  }
  .data-table th,
  .data-table td {
    padding: 8px 10px;
    text-align: left;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
  }
  .data-table th {
    font-size: 12px;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    border-bottom: 1px solid var(--border-subtle);
  }
  .data-table tr:last-child td { border-bottom: none; }
  .data-table code.path {
    font-size: 12px;
    line-height: 1.4;
    background: transparent;
    padding: 0;
  }
  .data-table.knobs td:first-child {
    width: 180px;
    color: var(--text-secondary);
    font-weight: 600;
  }

  /* ── Diagrams ──────────────────────────────────────── */
  .diagram-frame {
    margin: 14px 0;
    padding: 14px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow-x: auto;
    color: var(--text-secondary);
  }
  .diagram-frame svg {
    display: block;
    width: 100%;
    height: auto;
    min-width: 480px;
  }
  /* X2: drop diagram min-width on phones to avoid horizontal scroll. */
  @media (max-width: 600px) {
    .diagram-frame svg { min-width: 0; }
  }
  .diagram-frame :global(.svg-title) {
    font: 600 13px ui-sans-serif, system-ui, sans-serif;
    fill: var(--text-primary);
  }
  .diagram-frame :global(.svg-sub) {
    font: 500 11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    fill: var(--text-secondary);
  }
  .diagram-frame :global(.svg-emph) { fill: var(--accent-teal); font-weight: 700; }

  /* ── PPO objective: formula + table side-by-side on wide ─ */
  .ppo-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 16px;
    align-items: start;
    margin: 8px 0 12px;
  }
  @media (min-width: 860px) {
    .ppo-grid { grid-template-columns: minmax(0, 1.05fr) minmax(0, 1fr); }
  }

  .loss-formula {
    display: flex;
    flex-direction: column;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
  }
  .loss-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    padding: 8px 14px;
    border-bottom: 1px solid var(--border-subtle);
    background: var(--bg-secondary);
  }
  .loss-block {
    padding: 14px 16px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 13.5px;
    line-height: 1.55;
    color: var(--text-primary);
    overflow-x: auto;
    white-space: pre;
    margin: 0;
  }
  .loss-block :global(.loss-coef) {
    color: var(--accent-gold);
    font-weight: 700;
  }
  .loss-block :global(.loss-comment) {
    color: var(--text-muted);
  }

  /* ── <details> ─────────────────────────────────────── */
  details {
    margin-top: 10px;
    padding: 8px 12px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
  }
  details > summary {
    cursor: pointer;
    font-weight: 600;
    color: var(--accent-teal);
    font-size: 13.5px;
    list-style: none;
    display: block;
  }
  details > summary::-webkit-details-marker { display: none; }
  details > summary::before { content: '▸ '; color: var(--text-muted); }
  details[open] > summary::before { content: '▾ '; }
  details > summary:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: 2px;
    border-radius: 2px;
  }
  details > p {
    margin-top: 8px;
    font-size: 13.5px;
    line-height: 1.55;
    color: var(--text-secondary);
    max-width: 70ch;
  }

  /* ── Footer ────────────────────────────────────────── */
  .about-footer {
    max-width: 920px;
    margin: 24px auto 0;
    padding: 12px 4px;
    color: var(--text-muted);
    font-size: 13px;
    text-align: center;
    border-top: 1px solid var(--border-subtle);
  }
  .about-footer .issue-link {
    margin-left: 8px;
    color: var(--accent-teal);
    text-decoration: underline;
    text-underline-offset: 2px;
    font-style: normal;
  }
  .about-footer .issue-link:hover { text-decoration-thickness: 2px; }
  .about-footer .issue-link:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: 2px;
    border-radius: 2px;
  }

  /* ── Right-rail TOC ───────────────────────────────── */
  .about-toc {
    position: sticky;
    top: 90px;
    align-self: start;
    padding: 14px 4px 14px 18px;
    border-left: 1px solid var(--border-subtle);
    max-height: calc(100dvh - 110px);
    overflow-y: auto;
  }
  .toc-heading {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 10px;
  }
  .toc-list {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  .toc-list li {
    margin: 0;
    padding: 0;
  }
  .toc-list a {
    display: block;
    padding: 5px 10px;
    font-size: 13px;
    color: var(--text-secondary);
    text-decoration: none;
    border-left: 2px solid transparent;
    margin-left: -1px;
    transition: color 0.15s, border-color 0.15s, background 0.15s;
    line-height: 1.45;
  }
  .toc-list a:hover {
    color: var(--text-primary);
    background: var(--bg-card);
  }
  .toc-list a:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: 2px;
    border-radius: 2px;
  }
  .toc-list li.active a {
    color: var(--accent-teal);
    border-left-color: var(--accent-teal);
    font-weight: 600;
  }

  /* Don't draw a focus ring on the panel itself — it's only focusable
     to be a skip-link target. */
  .about-view:focus { outline: none; }

  @media (prefers-reduced-motion: reduce) {
    .level-selector button,
    .level-progress-bar,
    .toc-list a {
      transition: none;
    }
  }

  /* ── Print: expand all levels, drop chrome ─────────── */
  @media print {
    .about-view {
      color: #000;
      background: #fff;
      overflow: visible;
      height: auto;
      padding: 0;
    }
    .level-bar,
    .about-toc,
    .level-section {
      display: none !important;
    }
    .about-layout {
      display: block;
      max-width: none;
    }
    .about-view :global([data-min-level]) { display: block !important; }
    .card {
      max-width: none;
      page-break-inside: avoid;
      break-inside: avoid;
      box-shadow: none;
      background: #fff;
      border: 1px solid #ccc;
      margin-bottom: 16px;
    }
    .card.hero,
    .card.research {
      background: #fff;
    }
    details > summary::before { content: ''; }
    details[open] > p,
    details > p { display: block; }
    details > summary { color: #000; }
    .about-footer .issue-link {
      color: #000;
      text-decoration: none;
    }
    .about-footer .issue-link::after {
      content: ' (https://github.com/tachyon-beep/keisei/issues/new)';
      color: #555;
      font-size: 11px;
    }
  }
</style>
