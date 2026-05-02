import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'

export default defineConfig({
  plugins: [svelte()],
  test: {
    environment: 'node',
  },
  build: {
    outDir: '../keisei/server/static',
    emptyOutDir: true,
  },
  server: {
    hmr: {
      protocol: 'ws',
      host: 'localhost',
    },
    proxy: {
      '/ws': {
        target: 'ws://localhost:8001',
        ws: true,
      },
      '/healthz': 'http://localhost:8001',
      '/audio': 'http://localhost:8001',
    },
  },
})
