import { defineConfig, loadEnv } from 'vite'
import { dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import react from '@vitejs/plugin-react'

const envDir = dirname(fileURLToPath(import.meta.url))

export default defineConfig(({ mode }) => {
  // Resolve mode once from Vite's normal environment loading rules. The same
  // VITE_* values are exposed to browser code through import.meta.env, so the
  // selected React app and its port/proxy cannot disagree about demo mode.
  const env = loadEnv(mode, envDir, 'VITE_')
  const realDemo = env.VITE_SERVY_REAL_DEMO === '1'

  return {
    plugins: [react()],
    base: './',
    server: {
      host: '127.0.0.1',
      port: realDemo ? 5175 : 5173,
      proxy: realDemo
        ? { '/demo-api': { target: 'http://127.0.0.1:8002', changeOrigin: true } }
        : { '/api': {
          target: process.env.SERVY_UI_TEST_API_TARGET || 'http://127.0.0.1:8000',
          changeOrigin: true,
        } },
    },
  }
})
