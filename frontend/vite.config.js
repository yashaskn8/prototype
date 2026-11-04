import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: './',
  server: {
    host: process.env.VITE_SERVY_REAL_DEMO === '1' ? '127.0.0.1' : true,
    port: process.env.VITE_SERVY_REAL_DEMO === '1' ? 5175 : 5173,
    proxy: process.env.VITE_SERVY_REAL_DEMO === '1'
      ? { '/demo-api': { target: 'http://127.0.0.1:8002', changeOrigin: true } }
      : { '/api': {
        target: process.env.SERVY_UI_TEST_API_TARGET || 'http://127.0.0.1:8000',
        changeOrigin: true,
      } },
  }
})
