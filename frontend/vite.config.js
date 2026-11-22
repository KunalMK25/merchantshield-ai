import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev server proxies API calls to the FastAPI backend on :8000.
const API = process.env.VITE_PROXY_TARGET || 'http://localhost:8000'
const apiPaths = ['/health', '/model', '/risk', '/audit-log', '/registry']

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: Object.fromEntries(apiPaths.map((p) => [p, { target: API, changeOrigin: true }])),
  },
})
