import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const proxyTarget = process.env.VITE_API_PROXY_TARGET
const port = Number(process.env.VITE_PORT ?? 5173)

export default defineConfig({
  plugins: [react()],
  server: {
    port,
    open: true,
    proxy: proxyTarget
      ? {
          '/api': {
            target: proxyTarget,
            changeOrigin: true,
          },
        }
      : undefined,
  },
})
