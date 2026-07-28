import { copyFileSync, createReadStream, statSync } from 'node:fs'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'

const proxyTarget = process.env.VITE_API_PROXY_TARGET
const port = Number(process.env.VITE_PORT ?? 5173)
const moveTablePath = fileURLToPath(new URL('../static/move_data/full.bin', import.meta.url))

function canonicalMoveTable(): Plugin {
  return {
    name: 'canonical-move-table',
    configureServer(server) {
      server.middlewares.use('/move-data.bin', (_request, response, next) => {
        response.setHeader('Content-Type', 'application/octet-stream')
        response.setHeader('Content-Length', statSync(moveTablePath).size)
        createReadStream(moveTablePath).on('error', next).pipe(response)
      })
    },
    writeBundle(options) {
      copyFileSync(moveTablePath, resolve(options.dir ?? 'dist', 'move-data.bin'))
    },
  }
}

export default defineConfig({
  plugins: [react(), canonicalMoveTable()],
  worker: {
    format: 'es',
  },
  resolve: {
    // Use ONNX Runtime Web's external-WASM entry point. The worker supplies
    // the paired model-specific loader and binary through env.wasm.wasmPaths.
    conditions: ['onnxruntime-web-use-extern-wasm'],
  },
  // The checked-in ONNX model pair is the only root static asset group. Keep
  // unrelated native move tables out of the Pages artifact. The canonical
  // full table is copied explicitly by canonicalMoveTable above.
  publicDir: '../static/browser',
  // GitHub Pages project sites are served below /<repository>/.
  base: process.env.VITE_BASE_PATH ?? '/',
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
