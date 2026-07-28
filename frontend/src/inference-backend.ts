import type { InferenceBackend } from './protocol'

export type InferenceConfiguration = {
  backend: InferenceBackend
  debug: boolean
  requestedBackend: InferenceBackend | null
}

type NavigatorWithWebGPU = Navigator & {
  gpu?: {
    requestAdapter(): Promise<unknown | null>
  }
}

function isIOSPlatform(): boolean {
  return /iPad|iPhone|iPod/.test(navigator.userAgent)
    || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1)
}

export function getInferenceConfiguration(): InferenceConfiguration {
  const parameters = new URLSearchParams(window.location.search)
  const backendParameter = parameters.get('backend')?.toLowerCase()
  const requestedBackend =
    backendParameter === 'webgpu' || backendParameter === 'wasm'
      ? backendParameter
      : null
  const supportsWebGPU = 'gpu' in navigator

  let backend: InferenceBackend
  if (requestedBackend === 'wasm') {
    backend = 'wasm'
  } else if (requestedBackend === 'webgpu') {
    backend = supportsWebGPU ? 'webgpu' : 'wasm'
  } else {
    backend = !isIOSPlatform() && supportsWebGPU ? 'webgpu' : 'wasm'
  }

  return {
    backend,
    debug: parameters.get('debug') === 'true',
    requestedBackend,
  }
}

export async function resolveInferenceBackend(
  candidate: InferenceBackend,
): Promise<InferenceBackend> {
  if (candidate === 'wasm') return 'wasm'

  const gpu = (navigator as NavigatorWithWebGPU).gpu
  if (!gpu) return 'wasm'
  try {
    return await gpu.requestAdapter() ? 'webgpu' : 'wasm'
  } catch {
    return 'wasm'
  }
}
