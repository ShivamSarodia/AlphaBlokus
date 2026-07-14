/// <reference lib="webworker" />

import * as ort from 'onnxruntime-web/webgpu'
import { evaluate, setBrowserInferenceSession } from './browser-inference'
import ortWasmUrl from './ort-wasm-url'
import { Decompress } from 'fzstd'
import initBrowserWasm, {
  BrowserGameBuilder as WasmBrowserGameBuilder,
} from './wasm/alphablokus_browser_wasm'
import { STRENGTHS, type LoadingProgress, type Seat, type Snapshot, type Strength, type WorkerCommand, type WorkerEvent } from './protocol'

declare const self: DedicatedWorkerGlobalScope

const BOARD_SIZE = 20

// Vite fingerprints this dependency. Tell ONNX Runtime Web its final URL
// instead of letting it resolve an unfingerprinted filename against the page.
ort.env.wasm.wasmPaths = { wasm: ortWasmUrl }

let session: ort.InferenceSession | null = null
let current: Snapshot | null = null
type BrowserGame = {
  state_json(): string
  current_player(): number
  valid_move_indexes(): Uint32Array
  choose_move(
    rollouts: number,
    evaluate: (board: Uint8Array, pieceAvailability: Uint8Array, policyIndexes: Uint32Array) => Promise<unknown>,
  ): Promise<string>
  apply_move(moveIndex: number): boolean
  undo(): boolean
  reset(): void
}
type BrowserGameBuilder = {
  total_profiles(): number
  completed_profiles(): number
  build_profiles(count: number): number
  finish(): BrowserGame
}
let game: BrowserGame | null = null

const emit = (event: WorkerEvent) => self.postMessage(event)
const emitLoading = (progress: LoadingProgress) => emit({ type: 'loading', progress })
const yieldToWorker = () => new Promise<void>((resolve) => setTimeout(resolve, 0))
const TABLE_CHUNK_BYTES = 512 * 1024

async function fetchWithProgress(url: string, label: string): Promise<Uint8Array> {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Could not download ${label.toLowerCase()}.`)

  const total = Number(response.headers.get('content-length')) || undefined
  if (!response.body) {
    const bytes = new Uint8Array(await response.arrayBuffer())
    emitLoading({ label, loaded: bytes.byteLength, total: bytes.byteLength })
    return bytes
  }

  emitLoading({ label, loaded: 0, total })
  const reader = response.body.getReader()
  const chunks: Uint8Array[] = []
  let loaded = 0
  let lastReported = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    if (!value) continue
    chunks.push(value)
    loaded += value.byteLength
    if (loaded - lastReported >= 256 * 1024) {
      emitLoading({ label, loaded, total })
      lastReported = loaded
    }
  }

  const bytes = new Uint8Array(loaded)
  let offset = 0
  for (const chunk of chunks) {
    bytes.set(chunk, offset)
    offset += chunk.byteLength
  }
  emitLoading({ label, loaded, total: total ?? loaded })
  return bytes
}

function joinChunks(chunks: Uint8Array[]): Uint8Array {
  const length = chunks.reduce((total, chunk) => total + chunk.byteLength, 0)
  const bytes = new Uint8Array(length)
  let offset = 0
  for (const chunk of chunks) {
    bytes.set(chunk, offset)
    offset += chunk.byteLength
  }
  return bytes
}

async function decompressWithNativeStream(compressed: Uint8Array): Promise<Uint8Array> {
  const decompressor = new DecompressionStream('zstd' as unknown as CompressionFormat)
  const writer = decompressor.writable.getWriter()
  const reader = decompressor.readable.getReader()
  const output: Uint8Array[] = []
  const consumeOutput = (async () => {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) return
      if (value) output.push(value)
    }
  })()
  for (let offset = 0; offset < compressed.byteLength; offset += TABLE_CHUNK_BYTES) {
    const end = Math.min(offset + TABLE_CHUNK_BYTES, compressed.byteLength)
    await writer.write(compressed.slice(offset, end))
    emitLoading({ label: 'Unpacking move table', loaded: end, total: compressed.byteLength })
    await yieldToWorker()
  }
  await writer.close()
  await consumeOutput
  return joinChunks(output)
}

async function decompressWithFallback(compressed: Uint8Array): Promise<Uint8Array> {
  const output: Uint8Array[] = []
  const stream = new Decompress((chunk) => output.push(chunk))
  for (let offset = 0; offset < compressed.byteLength; offset += TABLE_CHUNK_BYTES) {
    const end = Math.min(offset + TABLE_CHUNK_BYTES, compressed.byteLength)
    stream.push(compressed.subarray(offset, end), end === compressed.byteLength)
    emitLoading({ label: 'Unpacking move table', loaded: end, total: compressed.byteLength })
    await yieldToWorker()
  }
  return joinChunks(output)
}

async function decompressMoveTable(compressed: Uint8Array): Promise<Uint8Array> {
  emitLoading({ label: 'Unpacking move table', loaded: 0, total: compressed.byteLength })
  try {
    return await decompressWithNativeStream(compressed)
  } catch {
    return decompressWithFallback(compressed)
  }
}

function snapshot(seats: Seat[], strength: Strength): Snapshot {
  return {
    boardSize: BOARD_SIZE,
    board: Array.from({ length: BOARD_SIZE }, () => Array<number>(BOARD_SIZE).fill(-1)),
    currentPlayer: 0,
    seats,
    strength,
    thinking: false,
    gameOver: false,
    message: 'Choose a piece to begin.',
  }
}

async function loadBrowserGame(): Promise<void> {
  if (game) return
  const moveDataUrl = `${import.meta.env.BASE_URL}move-data.bin`
  emitLoading({ label: 'Preparing game engine' })
  await initBrowserWasm()
  const compressed = await fetchWithProgress(moveDataUrl, 'Downloading move table')
  const moveData = await decompressMoveTable(compressed)
  emitLoading({ label: 'Reading move table metadata' })
  const builder: BrowserGameBuilder = new WasmBrowserGameBuilder(moveData)
  const totalProfiles = builder.total_profiles()
  while (builder.completed_profiles() < totalProfiles) {
    const completed = builder.build_profiles(128)
    emitLoading({ label: 'Building game rules', loaded: completed, total: totalProfiles, unit: 'profiles' })
    await yieldToWorker()
  }
  game = builder.finish()
}

function buildSnapshot(seats: Seat[], strength: Strength, message: string): Snapshot {
  const activeGame = requireGame()
  const state = JSON.parse(activeGame.state_json())
  const board = Array.from({ length: BOARD_SIZE }, () => Array<number>(BOARD_SIZE).fill(-1))
  state.board.slices.forEach((slice: { cells: boolean[][] }, player: number) => {
    slice.cells.forEach((column, x) => column.forEach((filled, y) => {
      if (filled) board[y][x] = player
    }))
  })
  return {
    boardSize: BOARD_SIZE,
    board,
    currentPlayer: activeGame.current_player(),
    seats,
    strength,
    thinking: false,
    gameOver: activeGame.valid_move_indexes().length === 0,
    message,
  }
}

async function playBotTurns(seats: Seat[], strength: Strength): Promise<void> {
  const activeGame = requireGame()
  while (seats[activeGame.current_player()] === 'bot' && activeGame.valid_move_indexes().length > 0) {
    current = { ...buildSnapshot(seats, strength, 'AlphaBlokus is thinking…'), thinking: true }
    emit({ type: 'snapshot', snapshot: current })
    const result = JSON.parse(await activeGame.choose_move(STRENGTHS[strength].rollouts, evaluate))
    activeGame.apply_move(result.move_index)
  }
  current = buildSnapshot(seats, strength, activeGame.valid_move_indexes().length ? 'Your turn.' : 'Game over.')
  emit({ type: 'snapshot', snapshot: current })
}

function requireGame(): BrowserGame {
  if (!game) throw new Error('The browser game has not finished loading.')
  return game
}

async function loadModel(): Promise<void> {
  if (session) return
  const modelUrl = `${import.meta.env.BASE_URL}026025784.onnx`
  const externalDataUrl = `${import.meta.env.BASE_URL}026025784.onnx.data`
  const model = await fetchWithProgress(modelUrl, 'Downloading model definition')
  const externalData = await fetchWithProgress(externalDataUrl, 'Downloading model weights')
  emitLoading({ label: 'Preparing WebGPU model' })
  const options: ort.InferenceSession.SessionOptions = {
    executionProviders: [{ name: 'webgpu', preferredLayout: 'NCHW' }],
  }
  Object.assign(options, {
    externalData: [{ path: '026025784.onnx.data', data: externalData }],
  })
  session = await ort.InferenceSession.create(model, options)
  setBrowserInferenceSession(session)
}

self.onmessage = async ({ data }: MessageEvent<WorkerCommand>) => {
  try {
    switch (data.type) {
      case 'init':
        if (!('gpu' in navigator)) {
          throw new Error('WebGPU is required to run AlphaBlokus in this browser.')
        }
        emit({ type: 'ready', webgpu: true })
        return
      case 'start-game':
        await loadBrowserGame()
        // Avoid downloading public model weights for an all-human local game.
        if (data.seats.includes('bot')) await loadModel()
        current = snapshot(data.seats, data.strength)
        await playBotTurns(data.seats, data.strength)
        return
      case 'restart':
        if (game && current) {
          game.reset()
          await playBotTurns(current.seats, current.strength)
        }
        return
      case 'undo':
        if (game && current) {
          game.undo()
          current = buildSnapshot(current.seats, current.strength, 'Move undone.')
          emit({ type: 'snapshot', snapshot: current })
        }
        return
      case 'play-move':
        if (!game || !current) return
        if (current.seats[game.current_player()] !== 'human') throw new Error('Wait for the bot to finish its turn.')
        game.apply_move(data.moveIndex)
        await playBotTurns(current.seats, current.strength)
        return
      case 'save':
        if (game) emit({ type: 'save', json: game.state_json() })
        return
    }
  } catch (error) {
    emit({ type: 'error', message: error instanceof Error ? error.message : String(error) })
  }
}
