/// <reference lib="webworker" />

import * as ort from 'onnxruntime-web'
import { evaluate, setBrowserInferenceSession } from './browser-inference'
import ortWasmUrl from './ort-wasm-url'
import { Decompress } from 'fzstd'
import initBrowserWasm, {
  BrowserGameBuilder as WasmBrowserGameBuilder,
} from './wasm/alphablokus_browser_wasm'
import {
  STRENGTHS,
  type LoadingProgress,
  type Piece,
  type Placement,
  type PersistedGame,
  type Seat,
  type Snapshot,
  type WorkerCommand,
  type WorkerEvent,
} from './protocol'

declare const self: DedicatedWorkerGlobalScope

const BOARD_SIZE = 20

// Vite fingerprints these dependencies. Tell ONNX Runtime Web their final
// URLs instead of letting it resolve unfingerprinted filenames against the page.
ort.env.wasm.wasmPaths = ortWasmUrl

let session: ort.InferenceSession | null = null
let current: Snapshot | null = null
let moveHistory: number[] = []
type BrowserGame = {
  state_json(): string
  pieces_json(): string
  legal_placements_json(orientationId: number): string
  current_player(): number
  valid_move_indexes(): Uint32Array
  choose_move(
    rollouts: number,
    evaluate: (board: Uint8Array, pieceAvailability: Uint8Array, policyIndexes: Uint32Array) => Promise<unknown>,
    onProgress: (completed: number, total: number) => void,
  ): Promise<string>
  apply_move(moveIndex: number): boolean
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
const errorMessage = (error: unknown) => error instanceof Error ? error.message : String(error)
const emitLoading = (progress: LoadingProgress) => emit({ type: 'loading', progress })
const emitCurrent = () => {
  if (!current) return
  const persistedGame: PersistedGame = {
    version: 2,
    seats: current.seats,
    moves: [...moveHistory],
  }
  emit({ type: 'persist', game: persistedGame })
  emit({ type: 'snapshot', snapshot: current })
}
const yieldToWorker = () => new Promise<void>((resolve) => setTimeout(resolve, 0))
const TABLE_CHUNK_BYTES = 512 * 1024

self.addEventListener('unhandledrejection', (event) => {
  event.preventDefault()
  emit({ type: 'error', message: `Unhandled worker error: ${errorMessage(event.reason)}` })
})

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

function snapshot(seats: Seat[]): Snapshot {
  return {
    boardSize: BOARD_SIZE,
    board: Array.from({ length: BOARD_SIZE }, () => Array<number>(BOARD_SIZE).fill(-1)),
    currentPlayer: 0,
    seats,
    thinking: false,
    gameOver: false,
    message: 'Choose a piece to begin.',
    pieces: [],
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

function buildSnapshot(seats: Seat[], message: string): Snapshot {
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
    thinking: false,
    gameOver: activeGame.valid_move_indexes().length === 0,
    message,
    pieces: JSON.parse(activeGame.pieces_json()) as Piece[],
  }
}

async function playBotTurns(seats: Seat[]): Promise<void> {
  const activeGame = requireGame()
  while (seats[activeGame.current_player()] !== 'human' && activeGame.valid_move_indexes().length > 0) {
    const player = activeGame.current_player()
    const strength = seats[player]
    if (strength === 'human') break
    current = { ...buildSnapshot(seats, 'AlphaBlokus is thinking…'), thinking: true }
    emitCurrent()
    const rollouts = STRENGTHS[strength].rollouts
    const reportProgress = (completed: number, total: number) => {
      emit({
        type: 'bot-progress',
        progress: { player, completed, total },
      })
    }
    reportProgress(0, rollouts)
    const result = JSON.parse(await activeGame.choose_move(rollouts, evaluate, reportProgress)) as {
      move_index: number
    }
    activeGame.apply_move(result.move_index)
    moveHistory.push(result.move_index)
  }
  current = buildSnapshot(seats, activeGame.valid_move_indexes().length ? 'Your turn.' : 'Game over.')
  emitCurrent()
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
  emitLoading({ label: 'Preparing local model' })
  const options: ort.InferenceSession.SessionOptions = {
    executionProviders: ['wasm'],
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
        emit({ type: 'ready' })
        return
      case 'start-game':
        await loadBrowserGame()
        // Avoid downloading public model weights for an all-human local game.
        if (data.seats.some((seat) => seat !== 'human')) await loadModel()
        requireGame().reset()
        moveHistory = []
        current = snapshot(data.seats)
        await playBotTurns(data.seats)
        return
      case 'restore-game': {
        await loadBrowserGame()
        if (data.game.seats.some((seat) => seat !== 'human')) await loadModel()
        const activeGame = requireGame()
        activeGame.reset()
        moveHistory = []
        for (const moveIndex of data.game.moves) {
          activeGame.apply_move(moveIndex)
          moveHistory.push(moveIndex)
        }
        current = buildSnapshot(data.game.seats, 'Game restored.')
        await playBotTurns(data.game.seats)
        return
      }
      case 'select-orientation': {
        const activeGame = requireGame()
        const placements = (
          JSON.parse(activeGame.legal_placements_json(data.orientationId)) as {
            move_index: number
            cells: [number, number][]
          }[]
        ).map<Placement>((placement) => ({
          moveIndex: placement.move_index,
          cells: placement.cells,
        }))
        emit({
          type: 'placements',
          orientationId: data.orientationId,
          placements,
        })
        return
      }
      case 'play-move':
        if (!game || !current) return
        if (current.seats[game.current_player()] !== 'human') throw new Error('Wait for the bot to finish its turn.')
        game.apply_move(data.moveIndex)
        moveHistory.push(data.moveIndex)
        await playBotTurns(current.seats)
        return
    }
  } catch (error) {
    emit({ type: 'error', message: errorMessage(error) })
  }
}
