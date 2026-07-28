export const PLAYER_COLORS = ['#3b82f6', '#eab308', '#ef4444', '#22c55e'] as const
export const STRENGTHS = {
  quick: { label: 'Quick', rollouts: 100 },
  strong: { label: 'Balanced', rollouts: 500 },
  expert: { label: 'Strong', rollouts: 2000 },
} as const

export type Strength = keyof typeof STRENGTHS
export type Seat = 'human' | Strength

export type PersistedGame = {
  version: 2
  seats: Seat[]
  moves: number[]
}

export type PieceOrientation = {
  id: number
  width: number
  height: number
  cells: [number, number][]
  valid: boolean
}

export type Piece = {
  id: number
  squares: number
  available: boolean
  orientations: PieceOrientation[]
}

export type Placement = {
  moveIndex: number
  cells: [number, number][]
}

export type LoadingProgress = {
  label: string
  loaded?: number
  total?: number
  unit?: 'bytes' | 'profiles'
}

export type BotProgress = {
  player: number
  completed: number
  total: number
}

export type Snapshot = {
  boardSize: number
  board: number[][]
  currentPlayer: number
  seats: Seat[]
  thinking: boolean
  gameOver: boolean
  message: string
  pieces: Piece[]
}

export type WorkerCommand =
  | { type: 'init' }
  | { type: 'start-game'; seats: Seat[] }
  | { type: 'restore-game'; game: PersistedGame }
  | { type: 'select-orientation'; orientationId: number }
  | { type: 'play-move'; moveIndex: number }

export type WorkerEvent =
  | { type: 'ready' }
  | { type: 'loading'; progress: LoadingProgress }
  | { type: 'bot-progress'; progress: BotProgress }
  | { type: 'snapshot'; snapshot: Snapshot }
  | { type: 'placements'; orientationId: number; placements: Placement[] }
  | { type: 'persist'; game: PersistedGame }
  | { type: 'error'; message: string }
