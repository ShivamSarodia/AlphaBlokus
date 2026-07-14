export const PLAYER_COLORS = ['#3b82f6', '#eab308', '#ef4444', '#22c55e'] as const
export const STRENGTHS = {
  quick: { label: 'Quick', rollouts: 100 },
  strong: { label: 'Strong', rollouts: 500 },
  expert: { label: 'Expert', rollouts: 2000 },
} as const

export type Strength = keyof typeof STRENGTHS
export type Seat = 'human' | 'bot'

export type LoadingProgress = {
  label: string
  loaded?: number
  total?: number
  unit?: 'bytes' | 'profiles'
}

export type Snapshot = {
  boardSize: number
  board: number[][]
  currentPlayer: number
  seats: Seat[]
  strength: Strength
  thinking: boolean
  gameOver: boolean
  message: string
}

export type WorkerCommand =
  | { type: 'init' }
  | { type: 'start-game'; seats: Seat[]; strength: Strength }
  | { type: 'restart' }
  | { type: 'undo' }
  | { type: 'play-move'; moveIndex: number }
  | { type: 'save' }

export type WorkerEvent =
  | { type: 'ready'; webgpu: true }
  | { type: 'loading'; progress: LoadingProgress }
  | { type: 'snapshot'; snapshot: Snapshot }
  | { type: 'save'; json: string }
  | { type: 'error'; message: string }
