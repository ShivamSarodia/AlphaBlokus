import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import { isMobileDevice } from './device'
import { getInferenceConfiguration, resolveInferenceBackend } from './inference-backend'
import {
  PLAYER_COLORS,
  STRENGTHS,
  type BotProgress,
  type InferenceBackend,
  type LoadingProgress,
  type Piece,
  type PieceOrientation,
  type Placement,
  type PersistedGame,
  type Seat,
  type Snapshot,
  type Strength,
  type WorkerCommand,
  type WorkerEvent,
} from './protocol'

const send = (worker: Worker | null, command: WorkerCommand) => worker?.postMessage(command)
const GAME_STORAGE_KEY = 'alphablokus.game.v1'
const PLAYER_NAMES = ['Blue', 'Yellow', 'Red', 'Green'] as const
const START_CORNER_NAMES = ['top-left', 'top-right', 'bottom-right', 'bottom-left'] as const
const EDGE_DIRECTIONS = [[-1, 0], [1, 0], [0, -1], [0, 1]] as const
const CORNER_DIRECTIONS = [[-1, -1], [1, -1], [-1, 1], [1, 1]] as const

function invalidPlacementReason(
  cells: [number, number][],
  board: number[][],
  player: number,
): string {
  const boardSize = board.length
  const outsideBoard = cells.some(([x, y]) => x < 0 || y < 0 || x >= boardSize || y >= boardSize)
  if (outsideBoard) return 'The entire piece must stay on the board.'

  const overlapsPiece = cells.some(([x, y]) => board[y]?.[x] !== -1)
  if (overlapsPiece) return 'That space is already occupied.'

  const isFirstMove = !board.some((row) => row.includes(player))
  if (isFirstMove) {
    const corner = [
      [0, 0],
      [boardSize - 1, 0],
      [boardSize - 1, boardSize - 1],
      [0, boardSize - 1],
    ][player]
    const coversCorner = corner && cells.some(([x, y]) => x === corner[0] && y === corner[1])
    if (!coversCorner) {
      return `Your first move must cover the ${START_CORNER_NAMES[player]} corner.`
    }
  }

  const sharesOwnEdge = cells.some(([x, y]) =>
    EDGE_DIRECTIONS.some(([dx, dy]) => board[y + dy]?.[x + dx] === player),
  )
  if (sharesOwnEdge) {
    return `This piece cannot share an edge with another ${PLAYER_NAMES[player].toLowerCase()} piece.`
  }

  const touchesOwnCorner = cells.some(([x, y]) =>
    CORNER_DIRECTIONS.some(([dx, dy]) => board[y + dy]?.[x + dx] === player),
  )
  if (!isFirstMove && !touchesOwnCorner) {
    return `This piece must touch a corner of another ${PLAYER_NAMES[player].toLowerCase()} piece.`
  }

  return 'That placement is not valid.'
}

function readPersistedGame(): PersistedGame | null {
  try {
    const stored = localStorage.getItem(GAME_STORAGE_KEY)
    if (!stored) return null
    const game = JSON.parse(stored) as Partial<PersistedGame>
    const validSeats =
      Array.isArray(game.seats) &&
      game.seats.length === 4 &&
      game.seats.every((seat) => seat === 'human' || seat in STRENGTHS)
    const validMoves =
      Array.isArray(game.moves) &&
      game.moves.every((move) => Number.isInteger(move) && move >= 0)
    if (game.version !== 2 || !validSeats || !validMoves) {
      localStorage.removeItem(GAME_STORAGE_KEY)
      return null
    }
    return game as PersistedGame
  } catch {
    return null
  }
}

function persistGame(game: PersistedGame): void {
  try {
    localStorage.setItem(GAME_STORAGE_KEY, JSON.stringify(game))
  } catch {
    // The game remains playable if storage is unavailable or full.
  }
}

function clearPersistedGame(): void {
  try {
    localStorage.removeItem(GAME_STORAGE_KEY)
  } catch {
    // The in-memory game can still be replaced.
  }
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.ceil(bytes / 1024)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function formatProgress(progress: LoadingProgress): string | null {
  if (progress.loaded === undefined) return null
  if (progress.unit === 'profiles') {
    return `${progress.loaded.toLocaleString()}${progress.total ? ` / ${progress.total.toLocaleString()}` : ''} profiles`
  }
  return `${formatBytes(progress.loaded)}${progress.total ? ` / ${formatBytes(progress.total)}` : ''}`
}

function PiecePreview({
  orientation,
  color,
  cellSize = 9,
}: {
  orientation: PieceOrientation
  color: string
  cellSize?: number
}) {
  return (
    <span
      className="piece-preview"
      style={{
        gridTemplateColumns: `repeat(${orientation.width}, ${cellSize}px)`,
        gridTemplateRows: `repeat(${orientation.height}, ${cellSize}px)`,
      }}
      aria-hidden="true"
    >
      {orientation.cells.map(([x, y]) => (
        <span
          className="piece-preview__cell"
          key={`${x}-${y}`}
          style={{
            background: color,
            gridColumn: x + 1,
            gridRow: y + 1,
          }}
        />
      ))}
    </span>
  )
}

function orientationKey(cells: [number, number][]): string {
  return [...cells]
    .sort(([leftX, leftY], [rightX, rightY]) => leftY - rightY || leftX - rightX)
    .map(([x, y]) => `${x},${y}`)
    .join('|')
}

function transformedOrientation(
  piece: Piece,
  orientation: PieceOrientation,
  transform: 'rotate' | 'flip',
): PieceOrientation {
  const transformedCells: [number, number][] =
    transform === 'rotate'
      ? orientation.cells.map(([x, y]) => [orientation.height - 1 - y, x])
      : orientation.cells.map(([x, y]) => [orientation.width - 1 - x, y])
  const targetKey = orientationKey(transformedCells)
  return piece.orientations.find((candidate) => orientationKey(candidate.cells) === targetKey)
    ?? orientation
}

function defaultSeats(): Seat[] {
  const botStrength: Strength = isMobileDevice() ? 'quick' : 'strong'
  return ['human', botStrength, botStrength, botStrength]
}

export default function App() {
  const inference = useMemo(() => getInferenceConfiguration(), [])
  const worker = useRef<Worker | null>(null)
  const [workerGeneration, setWorkerGeneration] = useState(0)
  const [ready, setReady] = useState(false)
  const [selectedBackend, setSelectedBackend] = useState<InferenceBackend | null>(null)
  const [activeBackend, setActiveBackend] = useState<InferenceBackend | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [seats, setSeats] = useState<Seat[]>(defaultSeats)
  const [game, setGame] = useState<Snapshot | null>(null)
  const [savedGameToRestore, setSavedGameToRestore] = useState<PersistedGame | null>(
    () => readPersistedGame(),
  )
  const savedGameToRestoreRef = useRef(savedGameToRestore)
  const [loading, setLoading] = useState<LoadingProgress | null>(null)
  const [botProgress, setBotProgress] = useState<BotProgress | null>(null)
  const [selectedPieceId, setSelectedPieceId] = useState<number | null>(null)
  const [selectedOrientationId, setSelectedOrientationId] = useState<number | null>(null)
  const [placementResult, setPlacementResult] = useState<{
    orientationId: number
    placements: Placement[]
  } | null>(null)
  const [tentativeCells, setTentativeCells] = useState<[number, number][]>([])
  const [selectedPlacement, setSelectedPlacement] = useState<Placement | null>(null)
  const [placementFeedback, setPlacementFeedback] = useState<'valid' | 'invalid' | null>(null)
  const [isPieceTrayOpen, setIsPieceTrayOpen] = useState(false)
  const [isLoadingPlacements, setIsLoadingPlacements] = useState(false)

  useEffect(() => {
    let instance: Worker | null = null
    let stopped = false

    const stopWorker = () => {
      if (!instance) return
      instance.onmessage = null
      instance.onerror = null
      instance.onmessageerror = null
      instance.terminate()
      if (worker.current === instance) worker.current = null
      instance = null
    }

    const startWorker = async () => {
      if (stopped) return
      const backend = await resolveInferenceBackend(inference.backend)
      if (stopped) return
      setSelectedBackend(backend)
      const nextInstance = new Worker(new URL('./game.worker.ts', import.meta.url), { type: 'module' })
      instance = nextInstance
      worker.current = nextInstance
      nextInstance.onerror = (event) => {
        if (worker.current !== nextInstance) return
        setLoading(null)
        setBotProgress(null)
        setError(event.message || 'The game worker stopped unexpectedly.')
      }
      nextInstance.onmessageerror = () => {
        if (worker.current !== nextInstance) return
        setLoading(null)
        setBotProgress(null)
        setError('The game worker returned an unreadable response.')
      }
      nextInstance.onmessage = ({ data }: MessageEvent<WorkerEvent>) => {
        if (worker.current !== nextInstance) return
        if (data.type === 'ready') {
          setActiveBackend(data.backend)
          setReady(true)
          const savedGame = savedGameToRestoreRef.current
          if (savedGame) {
            setSeats(savedGame.seats)
            setLoading({ label: 'Restoring saved game' })
            send(nextInstance, { type: 'restore-game', game: savedGame })
          }
        }
        if (data.type === 'loading') setLoading(data.progress)
        if (data.type === 'bot-progress') setBotProgress(data.progress)
        if (data.type === 'snapshot') {
          setLoading(null)
          setBotProgress(null)
          setGame(data.snapshot)
          setSavedGameToRestore(null)
          savedGameToRestoreRef.current = null
          setSelectedPieceId(null)
          setSelectedOrientationId(null)
          setPlacementResult(null)
          setTentativeCells([])
          setSelectedPlacement(null)
          setPlacementFeedback(null)
          setIsPieceTrayOpen(false)
          setIsLoadingPlacements(false)
        }
        if (data.type === 'placements') {
          setPlacementResult({
            orientationId: data.orientationId,
            placements: data.placements,
          })
          setTentativeCells([])
          setSelectedPlacement(null)
          setPlacementFeedback(null)
          setIsLoadingPlacements(false)
        }
        if (data.type === 'persist') persistGame(data.game)
        if (data.type === 'error') {
          setLoading(null)
          setBotProgress(null)
          if (savedGameToRestoreRef.current) {
            clearPersistedGame()
            setSavedGameToRestore(null)
            savedGameToRestoreRef.current = null
          }
          setError(data.message)
        }
      }
      send(nextInstance, { type: 'init', backend })
    }

    const stopBeforeUnload = () => stopWorker()
    window.addEventListener('beforeunload', stopBeforeUnload)
    const startupTimer = window.setTimeout(startWorker, 0)

    return () => {
      stopped = true
      window.clearTimeout(startupTimer)
      window.removeEventListener('beforeunload', stopBeforeUnload)
      stopWorker()
    }
  }, [workerGeneration, inference.backend])

  const botCount = useMemo(() => seats.filter((seat) => seat !== 'human').length, [seats])
  const board = game?.board ?? Array.from({ length: 20 }, () => Array<number>(20).fill(-1))
  const selectedPiece =
    selectedPieceId === null ? null : game?.pieces.find((piece) => piece.id === selectedPieceId) ?? null
  const selectedOrientation =
    selectedOrientationId === null
      ? null
      : selectedPiece?.orientations.find((orientation) => orientation.id === selectedOrientationId) ?? null
  const activePlacements =
    placementResult?.orientationId === selectedOrientationId ? placementResult.placements : []
  const previewCells = useMemo(
    () => new Set(tentativeCells.map(([x, y]) => `${x}-${y}`)),
    [tentativeCells],
  )
  const isHumanTurn = Boolean(
    game && game.seats[game.currentPlayer] === 'human' && !game.gameOver && !game.thinking,
  )
  const remainingPieces = game?.pieces.filter((piece) => piece.available) ?? []
  const occupiedCells = useMemo(
    () => [0, 1, 2, 3].map(
      (player) => board.reduce(
        (total, row) => total + row.filter((owner) => owner === player).length,
        0,
      ),
    ),
    [board],
  )
  const winningScore = Math.max(...occupiedCells)
  const winningPlayers = occupiedCells
    .map((score, player) => ({ player, score }))
    .filter(({ score }) => score === winningScore)
    .map(({ player }) => player)
  const winnerLabel = winningPlayers.length === 1
    ? `${PLAYER_NAMES[winningPlayers[0]]} wins`
    : `${winningPlayers.map((player) => PLAYER_NAMES[player]).join(' & ')} tie`
  const highlightedPlayers = game?.gameOver ? winningPlayers : game ? [game.currentPlayer] : []

  const start = () => {
    setError(null)
    setBotProgress(null)
    setLoading({ label: 'Starting local game' })
    clearPersistedGame()
    send(worker.current, { type: 'start-game', seats })
  }

  const newGame = () => {
    const oldWorker = worker.current
    if (oldWorker) {
      worker.current = null
      oldWorker.onmessage = null
      oldWorker.terminate()
    }
    clearPersistedGame()
    setSavedGameToRestore(null)
    savedGameToRestoreRef.current = null
    setReady(false)
    setSelectedBackend(null)
    setActiveBackend(null)
    setError(null)
    setLoading(null)
    setBotProgress(null)
    setGame(null)
    setSelectedPieceId(null)
    setSelectedOrientationId(null)
    setPlacementResult(null)
    setTentativeCells([])
    setSelectedPlacement(null)
    setPlacementFeedback(null)
    setIsPieceTrayOpen(false)
    setIsLoadingPlacements(false)
    setWorkerGeneration((generation) => generation + 1)
  }

  const cancelPlacement = () => {
    setSelectedPieceId(null)
    setSelectedOrientationId(null)
    setPlacementResult(null)
    setTentativeCells([])
    setSelectedPlacement(null)
    setPlacementFeedback(null)
    setIsPieceTrayOpen(false)
    setIsLoadingPlacements(false)
  }

  const selectPiece = (piece: Piece) => {
    if (!piece.available) return
    const defaultOrientation =
      piece.orientations.find((orientation) => orientation.valid) ?? piece.orientations[0]
    setSelectedPieceId(piece.id)
    setPlacementResult(null)
    setTentativeCells([])
    setSelectedPlacement(null)
    setPlacementFeedback(null)
    setIsPieceTrayOpen(false)
    if (defaultOrientation) {
      setSelectedOrientationId(defaultOrientation.id)
      setIsLoadingPlacements(true)
      send(worker.current, {
        type: 'select-orientation',
        orientationId: defaultOrientation.id,
      })
    }
  }

  const selectOrientation = (orientation: PieceOrientation) => {
    setSelectedOrientationId(orientation.id)
    setPlacementResult(null)
    setTentativeCells([])
    setSelectedPlacement(null)
    setPlacementFeedback(null)
    setIsLoadingPlacements(true)
    setIsPieceTrayOpen(false)
    send(worker.current, { type: 'select-orientation', orientationId: orientation.id })
  }

  const transformSelectedPiece = (transform: 'rotate' | 'flip') => {
    if (!selectedPiece || !selectedOrientation) return
    selectOrientation(transformedOrientation(selectedPiece, selectedOrientation, transform))
  }

  const cellsAtBoardPosition = (x: number, y: number): [number, number][] => {
    if (!selectedOrientation || !game) return []
    const originX = Math.min(
      Math.max(x - Math.floor(selectedOrientation.width / 2), 0),
      game.boardSize - selectedOrientation.width,
    )
    const originY = Math.min(
      Math.max(y - Math.floor(selectedOrientation.height / 2), 0),
      game.boardSize - selectedOrientation.height,
    )
    return selectedOrientation.cells.map(([cellX, cellY]) => [
      originX + cellX,
      originY + cellY,
    ])
  }

  const previewBoardPosition = (x: number, y: number) => {
    if (!selectedOrientation || isLoadingPlacements || selectedPlacement) return
    setTentativeCells(cellsAtBoardPosition(x, y))
    setSelectedPlacement(null)
    setPlacementFeedback(null)
  }

  const testBoardPosition = (x: number, y: number) => {
    if (!selectedOrientation || isLoadingPlacements) return
    const cells = cellsAtBoardPosition(x, y)
    const key = orientationKey(cells)
    const match = activePlacements.find((placement) => orientationKey(placement.cells) === key) ?? null
    setTentativeCells(cells)
    setSelectedPlacement(match)
    setPlacementFeedback(match ? 'valid' : 'invalid')
  }

  const confirmPlacement = () => {
    if (!selectedPlacement) return
    send(worker.current, { type: 'play-move', moveIndex: selectedPlacement.moveIndex })
    cancelPlacement()
  }

  const displayedBackend = activeBackend ?? selectedBackend
  const debugPanel = inference.debug ? (
    <aside
      className="inference-debug"
      data-inference-backend={displayedBackend ?? 'detecting'}
      aria-label="Inference debug"
    >
      <span>Inference</span>
      <strong>
        {displayedBackend === 'webgpu' ? 'WebGPU' : displayedBackend === 'wasm' ? 'WASM' : 'Detecting…'}
      </strong>
    </aside>
  ) : null

  if (game) {
    return (
      <>
        <main className="game-shell">
        <section className="table-layout table-layout--game">
          <header className="game-header">
            <h1>AlphaBlokus</h1>
            <div className="header-actions">
              <button
                onClick={newGame}
              >
                New game
              </button>
            </div>
          </header>
          <div className="score-strip" aria-label="Points">
            {game.seats.map((seat, player) => {
              const progress = botProgress?.player === player ? botProgress : null
              return (
                <article
                  className={highlightedPlayers.includes(player) ? 'score-player score-player--current' : 'score-player'}
                  key={player}
                  aria-label={`${PLAYER_NAMES[player]}, ${occupiedCells[player]} points, ${seat === 'human' ? 'Human' : 'Bot'}${progress ? `, ${progress.completed} of ${progress.total} rollouts` : seat !== 'human' && player === game.currentPlayer && game.thinking ? ', thinking' : ''}`}
                  aria-current={!game.gameOver && player === game.currentPlayer ? 'true' : undefined}
                  style={highlightedPlayers.includes(player)
                    ? { backgroundColor: `${PLAYER_COLORS[player]}24` }
                    : undefined}
                >
                  <span className="score-player__summary">
                    <span className="swatch" style={{ background: PLAYER_COLORS[player] }} />
                    <strong style={{ color: PLAYER_COLORS[player] }}>
                      {occupiedCells[player]}<span> pts</span>
                    </strong>
                  </span>
                  <span className="score-player__type">
                    <span className="score-player__role">
                      {seat === 'human' ? 'Human' : 'Bot'}
                      {seat !== 'human' && player === game.currentPlayer && game.thinking && (
                        <span className="bot-thinking" aria-hidden="true" title="Thinking" />
                      )}
                    </span>
                    <span
                      className="score-player__progress"
                      aria-hidden={!progress}
                    >
                      {progress ? `${progress.completed} / ${progress.total}` : '\u00A0'}
                    </span>
                  </span>
                </article>
              )
            })}
          </div>
          <div className="table-stage">
            <div
              className="board-wrap"
              aria-label="Blokus board"
            >
              <div className="board" style={{ gridTemplateColumns: `repeat(${game.boardSize}, 1fr)` }}>
                {board.flatMap((row, y) => row.map((owner, x) => {
                  const isPreview = previewCells.has(`${x}-${y}`)
                  const background = isPreview
                    ? PLAYER_COLORS[game.currentPlayer]
                    : owner >= 0
                      ? PLAYER_COLORS[owner]
                      : undefined
                  return (
                    <button
                      type="button"
                      key={`${x}-${y}`}
                      className={`tile${selectedOrientation ? ' tile--interactive' : ''}${isPreview ? ' tile--preview' : ''}${isPreview && placementFeedback === 'invalid' ? ' tile--invalid' : ''}`}
                      style={background ? { background } : undefined}
                      onMouseEnter={() => previewBoardPosition(x, y)}
                      onClick={() => testBoardPosition(x, y)}
                      disabled={!selectedOrientation || isLoadingPlacements}
                      tabIndex={-1}
                      aria-label={`Place near row ${y + 1}, column ${x + 1}`}
                      aria-pressed={isPreview}
                    />
                  )
                }))}
              </div>
              {placementFeedback === 'valid' ? (
                <button
                  type="button"
                  className="board-feedback board-feedback--valid board-feedback--confirm"
                  style={{
                    backgroundColor: PLAYER_COLORS[game.currentPlayer],
                    borderColor: PLAYER_COLORS[game.currentPlayer],
                  }}
                  onClick={confirmPlacement}
                >
                  Confirm move
                </button>
              ) : (
                <div
                  className={`board-feedback${placementFeedback === 'invalid' ? ' board-feedback--invalid' : ''}`}
                  aria-live="polite"
                >
                  {placementFeedback === 'invalid'
                    ? invalidPlacementReason(tentativeCells, board, game.currentPlayer)
                    : ''}
                </div>
              )}
            </div>
          </div>
          {game.gameOver ? (
            <section className="game-over-panel" aria-labelledby="game-over-title">
              <span className="game-over-panel__eyebrow">Game over</span>
              <h2 id="game-over-title">{winnerLabel}</h2>
              <div className="game-over-panel__score">
                <span className="game-over-panel__swatches" aria-hidden="true">
                  {winningPlayers.map((player) => (
                    <span
                      className="swatch"
                      key={player}
                      style={{ background: PLAYER_COLORS[player] }}
                    />
                  ))}
                </span>
                <strong>{winningScore} pts</strong>
              </div>
              <button type="button" onClick={newGame}>
                New game
              </button>
            </section>
          ) : (
            <>
              <div
                className={`placement-toolbar${isHumanTurn ? '' : ' gameplay-controls--hidden'}`}
                aria-hidden={!isHumanTurn}
              >
                <div className="placement-toolbar__actions">
                  <button
                    type="button"
                    className="orientation-display"
                    onClick={cancelPlacement}
                    disabled={!isHumanTurn || !selectedPiece}
                    style={selectedPiece ? { borderColor: PLAYER_COLORS[game.currentPlayer] } : undefined}
                    aria-label={selectedPiece ? 'Clear selected piece' : 'No piece selected'}
                    title={selectedPiece ? 'Clear selected piece' : 'Choose a piece below'}
                  >
                    {selectedOrientation ? (
                      <PiecePreview
                        orientation={selectedOrientation}
                        color={PLAYER_COLORS[game.currentPlayer]}
                        cellSize={12}
                      />
                    ) : (
                      <span className="orientation-display__empty">—</span>
                    )}
                  </button>
                  <button
                    type="button"
                    className="transform-button"
                    onClick={() => transformSelectedPiece('rotate')}
                    disabled={!isHumanTurn || !selectedOrientation}
                  >
                    <span aria-hidden="true">↻</span>
                    Rotate
                  </button>
                  <button
                    type="button"
                    className="transform-button"
                    onClick={() => transformSelectedPiece('flip')}
                    disabled={!isHumanTurn || !selectedOrientation}
                  >
                    <span aria-hidden="true">⇄</span>
                    Flip
                  </button>
                </div>
              </div>
              <section
                className={`piece-dock${isPieceTrayOpen ? ' piece-dock--open' : ''}${selectedOrientation ? ' piece-dock--placing' : ''}${isHumanTurn ? '' : ' gameplay-controls--hidden'}`}
                aria-label="Piece selection"
                aria-hidden={!isHumanTurn}
              >
                <header className="piece-dock__header">
                  <span>Your pieces</span>
                  <strong>
                    {remainingPieces.length} {remainingPieces.length === 1 ? 'piece' : 'pieces'} remaining
                  </strong>
                </header>
                <div className="piece-dock__body">
                  <div className="piece-tray" role="group" aria-label="Remaining pieces">
                    {game.pieces.map((piece) => {
                      const previewOrientation = piece.orientations[0]
                      const selected = piece.id === selectedPieceId
                      return (
                        <button
                          type="button"
                          className={`piece-option${selected ? ' piece-option--selected' : ''}`}
                          key={piece.id}
                          onClick={() => selectPiece(piece)}
                          disabled={!isHumanTurn || !piece.available}
                          aria-pressed={selected}
                          aria-label={`Piece ${piece.id + 1}, ${piece.squares} squares${piece.available ? '' : ', played'}`}
                        >
                          {previewOrientation && (
                            <PiecePreview
                              orientation={previewOrientation}
                              color={PLAYER_COLORS[game.currentPlayer]}
                            />
                          )}
                          <span>{piece.squares}</span>
                        </button>
                      )
                    })}
                  </div>
                </div>
              </section>
            </>
          )}
        </section>
        </main>
        {debugPanel}
      </>
    )
  }

  return (
    <>
      <main className="setup-shell">
      <section className="setup-card">
        <h1>Play AlphaBlokus</h1>
        <div className="setup-intro">
          <p>
            <a
              href="https://github.com/ShivamSarodia/AlphaBlokus"
              target="_blank"
              rel="noreferrer"
            >
              AlphaBlokus
            </a>
            {' '}is an agent for the board game Blokus, implemented in Rust and trained
            purely on self-play.
          </p>
          <p>
            Inference runs locally on your machine, so expect runtime to vary significantly based
            on your device.
          </p>
        </div>
        {savedGameToRestore ? (
          <section className="restore-loading" aria-live="polite" aria-labelledby="restore-title">
            <div className="restore-loading__header">
              <h2 id="restore-title">Loading previous game…</h2>
              <button type="button" onClick={newGame}>
                Start new game instead
              </button>
            </div>
            <div className="loading" aria-label={loading?.label ?? 'Preparing saved game'}>
              <div className="loading-copy">
                <strong>{loading?.label ?? 'Preparing saved game'}</strong>
                {loading && formatProgress(loading) && <span>{formatProgress(loading)}</span>}
              </div>
              <div className="loading-track" aria-hidden="true">
                <span
                  className={loading?.total ? 'loading-fill' : 'loading-fill indeterminate'}
                  style={loading?.total
                    ? { width: `${Math.min(100, loading.loaded! / loading.total * 100)}%` }
                    : undefined}
                />
              </div>
            </div>
          </section>
        ) : (
          <>
            <h2>Choose your table</h2>
            <div className="seat-grid">
              {seats.map((seat, index) => (
                <label className="seat" key={index}>
                  <span className="swatch" style={{ background: PLAYER_COLORS[index] }} />
                  <span>{PLAYER_NAMES[index]}</span>
                  <select value={seat} onChange={(event) => setSeats((all) => all.map((value, i) => i === index ? event.target.value as Seat : value))}>
                    <option value="human">Human</option>
                    {(Object.keys(STRENGTHS) as Strength[]).map((name) => (
                      <option key={name} value={name}>AlphaBlokus ({STRENGTHS[name].label})</option>
                    ))}
                  </select>
                </label>
              ))}
            </div>
            {error && <p className="error">{error}</p>}
            {loading && <section className="loading" aria-live="polite" aria-label={loading.label}>
              <div className="loading-copy"><strong>{loading.label}</strong>{formatProgress(loading) && <span>{formatProgress(loading)}</span>}</div>
              <div className="loading-track" aria-hidden="true"><span className={loading.total ? 'loading-fill' : 'loading-fill indeterminate'} style={loading.total ? { width: `${Math.min(100, loading.loaded! / loading.total * 100)}%` } : undefined} /></div>
            </section>}
            <button className="start primary" disabled={!ready || !!loading} onClick={start}>{ready ? `Start ${botCount ? 'match' : 'local game'}` : 'Preparing…'}</button>
          </>
        )}
      </section>
      </main>
      {debugPanel}
    </>
  )
}
