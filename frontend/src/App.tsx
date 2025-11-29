// Vibe-coded frontend for the AlphaBlokus game server.

import { useCallback, useEffect, useMemo, useState } from 'react'
import './App.css'

const PREVIEW_CELL_SIZE = 16
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')
const GAME_ENDPOINT = API_BASE_URL ? `${API_BASE_URL}/game` : '/api/game'
const MOVE_ENDPOINT = API_BASE_URL ? `${API_BASE_URL}/human_move` : '/api/human_move'
const MOVE_CELLS_ENDPOINT = API_BASE_URL ? `${API_BASE_URL}/move` : '/api/move'
const MOVE_INDEX_ENDPOINT = API_BASE_URL ? `${API_BASE_URL}/move_index` : '/api/move_index'
const AGENT_MOVE_ENDPOINT = API_BASE_URL ? `${API_BASE_URL}/agent_move` : '/api/agent_move'
const RESET_ENDPOINT = API_BASE_URL ? `${API_BASE_URL}/reset` : '/api/reset'
const BACK_ENDPOINT = API_BASE_URL ? `${API_BASE_URL}/back` : '/api/back'
const SAVE_ENDPOINT = API_BASE_URL ? `${API_BASE_URL}/save_game_state` : '/api/save_game_state'
const PLAYER_COLORS = ['#2563eb', '#fbbf24', '#ef4444', '#22c55e']
const NUM_PLAYERS = 4

type BackendOrientation = {
  id: number
  width: number
  height: number
  cells: [number, number][]
  valid: boolean
}

type BackendPiece = {
  id: number
  squares: number
  orientations: BackendOrientation[]
}

type BackendBoardSlice = {
  size: number
  cells: boolean[][]
}

type BoardPoint = { row: number; col: number }
type SelectedOrientation = { pieceId: number; orientation: BackendOrientation }
type Placement = { pieceId: number; orientationId: number; cells: BoardPoint[] }
type GameStateResponse = {
  board_size: number
  pieces: BackendPiece[]
  board: BackendBoardSlice[]
  current_player: number
  agents?: string[]
  pending_agent?: string | null
  game_over?: boolean
  scores?: number[]
  tile_counts?: number[]
}

const createEmptyBoard = (size: number) =>
  Array.from({ length: size }, () => Array<string>(size).fill(''))

const cellKey = (row: number, col: number) => `${row}-${col}`

const buildBoardFromSlices = (slices: BackendBoardSlice[] | undefined, boardSize: number) => {
  const board = createEmptyBoard(boardSize)
  if (!slices) {
    return board
  }

  slices.forEach((slice, playerIndex) => {
    slice.cells.forEach((column, x) => {
      column.forEach((filled, y) => {
        if (filled) {
          board[y][x] = String(playerIndex)
        }
      })
    })
  })

  return board
}

type NormalizedGameState = {
  boardSize: number
  pieces: BackendPiece[]
  board: string[][]
  currentPlayer: number
  agents: string[]
  pendingAgent: string | null
  gameOver: boolean
  scores: number[] | null
  tileCounts: number[]
}

const normalizeGameState = (payload: GameStateResponse): NormalizedGameState => {
  const normalizedPieces = (payload.pieces ?? [])
    .map((piece) => ({
      ...piece,
      orientations: piece.orientations
        .map((orientation) => ({
          ...orientation,
          cells: orientation.cells.map(([x, y]) => [y, x] as [number, number]),
        }))
        .sort((a, b) => a.id - b.id),
    }))
    .sort((a, b) => a.id - b.id)

  const normalizedBoard = buildBoardFromSlices(payload.board, payload.board_size)
  const currentPlayer =
    typeof payload.current_player === 'number' ? payload.current_player : 0

  return {
    boardSize: payload.board_size,
    pieces: normalizedPieces,
    board: normalizedBoard,
    currentPlayer,
    agents: Array.isArray(payload.agents) ? payload.agents : [],
    pendingAgent:
      typeof payload.pending_agent === 'string' && payload.pending_agent.length > 0
        ? payload.pending_agent
        : null,
    gameOver: Boolean(payload.game_over),
    scores:
      Array.isArray(payload.scores) && payload.scores.length > 0
        ? payload.scores
        : null,
    tileCounts: Array.isArray(payload.tile_counts) ? payload.tile_counts : [],
  }
}

function App() {
  const [boardSize, setBoardSize] = useState<number | null>(null)
  const [pieces, setPieces] = useState<BackendPiece[]>([])
  const [board, setBoard] = useState<string[][] | null>(null)
  const [currentPlayer, setCurrentPlayer] = useState(0)
  const [expandedPieceId, setExpandedPieceId] = useState<number | null>(null)
  const [selectedOrientation, setSelectedOrientation] = useState<SelectedOrientation | null>(null)
  const [hoverCell, setHoverCell] = useState<BoardPoint | null>(null)
  const [pendingPlacement, setPendingPlacement] = useState<Placement | null>(null)
  const [isSubmittingMove, setIsSubmittingMove] = useState(false)
  const [hasLoaded, setHasLoaded] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [agents, setAgents] = useState<string[]>([])
  const [pendingAgent, setPendingAgent] = useState<string | null>(null)
  const [gameOver, setGameOver] = useState(false)
  const [scores, setScores] = useState<number[] | null>(null)
  const [tileCounts, setTileCounts] = useState<number[]>([])
  const [moveIndexInput, setMoveIndexInput] = useState('')
  const [isFetchingMoveByIndex, setIsFetchingMoveByIndex] = useState(false)
  const [pendingMoveIndex, setPendingMoveIndex] = useState<number | null>(null)
  const [isFetchingPendingMoveIndex, setIsFetchingPendingMoveIndex] = useState(false)
  const [playerAssignments, setPlayerAssignments] = useState<string[]>(() =>
    Array(NUM_PLAYERS).fill('human'),
  )
  const [autoPlayPaused, setAutoPlayPaused] = useState(false)
  const [isSavingGame, setIsSavingGame] = useState(false)

  const applyGameState = useCallback((normalized: NormalizedGameState) => {
    setPieces(normalized.pieces)
    setBoard(normalized.board)
    setBoardSize((current) => (normalized.boardSize !== current ? normalized.boardSize : current))
    setCurrentPlayer(normalized.currentPlayer)
    setAgents(normalized.agents)
    setPendingAgent(normalized.pendingAgent)
    setGameOver(normalized.gameOver)
    setScores(normalized.scores)
    setTileCounts(normalized.tileCounts)
    setHasLoaded(true)
  }, [])

  const fetchGameState = useCallback(async () => {
    const response = await fetch(GAME_ENDPOINT)
    if (!response.ok) {
      throw new Error('Game endpoint returned an error')
    }
    const payload = (await response.json()) as GameStateResponse
    return normalizeGameState(payload)
  }, [])

  useEffect(() => {
    let isMounted = true
    const loadGameState = async () => {
      try {
        const normalized = await fetchGameState()
        if (isMounted && Number.isFinite(normalized.boardSize) && normalized.boardSize > 0) {
          applyGameState(normalized)
        }
      } catch {
        // Ignore errors; fall back to default size.
      }
    }

    loadGameState()
    return () => {
      isMounted = false
    }
  }, [applyGameState, fetchGameState])

  useEffect(() => {
    if (!pendingAgent) {
      return
    }

    let isCancelled = false
    const interval = setInterval(async () => {
      try {
        const normalized = await fetchGameState()
        if (!isCancelled) {
          applyGameState(normalized)
          if (!normalized.pendingAgent) {
            clearInterval(interval)
          }
        }
      } catch {
        // Ignore polling errors; will retry on next tick.
      }
    }, 1500)

    return () => {
      isCancelled = true
      clearInterval(interval)
    }
  }, [pendingAgent, applyGameState, fetchGameState])

  useEffect(() => {
    setPlayerAssignments((prev) => {
      let changed = false
      const next = prev.map((assignment) => {
        if (assignment !== 'human' && !agents.includes(assignment)) {
          changed = true
          return 'human'
        }
        return assignment
      })
      return changed ? next : prev
    })
  }, [agents])

  useEffect(() => {
    if (boardSize === null) {
      setBoard(null)
    }
    setSelectedOrientation(null)
    setPendingPlacement(null)
    setPendingMoveIndex(null)
    setHoverCell(null)
    setErrorMessage(null)
  }, [boardSize])

  const preview = useMemo(() => {
    if (boardSize === null || !board) {
      return null
    }
    if (!selectedOrientation || !hoverCell || pendingPlacement) {
      return null
    }

    const cells = selectedOrientation.orientation.cells.map(([dr, dc]) => ({
      row: hoverCell.row + dr,
      col: hoverCell.col + dc,
    }))

    const isValid = cells.every(
      ({ row, col }) =>
        row >= 0 && row < boardSize && col >= 0 && col < boardSize && !board[row][col],
    )

    return {
      cells,
      isValid,
      color: PLAYER_COLORS[currentPlayer % PLAYER_COLORS.length] ?? PLAYER_COLORS[0],
      cellSet: new Set(cells.map(({ row, col }) => cellKey(row, col))),
    }
  }, [selectedOrientation, hoverCell, board, pendingPlacement, boardSize, currentPlayer])

  const pendingMap = useMemo(() => {
    if (!pendingPlacement) {
      return null
    }

    return new Set(pendingPlacement.cells.map(({ row, col }) => cellKey(row, col)))
  }, [pendingPlacement])

  useEffect(() => {
    setIsFetchingPendingMoveIndex(false)
    if (!pendingPlacement) {
      setPendingMoveIndex(null)
      setIsFetchingPendingMoveIndex(false)
      return
    }
    if (pendingMoveIndex !== null) {
      return
    }

    let isCancelled = false
    const fetchMoveIndex = async () => {
      setIsFetchingPendingMoveIndex(true)
      try {
        const response = await fetch(MOVE_INDEX_ENDPOINT, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            cells: pendingPlacement.cells.map(({ row, col }) => [col, row] as [number, number]),
          }),
        })

        if (!response.ok) {
          const body = await response.json().catch(async () => ({ error: await response.text() }))
          if (!isCancelled) {
            setErrorMessage(body?.error ?? 'Unable to find move index for this placement.')
          }
          return
        }

        const payload = (await response.json()) as { move_index?: number }
        if (!isCancelled) {
          if (typeof payload.move_index === 'number') {
            setPendingMoveIndex(payload.move_index)
          } else {
            setErrorMessage('Unable to find move index for this placement.')
          }
        }
      } catch (error) {
        console.error('Failed to fetch move index', error)
        if (!isCancelled) {
          setErrorMessage('Something went wrong finding the move index.')
        }
      } finally {
        if (!isCancelled) {
          setIsFetchingPendingMoveIndex(false)
        }
      }
    }

    fetchMoveIndex()
    return () => {
      isCancelled = true
    }
  }, [pendingPlacement, pendingMoveIndex])

  const isAgentThinking = Boolean(pendingAgent)
  const currentAssignment = playerAssignments[currentPlayer] ?? 'human'
  const autoAgentForCurrentPlayer =
    currentAssignment !== 'human' ? currentAssignment : null
  const isHumanTurn = autoAgentForCurrentPlayer === null
  const interactionLocked = isAgentThinking || gameOver
  const winningScore = useMemo(() => {
    if (!scores || scores.length === 0) {
      return null
    }
    return Math.max(...scores)
  }, [scores])

  const handleOrientationSelect = (pieceId: number, orientation: BackendOrientation) => {
    if (pendingPlacement || !orientation.valid || interactionLocked || !isHumanTurn) {
      return
    }
    setErrorMessage(null)
    setSelectedOrientation({ pieceId, orientation })
  }

  const handleCellClick = (row: number, col: number) => {
    if (
      !selectedOrientation ||
      pendingPlacement ||
      boardSize === null ||
      !board ||
      isSubmittingMove ||
      interactionLocked ||
      !isHumanTurn
    ) {
      return
    }

    const cells = selectedOrientation.orientation.cells.map(([dr, dc]) => ({
      row: row + dr,
      col: col + dc,
    }))

    const isValid =
      cells.length > 0 &&
      cells.every(
        ({ row: targetRow, col: targetCol }) =>
          targetRow >= 0 &&
          targetRow < boardSize &&
          targetCol >= 0 &&
          targetCol < boardSize &&
          !board[targetRow][targetCol],
      )

    if (!isValid) {
      return
    }

    setPendingPlacement({
      pieceId: selectedOrientation.pieceId,
      orientationId: selectedOrientation.orientation.id,
      cells,
    })
    setPendingMoveIndex(null)
    setErrorMessage(null)
  }

  const handlePreviewMoveIndex = async (event?: React.FormEvent<HTMLFormElement>) => {
    event?.preventDefault()
    if (interactionLocked || !isHumanTurn || isSubmittingMove) {
      return
    }

    const parsedIndex = Number.parseInt(moveIndexInput.trim(), 10)
    if (!Number.isFinite(parsedIndex) || parsedIndex < 0) {
      setErrorMessage('Enter a valid move index to preview.')
      return
    }

    setIsFetchingMoveByIndex(true)
    setErrorMessage(null)
    try {
      const response = await fetch(`${MOVE_CELLS_ENDPOINT}/${parsedIndex}/cells`)
      if (!response.ok) {
        const body = await response.json().catch(async () => ({ error: await response.text() }))
        setErrorMessage(body?.error ?? 'Unable to load move cells for that index.')
        return
      }

      const payload = (await response.json()) as { cells?: [number, number][] }
      const cells = (payload.cells ?? []).map(([col, row]) => ({ row, col }))
      if (cells.length === 0) {
        setErrorMessage('No cells returned for that move index.')
        return
      }

      setPendingPlacement({
        pieceId: -1,
        orientationId: parsedIndex,
        cells,
      })
      setPendingMoveIndex(parsedIndex)
      setSelectedOrientation(null)
      setExpandedPieceId(null)
      setHoverCell(null)
    } catch (error) {
      console.error('Failed to preview move index', error)
      setErrorMessage('Something went wrong loading that move.')
    } finally {
      setIsFetchingMoveByIndex(false)
    }
  }

  const handleConfirmPlacement = async () => {
    if (!pendingPlacement || isSubmittingMove || interactionLocked || !isHumanTurn) {
      return
    }

    setIsSubmittingMove(true)
    try {
      const response = await fetch(MOVE_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          cells: pendingPlacement.cells.map(({ row, col }) => [col, row] as [number, number]),
        }),
      })

      if (!response.ok) {
        const body = await response.json().catch(async () => ({ error: await response.text() }))
        setErrorMessage(body?.error ?? 'Move rejected by server.')
        setPendingPlacement(null)
        setSelectedOrientation(null)
        setPendingMoveIndex(null)
        return
      }

      const payload = (await response.json()) as GameStateResponse
      const normalized = normalizeGameState(payload)

      applyGameState(normalized)
      setPendingPlacement(null)
      setSelectedOrientation(null)
      setExpandedPieceId(null)
      setPendingMoveIndex(null)
      setErrorMessage(null)
    } catch (error) {
      console.error('Failed to submit move', error)
      setErrorMessage('Something went wrong placing the move.')
    } finally {
      setIsSubmittingMove(false)
    }
  }

  const requestAgentMove = useCallback(
    async (agentName: string | null) => {
      if (!agentName || isSubmittingMove || interactionLocked) {
        return
      }

      setIsSubmittingMove(true)
      setErrorMessage(null)
      try {
        const response = await fetch(AGENT_MOVE_ENDPOINT, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ agent: agentName }),
        })

        if (!response.ok) {
          const body = await response.json().catch(async () => ({ error: await response.text() }))
          setErrorMessage(body?.error ?? 'Agent request rejected by server.')
          return
        }

        setPendingAgent(agentName)
        const normalized = await fetchGameState().catch(() => null)
        if (normalized) {
          applyGameState(normalized)
        }
      } catch (error) {
        console.error('Failed to request agent move', error)
        setErrorMessage('Something went wrong requesting the agent move.')
      } finally {
        setIsSubmittingMove(false)
      }
    },
    [applyGameState, fetchGameState, interactionLocked, isSubmittingMove],
  )

  useEffect(() => {
    if (
      autoPlayPaused ||
      !autoAgentForCurrentPlayer ||
      pendingAgent ||
      interactionLocked ||
      isSubmittingMove
    ) {
      return
    }
    requestAgentMove(autoAgentForCurrentPlayer)
  }, [
    autoPlayPaused,
    autoAgentForCurrentPlayer,
    pendingAgent,
    interactionLocked,
    isSubmittingMove,
    currentPlayer,
    requestAgentMove,
  ])

  const handleResetGame = async () => {
    if (isSubmittingMove) {
      return
    }
    setIsSubmittingMove(true)
    setErrorMessage(null)
    try {
      const response = await fetch(RESET_ENDPOINT, { method: 'POST' })
      if (!response.ok) {
        const body = await response.json().catch(async () => ({ error: await response.text() }))
        setErrorMessage(body?.error ?? 'Failed to reset game.')
        return
      }
      const payload = (await response.json()) as GameStateResponse
      const normalized = normalizeGameState(payload)
      applyGameState(normalized)
      setPendingPlacement(null)
      setSelectedOrientation(null)
      setExpandedPieceId(null)
      setPendingMoveIndex(null)
      setPlayerAssignments(Array(NUM_PLAYERS).fill('human'))
    } catch (error) {
      console.error('Failed to reset game', error)
      setErrorMessage('Something went wrong resetting the game.')
    } finally {
      setIsSubmittingMove(false)
    }
  }

  const handleStepBack = async () => {
    if (isSubmittingMove) {
      return
    }
    setIsSubmittingMove(true)
    setErrorMessage(null)
    try {
      const response = await fetch(BACK_ENDPOINT, { method: 'POST' })
      if (!response.ok) {
        const body = await response.json().catch(async () => ({ error: await response.text() }))
        setErrorMessage(body?.error ?? 'Failed to go back.')
        return
      }
      const payload = (await response.json()) as GameStateResponse
      const normalized = normalizeGameState(payload)
      applyGameState(normalized)
      setPendingPlacement(null)
      setSelectedOrientation(null)
      setExpandedPieceId(null)
      setPendingMoveIndex(null)
      setAutoPlayPaused(true)
    } catch (error) {
      console.error('Failed to step back', error)
      setErrorMessage('Something went wrong going back a turn.')
    } finally {
      setIsSubmittingMove(false)
    }
  }

  const handleSaveGameState = async () => {
    if (isSavingGame) {
      return
    }
    setIsSavingGame(true)
    setErrorMessage(null)
    try {
      const response = await fetch(SAVE_ENDPOINT, { method: 'POST' })
      if (!response.ok) {
        const body = await response.json().catch(async () => ({ error: await response.text() }))
        setErrorMessage(body?.error ?? 'Failed to save game.')
      }
    } catch (error) {
      console.error('Failed to save game', error)
      setErrorMessage('Something went wrong saving the game.')
    } finally {
      setIsSavingGame(false)
    }
  }

  const handleUndoPlacement = () => {
    setPendingPlacement(null)
    setSelectedOrientation(null)
    setPendingMoveIndex(null)
    setErrorMessage(null)
  }

  if (boardSize === null || !board || !hasLoaded) {
    return (
      <main className="app">
        <header className="app__header">
          <div>
            <h1>AlphaBlokus</h1>
          </div>
        </header>
      </main>
    )
  }

  return (
    <main className="app">
      <header className="app__header">
        <div>
          <h1>AlphaBlokus</h1>
        </div>
        <div className="header-actions">
          <button
            type="button"
            className="secondary"
            onClick={handleSaveGameState}
            disabled={isSavingGame}
          >
            Save game
          </button>
          <button
            type="button"
            className="secondary"
            onClick={handleStepBack}
            disabled={isSubmittingMove}
          >
            Step back
          </button>
          <button
            type="button"
            className="secondary reset-button"
            onClick={handleResetGame}
            disabled={isSubmittingMove}
          >
            Restart game
          </button>
        </div>
      </header>

      <section className="layout">
        <div className="board-panel" onMouseLeave={() => setHoverCell(null)}>
          <form className="move-index-form" onSubmit={handlePreviewMoveIndex}>
            <label htmlFor="move-index-input">Preview move by index</label>
            <div className="move-index-form__controls">
              <input
                id="move-index-input"
                type="number"
                inputMode="numeric"
                pattern="[0-9]*"
                min={0}
                placeholder="e.g. 1234"
                value={moveIndexInput}
                onChange={(event) => setMoveIndexInput(event.target.value)}
                disabled={interactionLocked || isSubmittingMove || isFetchingMoveByIndex || !isHumanTurn}
              />
              <button
                type="submit"
                className="secondary"
                disabled={
                  interactionLocked || isSubmittingMove || isFetchingMoveByIndex || !isHumanTurn
                }
              >
                {isFetchingMoveByIndex ? 'Loading…' : 'Show move'}
              </button>
            </div>
          </form>
          <div
            className="board-grid"
            style={{ gridTemplateColumns: `repeat(${boardSize}, 28px)` }}
          >
            {board.map((row, rowIndex) =>
              row.map((value, colIndex) => {
                const key = cellKey(rowIndex, colIndex)
                const isFilled = Boolean(value)
                const isPending = pendingMap?.has(key)
                const isPreview = Boolean(preview?.cellSet.has(key))
                const showPreviewInvalid = isPreview && preview && !preview.isValid

                let className = 'cell'
                if (isFilled) className += ` cell--player-${value}`
                if (isPending) className += ' cell--pending'
                if (isPreview) {
                  className += preview?.isValid ? ' cell--preview' : ' cell--preview-invalid'
                }
                if (showPreviewInvalid) className += ' cell--preview-invalid'

                let inlineStyle: React.CSSProperties | undefined
                if (isPreview && preview) {
                  inlineStyle = preview.isValid
                    ? { background: preview.color }
                    : {
                      background: preview.color,
                      backgroundImage:
                        'repeating-linear-gradient(45deg, rgba(255,255,255,0.6) 0 6px, transparent 6px 12px)',
                    }
                } else if (isPending && pendingPlacement) {
                  const pendingColor =
                    PLAYER_COLORS[currentPlayer % PLAYER_COLORS.length] ?? PLAYER_COLORS[0]
                  inlineStyle = {
                    background: pendingColor,
                    backgroundImage:
                      'repeating-linear-gradient(45deg, rgba(255,255,255,0.35) 0 6px, transparent 6px 12px)',
                  }
                }

                return (
                  <button
                    key={key}
                    type="button"
                    className={className}
                    onMouseEnter={() => setHoverCell({ row: rowIndex, col: colIndex })}
                    onFocus={() => setHoverCell({ row: rowIndex, col: colIndex })}
                    onClick={() => handleCellClick(rowIndex, colIndex)}
                    disabled={Boolean(pendingPlacement) || isSubmittingMove || interactionLocked}
                    style={inlineStyle}
                    aria-label={`Row ${rowIndex + 1}, Column ${colIndex + 1}`}
                  />
                )
              }),
            )}
          </div>

          {pendingPlacement && (
            <div className="placement-actions">
              {pendingMoveIndex !== null ? (
                <span className="move-index-chip">Move #{pendingMoveIndex}</span>
              ) : (
                isFetchingPendingMoveIndex && (
                  <span className="move-index-chip move-index-chip--muted">Finding move index…</span>
                )
              )}
              <button
                type="button"
                className="primary"
                onClick={handleConfirmPlacement}
                disabled={isSubmittingMove}
                style={{ background: PLAYER_COLORS[currentPlayer % PLAYER_COLORS.length] ?? PLAYER_COLORS[0] }}
              >
                {isSubmittingMove ? 'Placing…' : 'Confirm placement'}
              </button>
              <button type="button" className="secondary" onClick={handleUndoPlacement}>
                Undo
              </button>
            </div>
          )}
          {tileCounts.length > 0 && (
            <div className="tile-counts tile-counts--inline">
              {tileCounts.map((count, index) => {
                const color = PLAYER_COLORS[index % PLAYER_COLORS.length] ?? '#475467'
                return (
                  <div key={index} className="tile-counts__entry">
                    <span style={{ color }}>Player {index + 1}</span>
                    <strong>{count}</strong>
                  </div>
                )
              })}
            </div>
          )}
          {!isHumanTurn && autoAgentForCurrentPlayer && !pendingAgent && (
            <p className="agent-message">
              Player {currentPlayer + 1} is controlled by <strong>{autoAgentForCurrentPlayer}</strong>.
            </p>
          )}
          {pendingAgent && (
            <p className="agent-message">Agent <strong>{pendingAgent}</strong> is selecting a move…</p>
          )}
          {gameOver && scores && (
            <div className="scoreboard">
              <h3>Final scores</h3>
              <ul className="scoreboard__list">
                {scores.map((score, index) => {
                  const color = PLAYER_COLORS[index % PLAYER_COLORS.length] ?? '#475467'
                  const isLeader = winningScore !== null && score === winningScore
                  return (
                    <li key={index} className={isLeader ? 'scoreboard__entry scoreboard__entry--leader' : 'scoreboard__entry'}>
                      <span style={{ color }}>Player {index + 1}</span>
                      <strong>{score.toFixed(2)}</strong>
                    </li>
                  )
                })}
              </ul>
            </div>
          )}
          {errorMessage && <p className="error-message">{errorMessage}</p>}
        </div>

        <div className="sidebar">
          <aside className="pieces-panel">
            <div className="pieces-header">
              <h2>Pieces</h2>
              <span>
                {
                  pieces.filter((piece) => piece.orientations.some((orientation) => orientation.valid))
                    .length
                }
                /{pieces.length || 0} playable
              </span>
            </div>

            <div className="pieces-list">
              {pieces.map((piece) => {
                const isExpanded = expandedPieceId === piece.id
                const isSelected = selectedOrientation?.pieceId === piece.id
                const previewOrientation = piece.orientations[0]
                const hasPlayableOrientation = piece.orientations.some((orientation) => orientation.valid)
                const playerColor =
                  PLAYER_COLORS[currentPlayer % PLAYER_COLORS.length] ?? PLAYER_COLORS[0]

                return (
                  <div
                    key={piece.id}
                    className={`piece-card${isSelected ? ' piece-card--active' : ''}${!hasPlayableOrientation ? ' piece-card--faded' : ''
                      }`}
                  >
                    <button
                      type="button"
                      className="piece-card__header"
                      onClick={() => setExpandedPieceId(isExpanded ? null : piece.id)}
                      aria-label={`Piece ${piece.id}`}
                    >
                      {previewOrientation ? (
                        <OrientationPreview
                          orientation={previewOrientation}
                          className="piece-card__preview"
                          color={playerColor}
                        />
                      ) : (
                        <span className="piece-card__preview empty">No orientations</span>
                      )}
                      <div className="piece-card__meta">
                        <span className="piece-card__count">{piece.squares}</span>
                      </div>
                      <span className="piece-card__toggle" aria-hidden="true">
                        {isExpanded ? '−' : '+'}
                      </span>
                    </button>

                    {isExpanded && (
                      <div className="orientation-grid">
                        {piece.orientations.map((orientation) => {
                          const isOrientationSelected =
                            selectedOrientation?.orientation.id === orientation.id
                          const disabled =
                            !orientation.valid ||
                            Boolean(pendingPlacement) ||
                            isSubmittingMove ||
                            interactionLocked
                          return (
                            <button
                              type="button"
                              key={orientation.id}
                              className={`orientation${isOrientationSelected ? ' orientation--selected' : ''
                                }${!orientation.valid ? ' orientation--disabled' : ''}`}
                              onClick={() => handleOrientationSelect(piece.id, orientation)}
                              disabled={disabled}
                              aria-pressed={isOrientationSelected}
                            >
                              <OrientationPreview orientation={orientation} color={playerColor} />
                              <span>{orientation.valid ? 'Playable' : 'Blocked'}</span>
                            </button>
                          )
                        })}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </aside>

          <section className="agent-panel">
            <div className="agent-panel__header">
              <div>
                <h2>Players</h2>
                {pendingAgent && (
                  <span className="agent-panel__status">Running: {pendingAgent}</span>
                )}
                {autoPlayPaused && (
                  <span className="agent-panel__status agent-panel__status--paused">
                    Auto play paused
                  </span>
                )}
              </div>
              <button
                type="button"
                className={`secondary agent-panel__pause${autoPlayPaused ? ' agent-panel__pause--active' : ''
                  }`}
                onClick={() => setAutoPlayPaused((prev) => !prev)}
              >
                {autoPlayPaused ? 'Resume auto play' : 'Pause auto play'}
              </button>
            </div>
            <ul className="agent-panel__list">
              {playerAssignments.map((assignment, playerIndex) => {
                const isCurrentPlayer = playerIndex === currentPlayer
                return (
                  <li
                    key={playerIndex}
                    className={`agent-panel__row${isCurrentPlayer ? ' agent-panel__row--current' : ''
                      }`}
                  >
                    <div className="agent-panel__player">
                      <span>Player {playerIndex + 1}</span>
                      <small>
                        {playerAssignments[playerIndex] === 'human'
                          ? 'Human'
                          : playerAssignments[playerIndex]}
                      </small>
                    </div>
                    <select
                      value={assignment}
                      onChange={(event) => {
                        const value = event.target.value
                        setPlayerAssignments((prev) =>
                          prev.map((entry, idx) => (idx === playerIndex ? value : entry)),
                        )
                      }}
                    >
                      <option value="human">Human</option>
                      {agents.map((agent) => (
                        <option key={agent} value={agent}>
                          {agent}
                        </option>
                      ))}
                    </select>
                  </li>
                )
              })}
            </ul>
          </section>
        </div>
      </section>
    </main>
  )
}

function OrientationPreview({
  orientation,
  className,
  color,
}: {
  orientation: BackendOrientation
  className?: string
  color?: string
}) {
  const columns = Math.max(orientation.width, 1)
  const rows = Math.max(orientation.height, 1)
  const totalCells = columns * rows
  const filled = new Set(orientation.cells.map(([row, col]) => cellKey(row, col)))
  const fillColor = color ?? '#4f46e5'

  const style = {
    gridTemplateColumns: `repeat(${columns}, ${PREVIEW_CELL_SIZE}px)`,
    width: columns * PREVIEW_CELL_SIZE,
  }

  return (
    <div
      className={`mini-grid${className ? ` ${className}` : ''}`}
      style={style}
    >
      {Array.from({ length: totalCells }).map((_, index) => {
        const row = Math.floor(index / columns)
        const col = index % columns
        const key = cellKey(row, col)
        const isFilled = filled.has(key)
        return (
          <span
            key={key}
            className={`mini-cell${isFilled ? ' mini-cell--filled' : ''}`}
            style={{
              width: PREVIEW_CELL_SIZE,
              height: PREVIEW_CELL_SIZE,
              background: isFilled ? fillColor : undefined,
            }}
          />
        )
      })}
    </div>
  )
}

export default App
