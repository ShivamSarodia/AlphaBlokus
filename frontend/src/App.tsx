import { useEffect, useMemo, useState } from 'react'
import './App.css'

const PREVIEW_CELL_SIZE = 16
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')
const GAME_ENDPOINT = API_BASE_URL ? `${API_BASE_URL}/game` : '/api/game'
const MOVE_ENDPOINT = API_BASE_URL ? `${API_BASE_URL}/move` : '/api/move'
const PLAYER_COLORS = ['#2563eb', '#fbbf24', '#ef4444', '#22c55e']

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

const normalizeGameState = (payload: GameStateResponse) => {
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

  useEffect(() => {
    let isMounted = true
    const loadGameState = async () => {
      try {
        const response = await fetch(GAME_ENDPOINT)
        if (!response.ok) {
          return
        }
        const payload = (await response.json()) as GameStateResponse
        if (isMounted && Number.isFinite(payload.board_size) && payload.board_size > 0) {
          const normalized = normalizeGameState(payload)
          setBoardSize(normalized.boardSize)
          setPieces(normalized.pieces)
          setBoard(normalized.board)
          setCurrentPlayer(normalized.currentPlayer)
          setHasLoaded(true)
        }
      } catch {
        // Ignore errors; fall back to default size.
      }
    }

    loadGameState()
    return () => {
      isMounted = false
    }
  }, [])

  useEffect(() => {
    if (boardSize === null) {
      setBoard(null)
    }
    setSelectedOrientation(null)
    setPendingPlacement(null)
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

  const handleOrientationSelect = (pieceId: number, orientation: BackendOrientation) => {
    if (pendingPlacement || !orientation.valid) {
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
      isSubmittingMove
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
    setErrorMessage(null)
  }

  const handleConfirmPlacement = async () => {
    if (!pendingPlacement || isSubmittingMove) {
      return
    }

    setIsSubmittingMove(true)
    try {
      const response = await fetch(MOVE_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cells: pendingPlacement.cells }),
      })

      if (!response.ok) {
        const body = await response.json().catch(async () => ({ error: await response.text() }))
        setErrorMessage(body?.error ?? 'Move rejected by server.')
        setPendingPlacement(null)
        setSelectedOrientation(null)
        return
      }

      const payload = (await response.json()) as GameStateResponse
      const normalized = normalizeGameState(payload)

      setPieces(normalized.pieces)
      setBoard(normalized.board)
      setBoardSize((current) => (normalized.boardSize !== current ? normalized.boardSize : current))
      setCurrentPlayer(normalized.currentPlayer)
      setHasLoaded(true)
      setPendingPlacement(null)
      setSelectedOrientation(null)
      setErrorMessage(null)
    } catch (error) {
      console.error('Failed to submit move', error)
      setErrorMessage('Something went wrong placing the move.')
    } finally {
      setIsSubmittingMove(false)
    }
  }

  const handleUndoPlacement = () => {
    setPendingPlacement(null)
    setSelectedOrientation(null)
    setErrorMessage(null)
  }

  if (boardSize === null || !board || !hasLoaded) {
    return (
      <main className="app">
        <header className="app__header">
          <div>
            <h1>AlphaBlokus</h1>
          </div>
          <div className="board-meta">
            <span>Board size</span>
            <strong>Loading…</strong>
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
        <div className="board-meta">
          <span>Board size</span>
          <strong>{boardSize} × {boardSize}</strong>
        </div>
      </header>

      <section className="layout">
        <div className="board-panel" onMouseLeave={() => setHoverCell(null)}>
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
                    disabled={Boolean(pendingPlacement) || isSubmittingMove}
                    style={inlineStyle}
                    aria-label={`Row ${rowIndex + 1}, Column ${colIndex + 1}`}
                  />
                )
              }),
            )}
          </div>

          {pendingPlacement && (
            <div className="placement-actions">
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
          {errorMessage && <p className="error-message">{errorMessage}</p>}
        </div>

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
              const playerColor =
                PLAYER_COLORS[currentPlayer % PLAYER_COLORS.length] ?? PLAYER_COLORS[0]

              return (
                <div
                  key={piece.id}
                  className={`piece-card${isSelected ? ' piece-card--active' : ''}`}
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
                      <span className="piece-card__label" style={{ color: playerColor }}>
                        Piece {piece.id}
                      </span>
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
                        const disabled = !orientation.valid || Boolean(pendingPlacement) || isSubmittingMove
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
