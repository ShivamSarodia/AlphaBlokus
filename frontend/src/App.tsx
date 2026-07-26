import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import {
  PLAYER_COLORS,
  STRENGTHS,
  type LoadingProgress,
  type Seat,
  type Snapshot,
  type Strength,
  type WorkerCommand,
  type WorkerEvent,
} from './protocol'

const send = (worker: Worker | null, command: WorkerCommand) => worker?.postMessage(command)

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

export default function App() {
  const worker = useRef<Worker | null>(null)
  const [ready, setReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [seats, setSeats] = useState<Seat[]>(['human', 'bot', 'bot', 'bot'])
  const [strength, setStrength] = useState<Strength>('strong')
  const [game, setGame] = useState<Snapshot | null>(null)
  const [loading, setLoading] = useState<LoadingProgress | null>(null)
  const [moveIndex, setMoveIndex] = useState('')

  useEffect(() => {
    const instance = new Worker(new URL('./game.worker.ts', import.meta.url), { type: 'module' })
    worker.current = instance
    instance.onmessage = ({ data }: MessageEvent<WorkerEvent>) => {
      if (data.type === 'ready') setReady(true)
      if (data.type === 'loading') setLoading(data.progress)
      if (data.type === 'snapshot') {
        setLoading(null)
        setGame(data.snapshot)
      }
      if (data.type === 'save') {
        const blob = new Blob([data.json], { type: 'application/json' })
        const link = document.createElement('a')
        link.href = URL.createObjectURL(blob)
        link.download = 'alphablokus-game.json'
        link.click()
        URL.revokeObjectURL(link.href)
      }
      if (data.type === 'error') {
        setLoading(null)
        setError(data.message)
      }
    }
    send(instance, { type: 'init' })
    return () => instance.terminate()
  }, [])

  const botCount = useMemo(() => seats.filter((seat) => seat === 'bot').length, [seats])
  const board = game?.board ?? Array.from({ length: 20 }, () => Array<number>(20).fill(-1))

  const start = () => {
    setError(null)
    setLoading({ label: 'Starting local game' })
    send(worker.current, { type: 'start-game', seats, strength })
  }

  if (game) {
    return (
      <main className="game-shell">
        <header className="game-header">
          <div><span className="eyebrow">LOCAL WEBGPU MATCH</span><h1>AlphaBlokus</h1></div>
          <div className="header-actions">
            <button onClick={() => send(worker.current, { type: 'save' })}>Save</button>
            <button onClick={() => send(worker.current, { type: 'undo' })}>Undo</button>
            <button onClick={() => send(worker.current, { type: 'restart' })}>Restart</button>
            <button className="primary" onClick={() => setGame(null)}>New game</button>
          </div>
        </header>
        <section className="table-layout">
          <div className="board-wrap" aria-label="Blokus board">
            <div className="board" style={{ gridTemplateColumns: `repeat(${game.boardSize}, 1fr)` }}>
              {board.flatMap((row, y) => row.map((owner, x) => (
                <span key={`${x}-${y}`} className="tile" style={owner >= 0 ? { background: PLAYER_COLORS[owner] } : undefined} />
              )))}
            </div>
            <p className="status">{game.message}</p>
            {game.seats[game.currentPlayer] === 'human' && !game.gameOver && <form className="move-form" onSubmit={(event) => { event.preventDefault(); const parsed = Number.parseInt(moveIndex, 10); if (Number.isInteger(parsed) && parsed >= 0) send(worker.current, { type: 'play-move', moveIndex: parsed }) }}><label>Move index <input value={moveIndex} onChange={(event) => setMoveIndex(event.target.value)} inputMode="numeric" /></label><button className="primary">Play move</button></form>}
          </div>
          <aside className="side-panel">
            <h2>Players</h2>
            {game.seats.map((seat, player) => (
              <div className="player-row" key={player}>
                <span className="swatch" style={{ background: PLAYER_COLORS[player] }} />
                <span>Player {player + 1}</span><strong>{seat === 'bot' ? 'AlphaBlokus' : 'Human'}</strong>
              </div>
            ))}
            <div className="strength-readout"><span>Strength</span><strong>{STRENGTHS[game.strength].label} · {STRENGTHS[game.strength].rollouts.toLocaleString()} rollouts</strong></div>
          </aside>
        </section>
      </main>
    )
  }

  return (
    <main className="setup-shell">
      <section className="setup-card">
        <span className="eyebrow">ON-DEVICE WEBGPU</span>
        <h1>Play AlphaBlokus</h1>
        <p>Your match, search tree, and model inference stay in this browser.</p>
        <h2>Choose your table</h2>
        <div className="seat-grid">
          {seats.map((seat, index) => (
            <label className="seat" key={index}>
              <span className="swatch" style={{ background: PLAYER_COLORS[index] }} />
              <span>Player {index + 1}</span>
              <select value={seat} onChange={(event) => setSeats((all) => all.map((value, i) => i === index ? event.target.value as Seat : value))}>
                <option value="human">Human</option><option value="bot">AlphaBlokus</option>
              </select>
            </label>
          ))}
        </div>
        <fieldset><legend>Bot strength</legend><div className="strengths">
          {(Object.keys(STRENGTHS) as Strength[]).map((name) => <button key={name} className={strength === name ? 'selected' : ''} onClick={() => setStrength(name)}><strong>{STRENGTHS[name].label}</strong><span>{STRENGTHS[name].rollouts.toLocaleString()} rollouts</span></button>)}
        </div></fieldset>
        {error && <p className="error">{error}</p>}
        {loading && <section className="loading" aria-live="polite" aria-label={loading.label}>
          <div className="loading-copy"><strong>{loading.label}</strong>{formatProgress(loading) && <span>{formatProgress(loading)}</span>}</div>
          <div className="loading-track" aria-hidden="true"><span className={loading.total ? 'loading-fill' : 'loading-fill indeterminate'} style={loading.total ? { width: `${Math.min(100, loading.loaded! / loading.total * 100)}%` } : undefined} /></div>
        </section>}
        <button className="start primary" disabled={!ready || !!loading} onClick={start}>{ready ? `Start ${botCount ? 'match' : 'local game'}` : 'Checking WebGPU…'}</button>
      </section>
    </main>
  )
}
