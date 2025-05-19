import { useState, useEffect } from 'react';

const BOARD_SIZE = 20;
const CELL_SIZE = 24;
const PLAYER_COLORS = {
  0: 'bg-blue-500',   // First layer
  1: 'bg-yellow-500', // Second layer
  2: 'bg-red-500',    // Third layer
  3: 'bg-green-500'   // Fourth layer
};

const PiecePreview = ({ preview, playerColor }) => {
  return (
    <div className="grid gap-0.5" style={{
      gridTemplateColumns: `repeat(${preview[0].length}, ${CELL_SIZE}px)`
    }}>
      {preview.map((row, i) => 
        row.map((cell, j) => (
          <div
            key={`${i}-${j}`}
            className={`w-6 h-6 ${cell ? playerColor : 'bg-gray-100'}`}
          />
        ))
      )}
    </div>
  );
};

const AvailablePieces = ({ pieces, selectedPiece, setSelectedPiece, currentPlayer, renderPiecePreview }) => {
  return (
    <div className="mb-6">
      <h2 className="text-xl font-bold mb-2">Available Pieces</h2>
      <div className="flex flex-wrap gap-4">
        {pieces.map((coords, index) => {
          const preview = renderPiecePreview(coords);
          return (
            <div 
              key={index}
              className={`cursor-pointer p-2 border-2 ${
                selectedPiece === index ? 'border-blue-500' : 'border-gray-300'
              }`}
              onClick={() => setSelectedPiece(index)}
            >
              <PiecePreview preview={preview} playerColor={PLAYER_COLORS[currentPlayer]} />
            </div>
          );
        })}
      </div>
    </div>
  );
};

const PieceControls = ({ selectedPiece, setRotation, setIsFlipped, isFlipped, pieces, getTransformedPiece, renderPiecePreview, currentPlayer }) => {
  if (selectedPiece === null) return null;
  
  return (
    <div className="mb-4">
      <h3 className="text-lg font-semibold mb-2">Selected Piece Controls</h3>
      <div className="flex gap-2">
        <button
          className="px-4 py-2 bg-blue-500 text-white rounded"
          onClick={() => setRotation((r) => (r + 1) % 4)}
        >
          Rotate
        </button>
        <button
          className="px-4 py-2 bg-blue-500 text-white rounded"
          onClick={() => setIsFlipped(!isFlipped)}
        >
          Flip
        </button>
      </div>
      
      <div className="mt-2">
        <h4 className="font-medium mb-1">Preview:</h4>
        <div className="inline-block border-2 border-gray-300 p-2">
          <PiecePreview 
            preview={renderPiecePreview(getTransformedPiece(pieces[selectedPiece]))} 
            playerColor={PLAYER_COLORS[currentPlayer]}
          />
        </div>
      </div>
    </div>
  );
};

const GameBoard = ({ getCellState, calculatePlacementCoords, playMove, setHoverPosition, lastMove }) => {
  return (
    <div className="inline-block border-2 border-gray-300">
      {Array(BOARD_SIZE).fill().map((_, i) => (
        <div key={i} className="flex">
          {Array(BOARD_SIZE).fill().map((_, j) => {
            const cellState = getCellState(i, j);
            const showLastMove = lastMove && lastMove[i][j];
            return (
              <div
                key={`${i}-${j}`}
                className={`w-6 h-6 border border-gray-200 relative ${
                  cellState.filled ? PLAYER_COLORS[cellState.player] : 
                  'bg-white hover:bg-gray-100'
                }`}
                onClick={async () => {
                  const coords = calculatePlacementCoords(j, i);
                  if (coords) {
                    await playMove(coords);
                  }
                }}
                onMouseEnter={() => setHoverPosition([i, j])}
                onMouseLeave={() => setHoverPosition(null)}
              >
                {showLastMove && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="w-2 h-2 bg-black rounded-full"></div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
};

const GameStatus = ({ score, predictedValues, gameOver, result }) => {
  return (
    <>
      {score !== null && (
        <div className="mt-4 p-2 bg-gray-100 rounded font-mono text-sm">
          Score: {JSON.stringify(score)}
        </div>
      )}
      {predictedValues !== null && (
        <div className="mt-4 p-2 bg-gray-100 rounded">
          <h4 className="font-medium mb-2">Predicted Values:</h4>
          <div className="flex h-24 items-end gap-2">
            {predictedValues.map((value, i) => (
              <div key={i} className="flex flex-col items-center">
                <div 
                  className={`w-12 ${i === 0 ? 'bg-blue-500' : i === 1 ? 'bg-yellow-500' : i === 2 ? 'bg-red-500' : 'bg-green-500'}`}
                  style={{ height: `${Math.max(value * 80, 4)}px` }}
                ></div>
                <div className="text-xs mt-1">{(value * 100).toFixed(1)}%</div>
              </div>
            ))}
          </div>
        </div>
      )}
      {gameOver && result && (
        <div className="mt-4 p-4 bg-yellow-100 rounded">
          <h3 className="font-bold">Game Over!</h3>
          <div className="mt-2 font-mono text-sm">
            Result: {JSON.stringify(result)}
          </div>
        </div>
      )}
    </>
  );
};

const BlokusGame = () => {
  const [boards, setBoards] = useState(Array(4).fill().map(() => 
    Array(BOARD_SIZE).fill().map(() => Array(BOARD_SIZE).fill(0))
  ));
  const [pieces, setPieces] = useState([]);
  const [selectedPiece, setSelectedPiece] = useState(null);
  const [rotation, setRotation] = useState(0);
  const [isFlipped, setIsFlipped] = useState(false);
  const [hoverPosition, setHoverPosition] = useState(null);
  const [currentPlayer, setCurrentPlayer] = useState(0);
  const [errorMessage, setErrorMessage] = useState('');
  const [score, setScore] = useState(null);
  const [predictedValues, setPredictedValues] = useState(null);
  const [gameOver, setGameOver] = useState(false);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isHumanTurn, setIsHumanTurn] = useState(true);
  const [lastMove, setLastMove] = useState(null);

  const loadGameState = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:8080/state');
      const state = await response.json();
      
      if (!state.board || 
          !Array.isArray(state.board) || 
          state.board.length !== 4 || 
          !state.board.every(board => 
            Array.isArray(board) && 
            board.length === BOARD_SIZE && 
            board.every(row => 
              Array.isArray(row) && 
              row.length === BOARD_SIZE && 
              row.every(cell => cell === 0 || cell === 1)
            )
          ) ||
          typeof state.player !== 'number' ||
          !Number.isInteger(state.player) ||
          state.player < 0 ||
          state.player > 3 ||
          !state.pieces ||
          !validatePieces(state.pieces)) {
        throw new Error('Invalid game state received from server');
      }

      setBoards(state.board);
      setCurrentPlayer(state.player);
      setPieces(state.pieces);
      setSelectedPiece(null);
      setScore(state.score);
      setPredictedValues(state.predicted_values);
      setGameOver(state.game_over);
      setResult(state.result);
      setIsHumanTurn(state.is_human_turn);
      setLastMove(state.last_move);
      setErrorMessage('');
    } catch (e) {
      setErrorMessage(`Failed to load game state: ${e.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadGameState();
  }, []);

  const validatePieces = (pieces) => {
    return Array.isArray(pieces) && 
           pieces.every(piece => 
             Array.isArray(piece) && 
             piece.every(coord => 
               Array.isArray(coord) && 
               coord.length === 2 && 
               Number.isInteger(coord[0]) && 
               Number.isInteger(coord[1])
             )
           );
  };

  const getPieceBounds = (coords) => {
    const xs = coords.map(([x]) => x);
    const ys = coords.map(([, y]) => y);
    return {
      minX: Math.min(...xs),
      maxX: Math.max(...xs),
      minY: Math.min(...ys),
      maxY: Math.max(...ys),
      width: Math.max(...xs) - Math.min(...xs) + 1,
      height: Math.max(...ys) - Math.min(...ys) + 1
    };
  };

  const rotatePiece = (coords) => {
    return coords.map(([x, y]) => [-y, x]);
  };

  const flipPiece = (coords) => {
    return coords.map(([x, y]) => [-x, y]);
  };

  const normalizeCoords = (coords) => {
    const bounds = getPieceBounds(coords);
    return coords.map(([x, y]) => [
      x - (bounds.minX < 0 ? bounds.minX : 0),
      y - (bounds.minY < 0 ? bounds.minY : 0)
    ]);
  };

  const getTransformedPiece = (coords) => {
    if (!coords) return null;
    let transformed = [...coords];
    
    for (let i = 0; i < rotation; i++) {
      transformed = rotatePiece(transformed);
    }
    
    if (isFlipped) {
      transformed = flipPiece(transformed);
    }
    
    transformed = normalizeCoords(transformed);
    return transformed;
  };

  const calculatePlacementCoords = (boardX, boardY) => {
    if (selectedPiece === null) return;
    
    const piece = getTransformedPiece(pieces[selectedPiece]);
    return piece.map(([x, y]) => [boardY + y, boardX + x])
                .filter(([y, x]) => x >= 0 && x < BOARD_SIZE && y >= 0 && y < BOARD_SIZE);
  };

  const renderPiecePreview = (coords) => {
    const bounds = getPieceBounds(coords);
    const grid = Array(bounds.height).fill().map(() => Array(bounds.width).fill(0));
    
    coords.forEach(([x, y]) => {
      const normalizedX = x - bounds.minX;
      const normalizedY = y - bounds.minY;
      grid[normalizedY][normalizedX] = 1;
    });

    return grid;
  };

  const shouldShowPreview = (row, col) => {
    if (selectedPiece === null || !hoverPosition) return false;
    
    const piece = getTransformedPiece(pieces[selectedPiece]);
    const [hoverY, hoverX] = hoverPosition;
    
    return piece.some(([x, y]) => {
      const previewX = hoverX + x;
      const previewY = hoverY + y;
      return previewX === col && previewY === row;
    });
  };

  const getCellState = (row, col) => {
    if (shouldShowPreview(row, col)) {
      return { filled: true, player: currentPlayer };
    }
    
    for (let player = 0; player < 4; player++) {
      if (boards[player][row][col]) {
        return { filled: true, player };
      }
    }
    
    return { filled: false };
  };

  const playMove = async (coords) => {
    setIsLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:8080/move', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(coords)
      });
      
      if (!response.ok) {
        throw new Error('Failed to play move');
      }

      const newState = await response.json();
      setBoards(newState.board);
      setCurrentPlayer(newState.player);
      setPieces(newState.pieces);
      setSelectedPiece(null);
      setScore(newState.score);
      setPredictedValues(newState.predicted_values);
      setGameOver(newState.game_over);
      setResult(newState.result);
      setIsHumanTurn(newState.is_human_turn);
      setLastMove(newState.last_move);
      setErrorMessage('');
    } catch (e) {
      setErrorMessage(`Failed to play move: ${e.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="p-4">
      {errorMessage && (
        <div className="mb-4 p-4 bg-red-100 text-red-700 rounded">
          {errorMessage}
        </div>
      )}

      {isLoading && (
        <div className="mb-4 p-4 bg-blue-100 text-blue-700 rounded">
          Loading...
        </div>
      )}

      {isHumanTurn ? (
        <>
          <AvailablePieces 
            pieces={pieces}
            selectedPiece={selectedPiece}
            setSelectedPiece={setSelectedPiece}
            currentPlayer={currentPlayer}
            renderPiecePreview={renderPiecePreview}
          />

          <PieceControls
            selectedPiece={selectedPiece}
            setRotation={setRotation}
            setIsFlipped={setIsFlipped}
            isFlipped={isFlipped}
            pieces={pieces}
            getTransformedPiece={getTransformedPiece}
            renderPiecePreview={renderPiecePreview}
            currentPlayer={currentPlayer}
          />
        </>
      ) : (
        <button
          className="mb-4 px-4 py-2 bg-blue-500 text-white rounded"
          onClick={() => playMove([])}
        >
          Play AI Move
        </button>
      )}

      <div>
        <h2 className="text-xl font-bold mb-2">Game Board</h2>
        <GameBoard
          getCellState={getCellState}
          calculatePlacementCoords={calculatePlacementCoords}
          playMove={playMove}
          setHoverPosition={setHoverPosition}
          lastMove={lastMove}
        />
        <GameStatus
          score={score}
          predictedValues={predictedValues}
          gameOver={gameOver}
          result={result}
        />
      </div>
    </div>
  );
};

export default BlokusGame;