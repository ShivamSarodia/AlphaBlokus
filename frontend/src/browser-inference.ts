import * as ort from 'onnxruntime-web/webgpu'

const BOARD_SIZE = 20
const NUM_PIECES = 21

let session: ort.InferenceSession | null = null

export function setBrowserInferenceSession(nextSession: ort.InferenceSession): void {
  session = nextSession
}

/**
 * The Rust/WASM MCTS loop awaits this promise directly. Keeping this module
 * limited to ONNX Runtime Web makes JavaScript the inference provider rather
 * than the owner of MCTS traversal and backpropagation.
 */
export async function evaluate(
  board: Uint8Array,
  pieceAvailability: Uint8Array,
  policyIndexes: Uint32Array,
): Promise<{ value_logits: Float32Array; policy_logits: Float32Array }> {
  if (!session) throw new Error('The bot model is not initialized.')

  const boardTensor = new ort.Tensor(
    'float32',
    Float32Array.from(board),
    [1, 4, BOARD_SIZE, BOARD_SIZE],
  )
  const pieceAvailabilityTensor = new ort.Tensor(
    'float32',
    Float32Array.from(pieceAvailability),
    [1, 4, NUM_PIECES],
  )

  try {
    const outputs = await session.run({
      board: boardTensor,
      piece_availability: pieceAvailabilityTensor,
    })
    try {
      const fullPolicy = outputs.policy.data as Float32Array
      return {
        value_logits: Float32Array.from(outputs.value.data as Float32Array),
        policy_logits: Float32Array.from(policyIndexes, (index) => fullPolicy[index]),
      }
    } finally {
      outputs.value.dispose()
      outputs.policy.dispose()
    }
  } finally {
    boardTensor.dispose()
    pieceAvailabilityTensor.dispose()
  }
}
