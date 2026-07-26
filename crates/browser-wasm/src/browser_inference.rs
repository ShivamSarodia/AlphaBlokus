use alphablokus_game_core::{config::GameConfig, game::Board};
use alphablokus_mcts_core::{InferenceClient, Request, Response, softmax_inplace};
use anyhow::{Result, anyhow};
use js_sys::{Float32Array, Function, Promise, Reflect, Uint8Array, Uint32Array};
use std::{future::Future, pin::Pin};
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::JsFuture;

const BOARD_SIZE: usize = 20;

/// Adapts the application's existing inference client contract to the
/// Promise-returning ONNX Runtime Web function owned by the browser worker.
#[derive(Clone)]
pub struct BrowserInferenceClient {
    evaluate: Function,
    game_config: &'static GameConfig,
}

impl BrowserInferenceClient {
    pub fn new(evaluate: Function, game_config: &'static GameConfig) -> Self {
        Self {
            evaluate,
            game_config,
        }
    }
}

impl InferenceClient for BrowserInferenceClient {
    type EvaluationFuture<'a> = Pin<Box<dyn Future<Output = Result<Response>> + 'a>>;

    fn evaluate(&self, request: Request) -> Self::EvaluationFuture<'_> {
        Box::pin(async move {
            let board = board_values(&request.board);
            let piece_availability = request
                .piece_availability
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            let move_profiles = self.game_config.move_profiles()?;
            let policy_indexes = request
                .valid_move_indexes
                .into_iter()
                .map(|move_index| {
                    let profile = move_profiles.get(move_index);
                    u32::try_from(
                        profile.piece_orientation_index * BOARD_SIZE * BOARD_SIZE
                            + profile.center.0 * BOARD_SIZE
                            + profile.center.1,
                    )
                    .map_err(|error| anyhow!(error))
                })
                .collect::<Result<Vec<_>>>()?;
            let promise = self
                .evaluate
                .call3(
                    &JsValue::UNDEFINED,
                    &Uint8Array::from(board.as_slice()),
                    &Uint8Array::from(piece_availability.as_slice()),
                    &Uint32Array::from(policy_indexes.as_slice()),
                )
                .map_err(js_error)?;
            let result = JsFuture::from(Promise::resolve(&promise))
                .await
                .map_err(js_error)?;
            let mut value: [f32; 4] = result_float32_array(&result, "value_logits")?
                .try_into()
                .map_err(|_| anyhow!("expected four value logits"))?;
            let mut policy = result_float32_array(&result, "policy_logits")?;
            softmax_inplace(&mut value);
            softmax_inplace(&mut policy);
            Ok(Response { value, policy })
        })
    }
}

fn board_values(board: &Board) -> Vec<u8> {
    let mut values = vec![0; 4 * BOARD_SIZE * BOARD_SIZE];
    for channel in 0..4 {
        for x in 0..BOARD_SIZE {
            for y in 0..BOARD_SIZE {
                if board.slice(channel).get((x, y)) {
                    values[channel * BOARD_SIZE * BOARD_SIZE + x * BOARD_SIZE + y] = 1;
                }
            }
        }
    }
    values
}

fn result_float32_array(result: &JsValue, property: &str) -> Result<Vec<f32>> {
    let value = Reflect::get(result, &JsValue::from_str(property)).map_err(js_error)?;
    Ok(Float32Array::new(&value).to_vec())
}

fn js_error(error: JsValue) -> anyhow::Error {
    anyhow!("browser inference error: {error:?}")
}
