//! Browser-facing game state backed by AlphaBlokus' native rules and move table.

mod browser_inference;

use std::path::PathBuf;
use std::sync::OnceLock;

use crate::browser_inference::BrowserInferenceClient;
use alphablokus_game_core::{
    config::GameConfig,
    game::{
        GameStatus, SerializableState, State,
        move_data::codec::{MoveDataDecoder, decoder_from_slice},
    },
};
use alphablokus_mcts_core::{DefaultExploitationValue, MCTSConfig, MCTSSearch};
use js_sys::{Function, Promise};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

const BOARD_SIZE: usize = 20;
const NUM_MOVES: usize = 30_433;
const NUM_PIECES: usize = 21;
const NUM_PIECE_ORIENTATIONS: usize = 91;

#[wasm_bindgen]
pub struct BrowserGame {
    game_config: &'static GameConfig,
    history: Vec<State>,
}

#[wasm_bindgen]
pub struct BrowserGameBuilder {
    game_config: Option<GameConfig>,
    move_data_decoder: Option<MoveDataDecoder>,
}

#[wasm_bindgen]
impl BrowserGame {
    /// Construct a game from a decompressed canonical move-table payload.
    #[wasm_bindgen(constructor)]
    pub fn new(move_data: &[u8]) -> Result<BrowserGame, JsValue> {
        let mut builder = BrowserGameBuilder::new(move_data)?;
        let total = builder.total_profiles();
        builder.build_profiles(total)?;
        builder.finish()
    }

    fn from_config(config: GameConfig) -> Result<BrowserGame, JsValue> {
        let config = Box::leak(Box::new(config));
        let initial = State::new(config).map_err(js_error)?;
        Ok(Self {
            game_config: config,
            history: vec![initial],
        })
    }
}

#[wasm_bindgen]
impl BrowserGameBuilder {
    #[wasm_bindgen(constructor)]
    pub fn new(move_data: &[u8]) -> Result<BrowserGameBuilder, JsValue> {
        let config = GameConfig {
            board_size: BOARD_SIZE,
            num_moves: NUM_MOVES,
            num_pieces: NUM_PIECES,
            num_piece_orientations: NUM_PIECE_ORIENTATIONS,
            move_data_file: PathBuf::new(),
            move_data: None,
        };
        let move_data_decoder = decoder_from_slice(move_data, &config).map_err(js_error)?;
        Ok(Self {
            game_config: Some(config),
            move_data_decoder: Some(move_data_decoder),
        })
    }

    pub fn total_profiles(&self) -> usize {
        self.move_data_decoder
            .as_ref()
            .map(MoveDataDecoder::total_profiles)
            .unwrap_or(0)
    }

    pub fn completed_profiles(&self) -> usize {
        self.move_data_decoder
            .as_ref()
            .map(MoveDataDecoder::completed_profiles)
            .unwrap_or(0)
    }

    pub fn build_profiles(&mut self, count: usize) -> Result<usize, JsValue> {
        let config = self
            .game_config
            .as_ref()
            .ok_or_else(|| JsValue::from_str("browser game construction has already finished"))?;
        let decoder = self
            .move_data_decoder
            .as_mut()
            .ok_or_else(|| JsValue::from_str("browser game construction has already finished"))?;
        decoder.build_profiles(count, config).map_err(js_error)?;
        Ok(decoder.completed_profiles())
    }

    pub fn finish(&mut self) -> Result<BrowserGame, JsValue> {
        let mut config = self
            .game_config
            .take()
            .ok_or_else(|| JsValue::from_str("browser game construction has already finished"))?;
        let decoder = self
            .move_data_decoder
            .take()
            .ok_or_else(|| JsValue::from_str("browser game construction has already finished"))?;
        config.move_data = Some(decoder.finish(&config).map_err(js_error)?);
        BrowserGame::from_config(config)
    }
}

#[wasm_bindgen]
impl BrowserGame {
    pub fn board_size(&self) -> usize {
        self.game_config.board_size
    }

    pub fn current_player(&self) -> usize {
        self.state().player()
    }

    pub fn turn(&self) -> u16 {
        self.state().turn()
    }

    pub fn valid_move_indexes(&self) -> Vec<u32> {
        self.state()
            .valid_moves()
            .map(|index| index as u32)
            .collect()
    }

    pub fn move_cells_json(&self, move_index: usize) -> Result<String, JsValue> {
        let profile = self
            .game_config
            .move_profiles()
            .map_err(js_error)?
            .get(move_index);
        serde_json::to_string(&profile.occupied_cells.to_cells()).map_err(js_error)
    }

    pub fn apply_move(&mut self, move_index: usize) -> Result<bool, JsValue> {
        let mut next = self.state().clone();
        let status = next.apply_move(move_index).map_err(js_error)?;
        self.history.push(next);
        Ok(status == GameStatus::GameOver)
    }

    pub fn undo(&mut self) -> bool {
        if self.history.len() <= 1 {
            return false;
        }
        self.history.pop();
        true
    }

    pub fn reset(&mut self) -> Result<(), JsValue> {
        self.history = vec![State::new(self.game_config).map_err(js_error)?];
        Ok(())
    }

    pub fn state_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&SerializableState::from_state(self.state())).map_err(js_error)
    }

    pub fn result_json(&self) -> String {
        serde_json::to_string(&self.state().result()).expect("fixed-size result is serializable")
    }

    /// Runs the full search in Rust. The caller supplies the Promise-returning
    /// browser inference provider, and receives one promise for the final move.
    pub fn choose_move(&self, rollouts: u32, evaluate: Function) -> Promise {
        let state = self.state().clone();
        let mcts_config = match browser_mcts_config(rollouts) {
            Ok(config) => config,
            Err(error) => return Promise::reject(&error),
        };
        let game_config = self.game_config;
        let inference_client = BrowserInferenceClient::new(evaluate, game_config);
        future_to_promise(async move {
            let search = MCTSSearch::new(mcts_config, game_config, inference_client);
            let result = search.choose_move(&state).await.map_err(js_error)?;
            serde_json::to_string(&serde_json::json!({
                "move_index": result.move_index,
            }))
            .map(|json| JsValue::from_str(&json))
            .map_err(js_error)
        })
    }
}

fn browser_mcts_config(rollouts: u32) -> Result<&'static MCTSConfig, JsValue> {
    static QUICK: OnceLock<MCTSConfig> = OnceLock::new();
    static STRONG: OnceLock<MCTSConfig> = OnceLock::new();
    static EXPERT: OnceLock<MCTSConfig> = OnceLock::new();

    let config = match rollouts {
        100 => QUICK.get_or_init(|| create_browser_mcts_config(100)),
        500 => STRONG.get_or_init(|| create_browser_mcts_config(500)),
        2_000 => EXPERT.get_or_init(|| create_browser_mcts_config(2_000)),
        _ => return Err(JsValue::from_str("unknown AlphaBlokus strength preset")),
    };
    Ok(config)
}

fn create_browser_mcts_config(rollouts: u32) -> MCTSConfig {
    MCTSConfig {
        name: "browser".to_string(),
        // Browser play uses the full-move search path with root noise disabled,
        // and the selected rollout count for every turn.
        fast_move_probability: 0.0,
        fast_move_num_rollouts: rollouts,
        full_move_num_rollouts: rollouts,
        total_dirichlet_noise_alpha: 1.0,
        root_dirichlet_noise_fraction: 0.0,
        ucb_exploration_factor: 1.05,
        temperature_turn_cutoff: 0,
        move_selection_temperature: 0.0,
        default_exploitation_value: DefaultExploitationValue::NetworkValue,
        inference_config_name: String::new(),
        policy_inference_config_name: String::new(),
        value_inference_config_name: String::new(),
    }
}

impl BrowserGame {
    fn state(&self) -> &State {
        self.history
            .last()
            .expect("browser game always has an initial state")
    }
}

fn js_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

#[wasm_bindgen]
pub fn supported_strength_rollouts(name: &str) -> Result<u32, JsValue> {
    match name {
        "quick" => Ok(100),
        "strong" => Ok(500),
        "expert" => Ok(2_000),
        _ => Err(JsValue::from_str("unknown AlphaBlokus strength preset")),
    }
}

#[wasm_bindgen]
pub fn browser_runtime_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
