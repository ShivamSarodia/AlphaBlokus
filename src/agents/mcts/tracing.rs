use crate::config::{MCTSConfig, NUM_PLAYERS};
use crate::game::SerializableState;
use ahash::AHashMap as HashMap;
use serde::Serialize;
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;

#[allow(clippy::large_enum_variant)]
#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MCTSTrace {
    /// Published when the search starts (i.e. choose_move is called).
    StartedSearch {
        root_node_id: u64,
        state: SerializableState,
        search_id: u64,
        is_fast_move: bool,
        num_rollouts: u32,
    },
    /// Published when a node is created (i.e. Node::build_and_expand is called).
    CreatedNode {
        node_id: u64,
        search_id: u64,
        num_valid_moves: usize,
        move_index_to_array_index: HashMap<u16, u16>,
        array_index_to_move_index: Vec<u16>,
        array_index_to_player_pov_move_index: Vec<u16>,
    },
    /// Published when the network evaluation results are set for a node.
    NetworkEvalResult {
        node_id: u64,
        search_id: u64,
        value: [f32; NUM_PLAYERS],
        policy: Vec<f32>,
    },
    /// Published when a move is selected by UCB (i.e. Node::select_move_by_ucb is called).
    SelectedMoveByUcb {
        node_id: u64,
        search_id: u64,
        move_index: usize,
        array_index: usize,
        children_value_sums: Vec<[f32; NUM_PLAYERS]>,
        children_visit_counts: Vec<u16>,
        children_visit_counts_sum: u16,
        children_prior_probabilities: Vec<f32>,
        exploration_scores: Vec<f32>,
        exploitation_scores: Vec<f32>,
    },
    /// Published when a move is selected to play (i.e. Node::select_move_to_play is called).
    SelectedMoveToPlay {
        node_id: u64,
        search_id: u64,
        temperature: f32,
        children_visit_counts: Vec<u16>,
        move_index: usize,
        array_index: usize,
    },
    /// Published when a child node is added to a parent node.
    AddedChild {
        parent_node_id: u64,
        child_node_id: u64,
        search_id: u64,
        move_index: usize,
    },
}

pub async fn record_mcts_trace(mcts_trace: MCTSTrace, mcts_config: &MCTSConfig) {
    let trace_file = mcts_config.trace_file.as_ref().unwrap();

    // TODO: Don't re-open the file for each trace.
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .append(true)
        .open(trace_file)
        .await
        .unwrap();

    file.write_all(serde_json::to_string(&mcts_trace).unwrap().as_bytes())
        .await
        .unwrap();
    file.write_all("\n".as_bytes()).await.unwrap();
    file.flush().await.unwrap();
}
