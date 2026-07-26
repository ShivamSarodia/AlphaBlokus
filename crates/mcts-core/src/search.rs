use anyhow::{Context, Result, anyhow};
use log::trace;
use rand::Rng;

use super::node::Node;
use alphablokus_game_core::{
    config::GameConfig,
    game::{GameStatus, State},
};

use crate::{InferenceClient, MCTSConfig};

/// The selected move and the compact statistics needed by native self-play.
pub struct MCTSSearchResult {
    pub move_index: usize,
    pub is_fast_move: bool,
    pub stats: SearchStats,
}

#[derive(Debug, Clone)]
pub struct SearchStats {
    pub player: usize,
    pub player_pov_move_indexes: Vec<u16>,
    pub child_visit_counts: Vec<u16>,
    pub q_value: [f32; alphablokus_game_core::config::NUM_PLAYERS],
}

/// Runs the existing MCTS algorithm independently of the native Agent and
/// self-play recording lifecycle.
pub struct MCTSSearch<T: InferenceClient> {
    mcts_config: &'static MCTSConfig,
    game_config: &'static GameConfig,
    inference_client: T,
}

impl<T: InferenceClient> MCTSSearch<T> {
    pub fn new(
        mcts_config: &'static MCTSConfig,
        game_config: &'static GameConfig,
        inference_client: T,
    ) -> Self {
        Self {
            mcts_config,
            game_config,
            inference_client,
        }
    }

    async fn rollout_once(&self, state: &State, search_root: &mut Node) -> Result<()> {
        trace!("Rolling out once from state: {}", state);

        let mut moves_played = Vec::new();

        let value = {
            let mut current_state = state.clone();
            // This reborrows search_root as mutable to avoid moving it, since we'll
            // need it again below when backpropagating the value.
            let mut current_node = &mut *search_root;

            loop {
                trace!(
                    "Rollout traversal iteration. Moves played: {:?}, current state: {}",
                    moves_played, current_state,
                );

                // Select the next child node to explore.
                let move_index = current_node.select_move_by_ucb();

                // Play and record the selected move.
                let game_status = current_state
                    .apply_move(move_index)
                    .map_err(|err| anyhow!("Failed to apply move during rollout: {}", err))?;
                moves_played.push(move_index);

                // If the game is now over, we just assign values based on the final state.
                if game_status == GameStatus::GameOver {
                    trace!(
                        "Iteration terminated because game is over. Current state: {}",
                        current_state
                    );
                    break current_state.result();
                }

                // Try to find an existing child node for the selected move.
                if current_node.has_child(move_index) {
                    trace!(
                        "Proceeding to next iteration: found existing child node for move index: {}",
                        move_index
                    );
                    current_node = current_node.get_child_mut(move_index).ok_or_else(|| {
                        anyhow!(
                            "Expected child node for move {}, but none found",
                            move_index
                        )
                    })?;
                } else {
                    trace!(
                        "Expanding new node: no existing child node for move index: {}",
                        move_index
                    );
                    let new_node = Node::build_and_expand(
                        &current_state,
                        &self.inference_client,
                        self.mcts_config,
                        self.game_config,
                        false,
                    )
                    .await?;
                    let value = new_node.get_value_as_universal_pov();
                    current_node.add_child(move_index, new_node);
                    break value;
                }
            }
        };

        trace!("Backpropagating through moves played: {:?}", moves_played);

        // Now, backpropagate the value we just learned up the tree.
        let mut node: Option<&mut Node> = Some(&mut *search_root);
        for &move_index in moves_played.iter() {
            node.as_deref_mut()
                .context("expected node while backpropagating but found none")?
                .increment_child_value_sum(move_index, value);
            node.as_deref_mut()
                .context("expected node while backpropagating but found none")?
                .increment_child_visit_count(move_index);
            node = node
                .context("expected node while backpropagating but found none")?
                .get_child_mut(move_index);
        }

        Ok(())
    }

    pub async fn choose_move(&self, state: &State) -> Result<MCTSSearchResult> {
        let is_fast_move = rand::rng().random::<f32>() < self.mcts_config.fast_move_probability;
        let num_rollouts = if is_fast_move {
            self.mcts_config.fast_move_num_rollouts
        } else {
            self.mcts_config.full_move_num_rollouts
        };

        // Create a new node to represent the root of the search tree. Start by expanding the
        // node immediately.
        let mut search_root = Node::build_and_expand(
            state,
            &self.inference_client,
            self.mcts_config,
            self.game_config,
            // Add noise only on full moves, not on fast moves.
            !is_fast_move,
        )
        .await?;

        // Run the rollouts, which formulates the search tree.
        for _ in 0..num_rollouts {
            self.rollout_once(state, &mut search_root).await?;
        }

        let move_index = search_root.select_move_to_play(state)?;

        Ok(MCTSSearchResult {
            move_index,
            is_fast_move,
            stats: SearchStats {
                player: search_root.player(),
                player_pov_move_indexes: search_root.player_pov_move_indexes().to_vec(),
                child_visit_counts: search_root.child_visit_counts().to_vec(),
                q_value: search_root.root_value_estimate_as_player_pov(),
            },
        })
    }
}
