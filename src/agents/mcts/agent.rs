use std::sync::Arc;

use crate::agents::Agent;
use crate::agents::mcts::node::Node;
use crate::config::{GameConfig, MCTSConfig};
use crate::game::{GameStatus, State};
use crate::inference;

pub struct MCTSAgent {
    mcts_config: Arc<MCTSConfig>,
    game_config: Arc<GameConfig>,
    inference_client: Arc<inference::Client>,
}

impl MCTSAgent {
    pub fn new(
        mcts_config: Arc<MCTSConfig>,
        game_config: Arc<GameConfig>,
        inference_client: Arc<inference::Client>,
    ) -> Self {
        Self {
            mcts_config,
            game_config,
            inference_client,
        }
    }

    async fn rollout_once(&self, state: &State<'_>, search_root: &mut Node) {
        let mut moves_played = Vec::new();

        let value = {
            let mut current_state = state.clone();
            // This reborrows search_root as mutable to avoid moving it, since we'll
            // need it again below when backpropagating the value.
            let mut current_node = &mut *search_root;

            loop {
                // Select the next child node to explore.
                let move_index = current_node.select_move_by_ucb();

                // Record the selected move, and play it.
                moves_played.push(move_index);
                let game_status = current_state.apply_move(move_index);

                // If the game is now over, we just assign values based on the final state.
                if game_status == GameStatus::GameOver {
                    break current_state.result();
                }

                // Try to find an existing child node for the selected move.
                if current_node.has_child(move_index) {
                    current_node = current_node.get_child_mut(move_index).unwrap();
                } else {
                    let new_node = Node::build_and_expand(
                        &current_state,
                        &self.inference_client,
                        Arc::clone(&self.mcts_config),
                        Arc::clone(&self.game_config),
                        false,
                    )
                    .await;
                    let value = new_node.get_value_as_universal_pov();
                    current_node.add_child(move_index, new_node);
                    break value;
                }
            }
        };

        // Now, backpropagate the value we just learned up the tree.
        let mut node = &mut *search_root;
        for move_index in moves_played.iter().copied() {
            node.increment_child_value_sum(move_index, value);
            node.increment_child_visit_count(move_index);
            node = node.get_child_mut(move_index).unwrap();
        }
    }
}

impl Agent for MCTSAgent {
    async fn choose_move(&self, state: &State<'_>) -> usize {
        // Create a new node to represent the root of the search tree. Start by expanding the
        // node immediately.
        let mut search_root = Node::build_and_expand(
            state,
            &self.inference_client,
            Arc::clone(&self.mcts_config),
            Arc::clone(&self.game_config),
            true,
        )
        .await;

        // Run the rollouts, which formulates the search tree.
        for _ in 0..self.mcts_config.num_rollouts {
            self.rollout_once(state, &mut search_root).await;
        }

        search_root.select_move_to_play(state)
    }
}
