use std::sync::Arc;

use log::trace;
use rand::Rng;

use crate::agents::Agent;
use crate::agents::mcts::node::Node;
use crate::agents::mcts::tracing::{MCTSTrace, record_mcts_trace};
use crate::config::{GameConfig, MCTSConfig};
use crate::game::{GameStatus, SerializableState, State};
use crate::inference;
use crate::recorder::MCTSData;
use async_trait::async_trait;

pub struct MCTSAgent<T: inference::Client + Send + Sync> {
    pub name: String,
    mcts_config: &'static MCTSConfig,
    game_config: &'static GameConfig,
    inference_client: Arc<T>,
    game_id: u64,
    /// The agent is responsible for accumulating MCTS data from rollouts.
    /// The data in this vector will have 0 values for the game result, and that field
    /// will be populated when the game is over.
    mcts_data: Vec<MCTSData>,
}

impl<T: inference::Client + Send + Sync> MCTSAgent<T> {
    pub fn new(
        mcts_config: &'static MCTSConfig,
        game_config: &'static GameConfig,
        inference_client: Arc<T>,
    ) -> Self {
        Self {
            name: mcts_config.name.clone(),
            mcts_config,
            game_config,
            inference_client,
            game_id: rand::rng().random::<u64>(),
            mcts_data: Vec::new(),
        }
    }

    async fn rollout_once(&self, state: &State, search_root: &mut Node) {
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
                let move_index = current_node.select_move_by_ucb().await;

                // Play and record the selected move.
                let game_status = current_state.apply_move(move_index);
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
                    current_node = current_node.get_child_mut(move_index).unwrap();
                } else {
                    trace!(
                        "Expanding new node: no existing child node for move index: {}",
                        move_index
                    );
                    let new_node = Node::build_and_expand(
                        &current_state,
                        current_node.search_id,
                        self.inference_client.as_ref(),
                        self.mcts_config,
                        self.game_config,
                        false,
                    )
                    .await;
                    let new_node_id = new_node.id;
                    let value = new_node.get_value_as_universal_pov();
                    current_node.add_child(move_index, new_node);
                    if self.mcts_config.tracing_enabled() {
                        record_mcts_trace(
                            MCTSTrace::AddedChild {
                                parent_node_id: current_node.id,
                                child_node_id: new_node_id,
                                search_id: current_node.search_id,
                                move_index,
                            },
                            self.mcts_config,
                        )
                        .await;
                    }
                    break value;
                }
            }
        };

        trace!("Backpropagating through moves played: {:?}", moves_played);

        // Now, backpropagate the value we just learned up the tree.
        let mut node: Option<&mut Node> = Some(&mut *search_root);
        for &move_index in moves_played.iter() {
            node.as_deref_mut()
                .unwrap()
                .increment_child_value_sum(move_index, value);
            node.as_deref_mut()
                .unwrap()
                .increment_child_visit_count(move_index);
            node = node.unwrap().get_child_mut(move_index);
        }
    }
}

#[async_trait]
impl<T: inference::Client + Send + Sync> Agent for MCTSAgent<T> {
    fn name(&self) -> &str {
        &self.name
    }

    async fn choose_move(&mut self, state: &State) -> usize {
        let search_id: u64 = chrono::Utc::now()
            .timestamp_nanos_opt()
            .unwrap()
            .try_into()
            .unwrap();
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
            search_id,
            self.inference_client.as_ref(),
            self.mcts_config,
            self.game_config,
            // Add noise only on full moves, not on fast moves.
            !is_fast_move,
        )
        .await;

        if self.mcts_config.tracing_enabled() {
            record_mcts_trace(
                MCTSTrace::StartedSearch {
                    state: SerializableState::from_state(state),
                    search_id,
                    root_node_id: search_root.id,
                    is_fast_move,
                    num_rollouts,
                },
                self.mcts_config,
            )
            .await;
        }

        // Run the rollouts, which formulates the search tree.
        for _ in 0..num_rollouts {
            self.rollout_once(state, &mut search_root).await;
        }

        let move_index = search_root.select_move_to_play(state).await;

        if !is_fast_move {
            self.mcts_data
                .push(search_root.generate_mcts_data(self.game_id, state));
        }

        move_index
    }

    fn flush_mcts_data(&mut self) -> Vec<MCTSData> {
        self.mcts_data.drain(..).collect()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::inference::softmax_inplace;
    use crate::{config::NUM_PLAYERS, testing};
    use itertools::Itertools;

    struct MockInferenceClient {
        pub requests: Mutex<Vec<inference::Request>>,
    }

    impl inference::Client for MockInferenceClient {
        async fn evaluate(&self, request: inference::Request) -> inference::Response {
            // Push the requests onto the vector.
            self.requests.lock().unwrap().push(request.clone());

            // Return a response where the current player's value is 1.0 and the other
            // players values are 0.0. It also returns a policy where move 1 is preferred.
            let value = [1.0, 0.0, 0.0, 0.0];
            let mut policy = vec![0.0; request.valid_move_indexes.len()];
            request
                .valid_move_indexes
                .iter()
                .position(|&move_index| move_index == 1008)
                .inspect(|&index_to_prefer| {
                    policy[index_to_prefer] = 1.0;
                });

            inference::Response { value, policy }
        }
    }

    #[tokio::test]
    async fn test_fast_move_behavior() {
        let game_config = testing::create_half_game_config();

        let fast_mcts_config: &'static MCTSConfig = Box::leak(Box::new(MCTSConfig {
            name: "test_fast".to_string(),
            fast_move_probability: 1.0,
            fast_move_num_rollouts: 1,
            full_move_num_rollouts: 4,
            total_dirichlet_noise_alpha: 1.0,
            root_dirichlet_noise_fraction: 0.0,
            ucb_exploration_factor: 1.0,
            temperature_turn_cutoff: 10,
            move_selection_temperature: 0.0,
            inference_config_name: "".to_string(),
            trace_file: None,
        }));
        let fast_client = Arc::new(MockInferenceClient {
            requests: Mutex::new(Vec::new()),
        });
        let mut fast_agent =
            MCTSAgent::new(fast_mcts_config, game_config, Arc::clone(&fast_client));
        let fast_state = State::new(game_config);
        fast_agent.choose_move(&fast_state).await;
        let fast_requests = fast_client.requests.lock().unwrap().len();
        assert!(fast_agent.flush_mcts_data().is_empty());

        // Two requests are made -- one for the initial node expansion, and
        // another for the single rollout.
        assert_eq!(fast_requests, 2);
    }

    #[tokio::test]
    async fn test_board_and_policy_rotations() {
        let mcts_config = testing::create_mcts_config(1, 0.0);

        // Generate a larger game config so that there's no concern about
        // one move blocking the others.
        let game_config = testing::create_half_game_config();

        let mock_client = Arc::new(MockInferenceClient {
            requests: Mutex::new(Vec::new()),
        });

        let mut agent = MCTSAgent::new(mcts_config, game_config, Arc::clone(&mock_client));

        let mut state = State::new(&game_config);
        let move_index_0 = agent.choose_move(&state).await;
        let move_profile_0 = game_config.move_profiles().get(move_index_0);

        // For the player 0 state, the move that's played should match the one preferred by the
        // policy.
        assert_eq!(
            move_index_0,
            state.valid_moves().collect::<Vec<usize>>()[10]
        );

        // On the first request, the valid move indexes and board should just match the state's
        // valid moves and board.
        let request_0 = mock_client.requests.lock().unwrap()[0].clone();
        assert_eq!(
            request_0.valid_move_indexes,
            state.valid_moves().collect::<Vec<usize>>(),
        );
        println!("State: {}", state);
        println!("Request 0 board: {}", request_0.board);
        assert_eq!(request_0.board, *state.board());

        // Now, apply the move and run a second rollout on the new state.
        mock_client.requests.lock().unwrap().clear();
        state.apply_move(move_index_0);
        let move_index_1 = agent.choose_move(&state).await;
        let move_profile_1 = game_config.move_profiles().get(move_index_1);

        // On the second request, the valid move indexes should match the first request
        // because from the player's own perspective, the legal moves are the same in both
        // cases.
        let request_1 = mock_client.requests.lock().unwrap()[0].clone();
        assert_eq!(
            request_0
                .valid_move_indexes
                .iter()
                .cloned()
                .sorted()
                .collect::<Vec<usize>>(),
            request_1
                .valid_move_indexes
                .iter()
                .cloned()
                .sorted()
                .collect::<Vec<usize>>(),
        );
        println!("State: {}", state);
        println!("Request 1 board: {}", request_1.board);
        // The player who just went should always be at slice 3 and spot (0, 9).
        assert_eq!(request_1.board.slice(0).count(), 0);
        assert_eq!(request_1.board.slice(1).count(), 0);
        assert_eq!(request_1.board.slice(2).count(), 0);
        assert_eq!(request_1.board.slice(3).count(), 5);
        assert_eq!(request_1.board.slice(3).get((0, 9)), true);

        // Confirm that the piece selected is the same as in the first rollout.
        assert_eq!(move_profile_0.piece_index, move_profile_1.piece_index);

        mock_client.requests.lock().unwrap().clear();
        state.apply_move(move_index_1);
        let move_index_2 = agent.choose_move(&state).await;
        let move_profile_2 = game_config.move_profiles().get(move_index_2);

        let request_2 = mock_client.requests.lock().unwrap()[0].clone();
        assert_eq!(
            request_0
                .valid_move_indexes
                .iter()
                .cloned()
                .sorted()
                .collect::<Vec<usize>>(),
            request_2
                .valid_move_indexes
                .iter()
                .cloned()
                .sorted()
                .collect::<Vec<usize>>(),
        );
        println!("State: {}", state);
        println!("Request 2 board: {}", request_2.board);
        assert_eq!(request_2.board.slice(0).count(), 0);
        assert_eq!(request_2.board.slice(1).count(), 0);
        assert_eq!(request_2.board.slice(2).count(), 5);
        assert_eq!(request_2.board.slice(2).get((9, 9)), true);
        assert_eq!(request_2.board.slice(3).count(), 5);
        assert_eq!(request_2.board.slice(3).get((0, 9)), true);

        assert_eq!(move_profile_0.piece_index, move_profile_2.piece_index);

        // Now apply the move to get to player 3.
        mock_client.requests.lock().unwrap().clear();
        state.apply_move(move_index_2);

        let move_index_3 = agent.choose_move(&state).await;
        let move_profile_3 = game_config.move_profiles().get(move_index_3);

        let request_3 = mock_client.requests.lock().unwrap()[0].clone();
        assert_eq!(
            request_0
                .valid_move_indexes
                .iter()
                .cloned()
                .sorted()
                .collect::<Vec<usize>>(),
            request_3
                .valid_move_indexes
                .iter()
                .cloned()
                .sorted()
                .collect::<Vec<usize>>(),
        );
        println!("State: {}", state);
        println!("Request 3 board: {}", request_3.board);
        assert_eq!(request_3.board.slice(0).count(), 0);
        assert_eq!(request_3.board.slice(1).count(), 5);
        assert_eq!(request_3.board.slice(1).get((9, 0)), true);
        assert_eq!(request_3.board.slice(2).count(), 5);
        assert_eq!(request_3.board.slice(2).get((9, 9)), true);
        assert_eq!(request_3.board.slice(3).count(), 5);
        assert_eq!(request_3.board.slice(3).get((0, 9)), true);

        assert_eq!(move_profile_0.piece_index, move_profile_3.piece_index);

        state.apply_move(move_index_3);
        assert_eq!(state.player(), 0);
        assert_eq!(state.board().slice(0).get((0, 0)), true);
        assert_eq!(state.board().slice(1).count(), 5);
        assert_eq!(state.board().slice(1).get((9, 0)), true);
        assert_eq!(state.board().slice(2).count(), 5);
        assert_eq!(state.board().slice(2).get((9, 9)), true);
        assert_eq!(state.board().slice(3).count(), 5);
        assert_eq!(state.board().slice(3).get((0, 9)), true);
    }

    #[tokio::test]
    async fn test_value_rotation() {
        let mcts_config = testing::create_mcts_config(1, 0.0);

        // Generate a larger game config so that there's no concern about
        // one move blocking the others.
        let game_config = testing::create_half_game_config();

        let mock_client = Arc::new(MockInferenceClient {
            requests: Mutex::new(Vec::new()),
        });

        for player in 0..NUM_PLAYERS {
            let mut state = State::new(&game_config);
            for _ in 0..player {
                state.apply_move(state.first_valid_move().unwrap());
            }

            let search_root = Node::build_and_expand(
                &state,
                0,
                mock_client.as_ref(),
                &mcts_config,
                &game_config,
                true,
            )
            .await;

            let universal_value = search_root.get_value_as_universal_pov();
            for i in 0..NUM_PLAYERS {
                if i == player {
                    assert_eq!(universal_value[i], 1.0);
                } else {
                    assert_eq!(universal_value[i], 0.0);
                }
            }
        }
    }

    struct ValuesInferenceClient {}
    impl inference::Client for ValuesInferenceClient {
        async fn evaluate(&self, request: inference::Request) -> inference::Response {
            // Define "value" to prefer players with fewer pieces on the board, so that
            // if MCTS is working correctly all four players will play the single piece
            // move.

            let policy = vec![0.0; request.valid_move_indexes.len()];
            let mut value = [
                -(request.board.slice(0).count() as f32),
                -(request.board.slice(1).count() as f32),
                -(request.board.slice(2).count() as f32),
                -(request.board.slice(3).count() as f32),
            ];
            softmax_inplace(&mut value);

            inference::Response { value, policy }
        }
    }

    #[tokio::test]
    async fn test_values_used_in_search() {
        // Play a good number of rollouts to ensure we land on the conclusion move
        // that has the best value (i.e. the one-square).
        let mcts_config = testing::create_mcts_config(100, 0.0);
        let game_config = testing::create_half_game_config();
        let mock_client = Arc::new(ValuesInferenceClient {});

        let mut agent = MCTSAgent::new(mcts_config, game_config, Arc::clone(&mock_client));
        let mut state = State::new(&game_config);

        for player in 0..4 {
            assert_eq!(state.player(), player);
            let move_index = agent.choose_move(&state).await;
            let move_profile = game_config.move_profiles().get(move_index);
            assert_eq!(move_profile.occupied_cells.count(), 1);

            state.apply_move(move_index);
        }
    }

    #[tokio::test]
    async fn test_select_move_to_play_with_temperature() {
        // Test that select_move_to_play works when temperature is non-zero.
        // We don't verify the exact distribution, just that it doesn't crash
        // and returns a valid move.
        let mcts_config = testing::create_mcts_config(50, 1.0); // Non-zero temperature
        let game_config = testing::create_half_game_config();
        let mock_client = Arc::new(MockInferenceClient {
            requests: Mutex::new(Vec::new()),
        });

        let mut agent = MCTSAgent::new(mcts_config, game_config, Arc::clone(&mock_client));

        let state = State::new(&game_config);

        // Run the test multiple times to make sure it consistently works
        for _ in 0..5 {
            let move_index = agent.choose_move(&state).await;
            assert!(state.is_valid_move(move_index));
        }
    }
}
