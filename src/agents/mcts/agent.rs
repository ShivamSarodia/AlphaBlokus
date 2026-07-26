use std::sync::Arc;

use anyhow::Result;
use mcts_core::{InferenceClient, MCTSSearch, SearchStats};
use rand::Rng;

use crate::agents::Agent;
use crate::config::NUM_PLAYERS;
use crate::config::{GameConfig, MCTSConfig};
use crate::game::State;
use crate::recorder::MCTSData;
use async_trait::async_trait;

pub struct MCTSAgent<T: InferenceClient + Send + Sync> {
    pub name: String,
    search: MCTSSearch<Arc<T>>,
    game_config: &'static GameConfig,
    game_id: u64,
    /// The agent is responsible for accumulating MCTS data from rollouts.
    /// The data in this vector will have 0 values for the game result, and that field
    /// will be populated when the game is over.
    mcts_data: Vec<MCTSData>,
}

impl<T: InferenceClient + Send + Sync> MCTSAgent<T> {
    pub fn new(
        mcts_config: &'static MCTSConfig,
        game_config: &'static GameConfig,
        inference_client: Arc<T>,
    ) -> Self {
        Self {
            name: mcts_config.name.clone(),
            search: MCTSSearch::new(mcts_config, game_config, inference_client),
            game_config,
            game_id: rand::rng().random::<u64>(),
            mcts_data: Vec::new(),
        }
    }
}

#[async_trait]
impl<T> Agent for MCTSAgent<T>
where
    T: InferenceClient + Send + Sync + 'static,
    for<'a> T::EvaluationFuture<'a>: Send,
{
    fn name(&self) -> &str {
        &self.name
    }

    async fn choose_move(&mut self, state: &State) -> anyhow::Result<usize> {
        let result = self.search.choose_move(state).await?;
        if !result.is_fast_move {
            self.mcts_data.push(generate_mcts_data(
                &result.stats,
                self.game_id,
                state,
                self.game_config,
            )?);
        }
        Ok(result.move_index)
    }

    fn flush_mcts_data(&mut self) -> Vec<MCTSData> {
        self.mcts_data.drain(..).collect()
    }
}

fn generate_mcts_data(
    search_stats: &SearchStats,
    game_id: u64,
    state: &State,
    game_config: &'static GameConfig,
) -> Result<MCTSData> {
    let move_profiles = game_config.move_profiles()?;
    Ok(MCTSData {
        player: search_stats.player,
        turn: state.turn(),
        game_id,
        board: state
            .board()
            .clone_with_player_pov(search_stats.player as i32),
        valid_moves: search_stats
            .player_pov_move_indexes
            .iter()
            .map(|&x| x.into())
            .collect(),
        valid_move_tuples: search_stats
            .player_pov_move_indexes
            .iter()
            .map(|&index| {
                let move_profile = move_profiles.get(index);
                (
                    move_profile.piece_orientation_index,
                    move_profile.center.0,
                    move_profile.center.1,
                )
            })
            .collect::<Vec<(usize, usize, usize)>>(),
        visit_counts: search_stats
            .child_visit_counts
            .iter()
            .map(|&x| x as u32)
            .collect(),
        // This will be populated externally when the game is over.
        game_result: [0.0; NUM_PLAYERS],
        q_value: search_stats.q_value,
        piece_availability: state.piece_availability_player_pov(search_stats.player),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::config::DefaultExploitationValue;
    use crate::inference;
    use crate::inference::softmax_inplace;
    use crate::{config::NUM_PLAYERS, testing};
    use itertools::Itertools;

    struct MockInferenceClient {
        pub requests: Mutex<Vec<inference::Request>>,
    }

    impl inference::InferenceClient for MockInferenceClient {
        type EvaluationFuture<'a> = std::future::Ready<anyhow::Result<inference::Response>>;

        fn evaluate(&self, request: inference::Request) -> Self::EvaluationFuture<'_> {
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

            std::future::ready(Ok(inference::Response { value, policy }))
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
            policy_inference_config_name: "".to_string(),
            value_inference_config_name: "".to_string(),
            default_exploitation_value: DefaultExploitationValue::NetworkValue,
        }));
        let fast_client = Arc::new(MockInferenceClient {
            requests: Mutex::new(Vec::new()),
        });
        let mut fast_agent =
            MCTSAgent::new(fast_mcts_config, game_config, Arc::clone(&fast_client));
        let fast_state = State::new(game_config).unwrap();
        fast_agent.choose_move(&fast_state).await.unwrap();
        let fast_requests = fast_client.requests.lock().unwrap().len();
        assert!(fast_agent.flush_mcts_data().is_empty());

        // Two requests are made -- one for the initial node expansion, and
        // another for the single rollout.
        assert_eq!(fast_requests, 2);
    }

    #[tokio::test]
    async fn test_mcts_data_includes_piece_availability() {
        let mcts_config = testing::create_mcts_config(1, 0.0);
        let game_config = testing::create_half_game_config();
        let mock_client = Arc::new(MockInferenceClient {
            requests: Mutex::new(Vec::new()),
        });
        let mut agent = MCTSAgent::new(mcts_config, game_config, mock_client);
        let state = State::new(game_config).unwrap();

        agent.choose_move(&state).await.unwrap();
        let data = agent.flush_mcts_data();

        assert_eq!(data.len(), 1);
        assert_eq!(
            data[0].piece_availability,
            state.piece_availability_player_pov(state.player())
        );
        assert_eq!(data[0].piece_availability.len(), NUM_PLAYERS);
        for row in &data[0].piece_availability {
            assert_eq!(row.len(), game_config.num_pieces);
        }
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

        let mut state = State::new(&game_config).unwrap();
        let move_index_0 = agent.choose_move(&state).await.unwrap();
        let move_profile_0 = game_config.move_profiles().unwrap().get(move_index_0);

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
        state.apply_move(move_index_0).unwrap();
        let move_index_1 = agent.choose_move(&state).await.unwrap();
        let move_profile_1 = game_config.move_profiles().unwrap().get(move_index_1);

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
        state.apply_move(move_index_1).unwrap();
        let move_index_2 = agent.choose_move(&state).await.unwrap();
        let move_profile_2 = game_config.move_profiles().unwrap().get(move_index_2);

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
        state.apply_move(move_index_2).unwrap();

        let move_index_3 = agent.choose_move(&state).await.unwrap();
        let move_profile_3 = game_config.move_profiles().unwrap().get(move_index_3);

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

        state.apply_move(move_index_3).unwrap();
        assert_eq!(state.player(), 0);
        assert_eq!(state.board().slice(0).get((0, 0)), true);
        assert_eq!(state.board().slice(1).count(), 5);
        assert_eq!(state.board().slice(1).get((9, 0)), true);
        assert_eq!(state.board().slice(2).count(), 5);
        assert_eq!(state.board().slice(2).get((9, 9)), true);
        assert_eq!(state.board().slice(3).count(), 5);
        assert_eq!(state.board().slice(3).get((0, 9)), true);
    }

    struct ValuesInferenceClient {}
    impl inference::InferenceClient for ValuesInferenceClient {
        type EvaluationFuture<'a> = std::future::Ready<anyhow::Result<inference::Response>>;

        fn evaluate(&self, request: inference::Request) -> Self::EvaluationFuture<'_> {
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

            std::future::ready(Ok(inference::Response { value, policy }))
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
        let mut state = State::new(&game_config).unwrap();

        for player in 0..4 {
            assert_eq!(state.player(), player);
            let move_index = agent.choose_move(&state).await.unwrap();
            let move_profile = game_config.move_profiles().unwrap().get(move_index);
            assert_eq!(move_profile.occupied_cells.count(), 1);

            state.apply_move(move_index).unwrap();
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

        let state = State::new(&game_config).unwrap();

        // Run the test multiple times to make sure it consistently works
        for _ in 0..5 {
            let move_index = agent.choose_move(&state).await.unwrap();
            assert!(state.is_valid_move(move_index));
        }
    }
}
