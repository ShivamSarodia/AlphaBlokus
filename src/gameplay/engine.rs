use std::collections::HashMap;
use std::sync::Arc;

use tokio::task::JoinSet;

use crate::agents::{Agent, MCTSAgent, RandomAgent};
use crate::config::{AgentConfig, AgentGroupConfig, GameConfig, NUM_PLAYERS};
use crate::game::{GameStatus, State};
use crate::inference::DefaultClient;
use crate::recorder::Recorder;

pub struct Engine {
    num_concurrent_games: u32,
    num_total_games: u32,
    inference_clients: HashMap<String, Arc<DefaultClient>>,
    game_config: &'static GameConfig,
    agent_group_config: &'static AgentGroupConfig,
    num_started_games: u32,
    num_finished_games: u32,
    recorder: Recorder,
}

impl Engine {
    pub fn new(
        num_concurrent_games: u32,
        num_total_games: u32,
        inference_clients: HashMap<String, Arc<DefaultClient>>,
        game_config: &'static GameConfig,
        agent_group_config: &'static AgentGroupConfig,
        recorder: Recorder,
    ) -> Self {
        Self {
            num_concurrent_games,
            num_total_games,
            inference_clients,
            game_config,
            agent_group_config,
            num_started_games: 0,
            num_finished_games: 0,
            recorder,
        }
    }

    fn maybe_spawn_game_on_join_set(&mut self, join_set: &mut JoinSet<()>) {
        // Only spawn a game if we haven't already spawned the requested
        // number of games.
        if self.num_started_games >= self.num_total_games {
            return;
        }
        self.num_started_games += 1;

        join_set.spawn({
            let game_config = self.game_config;
            let (agent_vector, player_to_agent_index) = self.generate_agents();

            // The recorder itself is quite lightweight (just a MPSC channel), so it's fine to
            // clone here.
            let recorder = self.recorder.clone();
            async move {
                play_one_game(game_config, agent_vector, player_to_agent_index, recorder).await;
            }
        });
    }

    /// Return a vector of agents and an array mapping each player to the index of its
    /// agent in the vector.
    fn generate_agents(&self) -> (Vec<Box<dyn Agent>>, [usize; NUM_PLAYERS]) {
        match self.agent_group_config {
            AgentGroupConfig::Single(agent_config) => (
                vec![self.generate_single_agent(agent_config)],
                [0; NUM_PLAYERS],
            ),
        }
    }

    fn generate_single_agent(&self, agent_config: &'static AgentConfig) -> Box<dyn Agent> {
        match agent_config {
            AgentConfig::MCTS(mcts_config) => Box::new(MCTSAgent::new(
                mcts_config,
                self.game_config,
                Arc::clone(&self.inference_clients[&mcts_config.inference_config_name]),
            )),
            AgentConfig::Random => Box::new(RandomAgent::default()),
        }
    }

    pub async fn play_games(&mut self) -> u32 {
        let mut join_set = JoinSet::new();

        for _ in 0..self.num_concurrent_games {
            self.maybe_spawn_game_on_join_set(&mut join_set);
        }

        while let Some(result) = join_set.join_next().await {
            self.num_finished_games += 1;

            // Raise any error from the join_next.
            result.unwrap();

            println!(
                "Finished game. (total started: {:?}, total finished: {:?})",
                self.num_started_games, self.num_finished_games
            );
            self.maybe_spawn_game_on_join_set(&mut join_set);
        }

        self.num_finished_games
    }
}

pub async fn play_one_game(
    game_config: &'static GameConfig,
    mut agents: Vec<Box<dyn Agent>>,
    player_to_agent_index: [usize; NUM_PLAYERS],
    recorder: Recorder,
) {
    let mut state = State::new(game_config);
    loop {
        let agent = &mut agents[player_to_agent_index[state.player()]];
        let move_index = agent.choose_move(&state).await;
        let game_state = state.apply_move(move_index);
        if game_state == GameStatus::GameOver {
            break;
        }
    }

    // Accumulate game data from all agents. (In practice, usually only one
    // agent will have MCTS data to report.)
    let mut mcts_data = Vec::new();
    for mut agent in agents {
        mcts_data.extend(agent.flush_mcts_data());
    }

    // Populate the game result in all game data.
    let result = state.result();
    for gd in &mut mcts_data {
        // Set the game result from the state result.
        gd.game_result = result;

        // Then, rotate the game result into the player's POV.
        // E.g. if the current player is 1, then the value in result[1] should
        // be moved into result[0].
        gd.game_result.rotate_left(gd.player);
    }

    // Queue up the game data for recording.
    recorder.push_mcts_data(mcts_data);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::MCTSConfig, inference::OrtExecutor, recorder::read_mcts_data_from_disk, testing,
    };

    #[tokio::test]
    async fn test_play_games() {
        let expected_num_finished_games = 50;
        let game_config = testing::create_game_config();
        let directory = testing::create_tmp_directory();
        let mut engine = Engine::new(
            5,
            expected_num_finished_games,
            HashMap::new(),
            game_config,
            &AgentGroupConfig::Single(AgentConfig::Random),
            Recorder::build_and_start(1, directory),
        );

        let num_finished_games = engine.play_games().await;
        assert_eq!(num_finished_games, expected_num_finished_games);
    }

    #[tokio::test]
    async fn test_mcts_data_writing() {
        let game_config = testing::create_game_config();
        let directory = testing::create_tmp_directory();
        let inference_client = Arc::new(DefaultClient::build_and_start(
            OrtExecutor::build(&"static/networks/trivial_net_tiny.onnx", game_config),
            100,
            1,
        ));
        let recorder = Recorder::build_and_start(1, directory.clone());
        let agent_group_config = AgentGroupConfig::Single(AgentConfig::MCTS(MCTSConfig {
            num_rollouts: 10,
            total_dirichlet_noise_alpha: 1.0,
            root_dirichlet_noise_fraction: 0.0,
            ucb_exploration_factor: 1.0,
            temperature_turn_cutoff: 10,
            move_selection_temperature: 0.0,
            inference_config_name: "default".to_string(),
        }));
        let mut engine = Engine::new(
            1,
            1,
            HashMap::from([("default".to_string(), Arc::clone(&inference_client))]),
            game_config,
            Box::leak(Box::new(agent_group_config)),
            recorder,
        );

        engine.play_games().await;

        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        // Ensure there's a file written.
        let mut files = std::fs::read_dir(&directory).unwrap();
        let file = files.next().unwrap().unwrap();

        // Read the written file.
        let mcts_data_vec = read_mcts_data_from_disk(file.path().to_str().unwrap()).unwrap();

        let first_mcts_data = &mcts_data_vec[0];

        // Compute the player's scores.
        assert_eq!(first_mcts_data.player, 0);
        let mut player_scores = [0.0; NUM_PLAYERS];
        for player in 0..NUM_PLAYERS {
            player_scores[player] += first_mcts_data.game_result[player];
        }

        // The player scores should match up for all rows of data.
        for mcts_data in &mcts_data_vec {
            assert_eq!(mcts_data.game_result[0], player_scores[mcts_data.player]);
            assert_eq!(
                mcts_data.game_result[1],
                player_scores[(mcts_data.player + 1) % NUM_PLAYERS]
            );
            assert_eq!(
                mcts_data.game_result[2],
                player_scores[(mcts_data.player + 2) % NUM_PLAYERS]
            );
            assert_eq!(
                mcts_data.game_result[3],
                player_scores[(mcts_data.player + 3) % NUM_PLAYERS]
            );
        }

        // Identify the move index of Player 0 starting with a 1x1 square.
        let initial_moves_enabled = &game_config
            .move_data
            .as_ref()
            .unwrap()
            .initial_moves_enabled[0];
        let one_cell_move_index = game_config
            .move_profiles()
            .iter()
            .find(|mp| mp.occupied_cells.count() == 1 && initial_moves_enabled.contains(mp.index))
            .unwrap()
            .index;

        let mut players_seen = [false; NUM_PLAYERS];
        for mcts_data in &mcts_data_vec {
            if players_seen[mcts_data.player] {
                continue;
            }

            // The first time we see a player, they must have 0 as a valid move.
            // This tests that the reported valid moves are rotated into the player's POV
            // correctly.
            assert!(mcts_data.valid_moves.contains(&one_cell_move_index));

            players_seen[mcts_data.player] = true;
        }
    }
}
