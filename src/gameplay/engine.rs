use std::collections::HashMap;
use std::sync::Arc;

use tokio::task::JoinSet;

use crate::agents::{Agent, MCTSAgent, RandomAgent};
use crate::config::{AgentConfig, AgentGroupConfig, GameConfig, NUM_PLAYERS};
use crate::game::{GameStatus, State};
use crate::inference::DefaultClient;

pub struct Engine {
    num_concurrent_games: u32,
    num_total_games: u32,
    inference_clients: HashMap<String, Arc<DefaultClient>>,
    game_config: &'static GameConfig,
    agent_group_config: &'static AgentGroupConfig,
    num_finished_games: u32,
}

impl Engine {
    pub fn new(
        num_concurrent_games: u32,
        num_total_games: u32,
        inference_clients: HashMap<String, Arc<DefaultClient>>,
        game_config: &'static GameConfig,
        agent_group_config: &'static AgentGroupConfig,
    ) -> Self {
        Self {
            num_concurrent_games,
            num_total_games,
            inference_clients,
            game_config,
            agent_group_config,
            num_finished_games: 0,
        }
    }

    fn maybe_spawn_game_on_join_set(&mut self, join_set: &mut JoinSet<()>) {
        // Only spawn a game if we haven't already spawned the requested
        // number of games.
        if self.num_finished_games >= self.num_total_games {
            return;
        }
        self.num_finished_games += 1;

        join_set.spawn({
            let game_config = self.game_config;
            let agents = self.generate_agents();
            async move {
                play_one_game(game_config, agents).await;
            }
        });
    }

    fn generate_agents(&self) -> [Box<dyn Agent>; NUM_PLAYERS] {
        match self.agent_group_config {
            AgentGroupConfig::Single(agent_config) => {
                std::array::from_fn(|_| self.generate_single_agent(agent_config))
            }
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
            // Raise any error from the join_next.
            result.unwrap();

            println!("Finished game");
            self.maybe_spawn_game_on_join_set(&mut join_set);
        }

        self.num_finished_games
    }
}

pub async fn play_one_game(
    game_config: &'static GameConfig,
    mut agents: [Box<dyn Agent>; NUM_PLAYERS],
) {
    let mut state = State::new(game_config);
    loop {
        let agent = &mut agents[state.player()];
        let move_index = agent.choose_move(&state).await;
        let game_state = state.apply_move(move_index);
        if game_state == GameStatus::GameOver {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing;

    #[tokio::test]
    async fn test_play_games() {
        let expected_num_finished_games = 50;
        let game_config = testing::create_game_config();
        let mut engine = Engine::new(
            5,
            expected_num_finished_games,
            HashMap::new(),
            game_config,
            &AgentGroupConfig::Single(AgentConfig::Random),
        );

        let num_finished_games = engine.play_games().await;
        assert_eq!(num_finished_games, expected_num_finished_games);
    }

    mod generate_agents {
        use super::*;

        #[test]
        fn test_single_agent_group_config() {
            let game_config = testing::create_game_config();
            let engine = Engine::new(
                5,
                50,
                HashMap::new(),
                game_config,
                &AgentGroupConfig::Single(AgentConfig::Random),
            );

            let agents = engine.generate_agents();
            assert_eq!(agents.len(), 4);
        }
    }
}
