use tokio::task::JoinSet;

use crate::agents::{Agent, RandomAgent};
use crate::config::GameConfig;
use crate::game::{GameStatus, State};

pub struct Engine {
    num_concurrent_games: u32,
    num_total_games: u32,
    game_config: &'static GameConfig,
    num_finished_games: u32,
}

impl Engine {
    pub fn new(
        num_concurrent_games: u32,
        num_total_games: u32,
        game_config: &'static GameConfig,
    ) -> Self {
        Self {
            num_concurrent_games,
            num_total_games,
            game_config,
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
            async move {
                play_one_game(game_config).await;
            }
        });
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

pub async fn play_one_game(game_config: &'static GameConfig) {
    let mut agents = [
        RandomAgent::default(),
        RandomAgent::default(),
        RandomAgent::default(),
        RandomAgent::default(),
    ];

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
