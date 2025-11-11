use crate::{game::State, recorder::MCTSData};
use async_trait::async_trait;

#[async_trait]
pub trait Agent: Send + Sync {
    async fn choose_move(&mut self, state: &State) -> usize;

    /// Return a vector of game data up to this point. The game result
    /// will be empty and is expected to be populated by the caller.
    fn flush_mcts_data(&mut self) -> Vec<MCTSData> {
        Vec::new()
    }

    /// Return the name of this agent.
    fn name(&self) -> &str;
}
