use crate::{game::State, recorder::MCTSData};
use async_trait::async_trait;

#[async_trait]
pub trait Agent: Send + Sync {
    /// Ask this agent to choose a move for the current player.
    ///
    /// The agent can assume the move selected is then played by the engine.
    async fn choose_move(&mut self, state: &State) -> usize;

    /// Report a move made by a different agent instance to this agent.
    ///
    /// The state provided is the state before the move was applied.
    async fn report_move(&mut self, _state: &State, _move_index: usize) {}

    /// Return a vector of game data up to this point. The game result
    /// will be empty and is expected to be populated by the caller.
    fn flush_mcts_data(&mut self) -> Vec<MCTSData> {
        Vec::new()
    }

    /// Return the name of this agent.
    fn name(&self) -> &str;
}
