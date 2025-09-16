use crate::game::State;
use async_trait::async_trait;

#[async_trait]
pub trait Agent: Send + Sync {
    async fn choose_move(&self, state: &State) -> usize;
}
