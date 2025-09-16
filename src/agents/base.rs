use crate::game::State;
use async_trait::async_trait;

#[async_trait]
pub trait Agent {
    async fn choose_move(&self, state: &State) -> usize;
}
