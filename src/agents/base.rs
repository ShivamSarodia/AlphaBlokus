use crate::game::State;

#[allow(async_fn_in_trait)]
pub trait Agent {
    async fn choose_move(&self, state: &State) -> usize;
}
