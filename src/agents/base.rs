use crate::game::State;

pub trait Agent {
    fn choose_move(&self, state: &State) -> usize;
}
