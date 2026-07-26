use std::{future::Future, sync::Arc};

use alphablokus_game_core::{config::NUM_PLAYERS, game::Board};
use anyhow::Result;

#[derive(Debug, Clone, Hash)]
pub struct Request {
    pub board: Board,
    pub valid_move_indexes: Vec<usize>,
    pub piece_availability: Vec<Vec<u8>>,
}

#[derive(Debug, Clone)]
pub struct Response {
    pub value: [f32; NUM_PLAYERS],
    pub policy: Vec<f32>,
}

/// Inference boundary shared by native and browser runtimes.
pub trait InferenceClient {
    type EvaluationFuture<'a>: Future<Output = Result<Response>> + 'a
    where
        Self: 'a;

    fn evaluate(&self, request: Request) -> Self::EvaluationFuture<'_>;
}

impl<T: InferenceClient + ?Sized> InferenceClient for Arc<T> {
    type EvaluationFuture<'a>
        = T::EvaluationFuture<'a>
    where
        T: 'a;

    fn evaluate(&self, request: Request) -> Self::EvaluationFuture<'_> {
        self.as_ref().evaluate(request)
    }
}

pub fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max_x = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    for value in x.iter_mut() {
        *value = (*value - max_x).exp();
    }

    let sum: f32 = x.iter().sum();
    if sum == 0.0 {
        let probability = 1.0 / x.len() as f32;
        x.fill(probability);
    } else {
        for value in x.iter_mut() {
            *value /= sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmaxes_in_place() {
        let mut values = [1.0, 2.0, 3.0];
        softmax_inplace(&mut values);
        assert_eq!(values, [0.09003057, 0.24472847, 0.66524094]);
    }
}
