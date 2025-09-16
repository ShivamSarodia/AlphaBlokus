use itertools::Itertools;
use log::debug;
use rand_distr::Distribution;
use rand_distr::weighted::WeightedIndex;
use std::collections::HashMap;

use crate::config::GameConfig;
use crate::game::move_data::move_index_to_player_pov;
use crate::inference;
use crate::{config::MCTSConfig, config::NUM_PLAYERS, game::State};

pub struct Node {
    /// Player to move.
    player: usize,
    /// Number of valid moves for the player at this node.
    num_valid_moves: usize,
    /// Value of the node from network evaluation.
    value: [f32; NUM_PLAYERS],
    /// Mapping from move index to array index for this node. That is, only valid moves are
    /// stored in the Vec<> below to save memory, and this map is used to convert from a move
    /// index like 15,423 to the array index like 21 at which that move is stored.
    move_index_to_array_index: HashMap<usize, usize>,
    /// Mapping from array index to move index for this node.
    array_index_to_move_index: Vec<usize>,
    /// Mapping from array index to move index from player POV for this node.
    array_index_to_player_pov_move_index: Vec<usize>,
    /// Summed values for each child of this node, computed over time
    /// from backpropagation.
    children_value_sums: Vec<[f32; NUM_PLAYERS]>,
    /// Visit counts for each child of this node over backpropagation.
    children_visit_counts: Vec<u32>,
    /// Sum of the visit counts for all children of this node.
    children_visit_counts_sum: u32,
    /// Prior probabilities for each child of this node, computed from
    /// network policy evaluation.
    children_prior_probabilities: Vec<f32>,
    /// Children of this node. This vector is populated as children get get created through
    /// backpropagation.
    children: Vec<Option<Node>>,
    game_config: &'static GameConfig,
    mcts_config: &'static MCTSConfig,
}

impl Node {
    pub async fn build_and_expand<T: inference::Client>(
        state: &State<'_>,
        inference_client: &T,
        mcts_config: &'static MCTSConfig,
        game_config: &'static GameConfig,
        add_noise: bool,
    ) -> Self {
        let mut result = Self {
            // Initialized here.
            player: state.player(),
            game_config,
            mcts_config,
            // Initialized by initialize_move_mappings
            num_valid_moves: 0,
            move_index_to_array_index: HashMap::new(),
            array_index_to_move_index: Vec::new(),
            array_index_to_player_pov_move_index: Vec::new(),
            // Initialized by initialize_children
            children: Vec::new(),
            children_value_sums: vec![[0.0; NUM_PLAYERS]; 0],
            children_visit_counts: vec![0; 0],
            children_visit_counts_sum: 0,
            // Initialized by initialize_inference_results
            value: [0.0; 4],
            children_prior_probabilities: Vec::new(),
        };
        result.initialize_move_mappings(state);
        result.initialize_children();
        result
            .initialize_inference_results(state, inference_client)
            .await;
        if add_noise {
            result.add_noise();
        }
        result
    }

    fn add_noise(&mut self) {
        // Adds Dirichlet noise to the prior probabilities. The noise is computing using a Gamma
        // distribution because the Dirichlet distribution provided by rand_distr requires a compile
        // time constant distribution size.
        let per_move_alpha =
            self.mcts_config.total_dirichlet_noise_alpha / (self.num_valid_moves as f32);
        let gamma_dist = rand_distr::Gamma::<f32>::new(per_move_alpha, 1.0).unwrap();

        let unnormalized_dirichlet = (0..self.num_valid_moves)
            .map(|_| gamma_dist.sample(&mut rand::rng()))
            .collect::<Vec<f32>>();
        let normalization_factor = unnormalized_dirichlet.iter().sum::<f32>();

        self.children_prior_probabilities
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| {
                let noise = unnormalized_dirichlet[i] / normalization_factor;
                *x = *x * (1.0 - self.mcts_config.root_dirichlet_noise_fraction)
                    + noise * self.mcts_config.root_dirichlet_noise_fraction;
            });
    }

    fn initialize_move_mappings(&mut self, state: &State<'_>) {
        self.move_index_to_array_index = HashMap::new();
        self.array_index_to_move_index = Vec::new();
        self.array_index_to_player_pov_move_index = Vec::new();
        state
            .valid_moves()
            // I don't think sorting is technically necessary here, but it may reduce risk
            // of non-determinism.
            .sorted()
            .enumerate()
            .for_each(|(array_index, move_index)| {
                self.move_index_to_array_index
                    .insert(move_index, array_index);
                self.array_index_to_move_index.push(move_index);
                self.array_index_to_player_pov_move_index
                    .push(move_index_to_player_pov(
                        move_index,
                        self.player,
                        self.game_config.move_profiles(),
                    ));
            });
        self.num_valid_moves = self.array_index_to_move_index.len();
    }

    fn initialize_children(&mut self) {
        self.children = Vec::with_capacity(self.num_valid_moves);
        for _ in 0..self.num_valid_moves {
            self.children.push(None);
        }
        self.children_value_sums = vec![[0.0; NUM_PLAYERS]; self.num_valid_moves];
        self.children_visit_counts = vec![0; self.num_valid_moves];
        self.children_visit_counts_sum = 0;
    }

    async fn initialize_inference_results<T: inference::Client>(
        &mut self,
        state: &State<'_>,
        inference_client: &T,
    ) {
        // Pass the board and player POV move indexes to the network from the player's
        // own perspective.

        // Perform inference on this board.
        let inference_result = inference_client
            .evaluate(inference::Request {
                board: state.board().clone_with_player_pov(self.player as i32),
                valid_move_indexes: self.array_index_to_player_pov_move_index.clone(),
            })
            .await;

        // Store the results back into the node.
        self.set_value_from_player_pov(inference_result.value);
        self.set_prior_probabilities(inference_result.policy);
    }

    pub fn get_value_as_universal_pov(&self) -> [f32; NUM_PLAYERS] {
        self.value
    }

    #[allow(dead_code)]
    pub fn get_value_as_player_pov(&self) -> [f32; NUM_PLAYERS] {
        let mut value_clone = self.value;
        // Suppose self.player is 1. Then, self.value[1] should be returned as
        // result[0]. Rotating left accomplishes this.
        value_clone.rotate_left(self.player);
        value_clone
    }

    #[allow(dead_code)]
    fn set_value_from_universal_pov(&mut self, value: [f32; NUM_PLAYERS]) {
        self.value = value;
    }

    fn set_value_from_player_pov(&mut self, value: [f32; NUM_PLAYERS]) {
        self.value = value;
        // Suppose self.player is 1. Then, value[0] is the value for player 1. By
        // rotating right, we correctly move player 1's value in value[1].
        self.value.rotate_right(self.player);
    }

    /// Sets the prior probabilities for the child nodes. Expects `policy` to be provided
    /// in the same order as the move indexes in `self.array_index_to_move_index`.
    fn set_prior_probabilities(&mut self, policy: Vec<f32>) {
        if policy.len() != self.num_valid_moves {
            panic!(
                "Policy length {} does not match number of valid moves {}",
                policy.len(),
                self.num_valid_moves
            );
        }
        self.children_prior_probabilities = policy;
    }

    /// Returns the move index (in universal perspective) with the highest UCB score at
    /// this node.
    pub fn select_move_by_ucb(&self) -> usize {
        let mut max_score = f32::NEG_INFINITY;
        let mut max_index = 0;

        let exploration_scores = self.exploration_scores();
        let exploitation_scores = self.exploitation_scores();

        for index in 0..self.num_valid_moves {
            let score = exploration_scores[index] + exploitation_scores[index];
            if score > max_score {
                max_score = score;
                max_index = index;
            }
        }

        self.array_index_to_move_index[max_index]
    }

    fn exploration_scores(&self) -> Vec<f32> {
        let norm_factor = ((self.children_visit_counts_sum + 1) as f32).sqrt();
        self.children_prior_probabilities
            .iter()
            .enumerate()
            .map(|(i, prior_probability)| {
                (self.mcts_config.ucb_exploration_factor * prior_probability * norm_factor)
                    / (self.children_visit_counts[i] + 1) as f32
            })
            .collect::<Vec<f32>>()
    }

    fn exploitation_scores(&self) -> Vec<f32> {
        // Exploitation scores are between 0 and 1. 0 means the player has lost every game from this move,
        // while 1 means the player has won every game from this move.

        let mut result = Vec::with_capacity(self.num_valid_moves);
        for array_index in 0..self.num_valid_moves {
            let visit_count = self.children_visit_counts[array_index];
            if visit_count == 0 {
                // If this move has never been tried, assign a score based on the neural network
                // evaluation of this node.
                result.push(self.value[self.player]);
            } else {
                // Otherwise, assign a score by averaging the values of the child nodes.
                let score = self.children_value_sums[array_index][self.player] / visit_count as f32;
                result.push(score);
            }
        }
        result
    }

    pub fn select_move_to_play(&self, state: &State<'_>) -> usize {
        let temperature = if state.turn() < self.mcts_config.temperature_turn_cutoff {
            self.mcts_config.move_selection_temperature
        } else {
            0.0
        };

        // With temperature 0, just select the move with highest visit count.
        let array_index = if temperature.abs() < f32::EPSILON {
            self.children_visit_counts
                .iter()
                .enumerate()
                .max_by_key(|&(_, x)| x)
                .unwrap()
                .0
        } else {
            // Otherwise, remotely sample.
            let unnormalized_weights = self
                .children_visit_counts
                .iter()
                .map(|&x| (x as f32).powf(1.0 / temperature))
                .collect::<Vec<f32>>();

            let normalization_factor = unnormalized_weights.iter().sum::<f32>();
            let weights = unnormalized_weights
                .iter()
                .map(|x| x / normalization_factor)
                .collect::<Vec<f32>>();

            debug!(
                "Selecting move to play.\nWeights: {:?}\nUnnormalized Weights: {:?}\nPrior Probabilities: {:?}\nValue Sums: {:?}\nVisit Counts: {:?}\nMove Indexes: {:?}",
                weights,
                unnormalized_weights,
                self.children_prior_probabilities,
                self.children_value_sums,
                self.children_visit_counts,
                self.array_index_to_move_index
            );
            let dist = WeightedIndex::new(&weights).unwrap();
            dist.sample(&mut rand::rng())
        };

        self.array_index_to_move_index[array_index]
    }

    #[inline]
    pub fn get_array_index(&self, move_index: usize) -> usize {
        *self.move_index_to_array_index.get(&move_index).unwrap()
    }

    pub fn get_child(&self, move_index: usize) -> Option<&Node> {
        self.children[self.get_array_index(move_index)].as_ref()
    }

    pub fn get_child_mut(&mut self, move_index: usize) -> Option<&mut Node> {
        let array_index = self.get_array_index(move_index);
        self.children[array_index].as_mut()
    }

    pub fn has_child(&self, move_index: usize) -> bool {
        self.get_child(move_index).is_some()
    }

    pub fn add_child(&mut self, move_index: usize, child_node: Self) {
        let array_index = self.get_array_index(move_index);
        self.children[array_index] = Some(child_node);
    }

    pub fn increment_child_value_sum(&mut self, move_index: usize, values: [f32; NUM_PLAYERS]) {
        let array_index = self.get_array_index(move_index);
        values.iter().enumerate().for_each(|(i, &value)| {
            self.children_value_sums[array_index][i] += value;
        });
    }

    pub fn increment_child_visit_count(&mut self, move_index: usize) {
        let array_index = self.get_array_index(move_index);
        self.children_visit_counts[array_index] += 1;
        self.children_visit_counts_sum += 1;
    }
}
