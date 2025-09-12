use crate::config::{GameConfig, NUM_PLAYERS};
use crate::game::MovesBitSet;
use crate::game::display::{BoardDisplay, BoardDisplayLayer, BoardDisplayShape};
use crate::game::{Board, BoardSlice};

use std::fmt;

#[derive(Debug, PartialEq, Eq)]
pub enum GameStatus {
    InProgress,
    GameOver,
}

pub struct State<'c> {
    board: Board,
    player: usize,
    turn: u8,
    // Stored as a tuple of (move_index, player).
    last_move_played: Option<(usize, usize)>,
    moves_enabled: [MovesBitSet; NUM_PLAYERS],
    moves_ruled_out: [MovesBitSet; NUM_PLAYERS],
    game_config: &'c GameConfig,
}

impl<'c> State<'c> {
    pub fn new(game_config: &'c GameConfig) -> Self {
        State {
            board: Board::new(game_config),
            player: 0,
            turn: 0,
            last_move_played: None,
            moves_enabled: game_config.cloned_initial_moves_enabled(),
            moves_ruled_out: std::array::from_fn(|_| MovesBitSet::new(game_config.num_moves)),
            game_config,
        }
    }

    pub fn apply_move(&mut self, move_index: usize) -> GameStatus {
        let move_profile = self.game_config.move_profiles().get(move_index);

        // Update the board occupancies with the newly occupied cells of the
        // selected move.
        self.board.add(self.player, &move_profile.occupied_cells);

        // Update moves enabled for player.
        self.moves_enabled[self.player].add_mut(&move_profile.moves_enabled_for_self);

        // Update moves ruled out for the player.
        self.moves_ruled_out[self.player].add_mut(&move_profile.moves_ruled_out_for_self);

        // Update moves ruled out for other players.
        for other_player in 0..NUM_PLAYERS {
            if other_player != self.player {
                self.moves_ruled_out[other_player]
                    .add_mut(&move_profile.moves_ruled_out_for_others);
            }
        }

        // Update turn
        self.turn += 1;

        // Update the last move played, for display purposes.
        self.last_move_played = Some((move_index, self.player));

        // Set the player to the next player who has valid moves.
        // If no player does, return GAME_OVER.
        for _ in 0..NUM_PLAYERS {
            self.player = (self.player + 1) % NUM_PLAYERS;
            if self.any_valid_moves() {
                return GameStatus::InProgress;
            }
        }
        GameStatus::GameOver
    }

    pub fn valid_moves(&self) -> impl Iterator<Item = usize> {
        self.moves_enabled[self.player].subtract_iter(&self.moves_ruled_out[self.player])
    }

    pub fn any_valid_moves(&self) -> bool {
        return self.valid_moves().next().is_some();
    }

    pub fn first_valid_move(&self) -> Option<usize> {
        self.valid_moves().next()
    }

    pub fn player(&self) -> usize {
        self.player
    }

    pub fn turn(&self) -> u8 {
        self.turn
    }

    pub fn board(&self) -> &Board {
        &self.board
    }
}

impl<'c> fmt::Display for State<'c> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut layers = Vec::new();
        let mut latest_move_slice = BoardSlice::new(self.game_config.board_size);

        if let Some((move_index, player)) = self.last_move_played {
            let move_profile = self.game_config.move_profiles().get(move_index);
            latest_move_slice.add(&move_profile.occupied_cells);
            layers.push(BoardDisplayLayer {
                color: BoardDisplay::player_to_color(player),
                shape: BoardDisplayShape::Secondary,
                board_slice: &latest_move_slice,
            });
        }

        for player in 0..NUM_PLAYERS {
            layers.push(BoardDisplayLayer {
                color: BoardDisplay::player_to_color(player),
                board_slice: self.board.slice(player),
                shape: BoardDisplayShape::Primary,
            });
        }

        let board_display = BoardDisplay::new(layers).render();

        f.write_str(&format!(
            "Player: {}, Turn: {}\n{}",
            self.player, self.turn, board_display
        ))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::create_game_config;

    #[test]
    fn test_state_new_initialization() {
        let config = create_game_config();
        let state = State::new(config);

        assert_eq!(state.player(), 0);
        assert_eq!(state.turn(), 0);
        assert!(state.any_valid_moves());
    }

    #[test]
    fn test_apply_move_updates_state() {
        let config = create_game_config();
        let mut state = State::new(config);

        let move_index = state.first_valid_move().expect("Should have valid moves");
        let status = state.apply_move(move_index);

        // Should still be in progress after first move
        assert_eq!(status, GameStatus::InProgress);

        // Turn should increment
        assert_eq!(state.turn(), 1);

        // Player should change
        assert_ne!(state.player(), 0);
    }

    #[test]
    fn test_apply_multiple_moves() {
        let config = create_game_config();
        let mut state = State::new(config);

        loop {
            let move_index = state.first_valid_move().expect("Should have valid moves");
            let game_state = state.apply_move(move_index);
            if game_state == GameStatus::GameOver {
                break;
            }
        }

        assert_eq!(state.any_valid_moves(), false);
        assert_eq!(state.valid_moves().count(), 0);
    }

    #[test]
    fn test_display_implementation() {
        let config = create_game_config();
        let mut state = State::new(config);

        let display_string = format!("{}", state);
        assert!(!display_string.is_empty());

        let move_index = state.first_valid_move().expect("Should have valid moves");

        state.apply_move(move_index);

        let display_string = format!("{}", state);
        assert!(!display_string.is_empty());
    }

    #[test]
    fn test_first_valid_move_same() {
        let config = create_game_config();
        let state = State::new(config);

        let move_index_1 = state.first_valid_move().expect("Should have valid moves");
        let move_index_2 = state.first_valid_move().expect("Should have valid moves");

        assert_eq!(move_index_1, move_index_2);
    }
}
