//! Architecture-stable serialization for the runtime-optimized move table.
//!
//! `MoveData` is the runtime-optimized representation used by the application.
//! The structs in this module define a versioned wire schema using fixed-width
//! integers and explicit `u64` bit blocks, so native and WASM builds
//! reconstruct identical `MoveData` values from the same bytes.

use std::io::{Read, Write};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

use crate::config::{GameConfig, NUM_PLAYERS};
use crate::game::{BoardSlice, MovesArray, MovesBitSet};

use super::{MoveData, MoveProfile};

const FORMAT_VERSION: u16 = 1;

#[derive(Serialize, Deserialize)]
struct DiskMoveData {
    version: u16,
    board_size: u16,
    num_moves: u32,
    num_pieces: u16,
    num_piece_orientations: u16,
    profiles: Vec<DiskMoveProfile>,
    initial_moves_enabled: [Vec<u64>; NUM_PLAYERS],
}

#[derive(Serialize, Deserialize)]
struct DiskMoveProfile {
    index: u32,
    occupied_cells: Vec<[u16; 2]>,
    center: [u16; 2],
    piece_orientation_index: u16,
    piece_index: u16,
    rotated_move_indexes: [u32; NUM_PLAYERS],
    moves_ruled_out_for_self: Vec<u64>,
    moves_ruled_out_for_others: Vec<u64>,
    moves_enabled_for_self: Vec<u64>,
}

/// Incrementally reconstructs the runtime-optimized table from the private
/// disk schema. Browser callers can yield between batches for progress UI.
pub struct MoveDataDecoder {
    profiles: std::vec::IntoIter<DiskMoveProfile>,
    built_profiles: Vec<MoveProfile>,
    initial_moves_enabled: [Vec<u64>; NUM_PLAYERS],
    total_profiles: usize,
}

impl MoveDataDecoder {
    pub fn total_profiles(&self) -> usize {
        self.total_profiles
    }

    pub fn completed_profiles(&self) -> usize {
        self.built_profiles.len()
    }

    pub fn remaining_profiles(&self) -> usize {
        self.profiles.len()
    }

    pub fn build_profiles(&mut self, count: usize, config: &GameConfig) -> Result<()> {
        for profile in self.profiles.by_ref().take(count) {
            let expected_index = self.built_profiles.len();
            ensure!(
                profile.index as usize == expected_index,
                "move profile index {} is out of sequence; expected {expected_index}",
                profile.index
            );
            self.built_profiles.push(
                profile
                    .into_runtime(config)
                    .with_context(|| format!("failed to decode move profile {expected_index}"))?,
            );
        }
        Ok(())
    }

    pub fn finish(self, config: &GameConfig) -> Result<MoveData> {
        ensure!(
            self.profiles.as_slice().is_empty(),
            "move table decoding is incomplete"
        );
        let [player_0, player_1, player_2, player_3] = self.initial_moves_enabled;
        Ok(MoveData {
            profiles: MovesArray::new_from_vec(self.built_profiles, config)?,
            initial_moves_enabled: [
                decode_bitset(player_0, config.num_moves)?,
                decode_bitset(player_1, config.num_moves)?,
                decode_bitset(player_2, config.num_moves)?,
                decode_bitset(player_3, config.num_moves)?,
            ],
        })
    }
}

impl DiskMoveData {
    fn from_runtime(data: &MoveData, config: &GameConfig) -> Result<Self> {
        Ok(Self {
            version: FORMAT_VERSION,
            board_size: config.board_size.try_into()?,
            num_moves: config.num_moves.try_into()?,
            num_pieces: config.num_pieces.try_into()?,
            num_piece_orientations: config.num_piece_orientations.try_into()?,
            profiles: data
                .profiles
                .iter()
                .map(DiskMoveProfile::from_runtime)
                .collect::<Result<_>>()?,
            initial_moves_enabled: std::array::from_fn(|player| {
                encode_bitset(&data.initial_moves_enabled[player], config.num_moves)
            }),
        })
    }

    fn into_decoder(self, config: &GameConfig) -> Result<MoveDataDecoder> {
        ensure!(
            self.version == FORMAT_VERSION,
            "unsupported move table format version: {}",
            self.version
        );
        ensure!(
            self.board_size as usize == config.board_size,
            "board size mismatch"
        );
        ensure!(
            self.num_moves as usize == config.num_moves,
            "move count mismatch"
        );
        ensure!(
            self.num_pieces as usize == config.num_pieces,
            "piece count mismatch"
        );
        ensure!(
            self.num_piece_orientations as usize == config.num_piece_orientations,
            "piece orientation count mismatch"
        );
        ensure!(
            self.profiles.len() == config.num_moves,
            "move profile count mismatch"
        );
        Ok(MoveDataDecoder {
            total_profiles: self.profiles.len(),
            profiles: self.profiles.into_iter(),
            built_profiles: Vec::with_capacity(config.num_moves),
            initial_moves_enabled: self.initial_moves_enabled,
        })
    }
}

impl DiskMoveProfile {
    fn from_runtime(profile: &MoveProfile) -> Result<Self> {
        let [rotation_0, rotation_1, rotation_2, rotation_3] = profile.rotated_move_indexes;
        Ok(Self {
            index: profile.index.try_into()?,
            occupied_cells: profile
                .occupied_cells
                .to_cells()
                .into_iter()
                .map(|(x, y)| Ok([x.try_into()?, y.try_into()?]))
                .collect::<Result<_>>()?,
            center: [profile.center.0.try_into()?, profile.center.1.try_into()?],
            piece_orientation_index: profile.piece_orientation_index.try_into()?,
            piece_index: profile.piece_index.try_into()?,
            rotated_move_indexes: [
                rotation_0.try_into()?,
                rotation_1.try_into()?,
                rotation_2.try_into()?,
                rotation_3.try_into()?,
            ],
            moves_ruled_out_for_self: encode_bitset(
                &profile.moves_ruled_out_for_self,
                profile.moves_ruled_out_for_self.num_moves(),
            ),
            moves_ruled_out_for_others: encode_bitset(
                &profile.moves_ruled_out_for_others,
                profile.moves_ruled_out_for_others.num_moves(),
            ),
            moves_enabled_for_self: encode_bitset(
                &profile.moves_enabled_for_self,
                profile.moves_enabled_for_self.num_moves(),
            ),
        })
    }

    fn into_runtime(self, config: &GameConfig) -> Result<MoveProfile> {
        ensure!(
            (self.index as usize) < config.num_moves,
            "move index is out of range"
        );
        ensure!(
            (self.piece_orientation_index as usize) < config.num_piece_orientations,
            "piece orientation index is out of range"
        );
        ensure!(
            (self.piece_index as usize) < config.num_pieces,
            "piece index is out of range"
        );
        let cells = self
            .occupied_cells
            .into_iter()
            .map(|[x, y]| [x as usize, y as usize])
            .collect::<Vec<_>>();
        ensure!(
            cells
                .iter()
                .all(|[x, y]| *x < config.board_size && *y < config.board_size),
            "occupied cell is outside the board"
        );
        let center = (self.center[0] as usize, self.center[1] as usize);
        ensure!(
            center.0 < config.board_size && center.1 < config.board_size,
            "move center is outside the board"
        );
        let rotated_move_indexes = self.rotated_move_indexes.map(|index| index as usize);
        ensure!(
            rotated_move_indexes
                .iter()
                .all(|&index| index < config.num_moves),
            "rotated move index is out of range"
        );
        Ok(MoveProfile {
            index: self.index as usize,
            occupied_cells: BoardSlice::from_cells(config.board_size, &cells),
            center,
            piece_orientation_index: self.piece_orientation_index as usize,
            piece_index: self.piece_index as usize,
            rotated_move_indexes,
            moves_ruled_out_for_self: decode_bitset(
                self.moves_ruled_out_for_self,
                config.num_moves,
            )?,
            moves_ruled_out_for_others: decode_bitset(
                self.moves_ruled_out_for_others,
                config.num_moves,
            )?,
            moves_enabled_for_self: decode_bitset(self.moves_enabled_for_self, config.num_moves)?,
        })
    }
}

pub fn decoder_from_slice(bytes: &[u8], config: &GameConfig) -> Result<MoveDataDecoder> {
    let disk: DiskMoveData = rmp_serde::from_slice(bytes)?;
    disk.into_decoder(config)
}

pub fn decode(bytes: &[u8], config: &GameConfig) -> Result<MoveData> {
    let mut decoder = decoder_from_slice(bytes, config)?;
    decoder.build_profiles(decoder.remaining_profiles(), config)?;
    decoder.finish(config)
}

pub fn encode<W: Write>(writer: &mut W, data: &MoveData, config: &GameConfig) -> Result<()> {
    rmp_serde::encode::write(writer, &DiskMoveData::from_runtime(data, config)?)?;
    Ok(())
}

pub fn decode_reader<R: Read>(reader: &mut R, config: &GameConfig) -> Result<MoveData> {
    let disk: DiskMoveData = rmp_serde::decode::from_read(reader)?;
    let mut decoder = disk.into_decoder(config)?;
    decoder.build_profiles(decoder.remaining_profiles(), config)?;
    decoder.finish(config)
}

fn encode_bitset(values: &MovesBitSet, num_moves: usize) -> Vec<u64> {
    let mut blocks = vec![0; num_moves.div_ceil(64)];
    for index in values.iter() {
        blocks[index / 64] |= 1u64 << (index % 64);
    }
    blocks
}

fn decode_bitset(blocks: Vec<u64>, num_moves: usize) -> Result<MovesBitSet> {
    ensure!(
        blocks.len() == num_moves.div_ceil(64),
        "move bitset block count mismatch"
    );
    if let Some(&last) = blocks.last() {
        let used_bits = num_moves % 64;
        if used_bits != 0 {
            ensure!(last >> used_bits == 0, "move bitset has out-of-range bits");
        }
    }
    let mut values = MovesBitSet::new(num_moves);
    for (block_index, mut block) in blocks.into_iter().enumerate() {
        while block != 0 {
            let offset = block.trailing_zeros() as usize;
            values.insert(block_index * 64 + offset);
            block &= block - 1;
        }
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::move_data_tools::generate;
    use crate::testing::create_game_config_without_data;

    #[test]
    fn stable_codec_round_trips_every_runtime_bit() -> Result<()> {
        let config = create_game_config_without_data();
        let original = generate(config)?;
        let mut encoded = Vec::new();
        encode(&mut encoded, &original, config)?;

        let decoded = decode(&encoded, config)?;
        assert_move_data_eq(&original, &decoded, config.num_moves);

        let mut reencoded = Vec::new();
        encode(&mut reencoded, &decoded, config)?;
        assert_eq!(encoded, reencoded);
        Ok(())
    }

    fn assert_move_data_eq(before: &MoveData, after: &MoveData, num_moves: usize) {
        assert_eq!(before.initial_moves_enabled, after.initial_moves_enabled);
        for index in 0..num_moves {
            let before = before.profiles.get(index);
            let after = after.profiles.get(index);
            assert_eq!(before.index, after.index);
            assert_eq!(before.occupied_cells, after.occupied_cells);
            assert_eq!(before.center, after.center);
            assert_eq!(
                before.piece_orientation_index,
                after.piece_orientation_index
            );
            assert_eq!(before.piece_index, after.piece_index);
            assert_eq!(before.rotated_move_indexes, after.rotated_move_indexes);
            assert_eq!(
                before.moves_ruled_out_for_self,
                after.moves_ruled_out_for_self
            );
            assert_eq!(
                before.moves_ruled_out_for_others,
                after.moves_ruled_out_for_others
            );
            assert_eq!(before.moves_enabled_for_self, after.moves_enabled_for_self);
        }
    }
}
