//! Native-only move-table generation and filesystem adapters.
//!
//! The `move_data` module intentionally remains target-neutral: it owns the
//! runtime data model and stable decoder used by both native and WASM builds.
//! This module contains preprocessing, compressed-file I/O, and progress UI.

mod generate;
mod initial_moves_enabled;
mod pieces;
mod serialize;
mod stage_1;
mod stage_2;
mod stage_3;
mod stage_4;

pub use generate::generate;
pub use serialize::{load, save};
