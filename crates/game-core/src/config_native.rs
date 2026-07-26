//! Native-only `GameConfig` adapters.

use anyhow::{Context, Result};

use crate::config::GameConfig;
use crate::game::move_data_tools;

impl GameConfig {
    /// Load the architecture-stable move-table format from disk.
    pub fn load_move_profiles(&mut self) -> Result<()> {
        let move_data =
            move_data_tools::load(self.move_data_file.as_path(), self).with_context(|| {
                format!(
                    "Failed to load move profiles from file: {}",
                    self.move_data_file.display()
                )
            })?;
        self.move_data = Some(move_data);
        Ok(())
    }
}
