use anyhow::Result;
use bincode;
use std::fs::File;
use std::io::{BufWriter, Write};
use zstd::stream::write::Encoder;

use crate::game::MovesArray;
use crate::move_profile::MoveProfile;

pub fn save(move_profiles: MovesArray<MoveProfile>, output_file: &str) -> Result<()> {
    println!("Saving move profiles...");

    // Open file with buffered writer.
    let file = File::create(output_file)?;
    let buf = BufWriter::new(file);
    let mut enc = Encoder::new(buf, 6)?;

    let bincode_config = bincode::config::standard().with_variable_int_encoding();
    bincode::serde::encode_into_std_write(&move_profiles, &mut enc, bincode_config)?;

    let mut buf = enc.finish()?;
    buf.flush()?;

    println!("Wrote move profiles to disk at {}", output_file);

    println!("Finished!");
    Ok(())
}

pub fn load(_input_file: &str) -> Result<MovesArray<MoveProfile>> {
    panic!("Not implemented");
}
