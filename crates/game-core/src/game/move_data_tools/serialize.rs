use anyhow::Result;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use zstd::stream::{read::Decoder, write::Encoder};

use crate::config::GameConfig;
use crate::game::move_data::MoveData;
use crate::game::move_data::codec;

pub fn save<P: AsRef<Path>>(
    move_profiles: &MoveData,
    game_config: &GameConfig,
    output_file: P,
) -> Result<()> {
    let path = output_file.as_ref();
    println!("Saving move profiles...");

    // Open file with buffered writer.
    let file = File::create(path)?;
    let buf = BufWriter::new(file);
    let mut enc = Encoder::new(buf, 6)?;

    codec::encode(&mut enc, move_profiles, game_config)?;

    let mut buf = enc.finish()?;
    buf.flush()?;

    println!("Wrote move profiles to disk at {}", path.display());

    println!("Finished!");
    Ok(())
}

pub fn load<P: AsRef<Path>>(input_file: P, game_config: &GameConfig) -> Result<MoveData> {
    let path = input_file.as_ref();
    tracing::info!("Loading move profiles...");

    // Open file with buffered reader.
    let file = File::open(path)?;
    let buf = BufReader::new(file);
    let mut dec = Decoder::new(buf)?;

    let move_profiles = codec::decode_reader(&mut dec, game_config)?;

    tracing::info!("Loaded move profiles from disk at {}", path.display());
    Ok(move_profiles)
}
