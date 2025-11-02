use anyhow::Result;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use zstd::stream::{read::Decoder, write::Encoder};

use crate::game::move_data::MoveData;

pub fn save<P: AsRef<Path>>(move_profiles: MoveData, output_file: P) -> Result<()> {
    let path = output_file.as_ref();
    println!("Saving move profiles...");

    // Open file with buffered writer.
    let file = File::create(path)?;
    let buf = BufWriter::new(file);
    let mut enc = Encoder::new(buf, 6)?;

    rmp_serde::encode::write(&mut enc, &move_profiles)?;

    let mut buf = enc.finish()?;
    buf.flush()?;

    println!("Wrote move profiles to disk at {}", path.display());

    println!("Finished!");
    Ok(())
}

pub fn load<P: AsRef<Path>>(input_file: P) -> Result<MoveData> {
    let path = input_file.as_ref();
    tracing::info!("Loading move profiles...");

    // Open file with buffered reader.
    let file = File::open(path)?;
    let buf = BufReader::new(file);
    let mut dec = Decoder::new(buf)?;

    let move_profiles: MoveData = rmp_serde::decode::from_read(&mut dec)?;

    tracing::info!("Loaded move profiles from disk at {}", path.display());
    Ok(move_profiles)
}
