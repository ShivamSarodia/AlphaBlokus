use anyhow::Result;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use zstd::stream::{read::Decoder, write::Encoder};

use crate::game::move_data::MoveData;

pub fn save(move_profiles: MoveData, output_file: &str) -> Result<()> {
    println!("Saving move profiles...");

    // Open file with buffered writer.
    let file = File::create(output_file)?;
    let buf = BufWriter::new(file);
    let mut enc = Encoder::new(buf, 6)?;

    rmp_serde::encode::write(&mut enc, &move_profiles)?;

    let mut buf = enc.finish()?;
    buf.flush()?;

    println!("Wrote move profiles to disk at {}", output_file);

    println!("Finished!");
    Ok(())
}

pub fn load(input_file: &str) -> Result<MoveData> {
    println!("Loading move profiles...");

    // Open file with buffered reader.
    let file = File::open(input_file)?;
    let buf = BufReader::new(file);
    let mut dec = Decoder::new(buf)?;

    let move_profiles: MoveData = rmp_serde::decode::from_read(&mut dec)?;

    println!("Loaded move profiles from disk at {}", input_file);
    Ok(move_profiles)
}
