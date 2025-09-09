use anyhow::Result;
use bincode;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use zstd::stream::{read::Decoder, write::Encoder};

use crate::move_data::MoveData;

fn bincode_config() -> impl bincode::config::Config {
    bincode::config::standard().with_variable_int_encoding()
}

pub fn save(move_profiles: MoveData, output_file: &str) -> Result<()> {
    println!("Saving move profiles...");

    // Open file with buffered writer.
    let file = File::create(output_file)?;
    let buf = BufWriter::new(file);
    let mut enc = Encoder::new(buf, 6)?;

    bincode::serde::encode_into_std_write(&move_profiles, &mut enc, bincode_config())?;

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

    let move_profiles: MoveData = bincode::serde::decode_from_std_read(&mut dec, bincode_config())?;

    println!("Loaded move profiles from disk at {}", input_file);
    Ok(move_profiles)
}
