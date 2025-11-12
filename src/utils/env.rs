use anyhow::{Context, Result};

pub fn load_env() -> Result<()> {
    match dotenvy::dotenv() {
        Ok(_) => Ok(()),
        Err(e) if e.not_found() => {
            tracing::warn!("No .env file found. Skipped loading environment variables.");
            Ok(())
        }
        Err(e) => Err(anyhow::Error::new(e)).context("Failed to load .env file"),
    }
}
