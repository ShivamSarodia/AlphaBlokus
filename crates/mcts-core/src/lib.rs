//! Target-portable Monte Carlo tree search.

mod config;
mod inference;
mod node;
mod search;

pub use config::{DefaultExploitationValue, MCTSConfig};
pub use inference::{InferenceClient, Request, Response, softmax_inplace};
pub use search::{MCTSSearch, MCTSSearchResult, SearchStats};
