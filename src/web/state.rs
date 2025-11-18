use std::{collections::HashMap, sync::Arc};

use anyhow::{Context, Result};
use tokio::sync::{Mutex, MutexGuard, mpsc, oneshot};
use tokio_util::sync::CancellationToken;
use tracing::warn;

use crate::{
    agents::{Agent, MCTSAgent, RandomAgent},
    config::{AgentConfig, GameConfig, InferenceConfig, WebPlayConfig},
    game::State as BlokusState,
    inference::DefaultClient,
};

#[derive(Clone)]
pub struct AppState {
    pub config: &'static WebPlayConfig,
    session: Arc<Mutex<GameSession>>,
    agents: Arc<Vec<AgentEntry>>,
}

impl AppState {
    pub async fn build(config: &'static WebPlayConfig) -> Result<Self> {
        let inference_clients = build_inference_clients(&config.inference, &config.game).await?;
        let agent_entries = build_agent_entries(&config.agents, &config.game, &inference_clients)?;
        let agents = Arc::new(agent_entries);
        Ok(Self {
            config,
            session: Arc::new(Mutex::new(GameSession::new(&config.game))),
            agents,
        })
    }

    pub async fn session(&self) -> MutexGuard<'_, GameSession> {
        self.session.lock().await
    }

    pub fn agent_names(&self) -> Vec<String> {
        self.agents.iter().map(|agent| agent.name.clone()).collect()
    }

    pub async fn start_agent_move(&self, agent_name: &str) -> Result<(), AgentRunError> {
        let agent = self
            .agents
            .iter()
            .find(|candidate| candidate.name == agent_name)
            .cloned()
            .ok_or(AgentRunError::AgentNotFound)?;

        let snapshot = {
            let mut session = self.session().await;
            if session.has_pending_agent() {
                return Err(AgentRunError::AgentBusy);
            }
            if !session.state.any_valid_moves() {
                return Err(AgentRunError::NoMovesAvailable);
            }
            session.set_pending_agent(&agent.name);
            session.state.clone()
        };

        let (reply_sender, reply_receiver) = oneshot::channel();
        agent
            .sender
            .send(AgentJob {
                state: snapshot,
                reply: reply_sender,
            })
            .map_err(|_| AgentRunError::AgentFailed)?;

        let session = Arc::clone(&self.session);
        let agent_name = agent.name.clone();
        tokio::spawn(async move {
            let move_index = reply_receiver
                .await
                .expect("Agent failed to produce a move");

            let mut guard = session.lock().await;
            if guard.pending_agent() == Some(agent_name.as_str()) {
                if guard.state.is_valid_move(move_index) {
                    guard.state.apply_move(move_index);
                } else {
                    warn!(
                        agent = %agent_name,
                        move_index,
                        "Agent produced invalid move"
                    );
                }
                guard.clear_pending_agent();
            }
        });

        Ok(())
    }

    pub async fn reset(&self) {
        let mut session = self.session().await;
        session.reset();
    }
}

#[derive(Debug)]
pub enum AgentRunError {
    AgentNotFound,
    AgentBusy,
    NoMovesAvailable,
    AgentFailed,
}

pub struct GameSession {
    pub state: BlokusState,
    pending_agent: Option<String>,
    game_config: &'static GameConfig,
}

impl GameSession {
    fn new(game_config: &'static GameConfig) -> Self {
        Self {
            state: BlokusState::new(game_config),
            pending_agent: None,
            game_config,
        }
    }

    pub fn pending_agent(&self) -> Option<&str> {
        self.pending_agent.as_deref()
    }

    pub fn has_pending_agent(&self) -> bool {
        self.pending_agent.is_some()
    }

    fn set_pending_agent(&mut self, name: &str) {
        self.pending_agent = Some(name.to_string());
    }

    fn clear_pending_agent(&mut self) {
        self.pending_agent = None;
    }

    fn reset(&mut self) {
        self.state = BlokusState::new(self.game_config);
        self.clear_pending_agent();
    }
}

#[derive(Clone)]
struct AgentEntry {
    name: String,
    sender: mpsc::UnboundedSender<AgentJob>,
}

struct AgentJob {
    state: BlokusState,
    reply: oneshot::Sender<usize>,
}

async fn build_inference_clients(
    inference_configs: &'static [InferenceConfig],
    game_config: &'static GameConfig,
) -> Result<HashMap<String, Arc<DefaultClient>>> {
    let mut clients = HashMap::new();
    for inference in inference_configs {
        let client =
            DefaultClient::from_inference_config(inference, game_config, CancellationToken::new())
                .await;
        clients.insert(inference.name.clone(), Arc::new(client));
    }
    Ok(clients)
}

fn build_agent_entries(
    agent_configs: &'static [AgentConfig],
    game_config: &'static GameConfig,
    inference_clients: &HashMap<String, Arc<DefaultClient>>,
) -> Result<Vec<AgentEntry>> {
    agent_configs
        .iter()
        .map(|config| spawn_agent_runner(config, game_config, inference_clients))
        .collect()
}

fn spawn_agent_runner(
    config: &'static AgentConfig,
    game_config: &'static GameConfig,
    inference_clients: &HashMap<String, Arc<DefaultClient>>,
) -> Result<AgentEntry> {
    let (name, mut agent): (String, Box<dyn Agent>) = match config {
        AgentConfig::Random(random_config) => (
            random_config.name.clone(),
            Box::new(RandomAgent::new(random_config)),
        ),
        AgentConfig::MCTS(mcts_config) => {
            let client = inference_clients
                .get(&mcts_config.inference_config_name)
                .cloned()
                .with_context(|| {
                    format!(
                        "Inference config '{}' not found for agent '{}'",
                        mcts_config.inference_config_name, mcts_config.name
                    )
                })?;
            (
                mcts_config.name.clone(),
                Box::new(MCTSAgent::new(mcts_config, game_config, client)),
            )
        }
    };

    let (sender, mut receiver) = mpsc::unbounded_channel::<AgentJob>();
    let agent_name = name.clone();
    tokio::spawn(async move {
        while let Some(job) = receiver.recv().await {
            let move_index = agent.choose_move(&job.state).await;
            if job.reply.send(move_index).is_err() {
                warn!(agent = %agent_name, "Agent response receiver dropped");
            }
        }
    });

    Ok(AgentEntry { name, sender })
}
