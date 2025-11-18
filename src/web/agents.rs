use std::{collections::HashMap, sync::Arc};

use anyhow::{Context, Result};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;
use tracing::warn;

use crate::{
    agents::{Agent, MCTSAgent, RandomAgent},
    config::{AgentConfig, GameConfig, InferenceConfig},
    game::State as BlokusState,
    inference::DefaultClient,
};

#[derive(Clone)]
pub struct AgentRegistry {
    names: Arc<Vec<String>>,
    senders: Arc<HashMap<String, mpsc::UnboundedSender<AgentJob>>>,
}

impl AgentRegistry {
    pub async fn build(
        agent_configs: &'static [AgentConfig],
        game_config: &'static GameConfig,
        inference_configs: &'static [InferenceConfig],
    ) -> Result<Self> {
        let inference_clients = build_inference_clients(inference_configs, game_config).await?;

        let mut names = Vec::new();
        let mut senders = HashMap::new();
        for config in agent_configs {
            let entry = spawn_agent_runner(config, game_config, &inference_clients)?;
            names.push(entry.name.clone());
            senders.insert(entry.name, entry.sender);
        }

        Ok(Self {
            names: Arc::new(names),
            senders: Arc::new(senders),
        })
    }

    pub fn names(&self) -> Vec<String> {
        self.names.as_ref().clone()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.senders.contains_key(name)
    }

    pub fn request_move(
        &self,
        name: &str,
        state: BlokusState,
        reply: oneshot::Sender<usize>,
    ) -> Result<(), AgentRequestError> {
        let sender = self
            .senders
            .get(name)
            .cloned()
            .ok_or(AgentRequestError::AgentNotFound)?;
        sender
            .send(AgentJob { state, reply })
            .map_err(|_| AgentRequestError::AgentFailed)
    }
}

#[derive(Debug)]
pub enum AgentRequestError {
    AgentNotFound,
    AgentFailed,
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

fn spawn_agent_runner(
    config: &'static AgentConfig,
    game_config: &'static GameConfig,
    inference_clients: &HashMap<String, Arc<DefaultClient>>,
) -> Result<AgentEntry> {
    let (name, mut agent) = build_agent(config, game_config, inference_clients)?;
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

fn build_agent(
    config: &'static AgentConfig,
    game_config: &'static GameConfig,
    inference_clients: &HashMap<String, Arc<DefaultClient>>,
) -> Result<(String, Box<dyn Agent>)> {
    let agent: (String, Box<dyn Agent>) = match config {
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

    Ok(agent)
}
