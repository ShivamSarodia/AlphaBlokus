use ahash::AHashMap as HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, MutexGuard};
use tokio_util::sync::CancellationToken;

use crate::agents::Agent;
use crate::gameplay;
use crate::web::serve::ApiError;
use crate::{config::WebPlayConfig, game::State as BlokusState};

#[derive(Clone)]
pub struct AppState {
    // Config that originally started the server.
    pub config: &'static WebPlayConfig,

    // Agents that can play in this game.
    agents: AgentRegistry,

    // Session for the current game. Behind an Arc and Mutex to allow for concurrent
    // access.
    session: Arc<Mutex<GameSession>>,
}

pub struct GameSession {
    // State of the current Blokus game.
    pub blokus_state: BlokusState,

    // Name of the agent that is currently playing, if any.
    pub pending_agent: Option<String>,
}

#[derive(Clone)]
pub struct AgentRegistry {
    agents: HashMap<String, Arc<Mutex<Box<dyn Agent>>>>,
}

impl AppState {
    pub async fn build(config: &'static WebPlayConfig) -> Self {
        Self {
            config,
            agents: AgentRegistry::build(config).await,
            // TODO: We can probably implement some better synchronization here to ensure
            // that multiple agents / resets / etc can't conflict.
            session: Arc::new(Mutex::new(GameSession {
                blokus_state: BlokusState::new(&config.game),
                pending_agent: None,
            })),
        }
    }

    pub async fn reset(&self) {
        // TODO: Consider resetting the agent registry here as well.
        *self.session.lock().await = GameSession {
            blokus_state: BlokusState::new(&self.config.game),
            pending_agent: None,
        };
    }

    pub fn agent_names(&self) -> Vec<String> {
        self.agents.agent_names()
    }

    pub async fn session(&self) -> MutexGuard<'_, GameSession> {
        self.session.lock().await
    }

    pub async fn start_agent_move(&self, agent_name: &str) -> Result<(), ApiError> {
        // Confirm that we're ok to start the agent move and set the pending agent.
        let state = {
            let mut session = self.session().await;

            // Set the pending agent.
            if session.pending_agent.is_some() {
                return Err(ApiError::AgentBusy(
                    "An agent is already selecting a move".into(),
                ));
            }
            session.pending_agent = Some(agent_name.to_string());
            session.blokus_state.clone()
        };

        // Grab the agent from the registry.
        let agent = self.agents.get_agent(agent_name).await;
        let session = Arc::clone(&self.session);

        // Initiate a separate task to run that agent.
        tokio::spawn(async move {
            // Run the agent to chose a move.
            let move_index = agent.lock().await.choose_move(&state).await;

            // Apply the move to the state.
            let mut session = session.lock().await;
            session.blokus_state.apply_move(move_index);
            session.pending_agent = None;
        });

        Ok(())
    }
}

impl GameSession {
    pub async fn apply_human_move(&mut self, move_index: usize) -> Result<(), ApiError> {
        if self.pending_agent.is_some() {
            return Err(ApiError::AgentBusy(
                "Human player cannot make a move when an agent is selecting a move".into(),
            ));
        }
        if !self.blokus_state.is_valid_move(move_index) {
            return Err(ApiError::MoveNotAllowed(
                "The selected move is not valid".into(),
            ));
        }
        self.blokus_state.apply_move(move_index);

        Ok(())
    }
}

impl AgentRegistry {
    pub async fn build(config: &'static WebPlayConfig) -> Self {
        let inference_clients = gameplay::build_inference_clients(
            &config.inference,
            &config.game,
            CancellationToken::new(),
        )
        .await;

        let agents = HashMap::from_iter(
            config
                .agents
                .iter()
                .map(|agent_config| {
                    gameplay::build_agent(agent_config, &config.game, &inference_clients)
                })
                .map(|agent| (agent.name().to_string(), Arc::new(Mutex::new(agent)))),
        );

        Self { agents }
    }

    pub async fn get_agent(&self, agent_name: &str) -> Arc<Mutex<Box<dyn Agent>>> {
        Arc::clone(self.agents.get(agent_name).expect("Agent not found"))
    }

    fn agent_names(&self) -> Vec<String> {
        self.agents.keys().cloned().collect()
    }
}
