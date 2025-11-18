use std::sync::Arc;

use anyhow::Result;
use tokio::sync::{Mutex, MutexGuard, oneshot};
use tracing::warn;

use crate::{
    config::{GameConfig, WebPlayConfig},
    game::State as BlokusState,
};

use super::{
    agents::{AgentRegistry, AgentRequestError},
    response,
};

#[derive(Clone)]
pub struct AppState {
    pub config: &'static WebPlayConfig,
    session: Arc<Mutex<GameSession>>,
    agents: AgentRegistry,
}

impl AppState {
    pub async fn build(config: &'static WebPlayConfig) -> Result<Self> {
        let agents = AgentRegistry::build(&config.agents, &config.game, &config.inference).await?;
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
        self.agents.names()
    }

    pub fn game_response(&self, session: &GameSession) -> response::GameResponse {
        response::build_game_response(
            &session.state,
            self.config,
            self.agent_names(),
            session.pending_agent(),
        )
    }

    pub async fn start_agent_move(&self, agent_name: &str) -> Result<(), AgentRunError> {
        if !self.agents.contains(agent_name) {
            return Err(AgentRunError::AgentNotFound);
        }

        let snapshot = {
            let mut session = self.session().await;
            if session.has_pending_agent() {
                return Err(AgentRunError::AgentBusy);
            }
            if !session.state.any_valid_moves() {
                return Err(AgentRunError::NoMovesAvailable);
            }
            session.set_pending_agent(agent_name);
            session.state.clone()
        };

        let (reply_sender, reply_receiver) = oneshot::channel();
        if let Err(error) = self.agents.request_move(agent_name, snapshot, reply_sender) {
            let mut session = self.session().await;
            session.clear_pending_agent();
            return Err(error.into());
        }

        let session = Arc::clone(&self.session);
        let agent_name = agent_name.to_string();
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

impl From<AgentRequestError> for AgentRunError {
    fn from(error: AgentRequestError) -> Self {
        match error {
            AgentRequestError::AgentNotFound => AgentRunError::AgentNotFound,
            AgentRequestError::AgentFailed => AgentRunError::AgentFailed,
        }
    }
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
