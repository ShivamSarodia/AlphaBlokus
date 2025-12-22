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
    // States of the current Blokus game, from oldest to newest.
    // Current state is the last element.
    pub blokus_states: Vec<BlokusState>,

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
                blokus_states: vec![BlokusState::new(&config.game).unwrap()],
                pending_agent: None,
            })),
        }
    }

    pub async fn reset(&self) {
        // TODO: Consider resetting the agent registry here as well.
        *self.session.lock().await = GameSession {
            blokus_states: vec![BlokusState::new(&self.config.game).unwrap()],
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
            session.blokus_states.last().unwrap().clone()
        };

        // Grab the agent from the registry.
        let agent = self.agents.get_agent(agent_name);
        let other_agents = self
            .agents
            .agent_names()
            .iter()
            .filter(|name| *name != agent_name)
            .map(|name| self.agents.get_agent(name))
            .collect::<Vec<_>>();
        let session = Arc::clone(&self.session);

        // Initiate a separate task to run that agent.
        tokio::spawn(async move {
            // Run the agent to chose a move.
            let move_index = agent.lock().await.choose_move(&state).await.unwrap();

            // Apply the move to the state.
            let mut session = session.lock().await;
            let mut cloned_state = session.blokus_states.last().unwrap().clone();

            // Report the move to all the other agents.
            for other_agent in other_agents {
                other_agent
                    .lock()
                    .await
                    .report_move(&state, move_index)
                    .await
                    .unwrap();
            }

            cloned_state.apply_move(move_index).unwrap();
            session.blokus_states.push(cloned_state);
            session.pending_agent = None;
        });

        Ok(())
    }

    pub async fn report_human_move(&self, move_index: usize) -> Result<(), ApiError> {
        let state = {
            let session = self.session().await;
            session.blokus_states.last().unwrap().clone()
        };

        for agent in self.agents.iter() {
            agent
                .lock()
                .await
                .report_move(&state, move_index)
                .await
                .unwrap();
        }

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
        if !self.blokus_states.last().unwrap().is_valid_move(move_index) {
            return Err(ApiError::MoveNotAllowed(
                "The selected move is not valid".into(),
            ));
        }

        let mut cloned_state = self.blokus_states.last().unwrap().clone();
        cloned_state.apply_move(move_index).unwrap();
        self.blokus_states.push(cloned_state);

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
        .await
        .unwrap();

        let agents = HashMap::from_iter(
            config
                .agents
                .iter()
                .map(|agent_config| {
                    gameplay::build_agent(agent_config, &config.game, &inference_clients).unwrap()
                })
                .map(|agent| (agent.name().to_string(), Arc::new(Mutex::new(agent)))),
        );

        Self { agents }
    }

    pub fn get_agent(&self, agent_name: &str) -> Arc<Mutex<Box<dyn Agent>>> {
        Arc::clone(self.agents.get(agent_name).expect("Agent not found"))
    }

    fn agent_names(&self) -> Vec<String> {
        self.agents.keys().cloned().collect()
    }

    fn iter(&self) -> impl Iterator<Item = Arc<Mutex<Box<dyn Agent>>>> {
        self.agents.values().cloned()
    }
}
