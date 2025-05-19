import ray
import asyncio
import random
import time
import os
import copy
import pyinstrument
import json
from typing import Dict, Optional
from omegaconf import OmegaConf

from alpha_blokus.data_recorder import DataRecorder
from alpha_blokus.inference.client import InferenceClient
from alpha_blokus.agents import PentobiAgent, PolicySamplingAgent, RandomAgent
from alpha_blokus.mcts import MCTSAgent
from alpha_blokus.state import State
from alpha_blokus.event_logger import log_event

def generate_agent(
    agent_config: OmegaConf,
    inference_clients: Dict[str, InferenceClient],
    data_recorder: Optional[DataRecorder],
    recorder_game_id: Optional[int],
    cfg: OmegaConf,
):
    if agent_config["type"] == "mcts":
        network_name = agent_config["network"]
        return MCTSAgent(
            agent_config,
            inference_clients[network_name],
            data_recorder,
            recorder_game_id,
            cfg,
        )
    elif agent_config["type"] == "random":
        return RandomAgent(cfg)
    elif agent_config["type"] == "human":
        return None
    elif agent_config["type"] == "policy_sampling":
        network_name = agent_config["network"]
        return PolicySamplingAgent(
            agent_config,
            inference_clients[network_name],
        )
    elif agent_config["type"] == "pentobi":
        return PentobiAgent(
            agent_config,
            cfg,
        )
    else:
        raise "Unknown agent type."

@ray.remote
class GameplayActor:
    def __init__(self, actor_index: int, inference_clients: Dict[str, InferenceClient], cfg: OmegaConf):
        self.inference_clients = inference_clients
        self.data_recorder = DataRecorder(cfg)
        self.profiler = None
        # Store full config
        self.cfg = cfg
        # Filter agent configs based on this actor index
        self.agent_configs = [
            agent for agent in self.cfg.gameplay.agents.values()
            if (
                "gameplay_actors" not in agent
                or
                actor_index in agent.get("gameplay_actors")
            )
        ]

    async def run(self):
        print("Running gameplay process...")

        # Setup.
        if self.cfg.use_profiler:
            self.profiler = pyinstrument.Profiler()
            self.profiler.start()

        # Play games.
        await self.multi_continuously_play_games(self.cfg["gameplay"]["architecture"]["coroutines_per_process"])

    def cleanup(self):
        print("Cleaning up gameplay actor...")
        self.data_recorder.flush()
        if self.cfg.use_profiler:
            self.profiler.stop()
            path = os.path.join(self.output_data_dir, f"profiler/")
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, f"{random.getrandbits(30)}_gameplay.html")
            print(f"Writing profiler info to path: {path}")
            self.profiler.write_html(path)

    async def play_game(self):
        recorder_game_id = self.data_recorder.start_game()

        # TODO: Support different ways of selecting agents.
        agent_configs = self.agent_configs

        # Instantiate a single agent instance for each type of agent.
        # That is, if all four agents have the same name, we only instantiate one agent class.
        distinct_agent_names = set(agent["name"] for agent in agent_configs)
        agent_instances = {
            agent_name: generate_agent(
                next(filter(lambda x: x["name"] == agent_name, agent_configs)),
                self.inference_clients,
                self.data_recorder,
                recorder_game_id,
                self.cfg,
            )
            for agent_name in distinct_agent_names
        }

        agents = [
            agent_instances[agent_config["name"]]
            for agent_config in agent_configs
        ]

        game_over = False
        state = State(self.cfg)
        while not game_over:
            agent = agents[state.player]

            # Ask the current agent to select a move.
            move_index = await agent.select_move_index(state)

            # To all other agents, report the move that was selected.
            #
            # Couple ways this is useful:
            #   - For MCTS agents, when a move is made by another agent we
            #     know to remove the cached tree and start from scratch the
            #     next time the agent is used.
            #   - For a Pentobi agent, this information is used to update the
            #     game state.
            for other_agent in agent_instances.values():
                if other_agent is not agent:
                    await other_agent.report_move(state, move_index)
            
            # Make the selected move.
            game_over = state.play_move(move_index)
            if self.cfg.log_made_move:
                log_event("made_move")
        
        result = state.result()
        log_event("game_result", 
            {
                "scores": [
                    [agent_configs[i]["name"], state.result()[i]]
                    for i in range(4)
                ],
                "game_id": recorder_game_id,
            }
        )

        self.data_recorder.record_game_end(recorder_game_id, result)

    async def continuously_play_games(self):
        while True:
            log_event("game_start")
            start = time.time()
            await self.play_game()
            log_event("game_end", { "runtime": time.time() - start })

    async def multi_continuously_play_games(self, num_coroutines: int):
        # We need to call this in here so that uvloop has had a chance to set the event loop first.
        for client in self.inference_clients.values():
            client.init_in_process(asyncio.get_event_loop())

        await asyncio.gather(
            *[self.continuously_play_games() for _ in range(num_coroutines)]
        )