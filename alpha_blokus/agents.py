from typing import List
import numpy as np
import json
import asyncio
from omegaconf import OmegaConf

from alpha_blokus import player_pov_helpers
from alpha_blokus.inference.client import InferenceClient
from alpha_blokus.moves_data import moves_data
from alpha_blokus.state import State

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class RandomAgent:
    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg

    async def select_move_index(self, state: State):
        # Get all valid moves.
        valid_moves = np.flatnonzero(state.valid_moves_array())
        
        # Find the move with the highest number of newly occupied squares.
        best_move_score = -1
        for move_index in valid_moves:
            best_move_score = max(best_move_score, np.sum(moves_data(self.cfg)["new_occupieds"][move_index]))

        # Select a move randomly from all the best moves.
        best_moves = []
        for move_index in valid_moves:
            if np.sum(moves_data(self.cfg)["new_occupieds"][move_index]) == best_move_score:
                best_moves.append(move_index)

        return np.random.choice(best_moves)
    
    async def report_move(self, state: State, move_index: int):
        pass
    

class PolicySamplingAgent:
    def __init__(self, agent_config, inference_client: InferenceClient, cfg: OmegaConf):
        self.agent_config = agent_config
        self.inference_client = inference_client
        self.cfg = cfg

    async def select_move_index(self, state: State):
        array_index_to_move_index = np.flatnonzero(state.valid_moves_array())

        player_pov_occupancies = player_pov_helpers.occupancies_to_player_pov(
            state.occupancies,
            state.player,
        )
        player_pov_valid_move_indices = player_pov_helpers.moves_indices_to_player_pov(
            array_index_to_move_index,
            state.player,
            moves_data(self.cfg),
        )
        
        _, universal_children_prior_logits = await self.inference_client.evaluate(
            player_pov_occupancies,
            player_pov_valid_move_indices,
            state.turn,
        )
        universal_children_priors = softmax(universal_children_prior_logits)

        temperature = self.agent_config["move_selection_temperature"]
        if temperature == 0:
            return array_index_to_move_index[np.argmax(universal_children_priors)]
        else:
            weighted_probabilities = np.power(
                universal_children_priors,
                1 / temperature,
            )
            probabilities = weighted_probabilities / np.sum(weighted_probabilities)
            return np.random.choice(array_index_to_move_index, p=probabilities)
        
    async def report_move(self, state: State, move_index: int):
        pass

class PentobiAgent:
    def __init__(self, agent_config: OmegaConf, cfg: OmegaConf):
        self.agent_config = agent_config
        self.cfg = cfg

        self.process = None

    async def _initialize_process(self):
        self.process = await asyncio.create_subprocess_exec(
            self.agent_config["pentobi_bin_path"],
            "--level",
            str(self.agent_config["pentobi_level"]),
            "--book",
            self.agent_config["pentobi_opening_book"],
            "--noresign",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def _communicate(self, input: str):
        self.process.stdin.write(input.encode("utf-8"))
        await self.process.stdin.drain()

        stdout = b""
        while True:
            this_read = await self.process.stdout.readline()
            if this_read == b"\n":
                break
            stdout += this_read
        stdout = stdout.decode("utf-8")

        if not stdout.startswith("= "):
            raise Exception(f"Unexpected response from Pentobi: {stdout}")

        return stdout[2:]
    
    def _gtp_coordinates_to_move_index(self, coordinates: List[str]):
        occupancies = np.zeros((self.cfg.board_size, self.cfg.board_size))
        for coordinate in coordinates:
            x = ord(coordinate[0]) - ord("a")
            y = self.cfg.board_size - int(coordinate[1:])
            occupancies[y, x] = 1
        
        matches = np.all(moves_data(self.cfg)["new_occupieds"] == occupancies, axis=(1, 2))
        move_index = np.argmax(matches)
        return move_index
    
    def _move_index_to_gtp_coordinates(self, move_index: int):
        ys, xs = np.nonzero(moves_data(self.cfg)["new_occupieds"][move_index])
        coordinates = []
        for y, x in zip(ys, xs):
            coordinates.append(f"{chr(x + ord('a'))}{self.cfg.board_size - y}")
        return coordinates
    
    async def select_move_index(self, state: State):
        if self.process is None:
            await self._initialize_process()

        response = await self._communicate(f"genmove {state.player+1}\n")
        return self._gtp_coordinates_to_move_index(response.split(","))
    
    async def report_move(self, state: State, move_index: int):
        if self.process is None:
            await self._initialize_process()

        coordinates = ",".join(self._move_index_to_gtp_coordinates(move_index))
        await self._communicate(f"play {state.player+1} {coordinates}\n")