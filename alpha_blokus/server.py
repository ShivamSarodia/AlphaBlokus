from flask import Flask, request
from flask_cors import CORS
import ray
import asyncio
import numpy as np 

import alpha_blokus.player_pov_helpers as player_pov_helpers
from alpha_blokus.inference.actor import InferenceActor
from alpha_blokus.inference.client import InferenceClient
from alpha_blokus.gameplay_actor import generate_agent
from alpha_blokus.state import State
from alpha_blokus.moves_data import moves_data
async def serialize_state(state: State, inference_client: InferenceClient, agents, cfg):
    array_index_to_move_index = np.flatnonzero(state.valid_moves_array())        
    player_pov_occupancies = player_pov_helpers.occupancies_to_player_pov(
        state.occupancies,
        state.player,
    )
    player_pov_valid_move_indices = player_pov_helpers.moves_indices_to_player_pov(
        array_index_to_move_index,
        state.player,
        moves_data(cfg),
    )        
    player_pov_values, _ = await inference_client.evaluate(
        player_pov_occupancies,
        player_pov_valid_move_indices,
        state.turn,
    )
    values = player_pov_helpers.values_to_player_pov(player_pov_values, -state.player)

    # Convert the state into the json representation for the UI.

    moves_ruled_out = state.moves_ruled_out[state.player]
    moves_not_ruled_out = np.flatnonzero(~moves_ruled_out)
    pieces_available = np.unique(moves_data(cfg)["piece_indices"][moves_not_ruled_out])

    # TODO: DRY this, maybe store it in MOVES_DATA.
    PIECES = [
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
        [(0, 0), (1, 0), (1, 1), (2, 1), (3, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
        [(0, 0), (1, 0), (2, 0), (1, 1), (1, 2)],
        [(0, 0), (0, 1), (1, 1), (2, 1), (2, 0)],
        [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1)],
        [(0, 0), (1, 0), (2, 0), (3, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
        [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
        [(0, 0), (0, 1), (1, 1), (1, 0), (2, 0)],
        [(1, 1), (0, 1), (1, 0), (2, 1), (1, 2)],
        [(0, 0), (0, 1), (1, 1), (1, 2), (2, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(0, 0), (1, 0), (2, 0), (3, 0)],
        [(0, 0), (1, 0), (2, 0), (0, 1)],
        [(0, 0), (0, 1), (1, 1), (1, 0)],
        [(0, 0), (1, 0), (2, 0), (1, 1)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (1, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0)],
    ]

    pieces = []
    for piece_index in pieces_available:
        pieces.append(PIECES[piece_index])

    result = {
        "board": state.occupancies.astype(int).tolist(),
        "score": np.sum(state.occupancies, axis=(1, 2)).tolist(),
        "player": state.player,
        "pieces": pieces,
        "predicted_values": values.tolist(),
        "game_over": not state.valid_moves_array().any(),
        "result": state.result().tolist(),
        "is_human_turn": agents[state.player] is None,
    }
    if state.last_move_index is not None:
        result["last_move"] = moves_data(cfg)["new_occupieds"][state.last_move_index].tolist()

    return result

def run(cfg):
    ray.init(log_to_driver=True)

    inference_actor = InferenceActor.remote(cfg["networks"]["opponent"], cfg)
    inference_client = InferenceClient(inference_actor, 1, cfg)
    inference_client.init_in_process(asyncio.get_event_loop())

    agents = [
        generate_agent(agent_config, {"opponent": inference_client}, None, None, cfg)
        for agent_config in cfg["agents"]
    ]

    state = State(cfg)

    # Start the Flask server.
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})

    @app.route("/state")
    async def get_state():
        return await serialize_state(state, inference_client, agents, cfg)
    
    @app.route("/move", methods=["POST"])
    async def make_move():
        current_agent = agents[state.player]

        # If we're waiting for a human move, make the human move.
        if not current_agent:
            move_coordinates_occupied = request.get_json()
            move_new_occupieds = np.zeros((cfg["game"]["board_size"], cfg["game"]["board_size"]), dtype=bool)
            for x, y in move_coordinates_occupied:
                move_new_occupieds[x, y] = True

            move_index = np.where(
                np.all(
                    moves_data(cfg)["new_occupieds"] == move_new_occupieds,
                    axis=(1, 2),
                )
            )[0][0]
        
        else:
            move_index, _ = await current_agent.select_move_index(state, None)
        
        # Make the specified move.
        state.play_move(move_index)

        return await serialize_state(state, inference_client, agents, cfg)

    app.run(host="0.0.0.0", port=8080)
