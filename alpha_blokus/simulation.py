import ray
import time
import subprocess
import ray.exceptions
import traceback
import os
import glob
import json

import torch

from alpha_blokus.inference.client import InferenceClient
from alpha_blokus.inference.actor import InferenceActor
from alpha_blokus.neural_net import NeuralNet
from alpha_blokus.training.actor import TrainingActor
from alpha_blokus.gameplay_actor import GameplayActor
from datetime import datetime, timedelta


def run(cfg):
    ray.init(log_to_driver=cfg["log_to_console"])
    print("Running with working directory:", os.getcwd())

    # Start the Ray actor that runs GPU computations.
    inference_clients = {}
    for network_name, network_config in cfg.get("networks", {}).items():

        # Create the initial model if needed.
        if not os.path.exists(network_config["model_read_path"]):
            os.makedirs(network_config["model_read_path"], exist_ok=True)

        if (
            os.path.isdir(network_config["model_read_path"]) and 
            not os.listdir(network_config["model_read_path"]) and 
            network_config["initialize_model_if_empty"]
        ):
            initial_model_path = os.path.join(network_config["model_read_path"], "0.pt")
            print(f"Creating initial model at {initial_model_path}...")
            model = NeuralNet(network_config, cfg)
            torch.save(model.state_dict(), initial_model_path)
        
        # If we're expecting to play games, start an inference actor for each network.
        if cfg.get("gameplay"):
            print(f"Starting inference actor for network '{network_name}'...")
            inference_actor = InferenceActor.remote(network_config, cfg)
            inference_clients[network_name] = InferenceClient(inference_actor, network_config["batch_size"], cfg)

    # If we're supposed to be training, start the Ray actor that runs training.
    # if cfg.get("training"):
    #     gamedata_path = cfg["training"]["data_read_directory"]
    #     print(f"Starting training actor reading from {gamedata_path}...")
    #     training_actor = TrainingActor.remote(gamedata_path, cfg)
    #     training_actor.run.remote()

    # If we're supposed to be generating game data, start the Ray actor(s) for gameplay.
    if cfg.get("gameplay"):
        # Launch gameplay actors
        gameplay_actors = [
            GameplayActor.remote(i, inference_clients, cfg)
            for i in range(cfg["gameplay"]["architecture"]["gameplay_processes"])
        ]
        for gameplay_actor in gameplay_actors:
            gameplay_actor.run.remote()

    runtime = cfg["runtime"]
    if runtime > 0:
        finish_time = (datetime.now() + timedelta(seconds=runtime)).strftime("%I:%M:%S %p")
        print(f"Running for {runtime} seconds, finishing at {finish_time}...")
    else:
        print("Running indefinitely...")

    # Finally, run the main loop.
    start_time = time.time()
    logs_last_copied = start_time

    try:
        while True:
            current_time = time.time()

            # If it's time to wrap up, break.
            if runtime > 0 and current_time > start_time + runtime:
                break

            # If it's been a while since the logs were copied, copy them now.
            if current_time > logs_last_copied + cfg["log_flush_interval"]:
                copy_ray_logs()
                logs_last_copied = current_time

            # Sleep about 60 seconds before checking again.
            time.sleep(min(60, cfg["log_flush_interval"]))

    except KeyboardInterrupt:
        print("Got KeyboardInterrupt...")

    finally:
        print("Shutting down...")
        print("Cleaning up gameplay actors...")
        ray.get([
            gameplay_actor.cleanup.remote() for gameplay_actor in gameplay_actors
        ])
        print("Done cleaning up gameplay actors.")
        print("Shutting down Ray...")
        ray.shutdown()
        time.sleep(1)
        print("Done shutting down Ray. Run directory:", os.getcwd())
        copy_ray_logs() 
        print("Exiting.")

def copy_ray_logs():
    output_file_name = f"logs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')}.txt"

    print(f"Copying Ray logs...")
    with open(output_file_name, "w") as output_file:
        subprocess.run(
            "cat /tmp/ray/session_latest/logs/worker*.out",
            shell=True,
            stdout=output_file,
            stderr=subprocess.PIPE
        )
    print(f"Done copying Ray logs to {output_file_name}.")

    # Delete any previous logs
    for file in glob.glob("logs_*.txt"):
        if file.endswith(output_file_name):
            continue
        os.remove(file)
