import os
import time
import torch
import aim
from alpha_blokus.neural_net import NeuralNet
from alpha_blokus.event_logger import log_event
from alpha_blokus.training.helpers import TrainingLoop

def run(cfg):
    network_config = cfg["networks"][cfg["training"]["network_name"]]

    # Start with a randomly initialized model.
    model = NeuralNet(network_config, cfg)
    model.to(cfg["training"]["device"])
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

    # Initialize aim run for experiment tracking
    run_obj = aim.Run()
    run_obj["hparams"] = {
        "path": os.getcwd(),
        "network": network_config,
        "config": cfg["training"],
    }
    print("Starting training on run:", run_obj.hash)

    training_loop = TrainingLoop(
        initial_model=model,
        initial_lifetime_loaded_samples=0,
        optimizer=optimizer,
        device=cfg["training"]["device"],
        gamedata_dir=cfg["training"]["data_read_directory"],
        compute_top_one=True,
        use_logging=False,
        cfg=cfg,
    )
    previous_lifetime_loaded_samples = 0
    try:
        while True:
            action, result = training_loop.run_iteration()
            if action == "trained":
                # print("Trained")
                for key, value in result.items():
                    run_obj.track(
                        value,
                        name=key,
                        step=result["lifetime_trained_samples"],
                    )
            elif action == "read_new_data":
                # print("Reading new data")
                pass
            elif action == "no_new_data":
                # print("No new data")
                return
            lifetime_loaded_samples = training_loop.lifetime_loaded_samples
            if lifetime_loaded_samples // cfg["training"]["samples_per_generation"] > previous_lifetime_loaded_samples // cfg["training"]["samples_per_generation"]:
                model_name = str(lifetime_loaded_samples).zfill(9)
                model_dir = cfg["training"]["model_write_directory"]
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, f"{model_name}.pt")
                torch.save(model.state_dict(), model_path)
                run_obj.track(
                    lifetime_loaded_samples,
                    name="saved_model_samples",
                    step=training_loop.lifetime_trained_samples,
                )
            previous_lifetime_loaded_samples = lifetime_loaded_samples
    finally:
        # Make sure to close the aim run
        run_obj.close()
