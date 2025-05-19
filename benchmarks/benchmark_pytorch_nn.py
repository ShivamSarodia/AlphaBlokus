import os
import sys
import time
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from alpha_blokus.neural_net import NeuralNet

@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="benchmark",
)
def main(cfg: DictConfig):
    # Set up device and dtype
    device = torch.device(cfg.device)
    dtype = getattr(torch, cfg.dtype)

    # Initialize model
    model = NeuralNet(cfg.networks.main, cfg, flatten_policy=True)
    model = model.to(device=device)
    model = model.to(dtype=dtype)
    model.eval()

    # Benchmark parameters
    NUM_BATCHES_WARM_UP = cfg.num_batches_warm_up
    NUM_BATCHES_TO_EVALUATE = cfg.num_batches_to_evaluate
    BATCH_SIZE = cfg.batch_size

    def time_per_eval(num_batches, batch_size, dtype, model):
        random_arrays = np.random.random((num_batches, batch_size, 4, cfg.game.board_size, cfg.game.board_size))

        start = time.perf_counter()
        for i in range(num_batches):
            boards = torch.from_numpy(random_arrays[i]).to(device=device, dtype=dtype)
            with torch.no_grad():
                value, policy = model(boards)
                value = value.cpu().numpy()
                policy = policy.cpu().numpy()

        elapsed = time.perf_counter() - start
        return elapsed / (num_batches * batch_size)

    print("Warming up...")
    time_per_eval(NUM_BATCHES_WARM_UP, BATCH_SIZE, dtype, model)
    print("Evaluating...")
    elapsed = time_per_eval(NUM_BATCHES_TO_EVALUATE, BATCH_SIZE, dtype, model)
    print(f"Average time per evaluation: {elapsed:.6f} seconds")

if __name__ == "__main__":
    main()
