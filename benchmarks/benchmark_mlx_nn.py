import os 
import sys

os.environ["CONFIG_PATHS"] = "/Users/shivamsarodia/Dev/blockus/configs/self_play_20.yaml"
os.environ["CONFIG_OVERRIDES"] = 'game.moves_directory="/Users/shivamsarodia/Dev/blockus/data/moves_20"'
sys.path.append("/Users/shivamsarodia/Dev/blockus/src")

import argparse
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from typing import Dict 

from configuration import config
from neural_net_mlx import NeuralNetMLX

BOARD_SIZE = config()["game"]["board_size"]

parser = argparse.ArgumentParser(description="Benchmark neural network")
parser.add_argument('--batch-size', type=int, default=128, help='Batch size for evaluation')
parser.add_argument('--dtype', type=str, default='float16', help='Data type for tensors')

args = parser.parse_args()

BATCH_SIZE = args.batch_size
NUM_BATCHES_WARM_UP = 10
NUM_BATCHES_TO_EVALUATE = 500
DTYPE = getattr(mx, args.dtype)

model = NeuralNetMLX(config()["networks"]["default_1"])
model.set_dtype(DTYPE)
model.eval()

nn.quantize(model)
compiled_model = mx.compile(model)
# compiled_model = model

def time_per_eval(num_batches, batch_size, dtype, compiled_model):
    random_arrays = np.random.random((num_batches, batch_size, BOARD_SIZE, BOARD_SIZE, 4))
    boards = mx.array(random_arrays, dtype=dtype)

    start = time.perf_counter()
    for i in range(num_batches):
        result = compiled_model(boards[i])
        if isinstance(result, tuple):
            for x in result:
                # np.array(x)
                mx.eval(x)
        else:
            mx.eval(result)

    elapsed = time.perf_counter() - start

    return elapsed / (num_batches * batch_size)

print("Warming up...")
time_per_eval(NUM_BATCHES_WARM_UP, BATCH_SIZE, DTYPE, compiled_model)
print("Evaluating...")
print(time_per_eval(NUM_BATCHES_TO_EVALUATE, BATCH_SIZE, DTYPE, compiled_model))
