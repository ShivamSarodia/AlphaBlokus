import os 
import sys

os.environ["CONFIG_PATHS"] = "/Users/shivamsarodia/Dev/blockus/configs/self_play_20.yaml"
os.environ["CONFIG_OVERRIDES"] = 'game.moves_directory="/Users/shivamsarodia/Dev/blockus/data/moves_20"'
sys.path.append("/Users/shivamsarodia/Dev/blockus/src")

import argparse
import time
import numpy as np
from tinygrad import Tensor
from tinygrad import dtypes
from tinygrad import TinyJit
from typing import Dict 

from configuration import config
from neural_net_tg import NeuralNet

BOARD_SIZE = config()["game"]["board_size"]

parser = argparse.ArgumentParser(description="Benchmark neural network")
parser.add_argument('--batch-size', type=int, default=128, help='Batch size for evaluation')

args = parser.parse_args()

BATCH_SIZE = args.batch_size
NUM_BATCHES_WARM_UP = 10
NUM_BATCHES_TO_EVALUATE = 20
# DTYPE = getattr(dtypes, args.dtype)

from tinygrad import Device
print(Device.DEFAULT)

model = NeuralNet(config()["networks"]["default_1"])

def evaluate_model(input):
    results = model(input)
    return [result.realize() for result in results]

compiled_model = TinyJit(evaluate_model)

def time_per_eval(num_batches, batch_size, compiled_model):
    boards = [
        Tensor.randn(batch_size, 4, BOARD_SIZE, BOARD_SIZE, dtype=dtypes.int8)
        for _ in range(num_batches)
    ]
    start = time.perf_counter()
    for i in range(num_batches):
        result = compiled_model(boards[i])

    elapsed = time.perf_counter() - start

    return elapsed / (num_batches * batch_size)

print("Warming up...")
time_per_eval(NUM_BATCHES_WARM_UP, BATCH_SIZE, compiled_model)
print("Evaluating...")
print(time_per_eval(NUM_BATCHES_TO_EVALUATE, BATCH_SIZE, compiled_model))
