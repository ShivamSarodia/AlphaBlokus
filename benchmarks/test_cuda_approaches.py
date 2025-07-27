import torch
from torch import nn
import torch.nn.functional as F
import ray
import numpy as np
import time
import threading

# # # # # # # # # # # # # # # # # #
# Neural network components
# # # # # # # # # # # # # # # # # #

DEVICE = "cuda"
# DEVICE = "mps"

class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.convolutional_block = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
        )

    def forward(self, x):
        return F.relu(x + self.convolutional_block(x))

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolutional_block = nn.Sequential(
            nn.Conv2d(
                in_channels=5,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.residual_blocks = nn.ModuleList([
            ResidualBlock()
            for _ in range(10)
        ])
        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=16,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                16 * 20 * 20,
                64,
            ),
            nn.ReLU(),
            nn.Linear(
                64,
                4,
            ),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=91,
                kernel_size=3,
                stride=1,
                padding=3 // 2,
                # Include bias because we're not using batch norm.
                bias=True,
            ),
            nn.Flatten(),
        )

    def forward(self, occupancies):
        # Add an all-ones channel to the input for edge detection.
        ones = torch.ones(
            occupancies.shape[0],
            1,
            occupancies.shape[2],
            occupancies.shape[3],
            device=occupancies.device,
            dtype=torch.float16,
        )
        x = torch.cat([occupancies, ones], dim=1)  # Shape: (batch, 5, board_size, board_size)
        x = self.convolutional_block(x)
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        return (
            self.value_head(x),
            self.policy_head(x),
        )

# # # # # # # # # # # # # # #
# Inference Actor
# # # # # # # # # # # # # # #

@ray.remote(
    num_gpus=1,
    runtime_env={
        "nsight": {
        "gpu-metrics-devices": "all",
    }},
)
# @ray.remote
class InferenceActor:
    def __init__(self) -> None:
        self.model = NeuralNet()
        self.model.to(device=torch.device(DEVICE), dtype=torch.float16)
        self.model.eval()

        self.kernel_lock = threading.Lock()

        self.load_data_stream = torch.cuda.Stream()
        self.kernel_stream = torch.cuda.Stream()
        self.unload_data_stream = torch.cuda.Stream()

        self.time_in_lock = 0.0

    def evaluate_batch(self, boards):
        # TODO: The duration that we hold the kernel lock should be very short -- everything
        #       is happening async in here?
        with self.kernel_lock:
            start_time = time.perf_counter()
            with torch.cuda.stream(self.load_data_stream):
                boards_tensor = torch.from_numpy(boards.copy()).to(
                    dtype=torch.float16,
                    device=torch.device(DEVICE),
                    non_blocking=True,
                )

            self.kernel_stream.wait_stream(self.load_data_stream)
            with torch.cuda.stream(self.kernel_stream):
                with torch.inference_mode():
                    values_logits_tensor, policy = self.model(boards_tensor)
                    values = torch.softmax(values_logits_tensor, dim=1)

            self.unload_data_stream.wait_stream(self.kernel_stream)
            with torch.cuda.stream(self.unload_data_stream):
                values = values.to(device="cpu", non_blocking=True)
                policy = policy.to(device="cpu", non_blocking=True)

            end_time = time.perf_counter()

        self.time_in_lock += end_time - start_time

        # TODO: Add a RecordStream!?

        # Force the unload stream to finish before returning.
        self.unload_data_stream.synchronize()

        return values.numpy(), policy.numpy()

# # # # # # # # # # # # # # #
# Gameplay Actor
# # # # # # # # # # # # # # #
@ray.remote
class GameplayActor:
    def __init__(self, inference_actor):
        self.inference_actor = inference_actor

    def run(self):
        start_time = time.time()
        while time.time() - start_time < 60:
            boards = np.random.randint(0, 1, size=(128, 4, 20, 20))
            values, policy = ray.get(self.inference_actor.evaluate_batch.remote(boards))
            time.sleep(1e-3)

ray.init(log_to_driver=True)
inference_actor = InferenceActor.options(max_concurrency=50).remote()
gameplay_actors = [
    GameplayActor.remote(inference_actor)
    for _ in range(12)
]
for gameplay_actor in gameplay_actors:
    gameplay_actor.run.remote()

# Give the actors time to wrap up.
time.sleep(70)
