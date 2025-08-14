import torch
from torch import nn
import torch.nn.functional as F
import ray
import numpy as np
import time
import threading
import contextlib
from torch.profiler import profile, record_function, ProfilerActivity

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
        # return F.relu(x + self.convolutional_block(x))
        return self.convolutional_block(x)

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
        x = occupancies
        x = self.convolutional_block(x)
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        return x

model = NeuralNet().to(device=DEVICE, dtype=torch.float16)

total_start = time.perf_counter()
total_times = [0.0 for _ in range(10)]

# with profile(
#     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#     record_shapes=True,
#     profile_memory=True,
#     with_stack=True
# ) as prof:
with contextlib.nullcontext() as prof:
    for _ in range(1000):
        times = []

        times.append(time.perf_counter())

        stack = 5
        boards = np.random.randint(0, 1, size=(128, stack, 20, 20))

        times.append(time.perf_counter())

        # with torch.cuda.stream(self.load_data_stream):
        with record_function("to_gpu"):
            boards_tensor = torch.from_numpy(boards.copy()).to(
                dtype=torch.float16,
                device=torch.device(DEVICE),
                non_blocking=True,
            )

        times.append(time.perf_counter())

        with record_function("model_forward"):
            # with torch.cuda.stream(self.model_stream):
            # Run the model on the GPU.
            policy = model(boards_tensor)

        times.append(time.perf_counter())

        policy_cpu = policy.to(device="cpu", non_blocking=True)

        times.append(time.perf_counter())

        with record_function("synchronize"):
            torch.cuda.synchronize()

        times.append(time.perf_counter())

        for i in range(len(times) - 1):
            total_times[i] += (times[i+1] - times[i])

print("Times: ", total_times)
print("Total time: ", time.perf_counter() - total_start)

# at the end, dump a summary:
# print(prof.key_averages().table(
#     sort_by="cuda_time_total", row_limit=10
# ))
# # and if you want a Chrome‐trace:
# prof.export_chrome_trace("trace.json")
