import time
import torch
import numpy as np
from alpha_blokus.neural_net import NeuralNet

def run(cfg):
    """Benchmark neural network evaluation performance."""
    # Set up device and dtype
    device = torch.device(cfg.networks.main.device)
    dtype = getattr(torch, cfg.networks.main.inference_dtype)

    # Initialize model
    model = NeuralNet(cfg.networks.main, cfg, flatten_policy=True)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    # Benchmark parameters
    BATCH_SIZE = cfg.networks.main.batch_size
    BENCHMARK_DURATION = cfg.benchmark_duration
    NUM_WARMUP_BATCHES = cfg.num_warmup_batches
    
    print(f"Benchmarking neural network evaluation...")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Board size: {cfg.game.board_size}")
    print(f"Benchmark duration: {BENCHMARK_DURATION} seconds")
    print()

    # Pre-generate data for warmup and benchmark
    # Estimate how many batches we'll need (add some buffer)
    estimated_batches = int(BENCHMARK_DURATION * 500) + NUM_WARMUP_BATCHES + 1000  # Conservative estimate
    print(f"Pre-generating {estimated_batches} batches of random data...")

    pre_generated_data = []
    for i in range(estimated_batches):
        random_boards = np.random.random((BATCH_SIZE, 4, cfg.game.board_size, cfg.game.board_size))
        pre_generated_data.append(random_boards)
    
    print(f"Data generation complete. Starting benchmark...")
    print()

    # Warm up
    print("Warming up...")
    for i in range(NUM_WARMUP_BATCHES):
        boards = torch.from_numpy(pre_generated_data[i]).to(device=device, dtype=dtype)
        with torch.no_grad():
            value, policy = model(boards)
            value = value.cpu().numpy()
            policy = policy.cpu().numpy()

    # Benchmark with separate time tracking
    print("Running benchmark...")
    start_time = time.perf_counter()
    num_batches = 0
    total_evaluations = 0
    
    # Time counters
    total_host_to_device_time = 0.0
    total_model_eval_time = 0.0
    total_device_to_host_time = 0.0
    
    while time.perf_counter() - start_time < BENCHMARK_DURATION:
        # Use pre-generated data (cycle through if we run out)
        data_index = (NUM_WARMUP_BATCHES + num_batches) % len(pre_generated_data)
        
        # Time 1: Host to device transfer
        h2d_start = time.perf_counter()
        boards = torch.from_numpy(pre_generated_data[data_index]).to(device=device, dtype=dtype)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        h2d_end = time.perf_counter()
        total_host_to_device_time += (h2d_end - h2d_start)
        
        # Time 2: Model evaluation
        eval_start = time.perf_counter()
        with torch.inference_mode():
            value, policy = model(boards)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        eval_end = time.perf_counter()
        total_model_eval_time += (eval_end - eval_start)
        
        # Time 3: Device to host transfer
        d2h_start = time.perf_counter()
        value = value.cpu().numpy()
        policy = policy.cpu().numpy()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        d2h_end = time.perf_counter()
        total_device_to_host_time += (d2h_end - d2h_start)
        
        num_batches += 1
        total_evaluations += BATCH_SIZE
        
        if num_batches % 10 == 0:
            elapsed = time.perf_counter() - start_time
            evals_per_second = total_evaluations / elapsed
            print(f"Batch {num_batches}: {evals_per_second:.1f} evaluations/second (total time)")

    # Final results
    total_time = time.perf_counter() - start_time
    evaluations_per_second = total_evaluations / total_time
    avg_time_per_eval = total_time / total_evaluations
    
    # Calculate percentages
    h2d_percent = (total_host_to_device_time / total_time) * 100
    eval_percent = (total_model_eval_time / total_time) * 100
    d2h_percent = (total_device_to_host_time / total_time) * 100
    other_percent = 100 - (h2d_percent + eval_percent + d2h_percent)
    
    print()
    print("=== Benchmark Results ===")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Total batches: {num_batches}")
    print(f"Total evaluations: {total_evaluations}")
    print(f"Evaluations per second: {evaluations_per_second:.1f}")
    print(f"Average time per evaluation: {avg_time_per_eval:.6f} seconds")
    print(f"Average time per batch: {total_time / num_batches:.6f} seconds")
    print()
    print("=== Time Breakdown ===")
    print(f"Host-to-device transfer: {total_host_to_device_time:.3f}s ({h2d_percent:.1f}%)")
    print(f"Model evaluation:        {total_model_eval_time:.3f}s ({eval_percent:.1f}%)")
    print(f"Device-to-host transfer: {total_device_to_host_time:.3f}s ({d2h_percent:.1f}%)")
    print(f"Other overhead:          {total_time - total_host_to_device_time - total_model_eval_time - total_device_to_host_time:.3f}s ({other_percent:.1f}%)")
    print()
    print("=== Pure Model Performance ===")
    print(f"Pure model evaluations per second: {total_evaluations / total_model_eval_time:.1f}")
    print(f"Average pure model time per evaluation: {total_model_eval_time / total_evaluations:.6f} seconds") 