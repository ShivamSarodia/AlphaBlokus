# TODO: Before we start using this script, do some testing to make sure it works as expected.
#       I've eyeballed it but not confirmed correctness.

import argparse
import os
import time
from typing import Iterable

import torch
from prometheus_client import Gauge, start_http_server

from alphablokus.configs import GameConfig, NetworkConfig, TrainingLiveConfig
from alphablokus.data_loaders import (
    BufferedGameBatchDataset,
    WindowedS3FileProvider,
    build_streaming_dataloader,
    list_game_files_with_samples,
)
from alphablokus.train_utils import (
    get_loss,
    load_initial_state,
    log,
    save_model_and_state,
)


METRICS_PORT = int(os.getenv("TRAINING_METRICS_PORT", "9101"))

SAMPLES_AVAILABLE_GAUGE = Gauge(
    "alphablokus_training_samples_available",
    "Total gameplay samples currently available across data files.",
)
GAME_FILES_GAUGE = Gauge(
    "alphablokus_training_game_files_available",
    "Number of gameplay data files discovered in the latest poll.",
)
NEW_SAMPLES_GAUGE = Gauge(
    "alphablokus_training_new_samples_since_last_train",
    "New gameplay samples observed since the previous training iteration.",
)
SAMPLES_SELECTED_GAUGE = Gauge(
    "alphablokus_training_samples_selected_for_training",
    "Samples pulled into the latest training dataloader.",
)
SAMPLES_PROCESSED_GAUGE = Gauge(
    "alphablokus_training_samples_processed_in_last_poll",
    "Samples stepped through during the latest training poll.",
)
SAMPLES_TRAINED_GAUGE = Gauge(
    "alphablokus_training_samples_trained_total",
    "Total samples accounted for in the latest training state.",
)
SAMPLES_SINCE_SAVE_GAUGE = Gauge(
    "alphablokus_training_samples_since_last_save",
    "Samples accumulated since the last model checkpoint save.",
)
LAST_POLL_TS_GAUGE = Gauge(
    "alphablokus_training_last_poll_timestamp_seconds",
    "Unix timestamp of the latest poll for gameplay data.",
)
POLL_DURATION_GAUGE = Gauge(
    "alphablokus_training_poll_duration_seconds",
    "Time spent in the most recent training poll cycle.",
)


def start_metrics_server() -> None:
    start_http_server(METRICS_PORT)
    log(f"Prometheus metrics server running on port {METRICS_PORT}")


def publish_metrics(
    *,
    samples_total_available: int,
    new_samples: int,
    num_game_files: int,
    samples_selected_for_training: int,
    samples_processed_this_poll: int,
    samples_trained_total: int,
    samples_since_last_save: int,
    poll_duration_seconds: float,
) -> None:
    SAMPLES_AVAILABLE_GAUGE.set(samples_total_available)
    GAME_FILES_GAUGE.set(num_game_files)
    NEW_SAMPLES_GAUGE.set(new_samples)
    SAMPLES_SELECTED_GAUGE.set(samples_selected_for_training)
    SAMPLES_PROCESSED_GAUGE.set(samples_processed_this_poll)
    SAMPLES_TRAINED_GAUGE.set(samples_trained_total)
    SAMPLES_SINCE_SAVE_GAUGE.set(samples_since_last_save)
    LAST_POLL_TS_GAUGE.set(time.time())
    POLL_DURATION_GAUGE.set(poll_duration_seconds)


def train_for_samples(
    dataloader: Iterable,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    training_config: TrainingLiveConfig,
    max_samples: int,
) -> int:
    samples_trained = 0

    for batch in dataloader:
        loss, value_loss, policy_loss = get_loss(
            batch,
            model,
            device=training_config.device,
            policy_loss_weight=training_config.policy_loss_weight,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = batch[0].shape[0]
        samples_trained += batch_size

        if samples_trained >= max_samples:
            break

    log(
        f"Trained {samples_trained}/{max_samples} samples. "
        f"Loss: {loss.item():.4f} (Value: {value_loss.item():.4f}, Policy: {policy_loss.item():.4f})"
    )
    return samples_trained


def run_live_training(config_path: str) -> None:
    game_config = GameConfig(config_path)
    network_config = NetworkConfig(config_path)
    training_config = TrainingLiveConfig(config_path)

    if training_config.num_workers not in (0, 1):
        raise ValueError("Live training supports num_workers of 0 or 1.")

    model, optimizer, samples_last_trained = load_initial_state(
        network_config,
        game_config,
        training_config,
        training_config.training_directory,
        skip_loading_from_file=False,
    )
    samples_since_last_save = 0
    start_metrics_server()

    while True:
        poll_start = time.time()
        files = list_game_files_with_samples(training_config.game_data_directory)
        samples_total = sum(file_info.num_samples for file_info in files)
        new_samples = samples_total - samples_last_trained

        if new_samples <= 0:
            log("No new samples found.")
            time.sleep(training_config.poll_interval_seconds)
            continue

        log(f"Found {new_samples} new samples (total={samples_total}).")

        if samples_total > training_config.window_size * 2:
            window_size = training_config.window_size
        else:
            window_size = max(samples_total / 2, 50000)

        sampling_ratio = max(
            min(samples_total / window_size, training_config.sampling_ratio), 1.0
        )
        target_samples = int(new_samples * sampling_ratio)

        if target_samples == 0:
            log("Target samples is 0, skipping training.")
            time.sleep(training_config.poll_interval_seconds)
            continue

        log(
            f"Training on {target_samples} samples "
            f"(window_size={window_size}, sampling_ratio={sampling_ratio:.2f})."
        )

        file_provider = WindowedS3FileProvider(
            training_config.game_data_directory,
            window_size_samples=int(window_size),
        )
        dataset = BufferedGameBatchDataset(
            game_config,
            file_provider,
            training_config.batch_size,
            training_config.in_memory_shuffle_file_count,
            local_cache_dir=training_config.local_cache_dir or None,
            cleanup_local_files=training_config.cleanup_local_files,
        )
        dataloader = build_streaming_dataloader(
            dataset,
            num_workers=training_config.num_workers,
            prefetch_factor=training_config.prefetch_factor,
        )
        samples_trained = train_for_samples(
            dataloader,
            model,
            optimizer,
            training_config,
            target_samples,
        )

        samples_last_trained += new_samples
        samples_since_last_save += samples_trained
        poll_duration_seconds = time.time() - poll_start
        log(f"Poll finished in {poll_duration_seconds:.1f}s.")
        log(f"Trained {samples_trained} samples this poll.")

        publish_metrics(
            samples_total_available=samples_total,
            new_samples=new_samples,
            num_game_files=len(files),
            samples_selected_for_training=target_samples,
            samples_processed_this_poll=samples_trained,
            samples_trained_total=samples_last_trained,
            samples_since_last_save=samples_since_last_save,
            poll_duration_seconds=poll_duration_seconds,
        )

        if samples_since_last_save >= training_config.min_samples_for_save:
            save_model_and_state(
                model=model,
                optimizer=optimizer,
                name=samples_last_trained,
                model_directory=training_config.model_directory,
                training_directory=training_config.training_directory,
                device=training_config.device,
            )
            samples_since_last_save = 0

        time.sleep(training_config.poll_interval_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live training script")
    parser.add_argument("config", help="Path to config TOML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_live_training(args.config)


if __name__ == "__main__":
    main()
