import torch
import time
from dataclasses import dataclass
import sys
import os

from alphablokus.configs import (
    GameConfig,
    NetworkConfig,
    TrainingConfig,
    DirectoriesConfig,
)
from alphablokus.train_utils import (
    maybe_download_files,
    load_game_data,
    load_initial_state,
    list_game_data_files,
    train_loop,
)
from alphablokus.files import from_localized
from prometheus_client import Gauge, start_http_server


@dataclass
class TrainingState:
    """Tracks the state of the training process."""

    samples_last_trained: int
    samples_since_last_save: int = 0


@dataclass
class TrainingPollStats:
    """Snapshot of training poll metrics to export to Prometheus."""

    samples_total_available: int
    new_samples: int
    num_game_files: int
    samples_selected_for_training: int = 0
    samples_processed_this_poll: int = 0


# Load configs.
config_path = sys.argv[1]
game_config = GameConfig(config_path)
network_config = NetworkConfig(config_path)
training_config = TrainingConfig(config_path)
directories_config = DirectoriesConfig(config_path)


SIMULATED = False
METRICS_PORT = int(os.getenv("TRAINING_METRICS_PORT", "9101"))

# Prometheus metrics for monitoring training progress.
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
    "Samples stepped through during the latest training poll (accounts for epochs).",
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


def log(message: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def start_metrics_server():
    """Start the Prometheus metrics HTTP server."""
    start_http_server(METRICS_PORT)
    log(f"Prometheus metrics server running on port {METRICS_PORT}")


def publish_metrics(
    state: TrainingState,
    poll_stats: TrainingPollStats,
    poll_duration_seconds: float,
):
    """Push the latest training metrics to Prometheus gauges."""
    SAMPLES_AVAILABLE_GAUGE.set(poll_stats.samples_total_available)
    GAME_FILES_GAUGE.set(poll_stats.num_game_files)
    NEW_SAMPLES_GAUGE.set(poll_stats.new_samples)
    SAMPLES_SELECTED_GAUGE.set(poll_stats.samples_selected_for_training)
    SAMPLES_PROCESSED_GAUGE.set(poll_stats.samples_processed_this_poll)
    SAMPLES_TRAINED_GAUGE.set(state.samples_last_trained)
    SAMPLES_SINCE_SAVE_GAUGE.set(state.samples_since_last_save)
    LAST_POLL_TS_GAUGE.set(time.time())
    POLL_DURATION_GAUGE.set(poll_duration_seconds)


def train_on_new_samples(
    model, optimizer, samples_last_trained: int
) -> TrainingPollStats:
    """
    Trains on new samples available since samples_last_trained.
    Returns snapshot metrics about the poll/training cycle.
    """
    # List game files to get current total
    listed_game_data_files = list_game_data_files(directories_config)

    # If we're loading one file at a time, then only load one additional file since
    # the samples_last_trained.
    if SIMULATED:
        game_data_files = []
        num_samples_in_game_data_files = 0
        # Reverse the list here to load the oldest files first.
        for filename, num_samples_in_file in reversed(listed_game_data_files):
            game_data_files.append((filename, num_samples_in_file))
            num_samples_in_game_data_files += num_samples_in_file

            # Break as soon as game_data_files contains more samples than we
            # trained on last time, indicating we got one more file this time.
            if num_samples_in_game_data_files > samples_last_trained:
                break
        # Then reverse back, so the newest files are first.
        game_data_files = game_data_files[::-1]
    else:
        # Otherwise, just load all the files.
        game_data_files = listed_game_data_files

    samples_total = sum(
        num_samples_in_file for _, num_samples_in_file in game_data_files
    )

    # Compute the number of new samples available since we last trained and prep the poll stats.
    new_samples = samples_total - samples_last_trained
    poll_stats = TrainingPollStats(
        samples_total_available=samples_total,
        new_samples=new_samples,
        num_game_files=len(listed_game_data_files),
    )

    if new_samples == 0:
        return poll_stats

    log(f"Number of new samples available since last trained: {new_samples}")

    # Compute the number of samples to train on.
    num_samples = int(new_samples * training_config.sampling_ratio)
    log(f"Number of samples to train on: {num_samples}")

    if num_samples == 0:
        log("Number of samples to train on is 0, skipping training.")
        return poll_stats

    # Fetch game files.
    local_game_data_files = maybe_download_files(
        game_data_files,
        num_samples,
        training_config.window_size,
    )
    if not local_game_data_files:
        log("No game data files found, skipping training.")
        return poll_stats

    # Build a dataloader.
    dataloader = load_game_data(
        game_config, training_config.batch_size, local_game_data_files, num_samples
    )
    poll_stats.samples_selected_for_training = len(dataloader.dataset)
    poll_stats.samples_processed_this_poll = (
        poll_stats.samples_selected_for_training * training_config.num_epochs
    )
    train_loop(dataloader, model, optimizer, training_config)

    if not SIMULATED:
        log(f"Deleting {len(local_game_data_files)} game data files")
        for file_name in local_game_data_files:
            os.remove(file_name)

    return poll_stats


def save_model_and_state(model, optimizer, samples_total: int):
    """Saves the model and training state."""
    if SIMULATED:
        time_suffix = f"_{time.time():.0f}"
    else:
        time_suffix = ""

    # Save the model locally.
    if directories_config.model_directory.strip():
        onnx_path = (
            directories_config.model_directory
            + f"{samples_total:08d}{time_suffix}.onnx"
        )
        log(f"Saving model to: {onnx_path}")
        with from_localized(onnx_path) as onnx_path:
            model.save_onnx(onnx_path, training_config.device)
            model.train()
    else:
        log("No model directory set, skipping model save.")

    # Save training state so we can resume training later.
    if directories_config.training_directory.strip():
        training_state_path = (
            directories_config.training_directory
            + f"{samples_total:08d}{time_suffix}.pth"
        )
        log(f"Saving training state to: {training_state_path}")
        with from_localized(training_state_path) as training_state_path:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                training_state_path,
            )
    else:
        log("No training directory set, skipping training state save.")


def run():
    # Load the initial state of the model and optimizer.
    model, optimizer, samples_last_trained = load_initial_state(
        network_config,
        game_config,
        training_config,
        directories_config,
        skip_loading_from_file=SIMULATED,
    )

    # Initialize training state
    state = TrainingState(samples_last_trained=samples_last_trained)

    start_metrics_server()
    SAMPLES_TRAINED_GAUGE.set(state.samples_last_trained)
    SAMPLES_SINCE_SAVE_GAUGE.set(state.samples_since_last_save)

    log("Starting training loop...")
    if state.samples_last_trained == 0:
        log("Starting from scratch (no previous training state found)")
    else:
        log(f"Resuming from {state.samples_last_trained} samples")
    log(f"Polling every {training_config.poll_interval_seconds} seconds")
    log(
        f"Will save model after accumulating {training_config.min_samples_for_save} new samples"
    )

    while True:
        poll_start_time = time.time()
        # Train on any new samples
        poll_stats = train_on_new_samples(model, optimizer, state.samples_last_trained)
        poll_duration_seconds = time.time() - poll_start_time

        # Update tracking
        new_samples_trained = (
            poll_stats.samples_total_available - state.samples_last_trained
        )
        state.samples_since_last_save += new_samples_trained
        state.samples_last_trained = poll_stats.samples_total_available

        # Save if we've accumulated enough new samples
        if state.samples_since_last_save >= training_config.min_samples_for_save:
            log(
                f"\nAccumulated {state.samples_since_last_save} new samples since last save. Saving model..."
            )
            save_model_and_state(model, optimizer, poll_stats.samples_total_available)
            state.samples_since_last_save = 0
            log("Save complete!\n")
        else:
            log(
                f"Accumulated {state.samples_since_last_save} new samples since last save. Waiting for {training_config.min_samples_for_save - state.samples_since_last_save} more samples before saving."
            )

        publish_metrics(state, poll_stats, poll_duration_seconds)

        # If there were no new samples, sleep before polling again for any new data.
        if new_samples_trained == 0:
            if SIMULATED:
                break

            # Wait before polling again
            log(
                f"Waiting {training_config.poll_interval_seconds} seconds before next poll..."
            )
            time.sleep(training_config.poll_interval_seconds)

    assert SIMULATED, "Should only end while loop in SIMULATED mode."

    save_model_and_state(model, optimizer, state.samples_last_trained)


run()
