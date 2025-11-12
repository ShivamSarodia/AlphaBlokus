import torch
import time
from dataclasses import dataclass
import sys

from configs import GameConfig, NetworkConfig, TrainingConfig, DirectoriesConfig
from train_utils import (
    maybe_download_files,
    load_game_data,
    load_initial_state,
    list_game_data_files,
    train_loop,
)
from files import from_localized


@dataclass
class TrainingState:
    """Tracks the state of the training process."""

    samples_last_trained: int
    samples_since_last_save: int = 0


# Load configs.
config_path = sys.argv[1]
game_config = GameConfig(config_path)
network_config = NetworkConfig(config_path)
training_config = TrainingConfig(config_path)
directories_config = DirectoriesConfig(config_path)


def train_on_new_samples(model, optimizer, samples_last_trained: int) -> int:
    """
    Trains on new samples available since samples_last_trained.
    Returns the new total number of samples after training.
    """
    # List game files to get current total
    game_data_files = list_game_data_files(directories_config)
    samples_total = sum(
        num_samples_in_file for _, num_samples_in_file in game_data_files
    )

    # Compute the number of new samples available since we last trained.
    new_samples = samples_total - samples_last_trained
    if new_samples == 0:
        return samples_last_trained

    print(f"Number of new samples available since last trained: {new_samples}")

    # Compute the number of samples to train on.
    num_samples = int(new_samples * training_config.sampling_ratio)
    print(f"Number of samples to train on: {num_samples}")

    if num_samples == 0:
        print("Number of samples to train on is 0, skipping training.")
        return samples_last_trained

    # Fetch game files.
    local_game_data_files = maybe_download_files(
        game_data_files,
        num_samples,
        training_config.window_size,
    )
    if not local_game_data_files:
        print("No game data files found, skipping training.")
        return samples_last_trained

    # Build a dataloader.
    dataloader = load_game_data(
        game_config, training_config, local_game_data_files, num_samples
    )
    train_loop(dataloader, model, optimizer, training_config)

    return samples_total


def save_model_and_state(model, optimizer, samples_total: int):
    """Saves the model and training state."""
    # Save the model locally.
    if directories_config.model_directory.strip():
        onnx_path = directories_config.model_directory + f"{samples_total:08d}.onnx"
        print("Saving model to:", onnx_path)
        with from_localized(onnx_path) as onnx_path:
            model.save_onnx(onnx_path, training_config.device)
    else:
        print("No model directory set, skipping model save.")

    # Save training state so we can resume training later.
    if directories_config.training_directory.strip():
        training_state_path = (
            directories_config.training_directory + f"{samples_total:08d}.pth"
        )
        print("Saving training state to:", training_state_path)
        with from_localized(training_state_path) as training_state_path:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                training_state_path,
            )
    else:
        print("No training directory set, skipping training state save.")


def run():
    # Load the initial state of the model and optimizer.
    model, optimizer, samples_last_trained = load_initial_state(
        network_config, game_config, training_config, directories_config
    )

    # Initialize training state
    state = TrainingState(samples_last_trained=samples_last_trained)

    print("Starting training loop...")
    if state.samples_last_trained == 0:
        print("Starting from scratch (no previous training state found)")
    else:
        print(f"Resuming from {state.samples_last_trained} samples")
    print(f"Polling every {training_config.poll_interval_seconds} seconds")
    print(
        f"Will save model after accumulating {training_config.min_samples_for_save} new samples"
    )
    print()

    while True:
        # Train on any new samples
        samples_total = train_on_new_samples(
            model, optimizer, state.samples_last_trained
        )

        # Update tracking
        new_samples_trained = samples_total - state.samples_last_trained
        state.samples_since_last_save += new_samples_trained
        state.samples_last_trained = samples_total

        # Save if we've accumulated enough new samples
        if state.samples_since_last_save >= training_config.min_samples_for_save:
            print(
                f"\nAccumulated {state.samples_since_last_save} new samples since last save. Saving model..."
            )
            save_model_and_state(model, optimizer, samples_total)
            state.samples_since_last_save = 0
            print("Save complete!\n")
        else:
            print(
                f"Accumulated {state.samples_since_last_save} new samples since last save. Waiting for {training_config.min_samples_for_save - state.samples_since_last_save} more samples before saving."
            )

        # If there were no new samples, sleep before polling again for any new data.
        if new_samples_trained == 0:
            # Wait before polling again
            print(
                f"Waiting {training_config.poll_interval_seconds} seconds before next poll..."
            )
            time.sleep(training_config.poll_interval_seconds)


run()
