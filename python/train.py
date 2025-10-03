import torch

from configs import GameConfig, NetworkConfig, TrainingConfig, DirectoriesConfig
from train_utils import (
    maybe_download_files,
    load_game_data,
    load_initial_state,
    list_game_data_files,
    train_loop,
)
from files import from_localized

# Load configs.
config_path = "configs/training/half.toml"
game_config = GameConfig(config_path)
network_config = NetworkConfig(config_path)
training_config = TrainingConfig(config_path)
directories_config = DirectoriesConfig(config_path)

# List game files
game_data_files = list_game_data_files(directories_config)
samples_total = sum(num_samples_in_file for _, num_samples_in_file in game_data_files)

# Load the initial state of the model and optimizer.
model, optimizer, samples_last_trained = load_initial_state(
    network_config, game_config, training_config, directories_config
)

# Compute the number of new samples available since we last trained.
new_samples = samples_total - samples_last_trained
print(f"Number of new samples available since last trained: {new_samples}")

# Compute the number of samples to train on.
num_samples = int(new_samples * training_config.sampling_ratio)
print(f"Number of samples to train on: {num_samples}")


def run():
    if num_samples == 0:
        print("Number of samples to train on is 0, skipping training.")
        return

    # Fetch game files.
    local_game_data_files = maybe_download_files(
        game_data_files,
        num_samples,
        training_config.window_size,
    )
    if not local_game_data_files:
        print("No game data files found, skipping training.")
        return

    # Build a dataloader.
    dataloader = load_game_data(
        game_config, training_config, local_game_data_files, num_samples
    )
    train_loop(dataloader, model, optimizer, training_config)

    # Save the model locally.
    if directories_config.model_directory.strip():
        onnx_path = directories_config.model_directory + f"{samples_total:08d}.onnx"
        print("Saving model to:", onnx_path)
        with from_localized(onnx_path) as onnx_path:
            model.save_onnx(onnx_path)
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

    print("Done!")


run()
