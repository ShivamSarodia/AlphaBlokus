import torch

from configs import GameConfig, NetworkConfig, TrainingConfig, DirectoriesConfig
from train_utils import (
    maybe_download_files,
    load_game_data,
    get_loss,
    load_initial_state,
    list_game_data_files,
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

num_samples = int(new_samples * training_config.sampling_ratio)

# Download game files and build dataloader.
local_game_data_files = maybe_download_files(
    game_data_files,
    num_samples,
    training_config.window_size,
)
if local_game_data_files:
    dataloader = load_game_data(
        game_config, training_config, local_game_data_files, num_samples
    )

    # Train the model for the given number of epochs.
    for epoch in range(training_config.num_epochs):
        print(f"Epoch {epoch + 1} of {training_config.num_epochs}")

        for batch in dataloader:
            loss, value_loss, policy_loss = get_loss(batch, training_config, model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                f"Train loss: {loss.item()}. (Value loss: {value_loss.item()}, Policy loss: {policy_loss.item()})"
            )
else:
    print("No game data files found.")

# Save the model locally.
onnx_path = directories_config.model_directory + f"{samples_total:08d}.onnx"
print("Saving model to:", onnx_path)
with from_localized(onnx_path) as onnx_path:
    model.save_onnx(onnx_path)

# Save training state so we can resume training later.
training_state_path = directories_config.training_directory + f"{samples_total:08d}.pth"
print("Saving training state to:", training_state_path)
with from_localized(training_state_path) as training_state_path:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        training_state_path,
    )

print("Done!")
