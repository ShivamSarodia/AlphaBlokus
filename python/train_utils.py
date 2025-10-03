from typing import List, Tuple
import random
import zstandard
import msgpack
import torch
import torch.nn as nn

from configs import DirectoriesConfig, GameConfig, NetworkConfig, TrainingConfig
from res_net import NeuralNet
from files import latest_file, localize_file, list_files


def list_game_data_files(
    directories_config: DirectoriesConfig,
) -> List[Tuple[str, int]]:
    """
    Returns a list of game data files from the source directory.
    """
    print("Getting game data files from:", directories_config.game_data_directory)
    files = sorted(
        list_files(directories_config.game_data_directory, ".bin"), reverse=True
    )
    result = []
    for file in files:
        num_samples_in_file = int(file.split(".")[-2].split("_")[-1])
        result.append((file, num_samples_in_file))
    return result


def maybe_download_files(
    game_data_files, num_samples: int, window_size: int
) -> List[str]:
    """
    Returns a list of local file paths that represent at least the number of samples
    requested, pulled from the latest window size samples from the source directory.

    (If needed, this method downloads the files from the source directory to the local
    directory.)
    """
    files_in_window = []
    samples_in_window = 0
    samples_total = 0
    for file, num_samples_in_file in game_data_files:
        # If we don't have enough samples in the window, add the file to
        # the window.
        if samples_in_window < window_size:
            files_in_window.append((file, num_samples_in_file))
            samples_in_window += num_samples_in_file

        samples_total += num_samples_in_file

    print(
        f"Found {samples_total} total samples across {len(game_data_files)} game data files."
    )
    print(
        f"Assembled window with {samples_in_window} samples across {len(files_in_window)} files."
    )

    # Now, files_in_window is loaded with all files in the window. Collect a subset
    # of the files that contain the number of samples requested to actually download.
    # (If we don't have enough samples, we might not get the requested number of samples.)

    # Create a paired list and shuffle it to ensure fair random selection
    # (each file has equal probability of being selected regardless of size)
    random.shuffle(files_in_window)

    # Select files until we have at least num_samples
    selected_files = []
    total_samples = 0
    for file_path, sample_count in files_in_window:
        selected_files.append(file_path)
        total_samples += sample_count

        if total_samples >= num_samples:
            break

    print(f"Selected {len(selected_files)} game data files for download.")

    # Localize the selected files (download if needed)
    return [localize_file(file_path) for file_path in selected_files]


def load_initial_state(
    network_config: NetworkConfig,
    game_config: GameConfig,
    training_config: TrainingConfig,
    directories_config: DirectoriesConfig,
) -> tuple[NeuralNet, torch.optim.Optimizer, int]:
    """
    Loads the initial state of the model and optimizer from the training directory.
    """
    # Load the model and optimizer.
    model = NeuralNet(network_config, game_config).to(device=training_config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)

    initial_training_state = latest_file(directories_config.training_directory, ".pth")
    if initial_training_state is None:
        print("No training state found, starting from scratch.")
        samples = 0
    else:
        print("Loading training state from:", initial_training_state)
        samples = int(initial_training_state.split(".")[-2].split("/")[-1])
        initial_training_path = localize_file(initial_training_state)
        initial_training_state = torch.load(initial_training_path)
        model.load_state_dict(initial_training_state["model"])
        optimizer.load_state_dict(initial_training_state["optimizer"])

    return model, optimizer, samples


def load_game_data(
    game_config: GameConfig,
    training_config: TrainingConfig,
    local_file_paths: List[str],
    num_samples: int,
) -> torch.utils.data.DataLoader:
    """
    Loads the game data from the given local file paths.
    """
    board_inputs = []
    value_targets = []
    policy_targets = []

    for filename in local_file_paths:
        with zstandard.open(filename, "rb") as f:
            game_data_list = f.read()
            game_data_list = msgpack.unpackb(game_data_list)

            for game_data in game_data_list:
                board = [game_data["board"]["slices"][i]["cells"] for i in range(4)]
                board_inputs.append(torch.tensor(board, dtype=torch.float32))

                value_targets.append(torch.tensor(game_data["game_result"]))

                policy_target = torch.zeros(
                    (
                        game_config.num_piece_orientations,
                        game_config.board_size,
                        game_config.board_size,
                    )
                )

                for valid_move_tuple, visit_count in zip(
                    game_data["valid_move_tuples"], game_data["visit_counts"]
                ):
                    piece_orientation_index, center_x, center_y = valid_move_tuple
                    policy_target[piece_orientation_index, center_x, center_y] = (
                        visit_count
                    )

                policy_target = policy_target / policy_target.sum()
                policy_targets.append(policy_target)

    board_inputs = torch.stack(board_inputs)
    value_targets = torch.stack(value_targets)
    policy_targets = torch.stack(policy_targets)

    # Load the dataset and truncate to the number of samples requested.
    dataset = torch.utils.data.TensorDataset(
        board_inputs, value_targets, policy_targets
    )
    dataset = torch.utils.data.random_split(
        dataset, [num_samples, len(dataset) - num_samples]
    )[0]

    # Return a dataloader for the dataset.
    return torch.utils.data.DataLoader(
        dataset, batch_size=training_config.batch_size, shuffle=True
    )


def get_loss(
    batch, training_config: TrainingConfig, model: nn.Module
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Forward pass
    board, expected_value, expected_policy = batch
    board = board.to(training_config.device)
    expected_value = expected_value.to(training_config.device)
    expected_policy = expected_policy.to(training_config.device)
    pred_value, pred_policy = model(board)

    # Calculate value loss.
    value_loss = nn.CrossEntropyLoss()(pred_value, expected_value)

    # Calculate policy loss.
    pred_policy = pred_policy.view(pred_policy.shape[0], -1)
    expected_policy = expected_policy.view(expected_policy.shape[0], -1)
    policy_loss = training_config.policy_loss_weight * nn.CrossEntropyLoss()(
        pred_policy, expected_policy
    )

    return value_loss + policy_loss, value_loss, policy_loss
