from typing import List, Tuple
import random
import zstandard
import msgpack
import torch
import torch.nn as nn
import math

from alphablokus.configs import (
    DirectoriesConfig,
    GameConfig,
    NetworkConfig,
    TrainingConfig,
)
from alphablokus.files import latest_file, localize_file, list_files
from alphablokus.res_net import NeuralNet
from alphablokus.res_net_conv_value import NeuralNet as ResNetConvValueNet
from alphablokus.trivial_net import TrivialNet


def initialize_model(
    network_config: NetworkConfig,
    game_config: GameConfig,
) -> nn.Module:
    if network_config.model_class == "trivial":
        return TrivialNet(game_config)
    elif network_config.model_class == "resnet":
        return NeuralNet(network_config, game_config)
    elif network_config.model_class == "resnet_conv_value":
        return ResNetConvValueNet(network_config, game_config)
    else:
        raise ValueError(f"Invalid model class: {network_config.model_class}")


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

    print(
        f"Selected {len(selected_files)} game data files with {total_samples} samples for download."
    )

    # Localize the selected files (download if needed)
    print("Localizing files...")
    r = [localize_file(file_path) for file_path in selected_files]
    print("Files localized.")
    return r


def load_initial_state(
    network_config: NetworkConfig,
    game_config: GameConfig,
    training_config: TrainingConfig,
    directories_config: DirectoriesConfig,
    skip_loading_from_file: bool = False,
) -> tuple[nn.Module, torch.optim.Optimizer, int]:
    """
    Loads the initial state of the model and optimizer from the training directory.
    """
    # Load the model and optimizer.
    model = initialize_model(network_config, game_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
    samples = 0

    if not skip_loading_from_file:
        initial_training_state = latest_file(
            directories_config.training_directory, ".pth"
        )
        if initial_training_state is None:
            print("No training state found, starting from scratch.")

        else:
            print("Loading training state from:", initial_training_state)
            samples = int(initial_training_state.split(".")[-2].split("/")[-1])
            initial_training_path = localize_file(initial_training_state)
            initial_training_state = torch.load(
                initial_training_path, map_location=training_config.device
            )
            model.load_state_dict(initial_training_state["model"])
            optimizer.load_state_dict(initial_training_state["optimizer"])

    # Move the model to the appropriate device.
    model = model.to(device=training_config.device)

    # Move optimizer state to the same device as the model.
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(training_config.device)

    return model, optimizer, samples


def load_game_file(
    game_config: GameConfig,
    local_file_path: str,
) -> tuple[
    List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]
]:
    boards = []
    values = []
    policies = []
    valid_masks = []
    with zstandard.open(local_file_path, "rb") as f:
        game_data_list = f.read()
        game_data_list = msgpack.unpackb(game_data_list)

        for game_data in game_data_list:
            board = [game_data["board"]["slices"][i]["cells"] for i in range(4)]
            boards.append(torch.as_tensor(board, dtype=torch.float32))

            values.append(torch.as_tensor(game_data["game_result"]))

            policy_target = torch.zeros(
                (
                    game_config.num_piece_orientations,
                    game_config.board_size,
                    game_config.board_size,
                ),
                dtype=torch.float32,
            )

            valid_move_tuples = torch.as_tensor(
                game_data["valid_move_tuples"], dtype=torch.int32
            )
            visit_counts = torch.as_tensor(
                game_data["visit_counts"], dtype=torch.float32
            )

            valid_mask = torch.zeros(
                (
                    game_config.num_piece_orientations,
                    game_config.board_size,
                    game_config.board_size,
                ),
                dtype=torch.bool,
            )

            policy_target[
                valid_move_tuples[:, 0],
                valid_move_tuples[:, 1],
                valid_move_tuples[:, 2],
            ] = visit_counts

            policies.append(policy_target / policy_target.sum())
            valid_mask[
                valid_move_tuples[:, 0],
                valid_move_tuples[:, 1],
                valid_move_tuples[:, 2],
            ] = True
            valid_masks.append(valid_mask)

    return boards, values, policies, valid_masks


def load_game_files_to_tensor(
    game_config: GameConfig,
    local_file_paths: List[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    boards = []
    values = []
    policies = []
    valid_masks = []
    for file_path in local_file_paths:
        board, value, policy, valid_mask = load_game_file(game_config, file_path)
        boards += board
        values += value
        policies += policy
        valid_masks += valid_mask
    return (
        torch.stack(boards),
        torch.stack(values),
        torch.stack(policies),
        torch.stack(valid_masks),
    )


def load_game_data(
    game_config: GameConfig,
    batch_size: int,
    local_file_paths: List[str],
    num_samples: int,
) -> torch.utils.data.DataLoader:
    """
    Loads the game data from the given local file paths.
    """
    board_inputs = []
    value_targets = []
    policy_targets = []
    valid_policy_masks = []

    for filename in local_file_paths:
        print(f"Loading game file: {filename}")
        board, value, policy, valid_mask = load_game_file(game_config, filename)
        print(f"Loaded game file: {filename}")
        board_inputs += board
        value_targets += value
        policy_targets += policy
        valid_policy_masks += valid_mask

    board_inputs = torch.stack(board_inputs)
    value_targets = torch.stack(value_targets)
    policy_targets = torch.stack(policy_targets)
    valid_policy_masks = torch.stack(valid_policy_masks)

    # Load the dataset and truncate to the number of samples requested.
    dataset = torch.utils.data.TensorDataset(
        board_inputs, value_targets, policy_targets, valid_policy_masks
    )
    dataset = torch.utils.data.random_split(
        dataset, [num_samples, len(dataset) - num_samples]
    )[0]

    # Return a dataloader for the dataset.
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


class IterableGameDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        game_config: GameConfig,
        local_file_paths: List[str],
        shuffle_buffer_file_count: int,
    ):
        self.game_config = game_config
        self.local_file_paths = local_file_paths
        self.shuffle_buffer_file_count = shuffle_buffer_file_count

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            file_paths = self.local_file_paths.copy()
        else:
            num_workers = worker_info.num_workers
            per_worker = math.ceil(len(self.local_file_paths) / num_workers)
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(self.local_file_paths))
            file_paths = self.local_file_paths[start:end]

        while file_paths:
            # Load files into the shuffle buffer
            files_in_buffer = []
            while len(files_in_buffer) < self.shuffle_buffer_file_count and file_paths:
                files_in_buffer.append(file_paths.pop())

            print(f"Loading files into buffer (worker id: {worker_info.id})")
            boards, values, policies, valid_masks = load_game_files_to_tensor(
                self.game_config, files_in_buffer
            )
            print(f"Loaded files into buffer (worker id: {worker_info.id})")

            indices = random.sample(range(boards.shape[0]), boards.shape[0])
            for index in indices:
                yield (
                    boards[index],
                    values[index],
                    policies[index],
                    valid_masks[index],
                )


def get_loss(
    batch, training_config: TrainingConfig, model: nn.Module
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Forward pass
    board, expected_value, expected_policy, valid_policy_mask = batch
    board = board.to(training_config.device)
    expected_value = expected_value.to(training_config.device)
    expected_policy = expected_policy.to(training_config.device)
    valid_policy_mask = valid_policy_mask.to(training_config.device)
    pred_value, pred_policy = model(board)

    # Calculate value loss.
    value_loss = nn.CrossEntropyLoss()(pred_value, expected_value)

    # Calculate policy loss.
    pred_policy = pred_policy.view(pred_policy.shape[0], -1)
    expected_policy = expected_policy.view(expected_policy.shape[0], -1)
    valid_policy_mask = valid_policy_mask.view(valid_policy_mask.shape[0], -1)
    pred_policy[~valid_policy_mask] = -1e6

    policy_loss = training_config.policy_loss_weight * nn.CrossEntropyLoss()(
        pred_policy, expected_policy
    )

    return value_loss + policy_loss, value_loss, policy_loss


def train_loop(
    dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    training_config: TrainingConfig,
):
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
