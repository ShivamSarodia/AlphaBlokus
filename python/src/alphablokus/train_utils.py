import os
import torch
import torch.nn as nn
import time

from alphablokus.configs import GameConfig, NetworkConfig
from alphablokus.files import from_localized, is_s3, latest_file, localize_file
from alphablokus.res_net import NeuralNet
from alphablokus.res_net_bottleneck import NeuralNet as ResNetBottleneckNet
from alphablokus.res_net_bottleneck_double import NeuralNet as ResNetBottleneckDoubleNet
from alphablokus.res_net_se import NeuralNet as ResNetSENet
from alphablokus.res_net_global_pool import NeuralNet as ResNetGlobalPoolNet
from alphablokus.res_net_preact import NeuralNet as ResNetPreactNet
from alphablokus.trivial_net import TrivialNet


class TrainingError(Exception):
    pass


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def initialize_model(
    network_config: NetworkConfig,
    game_config: GameConfig,
) -> nn.Module:
    if network_config.model_class == "trivial":
        return TrivialNet(game_config)
    elif network_config.model_class == "resnet":
        return NeuralNet(network_config, game_config)
    elif network_config.model_class == "resnet_bottleneck":
        return ResNetBottleneckNet(network_config, game_config)
    elif network_config.model_class == "resnet_bottleneck_double":
        return ResNetBottleneckDoubleNet(network_config, game_config)
    elif network_config.model_class == "resnet_se":
        return ResNetSENet(network_config, game_config)
    elif network_config.model_class == "resnet_global_pool":
        return ResNetGlobalPoolNet(network_config, game_config)
    elif network_config.model_class == "resnet_preact":
        return ResNetPreactNet(network_config, game_config)
    else:
        raise ValueError(f"Invalid model class: {network_config.model_class}")


def load_initial_state(
    network_config: NetworkConfig,
    game_config: GameConfig,
    training_config,
    training_directory: str,
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
        initial_training_state = latest_file(training_directory, ".pth")
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


def _join_directory(directory: str, filename: str) -> str:
    if directory.endswith("/"):
        return f"{directory}{filename}"
    return f"{directory}/{filename}"


def _ensure_local_directory(directory: str) -> None:
    if directory and not is_s3(directory):
        os.makedirs(directory, exist_ok=True)


def save_model_and_state(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    name: str,
    model_directory: str,
    training_directory: str,
    device: str,
    add_timestamp: bool = False,
) -> None:
    """Saves the model and optimizer state to the configured directories."""
    name = str(name)

    if add_timestamp:
        name = f"{name}_{time.time():.0f}"

    if model_directory.strip():
        _ensure_local_directory(model_directory)
        onnx_path = _join_directory(model_directory, f"{name}.onnx")
        log(f"Saving model to: {onnx_path}")
        with from_localized(onnx_path) as onnx_path:
            model.save_onnx(onnx_path, device)
            model.train()
    else:
        log("No model directory set, skipping model save.")

    if training_directory.strip():
        _ensure_local_directory(training_directory)
        training_state_path = _join_directory(training_directory, f"{name}.pth")
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


def get_loss(
    batch,
    model: nn.Module,
    *,
    device: str,
    policy_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Forward pass
    board, expected_value, expected_policy, valid_policy_mask = batch
    board = board.to(device)
    expected_value = expected_value.to(device)
    expected_policy = expected_policy.to(device)
    valid_policy_mask = valid_policy_mask.to(device)
    pred_value, pred_policy = model(board)

    # Calculate value loss.
    value_loss = nn.CrossEntropyLoss()(pred_value, expected_value)

    # Calculate policy loss.
    pred_policy = pred_policy.view(pred_policy.shape[0], -1)
    expected_policy = expected_policy.view(expected_policy.shape[0], -1)
    valid_policy_mask = valid_policy_mask.view(valid_policy_mask.shape[0], -1)
    pred_policy[~valid_policy_mask] = -1e6

    policy_loss = policy_loss_weight * nn.CrossEntropyLoss()(
        pred_policy, expected_policy
    )

    total_loss = value_loss + policy_loss
    if total_loss.isnan().any():
        raise TrainingError("Loss is NaN")
    return total_loss, value_loss, policy_loss
