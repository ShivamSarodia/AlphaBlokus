import copy
import os
import time

import torch
import torch.nn as nn

from alphablokus.configs import GameConfig, NetworkConfig
from alphablokus.files import from_localized, is_s3, latest_file, localize_file
from alphablokus.res_net import NeuralNet
from alphablokus.res_net_old_value_head import NeuralNet as ResNetOldValueHeadNet
from alphablokus.trivial_net import TrivialNet
from alphablokus.log import log


class TrainingError(Exception):
    pass


def _copy_state_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _copy_state_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy_state_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_copy_state_to_cpu(v) for v in value)
    return copy.deepcopy(value)


def _move_state_to_device(value, device: str):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_state_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_state_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_state_to_device(v, device) for v in value)
    return value


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer, device: str
) -> None:
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def take_training_snapshot(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, dict]:
    return {
        "model": _copy_state_to_cpu(model.state_dict()),
        "optimizer": _copy_state_to_cpu(optimizer.state_dict()),
    }


def restore_training_snapshot(
    snapshot: dict[str, dict],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> None:
    model_state = _move_state_to_device(snapshot["model"], device)
    model.load_state_dict(model_state)
    optimizer.load_state_dict(snapshot["optimizer"])
    move_optimizer_state_to_device(optimizer, device)


def initialize_model(
    network_config: NetworkConfig,
    game_config: GameConfig,
) -> nn.Module:
    if network_config.model_class == "trivial":
        return TrivialNet(game_config)
    elif network_config.model_class == "resnet":
        return NeuralNet(network_config, game_config)
    elif network_config.model_class == "resnet_old_value_head":
        return ResNetOldValueHeadNet(network_config, game_config)
    else:
        raise ValueError(f"Invalid model class: {network_config.model_class}")


def get_sample_count_from_training_filename(training_file: str | None) -> int:
    if training_file is None:
        return 0

    filename = os.path.basename(training_file)
    stem, _ = os.path.splitext(filename)
    try:
        return int(stem)
    except ValueError as exc:
        raise ValueError(
            f"Training file name must be an integer sample count, got: {filename}"
        ) from exc


def load_initial_state(
    network_config: NetworkConfig,
    game_config: GameConfig,
    *,
    learning_rate: float,
    device: str,
    training_file: str | None,
    skip_loading_optimizer: bool = False,
    optimizer_type: str = "adam",
    optimizer_weight_decay: float = 0.0,
) -> tuple[nn.Module, torch.optim.Optimizer]:
    """
    Loads the initial state of the model and optimizer from the training directory.
    """
    # Load the model and optimizer.
    model = initialize_model(network_config, game_config)
    optimizer = build_optimizer(
        model,
        learning_rate=learning_rate,
        optimizer_type=optimizer_type,
        optimizer_weight_decay=optimizer_weight_decay,
    )
    if training_file is None:
        print("No training state found, starting from scratch.")
    else:
        print("Loading training state from:", training_file)
        initial_training_path = localize_file(training_file)
        initial_training_state = torch.load(initial_training_path, map_location=device)
        model.load_state_dict(initial_training_state["model"])
        if not skip_loading_optimizer:
            optimizer.load_state_dict(initial_training_state["optimizer"])

    # Move the model to the appropriate device.
    model = model.to(device=device)

    # Move optimizer state to the same device as the model.
    move_optimizer_state_to_device(optimizer, device)

    return model, optimizer


def build_optimizer(
    model: nn.Module,
    *,
    learning_rate: float,
    optimizer_type: str = "adam",
    optimizer_weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    if optimizer_type == "adam":
        assert optimizer_weight_decay == 0.0, (
            "optimizer_weight_decay must be 0.0 when optimizer_type is 'adam'"
        )
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=optimizer_weight_decay,
        )
    if optimizer_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=optimizer_weight_decay,
        )
    if optimizer_type == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.90,
            nesterov=True,
            weight_decay=optimizer_weight_decay,
        )
    raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


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
    value_head_l2: float = 0.0,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
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
    if value_head_l2 > 0.0:
        value_head = None
        if hasattr(model, "value_head"):
            value_head = model.value_head
        elif hasattr(model, "values"):
            value_head = model.values
        if value_head is not None:
            l2_term = sum((param**2).sum() for param in value_head.parameters())
            total_loss = total_loss + (value_head_l2 * l2_term)

    try:
        if torch.isnan(total_loss).any():
            log("!!! LOSS IS NaN !!!")
            return None, None, None
    except torch.AcceleratorError as e:
        log("!!! AcceleratorError: !!!")
        log(f"Error: {e}")
        return None, None, None

    return total_loss, value_loss, policy_loss
