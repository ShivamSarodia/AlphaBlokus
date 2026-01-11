import torch
from torch import nn
import torch.nn.functional as F

from alphablokus.configs import GameConfig, NetworkConfig
from alphablokus.save_onnx import SaveOnnxMixin


class ResidualBlock(nn.Module):
    def __init__(self, net_config: NetworkConfig):
        super().__init__()

        self.convolutional_block = nn.Sequential(
            nn.Conv2d(
                in_channels=net_config.main_body_channels,
                out_channels=net_config.main_body_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(net_config.main_body_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=net_config.main_body_channels,
                out_channels=net_config.main_body_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(net_config.main_body_channels),
        )

    def forward(self, x):
        return F.relu(x + self.convolutional_block(x))


class ValueHead(nn.Module):
    def __init__(self, net_config: NetworkConfig, game_config: GameConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=net_config.main_body_channels,
                out_channels=net_config.value_head_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(net_config.value_head_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(
                net_config.value_head_channels, net_config.value_head_flat_layer_width
            ),
            nn.ReLU(),
            nn.Linear(net_config.value_head_flat_layer_width, 4),
        )

    def forward(self, x):
        return self.layers(x)


class PolicyHead(nn.Module):
    def __init__(self, net_config: NetworkConfig, game_config: GameConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=net_config.main_body_channels,
                out_channels=net_config.policy_head_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(net_config.policy_head_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=net_config.policy_head_channels,
                out_channels=game_config.num_piece_orientations,
                kernel_size=net_config.policy_convolution_kernel,
                stride=1,
                padding=net_config.policy_convolution_kernel // 2,
                # Include bias because we're not using batch norm.
                bias=True,
            ),
        )

    def forward(self, x):
        return self.layers(x)


class NeuralNet(nn.Module, SaveOnnxMixin):
    def __init__(
        self,
        net_config: NetworkConfig,
        game_config: GameConfig,
    ):
        super().__init__()

        self.net_config = net_config
        self.game_config = game_config

        self.convolutional_block = nn.Sequential(
            nn.Conv2d(
                in_channels=6,
                out_channels=net_config.main_body_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(net_config.main_body_channels),
            nn.ReLU(),
        )
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(net_config) for _ in range(net_config.residual_blocks)]
        )
        self.value_head = ValueHead(net_config, game_config)
        self.policy_head = PolicyHead(net_config, game_config)

    def forward(self, board):
        # Add (x, y) positional channels in [0, 1]
        coords = torch.linspace(
            0.0,
            1.0,
            self.game_config.board_size,
            device=board.device,
            dtype=board.dtype,
        )
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")  # (N, N)
        pos = (
            torch.stack([xx, yy], dim=0).unsqueeze(0).expand(board.shape[0], -1, -1, -1)
        )
        x = torch.cat([board, pos], dim=1)  # (batch, 6, board_size, board_size)

        x = self.convolutional_block(x)
        for residual_block in self.residual_blocks:
            x = residual_block(x)

        return (
            self.value_head(x),
            self.policy_head(x),
        )


if __name__ == "__main__":
    import modelopt.torch.quantization as mtq

    from alphablokus.configs import TrainingStandaloneConfig
    import random

    from alphablokus.files import from_localized, list_files, localize_file
    from alphablokus.train_utils import load_game_files_to_tensor

    config_path = "configs/training/standalone_resnet.toml"
    checkpoint_path = "s3://alpha-blokus/full_v2/training_simulated/06081134_conv_value_position_1767797812.pth"
    output_path = (
        "s3://alpha-blokus/full_v2/models_untrained/res_net_cvp_auto_quantized.onnx"
    )
    device = "cpu"

    model = NeuralNet(NetworkConfig(config_path), GameConfig(config_path)).to(device)
    localized_checkpoint = localize_file(checkpoint_path)
    checkpoint = torch.load(localized_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    training_config = TrainingStandaloneConfig(config_path)
    training_config.device = device

    train_remote_files = list_files(training_config.remote_train_data_dir, ".bin")
    train_remote_files = sorted(train_remote_files, reverse=True)
    calibration_remote_files = random.sample(train_remote_files, k=10)

    calibration_local_files = [
        localize_file(filename, training_config.local_game_mirror)
        for filename in calibration_remote_files
    ]
    boards, values, policies, valid_masks = load_game_files_to_tensor(
        model.game_config, calibration_local_files
    )

    num_samples = boards.shape[0]
    indices = list(range(num_samples))
    random.shuffle(indices)

    calibration_batches = []
    batch_size = training_config.batch_size
    for start in range(0, num_samples, batch_size):
        batch_indices = indices[start : start + batch_size]
        if not batch_indices:
            break
        calibration_batches.append(
            (
                boards[batch_indices],
                values[batch_indices],
                policies[batch_indices],
                valid_masks[batch_indices],
            )
        )
        if len(calibration_batches) >= 100:
            break

    def forward_step(model, batch):
        board, _, _, _ = batch
        return model(board)

    def loss_func(output, batch):
        pred_value, pred_policy = output
        _, expected_value, expected_policy, valid_policy_mask = batch
        expected_value = expected_value.to(pred_value.device)
        expected_policy = expected_policy.to(pred_policy.device)
        valid_policy_mask = valid_policy_mask.to(pred_policy.device)

        value_loss = nn.CrossEntropyLoss()(pred_value, expected_value)

        pred_policy = pred_policy.view(pred_policy.shape[0], -1).clone()
        expected_policy = expected_policy.view(expected_policy.shape[0], -1)
        valid_policy_mask = valid_policy_mask.view(valid_policy_mask.shape[0], -1)
        pred_policy = pred_policy.masked_fill(~valid_policy_mask, -1e6)

        policy_loss = training_config.policy_loss_weight * nn.CrossEntropyLoss()(
            pred_policy, expected_policy
        )

        return value_loss + policy_loss

    int8_cfg = mtq.INT8_SMOOTHQUANT_CFG

    model.eval()
    quantized_model, _ = mtq.auto_quantize(
        model,
        constraints={"effective_bits": 8.0},
        quantization_formats=[int8_cfg],
        data_loader=calibration_batches,
        forward_step=forward_step,
        loss_func=loss_func,
        num_calib_steps=len(calibration_batches),
        num_score_steps=min(4, len(calibration_batches)),
        method="gradient",
    )

    mtq.print_quant_summary(quantized_model)
    from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer

    enabled_quantizers = []
    disabled_quantizers = []
    for name, module in quantized_model.named_modules():
        if isinstance(module, TensorQuantizer):
            if module.is_enabled:
                enabled_quantizers.append(name)
            else:
                disabled_quantizers.append(name)

    print("\nQuantizers enabled:")
    for name in enabled_quantizers:
        print(f"  + {name}")
    print("\nQuantizers disabled:")
    for name in disabled_quantizers:
        print(f"  - {name}")

    with from_localized(output_path) as localized_output_path:
        quantized_model.eval()
        dummy_input = (
            torch.randn(
                1,
                4,
                model.game_config.board_size,
                model.game_config.board_size,
                device=device,
            ),
        )
        torch.onnx.export(
            quantized_model,
            dummy_input,
            localized_output_path,
            dynamo=False,
            input_names=["board"],
            output_names=["value", "policy"],
            dynamic_axes={
                "board": {0: "batch_size"},
                "value": {0: "batch_size"},
                "policy": {0: "batch_size"},
            },
        )
