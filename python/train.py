import torch
import msgpack
import zstandard
import os
import torch.nn as nn

from configs import GameConfig, NetworkConfig, TrainingConfig
from res_net import NeuralNet

config_path = "configs/training/half.toml"
game_config = GameConfig(config_path)
network_config = NetworkConfig(config_path)
training_config = TrainingConfig(config_path)

game_data_directory = "data/scrap/games"

board_inputs = []
value_targets = []
policy_targets = []

for filename in os.listdir(game_data_directory):
    with zstandard.open(os.path.join(game_data_directory, filename), "rb") as f:
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
                policy_target[piece_orientation_index, center_x, center_y] = visit_count

            policy_target = policy_target / policy_target.sum()
            policy_targets.append(policy_target)

board_inputs = torch.stack(board_inputs)
value_targets = torch.stack(value_targets)
policy_targets = torch.stack(policy_targets)

dataset = torch.utils.data.TensorDataset(board_inputs, value_targets, policy_targets)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=training_config.batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=training_config.batch_size, shuffle=True
)

model = NeuralNet(network_config, game_config)

optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)

for epoch in range(training_config.num_epochs):
    print(f"Epoch {epoch + 1} of {training_config.num_epochs}")

    for batch in train_loader:
        board, expected_value, expected_policy = batch
        pred_value, pred_policy = model(board)

        value_loss = nn.CrossEntropyLoss()(pred_value, expected_value)

        pred_policy = pred_policy.view(pred_policy.shape[0], -1)
        expected_policy = expected_policy.view(expected_policy.shape[0], -1)
        policy_loss = training_config.policy_loss_weight * nn.CrossEntropyLoss()(
            pred_policy, expected_policy
        )
        loss = value_loss + policy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss.item())

    with torch.no_grad():
        test_loss = 0
        for batch in test_loader:
            board, expected_value, expected_policy = batch
            pred_value, pred_policy = model(board)

            value_loss = nn.CrossEntropyLoss()(pred_value, expected_value)

            pred_policy = pred_policy.view(pred_policy.shape[0], -1)
            expected_policy = expected_policy.view(expected_policy.shape[0], -1)
            policy_loss = training_config.policy_loss_weight * nn.CrossEntropyLoss()(
                pred_policy, expected_policy
            )
            loss = value_loss + policy_loss
            test_loss += loss.item()

        test_loss /= len(test_loader)
        print(f"Test loss: {test_loss}")
