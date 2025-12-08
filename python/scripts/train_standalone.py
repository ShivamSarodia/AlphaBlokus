from alphablokus.train_utils import (
    IterableGameDataset,
    localize_file,
    initialize_model,
    get_loss,
)
from alphablokus.files import list_files
from alphablokus.configs import GameConfig, NetworkConfig, TrainingConfig
import torch
import numpy as np
import aim
import dataclasses
from tqdm import tqdm
from typing import List

import random

random.seed(42)

game_config = GameConfig("../configs/training/full_vast.toml")
network_config = NetworkConfig("../configs/training/full_vast.toml")
training_config = TrainingConfig("../configs/training/full_vast.toml")

# ------------------------------------------------------------------------------------------------ #
# Select games from the full dataset for test and training.
# ------------------------------------------------------------------------------------------------ #
gamedata_files = sorted(list_files("s3://alpha-blokus/full/games/", ".bin"))

total_games = 0
selected_files = []

for gamedata_file in sorted(gamedata_files):
    num_games = int(gamedata_file.split("_")[-1].split(".")[0])
    total_games += num_games

    if total_games > 1.6e6:
        break

    selected_files.append(gamedata_file)

i = 0
test_files = []
train_files = []
for filename in tqdm(selected_files):
    local_filename = localize_file(
        filename, "/Users/shivamsarodia/Dev/AlphaBlokus/data/s3_mirrors/full/games"
    )

    if i % 10 == 0:
        test_files.append(local_filename)
    else:
        train_files.append(local_filename)
    i += 1

print("Num train files: ", len(train_files))
print("Num test files: ", len(test_files))

total_train_samples = sum(int(f.split("_")[-1].split(".")[0]) for f in train_files)
print("Total train samples: ", total_train_samples)

# ------------------------------------------------------------------------------------------------ #
# Train the model.
# ------------------------------------------------------------------------------------------------ #
aim_run = aim.Run(repo="/Users/shivamsarodia/Dev/AlphaBlokus/data/aim")
aim_run["network_config"] = dataclasses.asdict(network_config)
aim_run["training_config"] = dataclasses.asdict(training_config)

print("Run hash: ", aim_run.hash)

model = initialize_model(network_config, game_config)
model.to(training_config.device)

optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)


def test_model(total_samples_trained: int):
    print("Testing...")

    test_dataset = IterableGameDataset(game_config, test_files, 4)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128)

    total_test_loss = 0
    total_test_value_loss = 0
    total_test_policy_loss = 0
    total_test_samples = 0
    for batch in test_dataloader:
        test_loss, test_value_loss, test_policy_loss = get_loss(
            batch, training_config, model
        )
        total_test_loss += test_loss.item()
        total_test_value_loss += test_value_loss.item()
        total_test_policy_loss += test_policy_loss.item()
        total_test_samples += batch[0].shape[0]

    final_test_loss = total_test_loss / total_test_samples
    final_test_value_loss = total_test_value_loss / total_test_samples
    final_test_policy_loss = total_test_policy_loss / total_test_samples

    print(
        f"Test loss: {final_test_loss}. (Value loss: {final_test_value_loss}, Policy loss: {final_test_policy_loss})"
    )

    aim_run.track(
        final_test_loss,
        name="test_loss",
        step=total_samples_trained,
    )

    aim_run.track(
        final_test_value_loss,
        name="test_value_loss",
        step=total_samples_trained,
    )

    aim_run.track(
        final_test_policy_loss,
        name="test_policy_loss",
        step=total_samples_trained,
    )


def train_model():
    total_samples_trained = 0
    num_epochs = round(training_config.sampling_ratio)
    for epoch in range(num_epochs):
        print("Starting epoch ", epoch + 1)

        train_dataset = IterableGameDataset(game_config, train_files, 4)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128)

        print(f"Epoch {epoch + 1} of {num_epochs}")

        samples_trained_since_last_test = 0
        for batch in train_dataloader:
            batch_size = batch[0].shape[0]
            samples_trained_since_last_test += batch_size
            total_samples_trained += batch_size

            loss, value_loss, policy_loss = get_loss(batch, training_config, model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            average_loss = loss.item() / batch_size
            average_value_loss = value_loss.item() / batch_size
            average_policy_loss = policy_loss.item() / batch_size

            print(
                f"Train loss: {average_loss}. (Value loss: {average_value_loss}, Policy loss: {average_policy_loss})"
            )

            aim_run.track(
                average_loss,
                name="total_train_loss",
                step=total_samples_trained,
            )

            aim_run.track(
                average_value_loss,
                name="train_value_loss",
                step=total_samples_trained,
            )

            aim_run.track(
                average_policy_loss,
                name="train_policy_loss",
                step=total_samples_trained,
            )

            if samples_trained_since_last_test >= total_train_samples / 5:
                samples_trained_since_last_test = 0
                test_model(total_samples_trained)

    test_model(total_samples_trained)


train_model()
