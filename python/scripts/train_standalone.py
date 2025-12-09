import dataclasses
import random
import sys
from typing import Iterable, List, Tuple

import aim
import torch
from tqdm import tqdm

from alphablokus.configs import GameConfig, NetworkConfig, TrainingStandaloneConfig
from alphablokus.files import list_files
from alphablokus.train_utils import (
    IterableGameDataset,
    get_loss,
    initialize_model,
    localize_file,
)


def parse_num_games_from_filename(filename: str) -> int:
    """Extracts the number of games encoded in the filename suffix."""
    basename = filename.rsplit("/", 1)[-1]
    return int(basename.split(".")[0].split("_")[-1])


def select_game_files(remote_dir: str, max_total_games: int) -> List[str]:
    """Return the most recent remote file paths up to max_total_games samples."""
    gamedata_files = sorted(list_files(remote_dir, ".bin"), reverse=True)

    total_games = 0
    selected_files: List[str] = []

    # Walk newest files first.
    for gamedata_file in gamedata_files:
        num_games = parse_num_games_from_filename(gamedata_file)
        total_games += num_games
        if total_games > max_total_games:
            break
        selected_files.append(gamedata_file)

    return selected_files


def localize_and_split_files(
    remote_files: Iterable[str], local_dir: str, test_stride: int
) -> Tuple[List[str], List[str]]:
    """Download remote files to local_dir and split into train/test via stride."""
    train_files: List[str] = []
    test_files: List[str] = []

    for idx, filename in enumerate(tqdm(list(remote_files), desc="Localizing data")):
        local_filename = localize_file(filename, local_dir)
        (test_files if idx % test_stride == 0 else train_files).append(local_filename)

    print(f"Num train files: {len(train_files)}")
    print(f"Num test files: {len(test_files)}")

    return train_files, test_files


def count_total_samples(files: Iterable[str]) -> int:
    """Compute total sample count encoded in filenames."""
    return sum(parse_num_games_from_filename(f) for f in files)


def initialize_run(
    network_config: NetworkConfig, training_config: TrainingStandaloneConfig
):
    """Initialize Aim run with configs and return the run handle."""
    aim_run = aim.Run(repo=training_config.aim_repo_path)
    aim_run["network_config"] = dataclasses.asdict(network_config)
    aim_run["training_config"] = dataclasses.asdict(training_config)
    print("Run hash: ", aim_run.hash)
    return aim_run


def test_model(
    model,
    game_config: GameConfig,
    training_config: TrainingStandaloneConfig,
    test_files: List[str],
    aim_run,
    total_samples_trained: int,
):
    """Evaluate model on held-out data and log metrics."""
    print("Testing...")
    model.eval()

    test_dataset = IterableGameDataset(
        game_config, test_files, training_config.shuffle_buffer_file_count
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=training_config.batch_size
    )

    total_test_loss = 0.0
    total_test_value_loss = 0.0
    total_test_policy_loss = 0.0
    total_test_samples = 0

    with torch.no_grad():
        for batch in test_dataloader:
            test_loss, test_value_loss, test_policy_loss = get_loss(
                batch, training_config, model
            )
            batch_size = batch[0].shape[0]
            total_test_loss += test_loss.item() * batch_size
            total_test_value_loss += test_value_loss.item() * batch_size
            total_test_policy_loss += test_policy_loss.item() * batch_size
            total_test_samples += batch_size

    final_test_loss = total_test_loss / total_test_samples
    final_test_value_loss = total_test_value_loss / total_test_samples
    final_test_policy_loss = total_test_policy_loss / total_test_samples

    print(
        f"Test loss: {final_test_loss}. (Value loss: {final_test_value_loss}, Policy loss: {final_test_policy_loss})"
    )

    aim_run.track(
        final_test_loss,
        name="total_loss",
        step=total_samples_trained,
        context={"subset": "test"},
    )
    aim_run.track(
        final_test_value_loss,
        name="value_loss",
        step=total_samples_trained,
        context={"subset": "test"},
    )
    aim_run.track(
        final_test_policy_loss,
        name="policy_loss",
        step=total_samples_trained,
        context={"subset": "test"},
    )

    model.train()


def train_model(
    model,
    optimizer,
    game_config: GameConfig,
    training_config: TrainingStandaloneConfig,
    train_files: List[str],
    test_files: List[str],
    aim_run,
    total_train_samples: int,
):
    """Train for a fixed number of epochs with periodic evaluation."""
    total_samples_trained = 0
    num_epochs = training_config.num_epochs

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1} of {num_epochs}")

        train_dataset = IterableGameDataset(
            game_config, train_files, training_config.shuffle_buffer_file_count
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=training_config.batch_size
        )

        samples_trained_since_last_test = 0
        for batch in train_dataloader:
            batch_size = batch[0].shape[0]
            samples_trained_since_last_test += batch_size
            total_samples_trained += batch_size

            loss, value_loss, policy_loss = get_loss(batch, training_config, model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            average_loss = loss.item()
            average_value_loss = value_loss.item()
            average_policy_loss = policy_loss.item()

            print(
                f"Train loss: {average_loss}. (Value loss: {average_value_loss}, Policy loss: {average_policy_loss})"
            )

            aim_run.track(
                average_loss,
                name="total_loss",
                step=total_samples_trained,
                context={"subset": "train"},
            )
            aim_run.track(
                average_value_loss,
                name="value_loss",
                step=total_samples_trained,
                context={"subset": "train"},
            )
            aim_run.track(
                average_policy_loss,
                name="policy_loss",
                step=total_samples_trained,
                context={"subset": "train"},
            )

            if (
                samples_trained_since_last_test
                >= total_train_samples / training_config.num_tests_per_epoch
            ):
                samples_trained_since_last_test = 0
                test_model(
                    model,
                    game_config,
                    training_config,
                    test_files,
                    aim_run,
                    total_samples_trained,
                )

    test_model(
        model,
        game_config,
        training_config,
        test_files,
        aim_run,
        total_samples_trained,
    )


def load_configs(config_path: str):
    """Load game/network/standalone training configs from the same file."""
    game_config = GameConfig(config_path)
    network_config = NetworkConfig(config_path)
    training_config = TrainingStandaloneConfig(config_path)
    return game_config, network_config, training_config


def main(config_path: str):
    """Entrypoint for running a standalone training job."""
    game_config, network_config, training_config = load_configs(config_path)

    random.seed(42)

    selected_files = select_game_files(
        training_config.remote_game_dir, training_config.max_total_games
    )
    train_files, test_files = localize_and_split_files(
        selected_files,
        training_config.local_game_mirror,
        training_config.test_split_stride,
    )

    total_train_samples = count_total_samples(train_files)
    print("Total train samples: ", total_train_samples)

    aim_run = initialize_run(network_config, training_config)

    model = initialize_model(network_config, game_config)
    model.to(training_config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)

    train_model(
        model,
        optimizer,
        game_config,
        training_config,
        train_files,
        test_files,
        aim_run,
        total_train_samples,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/train_standalone.py <config_path>")
    main(sys.argv[1])
