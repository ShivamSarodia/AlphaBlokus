import dataclasses
import random
import sys
from typing import Iterable, List, Tuple

import aim
import torch
from tqdm import tqdm

from alphablokus.configs import GameConfig, NetworkConfig, TrainingStandaloneConfig
from alphablokus.files import list_files, parse_num_games_from_filename
from alphablokus.train_utils import (
    IterableGameDataset,
    get_loss,
    initialize_model,
    localize_file,
)


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
    aim_run["name"] = "baseline"
    print("Run hash: ", aim_run.hash)
    return aim_run


def test_model(
    model,
    training_config: TrainingStandaloneConfig,
    test_batch,
    aim_run,
    total_samples_trained: int,
):
    """Evaluate model on a single test batch and log metrics."""
    model.eval()

    with torch.no_grad():
        test_loss, test_value_loss, test_policy_loss = get_loss(
            test_batch, training_config, model
        )

    print(
        f"Step {total_samples_trained}: Test batch loss: {test_loss.item()}. (Value loss: {test_value_loss.item()}, Policy loss: {test_policy_loss.item()})"
    )

    aim_run.track(
        test_loss.item(),
        name="total_loss",
        step=total_samples_trained,
        context={"subset": "test"},
    )
    aim_run.track(
        test_value_loss.item(),
        name="value_loss",
        step=total_samples_trained,
        context={"subset": "test"},
    )
    aim_run.track(
        test_policy_loss.item(),
        name="policy_loss",
        step=total_samples_trained,
        context={"subset": "test"},
    )

    model.train()


def build_dataloader(
    game_config: GameConfig,
    files: List[str],
    training_config: TrainingStandaloneConfig,
) -> torch.utils.data.DataLoader:
    """Helper to build a dataloader for iterable game data."""
    dataset = IterableGameDataset(
        game_config, files, training_config.shuffle_buffer_file_count
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        persistent_workers=training_config.num_workers > 0,
        prefetch_factor=training_config.prefetch_factor,
    )


def train_model(
    model,
    optimizer,
    game_config: GameConfig,
    training_config: TrainingStandaloneConfig,
    train_files: List[str],
    test_files: List[str],
    aim_run,
):
    """Train for a fixed number of epochs with periodic evaluation."""
    total_samples_trained = 0
    num_epochs = training_config.num_epochs

    def get_test_iterator():
        random.shuffle(test_files)
        return iter(build_dataloader(game_config, test_files, training_config))

    test_iterator = get_test_iterator()

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1} of {num_epochs}")

        random.shuffle(train_files)

        train_dataloader = build_dataloader(game_config, train_files, training_config)

        train_batches_since_last_test = 0
        for batch in train_dataloader:
            batch_size = batch[0].shape[0]
            total_samples_trained += batch_size
            train_batches_since_last_test += 1

            loss, value_loss, policy_loss = get_loss(batch, training_config, model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Report on results only for full batches.
            if batch_size == training_config.batch_size:
                average_loss = loss.item()
                average_value_loss = value_loss.item()
                average_policy_loss = policy_loss.item()

                print(
                    f"Step: {total_samples_trained}. Train loss: {average_loss}. Value loss: {average_value_loss}. Policy loss: {average_policy_loss}."
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

            if train_batches_since_last_test >= training_config.train_batches_per_test:
                train_batches_since_last_test = 0
                try:
                    test_batch = next(test_iterator)
                except StopIteration:
                    test_iterator = get_test_iterator()
                    test_batch = next(test_iterator)
                test_model(
                    model,
                    training_config,
                    test_batch,
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
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/train_standalone.py <config_path>")
    main(sys.argv[1])
