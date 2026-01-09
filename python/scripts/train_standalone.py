import dataclasses
import random
import sys
from typing import List, Optional
import time

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
    save_model_and_state,
)


def log(message: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def initialize_run(
    network_config: NetworkConfig, training_config: TrainingStandaloneConfig
):
    """Initialize Aim run with configs and return the run handle."""
    aim_run = aim.Run(repo=training_config.aim_repo_path)
    aim_run["network_config"] = dataclasses.asdict(network_config)
    aim_run["training_config"] = dataclasses.asdict(training_config)
    aim_run["name"] = "baseline"
    log(f"Run hash: {aim_run.hash}")
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

    log(
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
    test_files: Optional[List[str]],
    aim_run,
):
    """Train for a fixed number of epochs with periodic evaluation."""
    total_samples_trained = 0
    num_epochs = training_config.num_epochs

    should_test = bool(test_files) and training_config.train_batches_per_test > 0
    test_iterator = None

    for epoch in range(num_epochs):
        log(f"Starting epoch {epoch + 1} of {num_epochs}")

        random.shuffle(train_files)

        train_dataloader = build_dataloader(game_config, train_files, training_config)

        train_batches_since_last_test = 0
        for batch in train_dataloader:
            batch_size = batch[0].shape[0]
            total_samples_trained += batch_size

            loss, value_loss, policy_loss = get_loss(batch, training_config, model)

            if loss.isnan().any():
                raise Exception("Loss is NaN")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Report on results only for full batches.
            if batch_size == training_config.batch_size:
                average_loss = loss.item()
                average_value_loss = value_loss.item()
                average_policy_loss = policy_loss.item()

                if total_samples_trained % 200000 <= batch_size:
                    log(
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

            if should_test:
                train_batches_since_last_test += 1
                if (
                    train_batches_since_last_test
                    >= training_config.train_batches_per_test
                ):
                    train_batches_since_last_test = 0
                    if test_iterator is None:
                        random.shuffle(test_files)
                        test_iterator = iter(
                            build_dataloader(game_config, test_files, training_config)
                        )
                    test_batch = next(test_iterator, None)
                    if test_batch is None:
                        random.shuffle(test_files)
                        test_iterator = iter(
                            build_dataloader(game_config, test_files, training_config)
                        )
                        test_batch = next(test_iterator, None)
                    if test_batch is None:
                        continue
                    test_model(
                        model,
                        training_config,
                        test_batch,
                        aim_run,
                        total_samples_trained,
                    )

    save_model_and_state(
        model,
        optimizer,
        "standalone",
        training_config.output_name,
        training_config.model_directory,
        training_config.training_directory,
        training_config.device,
        add_timestamp=True,
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

    train_remote_files = list_files(training_config.remote_train_data_dir, ".bin")

    # Grab the latest samples.
    train_remote_files = sorted(train_remote_files, reverse=True)
    if training_config.max_train_files > 0:
        train_remote_files = train_remote_files[: training_config.max_train_files]
    num_samples = sum(
        parse_num_games_from_filename(filename) for filename in train_remote_files
    )
    log(f"Number of samples: {num_samples}")

    train_local_files = []
    for filename in tqdm(train_remote_files, desc="Localizing data"):
        train_local_files.append(
            localize_file(filename, training_config.local_game_mirror)
        )

    test_local_files = None
    if training_config.remote_test_data_dir:
        test_remote_files = list_files(training_config.remote_test_data_dir, ".bin")
        test_local_files = []
        for filename in tqdm(test_remote_files, desc="Localizing test data"):
            test_local_files.append(
                localize_file(filename, training_config.local_game_mirror)
            )

    aim_run = initialize_run(network_config, training_config)

    model = initialize_model(network_config, game_config)
    model.to(training_config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)

    train_model(
        model,
        optimizer,
        game_config,
        training_config,
        train_local_files,
        test_local_files,
        aim_run,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/train_standalone.py <config_path>")
    main(sys.argv[1])
