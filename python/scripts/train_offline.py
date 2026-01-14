import argparse
import time

import torch
import random

from alphablokus.configs import GameConfig, NetworkConfig, TrainingOfflineConfig
from alphablokus.data_loaders import (
    BufferedGameBatchDataset,
    StaticListFileProvider,
    build_streaming_dataloader,
)
from alphablokus.files import list_files
from alphablokus.train_utils import (
    get_loss,
    initialize_model,
    log,
    save_model_and_state,
)


def run_offline_training(config_path: str) -> None:
    game_config = GameConfig(config_path)
    network_config = NetworkConfig(config_path)
    training_config = TrainingOfflineConfig(config_path)

    model = initialize_model(network_config, game_config).to(training_config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)

    train_files = list_files(training_config.game_data_directory, ".bin")
    train_files = sorted(train_files)

    ############################################################################################
    # Custom logic for training schedule.
    ############################################################################################

    is_four_times = random.randint(0, 1) > 0
    if is_four_times:
        windows = [
            train_files[-400:],
            train_files[-300:],
            train_files[-200:],
            train_files[-100:],
        ]
    else:
        windows = [
            train_files[-400:],
            train_files[-200:],
            train_files[-100:],
        ]
    train_files = [random.sample(window, len(window)) for window in windows]
    train_files = sum(train_files, [])

    ############################################################################################
    # End custom logic for training schedule.
    ############################################################################################

    log(f"Training on {len(train_files)} files.")

    log("Starting training pass.")
    file_provider = StaticListFileProvider(train_files)
    dataset = BufferedGameBatchDataset(
        game_config,
        file_provider,
        training_config.batch_size,
        training_config.in_memory_shuffle_file_count,
        local_cache_dir=training_config.local_game_mirror or None,
        cleanup_local_files=False,
    )
    dataloader = build_streaming_dataloader(
        dataset,
        num_workers=training_config.num_workers,
        prefetch_factor=training_config.prefetch_factor,
    )

    batches_seen = 0
    samples_trained = 0
    log_every_batches = 200

    for batch in dataloader:
        loss, value_loss, policy_loss = get_loss(
            batch,
            model,
            device=training_config.device,
            policy_loss_weight=training_config.policy_loss_weight,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = batch[0].shape[0]
        samples_trained += batch_size
        batches_seen += 1

        if batches_seen % log_every_batches == 0:
            log(
                f"Step {samples_trained}: loss={loss.item():.4f}, "
                f"value={value_loss.item():.4f}, policy={policy_loss.item():.4f}"
            )

    log(f"Finished training pass. Trained {samples_trained} samples.")

    save_model_and_state(
        model=model,
        optimizer=optimizer,
        name=training_config.output_name + f"_{is_four_times}",
        model_directory=training_config.model_directory,
        training_directory=training_config.training_directory,
        device=training_config.device,
        add_timestamp=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline training script")
    parser.add_argument("config", help="Path to config TOML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_offline_training(args.config)


if __name__ == "__main__":
    main()
