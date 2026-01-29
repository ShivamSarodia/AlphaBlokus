import argparse
from typing import Optional, Tuple
from collections import deque

import torch
import random
from torch.nn.utils import clip_grad_norm_

from alphablokus.configs import GameConfig, NetworkConfig, TrainingOfflineConfig
from alphablokus.data_loaders import (
    BufferedGameBatchDataset,
    StaticListFileProvider,
    build_streaming_dataloader,
)
from alphablokus.files import list_files, parse_num_games_from_filename
from alphablokus.train_utils import (
    get_loss,
    load_initial_state,
    restore_training_snapshot,
    save_model_and_state,
    take_training_snapshot,
)
from alphablokus.log import log


def build_sample_window(
    files_with_samples: list[tuple[str, int]],
    *,
    start_samples: int,
    end_samples: int,
    origin: str,
) -> list[str]:
    if origin not in ["start", "end"]:
        raise ValueError("origin must be 'start' or 'end'")

    if start_samples < end_samples:
        start_samples, end_samples = end_samples, start_samples

    if origin == "end":
        total_samples = sum(num_samples for _, num_samples in files_with_samples)
        start_samples, end_samples = (
            total_samples - end_samples,
            total_samples - start_samples,
        )
    else:
        total_samples = sum(num_samples for _, num_samples in files_with_samples)

    start_samples = max(0, min(start_samples, total_samples))
    end_samples = max(0, min(end_samples, total_samples))

    window_paths = []
    samples_seen = 0
    for path, num_samples in sorted(files_with_samples):
        samples_seen += num_samples
        if samples_seen <= end_samples:
            continue
        if samples_seen >= start_samples:
            break
        window_paths.append(path)
    return random.sample(window_paths, len(window_paths))


def run_offline_training(config_path: str) -> None:
    game_config = GameConfig(config_path)
    network_config = NetworkConfig(config_path)
    training_config = TrainingOfflineConfig(config_path)

    initial_state_file = training_config.initial_training_state_file.strip() or None
    model, optimizer = load_initial_state(
        network_config,
        game_config,
        learning_rate=training_config.learning_rate,
        device=training_config.device,
        training_file=initial_state_file,
        skip_loading_optimizer=not training_config.load_optimizer_from_initial_training_state,
        optimizer_type=training_config.optimizer_type,
        optimizer_weight_decay=training_config.optimizer_weight_decay,
    )

    all_files = list_files(training_config.game_data_directory, ".bin")
    all_files = sorted(all_files)
    file_infos = [(path, parse_num_games_from_filename(path)) for path in all_files]
    total_samples = sum(num_samples for _, num_samples in file_infos)

    ############################################################################################
    # Custom logic for training schedule.
    ############################################################################################

    random.seed(42)

    window_size = 2_000_000
    sample_ratio = 3
    train_files = []
    # Train on samples from 13.72M to 19M (17m + 2m window size)
    for start_samples in range(13_721_615, 17_000_000, int(window_size / sample_ratio)):
        train_files += build_sample_window(
            file_infos,
            start_samples=start_samples,
            end_samples=start_samples + window_size,
            origin="start",
        )
    total_train_samples = sum(
        parse_num_games_from_filename(path) for path in train_files
    )

    # train_files = []
    # n = len(all_files)
    # assert n > 300, "Needs at least 300 files."

    # # starts correspond to -300, -280, ..., -60 (all full 60-length windows)
    # for start in range(n - 300, n - 60 + 1, 20):
    #     this_window_files = all_files[start : start + 60]
    #     random.shuffle(this_window_files)
    #     train_files += this_window_files

    ############################################################################################
    # End custom logic for training schedule.
    ############################################################################################

    log(
        f"Training on {len(train_files)} files containing {total_train_samples} samples."
    )
    model.train()

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
    snapshot_history = deque(maxlen=11)

    for batch in dataloader:
        try:
            snapshot_history.append(take_training_snapshot(model, optimizer))

            loss, value_loss, policy_loss = get_loss(
                batch,
                model,
                device=training_config.device,
                policy_loss_weight=training_config.policy_loss_weight,
                value_head_l2=training_config.value_head_l2,
            )

            if loss is None:
                if snapshot_history:
                    rollback_batches = 10
                    if len(snapshot_history) < rollback_batches + 1:
                        rollback_batches = len(snapshot_history) - 1
                    restore_training_snapshot(
                        snapshot_history[0], model, optimizer, training_config.device
                    )
                    snapshot_history.clear()
                    log(
                        "!!! LOSS NOT COMPUTED: RESTORED MODEL/OPTIMIZER "
                        "FROM PREVIOUS BATCHES. CONTINUING. !!!"
                    )
                    continue
                log("!!! LOSS NOT COMPUTED: NO SNAPSHOT AVAILABLE. SKIPPING UPDATE. !!!")
                continue

            optimizer.zero_grad()
            loss.backward()
            if training_config.gradient_clip_norm is not None:
                clip_grad_norm_(model.parameters(), training_config.gradient_clip_norm)
            optimizer.step()

            batch_size = batch[0].shape[0]
            samples_trained += batch_size
            batches_seen += 1

            if batches_seen % log_every_batches == 0:
                log(
                    f"Step {samples_trained}: loss={loss.item():.4f}, "
                    f"value={value_loss.item():.4f}, policy={policy_loss.item():.4f}"
                )

            # TEMPORARY: Stop training after 3m samples.
            # if samples_trained >= 3_000_000:
            #     break

            if samples_trained % 3_000_000 < training_config.batch_size:
                log(f"Saving model and state at {samples_trained} samples.")
                save_model_and_state(
                    model=model,
                    optimizer=optimizer,
                    name=training_config.output_name + f"_tr{samples_trained:09d}",
                    model_directory=training_config.model_directory,
                    training_directory=training_config.training_directory,
                    device=training_config.device,
                    add_timestamp=True,
                )
        except torch.AcceleratorError as exc:
            log(
                "!!! AcceleratorError during batch; skipping batch. "
                f"Error: {exc}"
            )
            continue

    log(f"Finished training pass. Trained {samples_trained} samples.")

    save_model_and_state(
        model=model,
        optimizer=optimizer,
        name=training_config.output_name,
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
