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
    build_optimizer,
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


def select_train_files(
    file_infos: list[tuple[str, int]],
    training_config: TrainingOfflineConfig,
) -> list[str]:
    methodology = training_config.selection_methodology
    total_samples = sum(num_samples for _, num_samples in file_infos)

    if methodology == "custom":
        return build_train_files_custom(file_infos, total_samples)
    if methodology == "final_2.7m":
        return build_train_files_final_2p7m(file_infos, total_samples)
    if methodology == "windowed":
        return build_train_files_windowed(file_infos, total_samples)
    if methodology == "bulk":
        return build_train_files_bulk(file_infos, total_samples)
    if methodology == "dropoff":
        return build_train_files_dropoff(file_infos, total_samples)

    raise ValueError(f"Unknown selection_methodology='{methodology}'.")


def build_train_files_custom(
    file_infos: list[tuple[str, int]], total_samples: int
) -> list[str]:
    # Intentionally hand-editable custom schedule.
    return build_sample_window(
        file_infos,
        start_samples=2_700_000,
        end_samples=0,
        origin="end",
    )


def build_train_files_final_2p7m(
    file_infos: list[tuple[str, int]], total_samples: int
) -> list[str]:
    return build_sample_window(
        file_infos,
        start_samples=2_700_000,
        end_samples=0,
        origin="end",
    )


def build_train_files_dropoff(
    file_infos: list[tuple[str, int]], total_samples: int
) -> list[str]:
    # Build a weighted subset of about 2.7M samples from the most recent 5.0M.
    max_age_in_samples = 5_000_000
    target_window_samples = 2_700_000
    dropoff_power = 3.4
    sorted_infos = sorted(file_infos)
    recent_candidates: list[tuple[str, int, int]] = []
    samples_from_end = 0

    for path, num_samples in reversed(sorted_infos):
        if samples_from_end >= max_age_in_samples:
            break
        recent_candidates.append((path, num_samples, samples_from_end))
        samples_from_end += num_samples

    if not recent_candidates:
        return []

    newest_path, newest_samples, _ = recent_candidates[0]
    train_files: list[str] = [newest_path]
    selected_samples = newest_samples

    if selected_samples < target_window_samples:
        weighted_pool: list[tuple[float, str, int]] = []
        for path, num_samples, age_in_samples in recent_candidates[1:]:
            # Weight decays by sample age (not file index): newest=1.0, oldest~0.0.
            base_weight = max(1e-9, 1.0 - (age_in_samples / max_age_in_samples))
            weight = max(1e-9, base_weight**dropoff_power)
            # Weighted sampling without replacement (larger key = sampled earlier).
            key = random.random() ** (1.0 / weight)
            weighted_pool.append((key, path, num_samples))

        for _, path, num_samples in sorted(weighted_pool, reverse=True):
            train_files.append(path)
            selected_samples += num_samples
            if selected_samples >= target_window_samples:
                break

    return random.sample(train_files, len(train_files))


def build_train_files_windowed(
    file_infos: list[tuple[str, int]], total_samples: int
) -> list[str]:
    window_size = 2_000_000
    sample_ratio = 3
    train_files = []

    for start_samples in range(13_721_615 - window_size, 19_304_037 - window_size, int(window_size / sample_ratio)):
        train_files += build_sample_window(
            file_infos,
            start_samples=start_samples,
            end_samples=start_samples + window_size,
            origin="start",
        )

    return train_files


def build_train_files_bulk(
    file_infos: list[tuple[str, int]], total_samples: int
) -> list[str]:
    endpoint_from_start = 10_000_000

    train_files = []

    # Last 3/4
    train_files += build_sample_window(
        file_infos,
        start_samples=int(endpoint_from_start * 0.25),
        end_samples=endpoint_from_start,
        origin="start",
    )

    # Last 1/2
    train_files += build_sample_window(
        file_infos,
        start_samples=int(endpoint_from_start * 0.5),
        end_samples=endpoint_from_start,
        origin="start",
    )

    # Last 1/4
    train_files += build_sample_window(
        file_infos,
        start_samples=int(endpoint_from_start * 0.75),
        end_samples=endpoint_from_start,
        origin="start",
    )

    # Last 1/8th
    train_files += build_sample_window(
        file_infos,
        start_samples=int(endpoint_from_start * 0.875),
        end_samples=endpoint_from_start,
        origin="start",
    )

    return train_files


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

    all_files = sorted(list_files(training_config.game_data_directory, ".bin"))
    file_infos = [(path, parse_num_games_from_filename(path)) for path in all_files]
    total_samples = sum(num_samples for _, num_samples in file_infos)
    train_files = select_train_files(file_infos, training_config)

    total_train_samples = sum(
        parse_num_games_from_filename(path) for path in train_files
    )

    log(
        "Training with "
        f"selection_methodology='{training_config.selection_methodology}' "
        f"on {len(train_files)} files containing {total_train_samples} samples."
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
    # samples_seen_since_last_optimizer_reset = 0
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
                log(
                    "!!! LOSS NOT COMPUTED: NO SNAPSHOT AVAILABLE. SKIPPING UPDATE. !!!"
                )
                continue

            optimizer.zero_grad()
            loss.backward()
            if (
                training_config.gradient_clip_norm is not None
                and training_config.gradient_clip_norm > 0.0
            ):
                clip_grad_norm_(model.parameters(), training_config.gradient_clip_norm)
            optimizer.step()

            batch_size = batch[0].shape[0]
            samples_trained += batch_size
            # samples_seen_since_last_optimizer_reset += batch_size
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

            # if samples_seen_since_last_optimizer_reset >= window_size:
            #     optimizer = build_optimizer(
            #         model=model,
            #         optimizer_type=training_config.optimizer_type,
            #         optimizer_weight_decay=training_config.optimizer_weight_decay,
            #         learning_rate=training_config.learning_rate,
            #     )
            #     samples_seen_since_last_optimizer_reset = 0

        except torch.AcceleratorError as exc:
            log(f"!!! AcceleratorError during batch; skipping batch. Error: {exc}")
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
