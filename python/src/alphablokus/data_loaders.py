from __future__ import annotations

from dataclasses import dataclass
import os
import random
from typing import Iterable, List, Optional, Sequence

import torch

from alphablokus.configs import GameConfig
from alphablokus.files import (
    is_s3,
    list_files,
    localize_file,
    parse_num_games_from_filename,
)
from alphablokus.game_data import load_game_file
from alphablokus.log import log


@dataclass(frozen=True)
class GameFileInfo:
    path: str
    num_samples: int


@dataclass
class DataLoaderStats:
    files_loaded: int = 0
    samples_loaded: int = 0
    batches_yielded: int = 0
    samples_yielded: int = 0
    buffer_refills: int = 0


def list_game_files_with_samples(
    directory: str,
) -> List[GameFileInfo]:
    files = list_files(directory, ".bin")
    files = sorted(files, reverse=True)
    return [
        GameFileInfo(path=path, num_samples=parse_num_games_from_filename(path))
        for path in files
    ]


def build_sample_window(
    files: Sequence[GameFileInfo],
    window_size_samples: int,
) -> List[GameFileInfo]:
    window = []
    total_samples = 0
    for file_info in files:
        if total_samples >= window_size_samples:
            break
        window.append(file_info)
        total_samples += file_info.num_samples
    return window


def _stack_batch(samples: Sequence[tuple[torch.Tensor, ...]]):
    boards, values, policies, valid_masks = zip(*samples)
    return (
        torch.stack(boards),
        torch.stack(values),
        torch.stack(policies),
        torch.stack(valid_masks),
    )


class FileProvider:
    def next_files(
        self, count: int, worker_id: int = 0, num_workers: int = 1
    ) -> List[str]:
        raise NotImplementedError


class StaticListFileProvider(FileProvider):
    def __init__(self, file_paths: Sequence[str]):
        self._file_paths = list(file_paths)
        self._cursor = 0

    def next_files(
        self, count: int, worker_id: int = 0, num_workers: int = 1
    ) -> List[str]:
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        if worker_id < 0 or worker_id >= num_workers:
            raise ValueError("worker_id must be in [0, num_workers)")

        this_worker_paths = self._file_paths[worker_id::num_workers]

        if self._cursor >= len(this_worker_paths):
            return []
        start = self._cursor
        end = min(start + count, len(this_worker_paths))
        self._cursor = end
        return list(this_worker_paths[start:end])


class StaticWindowedS3FileProvider(FileProvider):
    def __init__(
        self,
        s3_prefix: str,
        *,
        window_size_samples: int,
    ):
        self.s3_prefix = s3_prefix
        self.window_size_samples = window_size_samples

        # Maintain an in-memory shuffled window of files.
        self.current_window_paths = []

    def _reload_window(self):
        """
        Query S3 to populate self.current_window_paths with the latest batch files
        from S3.
        """
        files = list_game_files_with_samples(self.s3_prefix)
        window_files = build_sample_window(files, self.window_size_samples)
        self.current_window_paths = [file_info.path for file_info in window_files]
        random.shuffle(self.current_window_paths)
        total_samples = sum(file_info.num_samples for file_info in window_files)
        log(
            f"Reloaded window with {len(self.current_window_paths)} files. Found {total_samples} samples."
        )

    def next_files(
        self, count: int, worker_id: int = 0, num_workers: int = 1
    ) -> List[str]:
        if num_workers != 1 or worker_id != 0:
            raise ValueError(
                "StaticWindowedS3FileProvider does not support multiple workers."
            )

        output = []
        while len(output) < count:
            if not self.current_window_paths:
                self._reload_window()
            output.append(self.current_window_paths.pop(0))
        return output


class BufferedGameBatchDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        game_config: GameConfig,
        file_provider: FileProvider,
        batch_size: int,
        in_memory_shuffle_file_count: int,
        *,
        local_cache_dir: Optional[str] = None,
        cleanup_local_files: bool = False,
        stats: Optional[DataLoaderStats] = None,
    ):
        self.game_config = game_config
        self.file_provider = file_provider
        self.batch_size = batch_size
        self.in_memory_shuffle_file_count = in_memory_shuffle_file_count
        self.local_cache_dir = local_cache_dir
        self.cleanup_local_files = cleanup_local_files
        self.stats = stats

    def _load_samples_from_file(
        self,
        path: str,
    ) -> List[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        local_path = localize_file(path, self.local_cache_dir)
        boards, values, policies, valid_masks = load_game_file(
            self.game_config, local_path
        )
        if self.stats is not None:
            self.stats.files_loaded += 1
            self.stats.samples_loaded += len(boards)
        if self.cleanup_local_files and is_s3(path) and local_path != path:
            try:
                os.remove(local_path)
            except OSError:
                pass
        return list(zip(boards, values, policies, valid_masks))

    def __iter__(self) -> Iterable[tuple[torch.Tensor, ...]]:
        worker_info = torch.utils.data.get_worker_info()
        rng = random.Random()
        if worker_info is not None:
            rng = random.Random(worker_info.seed)
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        buffer: List[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        while True:
            if len(buffer) < self.batch_size:
                next_files = self.file_provider.next_files(
                    self.in_memory_shuffle_file_count,
                    worker_id=worker_id,
                    num_workers=num_workers,
                )
                if not next_files:
                    break
                if self.stats is not None:
                    self.stats.buffer_refills += 1
                for path in next_files:
                    buffer.extend(self._load_samples_from_file(path))
                rng.shuffle(buffer)

            if len(buffer) < self.batch_size:
                break

            batch_samples = buffer[: self.batch_size]
            buffer = buffer[self.batch_size :]
            yield _stack_batch(batch_samples)
            if self.stats is not None:
                self.stats.batches_yielded += 1
                self.stats.samples_yielded += self.batch_size


def build_streaming_dataloader(
    dataset: torch.utils.data.IterableDataset,
    *,
    num_workers: int,
    prefetch_factor: int,
) -> torch.utils.data.DataLoader:
    kwargs = {
        "batch_size": None,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return torch.utils.data.DataLoader(dataset, **kwargs)
