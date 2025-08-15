import random
import os
import time

from alpha_blokus.model_stores.local_directory_model_store import (
    LocalDirectoryModelStore,
)


def test_local_directory_model_store():
    randomint = random.randint(0, 1_000_000)
    model_path = f"/tmp/alphablokus_test_{randomint}/"
    os.makedirs(model_path, exist_ok=True)

    model_extension = ".pt"
    cache_duration = 1.0
    recency_threshold = 0.5

    store = LocalDirectoryModelStore(
        model_path=model_path,
        model_extension=model_extension,
        cache_duration=cache_duration,
        recency_threshold=recency_threshold,
    )

    # Time = 0
    assert store.cached_list_model_files() == []

    store.create_model("model_1.pt", b"model_1")

    # After creating a model, it should appear in the cache.
    model_files = store.cached_list_model_files()
    assert len(model_files) == 1
    assert model_files[0].path == model_path + "model_1.pt"

    # But getting latest with include_recent=False should return None due to recency threshold
    assert store.cached_get_latest_model_file(include_recent=False) is None

    # Create the next model at 0.75s.
    time.sleep(0.75)
    store.create_model("model_2.pt", b"model_2")

    # Time = 1.05
    time.sleep(0.3)

    # Now, the cache will be refreshed. All models should be listed.
    assert [f.path for f in store.cached_list_model_files()] == [
        model_path + "model_1.pt",
        model_path + "model_2.pt",
    ]
    # But the latest model excluding recent should be model_1 due to recency threshold.
    latest_file = store.cached_get_latest_model_file(include_recent=False)
    assert latest_file is not None
    assert latest_file.path == model_path + "model_1.pt"

    time.sleep(0.5)

    # Now, model_2 should be old enough to be included in latest.
    assert [f.path for f in store.cached_list_model_files()] == [
        model_path + "model_1.pt",
        model_path + "model_2.pt",
    ]
    latest_file = store.cached_get_latest_model_file()
    assert latest_file is not None
    assert latest_file.path == model_path + "model_2.pt"
