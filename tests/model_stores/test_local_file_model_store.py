import os
import tempfile

from alpha_blokus.model_stores.local_file_model_store import LocalFileModelStore


def test_local_file_model_store():
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        f.write(b"test model data")
        temp_path = f.name

    # Initialize the store
    store = LocalFileModelStore(temp_path)

    # Test that we can list the single model file
    model_files = store.cached_list_model_files()
    assert len(model_files) == 1
    assert model_files[0].path == temp_path
    assert model_files[0].creation_time > 0

    # Test that we can get the latest model file
    latest_file = store.cached_get_latest_model_file()
    assert latest_file is not None
    assert latest_file.path == temp_path
    assert latest_file.creation_time == model_files[0].creation_time

    # Test with include_recent=False (should still return the file since recency_threshold=0)
    latest_file_no_recent = store.cached_get_latest_model_file(include_recent=False)
    assert latest_file_no_recent is not None
    assert latest_file_no_recent.path == temp_path

    # Clean up the temporary file
    os.unlink(temp_path)
