import os
from typing import List

from alpha_blokus.model_stores.base import BaseModelStore, ModelFile


class LocalFileModelStore(BaseModelStore):
    """
    A model store that returns just a single model.
    """

    def __init__(self, model_path: str):
        # The file model never updates, so we can cache it forever.
        super().__init__(cache_duration=1e9, recency_threshold=0)

        self.model_path = model_path
        assert os.path.isfile(model_path), "Model path must be a file"

    def _create_model(self, filename: str, contents: bytes):
        """
        Create a new model in the store.
        """
        raise ValueError("cannot create model in single-file store")

    def _list_model_files(self) -> List[ModelFile]:
        """
        List all models in the store.
        """
        stat = os.stat(self.model_path)
        return [
            ModelFile(
                path=self.model_path,
                creation_time=stat.st_mtime,
            )
        ]
