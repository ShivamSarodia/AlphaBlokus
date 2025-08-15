import os
from typing import List

from alpha_blokus.model_stores.base import BaseModelStore, ModelFile

class LocalDirectoryModelStore(BaseModelStore):
    """
    A model store that returns just a single model.
    """
    def __init__(
        self,
        model_path: str,
        model_extension: str,
        cache_duration: float,
        recency_threshold: float = 15,
    ):
        super().__init__(cache_duration=cache_duration, recency_threshold=recency_threshold)

        self.model_path = model_path
        self.model_extension = model_extension

        assert os.path.isdir(model_path), "Model path must be a directory"
            
    def _create_model(self, filename: str, contents: bytes):
        """
        Create a new model in the store.
        """
        with open(os.path.join(self.model_path, filename), "wb") as f:
            f.write(contents)

    def _list_model_files(self) -> List[ModelFile]:
        """
        List all models in the store.
        """
        model_files: List[ModelFile] = []
        with os.scandir(self.model_path) as entries:
            for entry in entries:
                if (
                    entry.is_file() and
                    entry.name.endswith(self.model_extension)
                ):
                    model_files.append(ModelFile(
                        path=entry.path,
                        creation_time=entry.stat().st_mtime,
                    ))
        return model_files