import time
from abc import ABC, abstractmethod
from typing import Optional, NamedTuple, List

class ModelFile(NamedTuple):
    path: str
    creation_time: float


class BaseModelStore(ABC):
    """
    An abstract base class for model stores.
    """
    def __init__(self, cache_duration: float, recency_threshold: float = 15):
        self.cache_duration = cache_duration
        self.recency_threshold = recency_threshold
        self.last_loaded_at = 0

        self.cached_model_files = []

    @abstractmethod
    def _list_model_files(self):
        """
        List all models in the store.
        """
        raise NotImplementedError

    @abstractmethod
    def _create_model(self, filename: str, contents: bytes):
        """
        Get the path to a model in the store.
        """
        raise NotImplementedError

    def create_model(self, filename: str, contents: bytes) -> ModelFile:
        """
        Create a new model in the store.
        """
        self._create_model(filename, contents)
        self.last_loaded_at = 0
        return self.cached_get_latest_model_file()

    def cached_list_model_files(self) -> List[ModelFile]:
        """
        List all models in the store.
        """
        expiry = self.last_loaded_at + self.cache_duration
        if time.time() < expiry:
            model_files = self.cached_model_files
        else:
            model_files = sorted(self._list_model_files(), key=lambda x: x.creation_time)
            self.cached_model_files = model_files
            self.last_loaded_at = time.time()

        return model_files

    def cached_get_latest_model_file(self, include_recent=True) -> Optional[ModelFile]:
        """
        Get the path to a model in the store.
        """
        model_files = self.cached_list_model_files()
        if not model_files:
            return None

        # Apply recency filtering if requested
        if not include_recent:
            model_files = [
                model_file for model_file in model_files
                if model_file.creation_time < time.time() - self.recency_threshold
            ]
            if not model_files:
                return None

        return max(model_files, key=lambda x: x.creation_time)