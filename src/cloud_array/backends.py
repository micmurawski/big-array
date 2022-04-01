import json
import os
from abc import ABCMeta, abstractmethod
from typing import AnyStr, Dict, Tuple

import numpy as np


class Backend(metaclass=ABCMeta):
    def __init__(self, path: AnyStr, config: Dict):
        self.config = config
        self.path = path

    @abstractmethod
    def save_chunk(self, number: int, chunk: np.array) -> None:
        pass

    @abstractmethod
    def save_metadata(self, metadata: Dict) -> None:
        pass

    @abstractmethod
    def read_chunk(self, number: int, dtype, shape: Tuple, ) -> np.array:
        pass

    @abstractmethod
    def read_metadata(self) -> Dict:
        pass


class LocalSystemBackend(Backend):
    def save_chunk(self, number: int, chunk: np.array) -> None:
        directory = os.path.join(self.path, str(number))
        if not os.path.exists(directory):
            os.makedirs(directory)
        chunk.tofile(os.path.join(directory, str(number)))

    def save_metadata(self, metadata: Dict) -> None:
        with open(os.path.join(self.path, "metadata.json"), "w") as f:
            f.write(json.dumps(metadata))

    def read_chunk(self, number: int, dtype, shape: Tuple) -> np.array:
        with open(os.path.join(self.path, str(number), str(number))) as f:
            return np.fromfile(f, dtype=dtype).reshape(shape)

    def read_metadata(self) -> Dict:
        with open(os.path.join(self.path, "metadata.json")) as f:
            return json.loads(f.read())


class S3Backend(Backend):
    def save_chunk(self, key, chunk: np.array) -> None:
        pass

    def save_metadata(self, metadata: Dict) -> None:
        pass

    def read_chunk(self, number: int, dtype, shape: Tuple) -> np.array:
        pass

    def read_metadata(self) -> Dict:
        pass


BACKENDS = {
    "s3": S3Backend
}


def get_backend(path: AnyStr, config: Dict):
    for k, v in BACKENDS.items():
        if path.startswith(k):
            return v(path, config)
    return LocalSystemBackend(path, config)
