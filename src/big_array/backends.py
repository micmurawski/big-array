import json
import os
from abc import ABCMeta, abstractclassmethod
from typing import AnyStr, Dict, Tuple

import numpy as np


class Backend(metaclass=ABCMeta):
    @abstractclassmethod
    def save_chunk(key, chunk: np.array, path: AnyStr, config: Dict) -> None:
        pass

    @abstractclassmethod
    def save_metadata(metadata: Dict, path: AnyStr, config: Dict) -> None:
        pass

    @abstractclassmethod
    def read_chunk(number: int, path: AnyStr, dtype, shape: Tuple, config: Dict) -> np.array:
        pass

    @abstractclassmethod
    def read_metadata(path: AnyStr, config: Dict) -> Dict:
        pass


class LocalSystemBackend(Backend):
    def save_chunk(key, chunk: np.array, path: AnyStr, config: Dict) -> None:
        directory = path.rsplit('/', 1)[0]
        if not os.path.exists(directory):
            os.makedirs(directory)
        chunk.tofile(path)

    def save_metadata(metadata: Dict, path: AnyStr, config: Dict) -> None:
        with open(os.path.join(path, "metadata.json"), "w") as f:
            f.write(json.dumps(metadata))

    def read_chunk(number: int, path: AnyStr, dtype, shape: Tuple, config: Dict) -> np.array:
        with open(path) as f:
            return np.fromfile(f, dtype=dtype).reshape(shape)

    def read_metadata(path: AnyStr, config: Dict) -> Dict:
        with open(os.path.join(path, "metadata.json")) as f:
            return json.loads(f.read())


class S3Backend(Backend):
    def save_chunk(key, chunk: np.array, path: AnyStr, config: Dict) -> None:
        pass

    def save_metadata(metadata: Dict, path: AnyStr, config: Dict) -> None:
        pass

    def read_chunk(number: int, path: AnyStr, dtype, shape: Tuple, config: Dict) -> np.array:
        pass

    def read_metadata(path: AnyStr, config: Dict) -> Dict:
        pass
