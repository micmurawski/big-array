import json
import os
from abc import ABCMeta, abstractclassmethod
from typing import AnyStr, Dict

import numpy as np


class Backend(metaclass=ABCMeta):
    @abstractclassmethod
    def save_chunk(chunk, number: int, path: AnyStr) -> None:
        pass

    @abstractclassmethod
    def save_metadata(path: AnyStr) -> None:
        pass

    @abstractclassmethod
    def read_chunk(number, path: AnyStr) -> None:
        pass

    @abstractclassmethod
    def read_metadata(path: AnyStr) -> Dict:
        pass


class LocalSystemBackend(Backend):
    def save_chunk(key, chunk: np.array, path: AnyStr) -> None:
        directory = path.rsplit('/', 1)[0]
        if not os.path.exists(directory):
            os.makedirs(directory)
        chunk.tofile(path)

    def save_metadata(metadata: Dict, path: AnyStr) -> None:
        with open(os.path.join(path, "metadata.json"), "w") as f:
            f.write(json.dumps(metadata))

    def read_chunk(number, path) -> np.array:
        with open(path, "b") as f:
            return np.array.fromfile(f)

    def read_metadata(path: AnyStr) -> Dict:
        with open(os.path.join(path, "metadata.json")) as f:
            return json.loads(f.read())
