import io
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
    def save_chunk(self, number: int, chunk: np.ndarray) -> None:
        pass

    def setitem_chunk(self, number: int, key: Tuple, chunk: np.ndarray) -> None:
        data = self.read_chunk(number)
        data.__setitem__(key, chunk)
        self.save_chunk(number, data)

    @abstractmethod
    def save_metadata(self, metadata: Dict) -> None:
        pass

    @abstractmethod
    def read_chunk(self, number: int) -> np.ndarray:
        pass

    @abstractmethod
    def read_metadata(self) -> Dict:
        pass


class LocalSystemBackend(Backend):
    def save_chunk(self, number: int, chunk: np.ndarray) -> None:
        directory = os.path.join(self.path, str(number))
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(os.path.join(directory, str(number)), chunk, allow_pickle=True)

    def save_metadata(self, metadata: Dict) -> None:
        with open(os.path.join(self.path, "metadata.json"), "w") as f:
            f.write(json.dumps(metadata))

    def setitem_chunk(self, number: int, key: Tuple, chunk: np.ndarray) -> None:
        data = self.read_chunk(number)
        data.__setitem__(key, chunk)
        self.save_chunk(number, data)

    def read_chunk(self, number: int) -> np.ndarray:
        return np.load(
            os.path.join(
                self.path,
                str(number),
                str(number) + ".npy"
            ),
            allow_pickle=True
        )

    def read_metadata(self) -> Dict:
        with open(os.path.join(self.path, "metadata.json")) as f:
            return json.loads(f.read())


class S3Backend(Backend):
    def __init__(self, path: AnyStr, config: Dict):
        super().__init__(path, config)
        import boto3
        self.client = boto3.client("s3", **config)

    def save_chunk(self, number: int, chunk: np.ndarray) -> None:
        path = os.path.join(
            self.path,
            str(number),
            str(number)+".npy"
        ).replace("s3://", "")
        bucket_name, key = path.split("/", 1)
        bytes_ = io.BytesIO()
        np.save(bytes_, chunk, allow_pickle=True)
        bytes_.seek(0)
        self.client.upload_fileobj(
            Fileobj=bytes_,
            Bucket=bucket_name,
            Key=key
        )

    def save_metadata(self, metadata: Dict) -> None:
        data = json.dumps(metadata)
        path = self.path.replace("s3://", "")
        bucket_name, key = path.split("/", 1)
        key = os.path.join(key, "metadata.json")
        self.client.put_object(
            Bucket=bucket_name, Key=key, Body=data)

    def read_chunk(self, number: int) -> np.ndarray:
        content = self.get_object(
            os.path.join(
                str(number),
                str(number)+".npy"
            )
        )
        with io.BytesIO(content["Body"].read()) as f:
            f.seek(0)
            return np.load(f)

    def read_metadata(self) -> Dict:
        content = self.get_object("metadata.json")['Body']
        return json.loads(content.read())

    def get_object(self, key: AnyStr):
        try:
            path = os.path.join(self.path.replace("s3://", ""), key)
            bucket_name, _key = path.split("/", 1)
            return self.client.get_object(Bucket=bucket_name, Key=_key)
        except Exception as e:
            raise Exception(bucket_name, _key) from e


BACKENDS = {
    "s3": S3Backend
}


def get_backend(path: AnyStr, config: Dict):
    for k, v in BACKENDS.items():
        if path.startswith(k):
            return v(path, config)
    return LocalSystemBackend(path, config)
