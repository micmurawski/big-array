from typing import AnyStr, Dict, Sequence, Tuple

import numpy as np

from cloud_array.backends import Backend, get_backend
from cloud_array.exceptions import CloudArrayException
from cloud_array.helpers import (chunk2list, collect, compute_index_of_slice, compute_number_of_chunks,
                                 generate_chunks_slices, get_chunk_slice_by_index, parse_key_to_slices)


class Chunk:
    def __init__(self, chunk_number: int, dtype, url: AnyStr, chunk_slice: Tuple[slice], backend: Backend) -> None:
        self.uri = url
        self.chunk_number = chunk_number
        self.backend = backend
        self._slice = chunk_slice
        self.dtype = dtype

    @property
    def shape(self):
        return tuple((s.stop-s.start)//(self.slice.step or 1) for s in self.slice)

    @shape.setter
    def shape(self, _):
        raise CloudArrayException("Cannot change value of shape.")

    @property
    def slice(self):
        return self._slice

    @slice.setter
    def slice(self, _):
        raise CloudArrayException("Cannot change value of slice")

    def save(self, data: np.ndarray) -> None:
        return self.backend.save_chunk(self.chunk_number, data)

    def __getitem__(self, key: Tuple) -> np.ndarray:
        return self.backend.read_chunk(self.chunk_number).__getitem__(key)

    def __setitem__(self, key: Tuple, data: np.ndarray) -> None:
        self.backend.setitem_chunk(self.chunk_number, key, data)


class CloudArray:
    def __init__(
        self, chunk_shape: Tuple[int], array: np.ndarray = None,
        shape: Tuple[int] = None, dtype=None, url: AnyStr = None, config={},
        backend: Backend = None
    ):
        self.chunk_shape = chunk_shape
        self.url = url
        self.array = array
        if array is None and dtype is None:
            raise CloudArrayException("Dtype must be defined.")
        if array is None and shape is None:
            raise CloudArrayException(
                "Shape must be defined by array or shape alone.")
        self._shape = array.shape if array is not None else shape
        self._dtype = array.dtype if array is not None else dtype
        self._chunks_number = self.count_number_of_chunks(
            self.shape,
            self.chunk_shape
        )
        self.backend = backend(
            url, config) if backend else get_backend(url, config)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, _):
        raise CloudArrayException("Cannot change value of shape.")

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, _):
        raise CloudArrayException("Cannot change value of dtype.")

    @property
    def chunks_number(self):
        return self._chunks_number

    @chunks_number.setter
    def chunks_number(self, _):
        raise CloudArrayException("Cannot change value of chunks_number.")

    @property
    def metadata(self) -> Dict:
        return self.backend.read_metadata()

    @metadata.setter
    def metadata(self, _):
        raise CloudArrayException("Cannot change value of metadata.")

    def get_metadata(self) -> dict:
        result = {
            "chunk_shape": self.chunk_shape,
            "dtype":  str(self.dtype),
            "chunks": {}
        }
        for i, chunk in enumerate(self.generate_chunks_slices()):
            for j, dim in enumerate(chunk):
                if dim.stop == self.shape[j]:
                    result["chunks"][i] = chunk2list(chunk)

        return result

    def generate_chunks_slices(self) -> Tuple[slice]:
        for _slice in generate_chunks_slices(self.shape, self.chunk_shape):
            yield _slice

    def get_chunk_slice_by_index(self, number: int) -> Tuple[slice]:
        return get_chunk_slice_by_index(self.shape, self.chunk_shape, number)

    @staticmethod
    def count_number_of_chunks(shape: Tuple[int], chunk_shape: Tuple[int]) -> int:
        return compute_number_of_chunks(shape, chunk_shape)

    def get_chunk(self, chunk_number: int) -> Chunk:
        chunk_slice = self.get_chunk_slice_by_index(chunk_number)
        return Chunk(
            chunk_number=chunk_number, url=self.url, chunk_slice=chunk_slice,
            dtype=self.dtype, backend=self.backend
        )

    def chunks(self) -> Chunk:
        for i in range(self.chunks_number):
            yield self.get_chunk(i)

    def save(self, array=None) -> None:
        if array is None and self.array is None:
            raise CloudArrayException("Array is not declared.")
        array = array or self.array
        metadata = self.get_metadata()
        self.backend.save_metadata(metadata)
        for chunk in self.chunks():
            chunk.save(array[chunk.slice])

    def __getitem__(self, key) -> np.ndarray:
        new_key = parse_key_to_slices(self.shape, self.chunk_shape, key)

        def _get_chunk_data_by_key(key: Sequence[slice]):
            idx = compute_index_of_slice(key, self.shape, self.chunk_shape)
            chunk = self.get_chunk(idx)
            return chunk[:, :, :]

        dataset = collect(
            slices=list(new_key),
            shape=self.shape,
            chunk_shape=self.chunk_shape,
            get_items=_get_chunk_data_by_key
        )
        return dataset.__getitem__(new_key)
