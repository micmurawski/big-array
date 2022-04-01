import os
from itertools import product
from typing import AnyStr, Dict, Tuple

import numpy as np

from cloud_array.backends import Backend, get_backend
from cloud_array.utils import (chunk2list, compute_key, initial_merge_of_chunks,
                               is_in, merge_datasets, sort_chunks)


class CloudArrayException(Exception):
    pass


class Chunk:
    def __init__(self, chunk_number: int, dtype, url: AnyStr, chunk_slice: Tuple[slice], backend: Backend) -> None:
        self.uri = url
        self.chunk_number = chunk_number
        self.backend = backend
        self._slice = chunk_slice
        self.dtype = dtype

    @property
    def shape(self):
        return tuple(s.stop-s.start for s in self.slice)

    @shape.setter
    def shape(self, _):
        raise CloudArrayException("Cannot change value of shape.")

    @property
    def slice(self):
        return self._slice

    @slice.setter
    def slice(self, _):
        raise CloudArrayException("Cannot change value of slice")

    def save(self, data: np.array) -> None:
        return self.backend.save_chunk(self.chunk_number, data)

    def __getitem__(self, key: Tuple) -> np.array:
        return self.backend.read_chunk(self.chunk_number, self.dtype, self.shape).__getitem__(key)


class CloudArray:
    def __init__(self, chunk_shape, array=None, shape=None, dtype=None, url=None, config={}):
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
        self.backend = get_backend(url, config)

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

    def generate_chunks_slices(self):
        _ranges = (
            range(0, a, c)
            for c, a in zip(self.chunk_shape, self.shape)
        )
        p = product(*_ranges)
        for i in p:
            _s = []
            for j in range(len(self.shape)):
                _s.append(
                    slice(
                        i[j],
                        self.shape[j] if i[j]+self.chunk_shape[j]
                        > self.shape[j] else i[j]+self.chunk_shape[j]
                    )
                )

            yield tuple(_s)

    def get_chunk_slice_by_number(self, number):
        _ranges = (range(0, a, c)
                   for c, a in zip(self.chunk_shape, self.shape))
        p = product(*_ranges)
        val = None
        for _ in range(number+1):
            val = next(p)
        _s = []
        for j in range(len(self.shape)):
            _s.append(
                slice(
                    val[j],
                    self.shape[j] if val[j]+self.chunk_shape[j]
                    > self.shape[j] else val[j]+self.chunk_shape[j])
            )
        return tuple(_s)

    @staticmethod
    def count_number_of_chunks(shape, chunk_shape):
        x = [round(shape[i]/chunk_shape[i]) for i in range(len(shape))]
        r = 1
        for i in x:
            if i != 0:
                r *= i
        return r

    def get_chunk(self, chunk_number: int):
        chunk_slice = self.get_chunk_slice_by_number(chunk_number)
        return Chunk(
            chunk_number=chunk_number, url=self.url, chunk_slice=chunk_slice,
            dtype=self.dtype, backend=self.backend
        )

    def save(self, array=None):
        if array is None and self.array is None:
            raise CloudArrayException("Array is not declared.")
        array = array or self.array
        metadata = self.get_metadata()
        self.backend.save_metadata(metadata)
        for i in range(self.chunks_number):
            chunk = self.get_chunk(i)
            chunk.save(array[chunk.slice])

    def __getitem__(self, key) -> np.array:
        key = list(key)
        for i in range(len(key)):
            key[i] = slice(
                key[i].start if key[i].start else 0,
                key[i].stop if key[i].stop else self.shape[i],
                None
            )
        chunks = []
        for i, s in enumerate(self.generate_chunks_slices()):
            if is_in(s, key):
                chunks.append((i, s))

        sorted_chunks = sort_chunks(chunks)
        datasets = initial_merge_of_chunks(self, sorted_chunks)
        datasets = merge_datasets(datasets)

        if len(datasets) == 1:
            new_key = compute_key(key, datasets[0][1])
            return datasets[0][0].__getitem__(new_key)
