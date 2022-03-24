import os
from itertools import product
from typing import AnyStr, Dict, Tuple, Union, Type

import numpy as np

from big_array.backends import Backend, LocalSystemBackend
from big_array.utils import chunk2list, is_in


class BigArrayException(Exception):
    pass


class Chunk:
    def __init__(self, chunk_number: int, url: AnyStr, chunk_slice: Tuple[slice], backend: Backend) -> None:
        self.uri = os.path.join(url, str(chunk_number), str(chunk_number))
        self.chunk_number = chunk_number
        self.backend = backend
        self.chunk_slice = chunk_slice

    @property
    def slice(self):
        return self.chunk_slice

    def __setitem__(self, key: Tuple[Type[slice]], data: np.array) -> None:
        return self.backend.save_chunk(key, data, self.uri)

    def __getitem__(self, key: Tuple[Type[slice]]) -> np.array:
        return self.backend.read_chunk(key)


class BigArray:
    def __init__(self, chunk_shape, array=None, shape=None, url=None, *arg, **kwargs):
        self.chunk_shape = chunk_shape
        self.url = url
        self.array = array
        if array is None and shape is None:
            raise BigArrayException(
                "Shape must be defined by array or shape alone.")
        self.shape = array.shape if array is not None else shape
        self.chunks_number = self.count_number_of_chunks(self.shape, self.chunk_shape)

    def set_shape(self, s):
        raise BigArrayException("You cannot change value of shape.")

    @property
    def metadata(self) -> Dict:
        return LocalSystemBackend.read_metadata(self.url)

    def get_metadata(self) -> dict:
        result = {
            "chunk_shape": self.chunk_shape,
            "chunks": {}
        }
        for i, chunk in enumerate(self.generate_chunks_slices()):
            for j, dim in enumerate(chunk):
                if dim.stop == self.shape[j]:
                    result["chunks"][i] = chunk2list(chunk)

        return result

    def generate_chunks_slices(self):
        _ranges = (range(0, a, c)
                   for c, a in zip(self.chunk_shape, self.shape))
        p = product(*_ranges)
        for i in p:
            _s = []
            for j in range(len(self.shape)):
                _s.append(
                    slice(
                        i[j],
                        self.shape[j] if i[j]+self.chunk_shape[j] 
                        > self.shape[j] else i[j]+self.chunk_shape[j])
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
        return Chunk(chunk_number, self.url, chunk_slice, LocalSystemBackend)


    def save(self, array=None):
        if array is None and self.array is None:
            raise BigArrayException("Array is not declared.")
        array = array or self.array
        metadata = self.get_metadata()
        LocalSystemBackend.save_metadata(metadata, self.url)
        for i in range(self.chunks_number):
            chunk = self.get_chunk(i)
            chunk[:,:,:] = array[chunk.slice]


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
                chunks.append(i)

