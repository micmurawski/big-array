from itertools import product
from math import ceil
from typing import AnyStr, Dict, List, Tuple

import numpy as np

from cloud_array.backends import Backend, get_backend
from cloud_array.exceptions import CloudArrayException
from cloud_array.utils import (chunk2list, compute_key, compute_number_of_chunks, get_chunk_index,
                               get_index_of_iter_product, merge_datasets, sort_chunks)


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
        _ranges = (
            range(0, a, c)
            for c, a in zip(self.chunk_shape, self.shape)
        )
        p = product(*_ranges)
        for i in p:
            yield tuple(
                slice(
                    i[j],
                    self.shape[j] if i[j]+self.chunk_shape[j]
                    > self.shape[j] else i[j]+self.chunk_shape[j]
                )
                for j in range(len(self.shape))
            )

    def get_chunk_slice_by_index(self, number: int) -> Tuple[slice]:
        p = tuple((0, a, c) for c, a in zip(self.chunk_shape, self.shape))
        val = get_index_of_iter_product(number, p)
        return tuple(
            slice(
                val[j],
                self.shape[j] if val[j]+self.chunk_shape[j]
                > self.shape[j] else val[j]+self.chunk_shape[j]
            )
            for j in range(len(self.shape))
        )

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

    def initial_merge_of_chunks(self, sorted_chunks) -> List[Tuple[np.ndarray, Tuple[slice]]]:
        datasets = []
        for x in sorted_chunks:
            data = None
            for i in x[0]:
                chunk_data = self.get_chunk(i)[:, :, :]
                if data is None:
                    data = chunk_data
                else:
                    data = np.concatenate(
                        (data, chunk_data),
                        axis=x[1]
                    )
            datasets.append(
                (data, x[2])
            )
        return datasets

    def parse_key_to_slices(self, key: Tuple[slice]):
        result = []
        for i in range(len(key)):
            val = key[i]
            if isinstance(val, int):
                if val < 0:
                    val = self.shape[i] + val
                result.append(
                    slice(val, val+1)
                )
            else:
                start = val.start or 0
                stop = val.stop or self.shape[i]
                if start > self.shape[i] or stop > self.shape[i]:
                    raise CloudArrayException(
                        f"Slice {key[i]} does not fit shape: {self.shape}.")
                if start >= stop:
                    raise CloudArrayException(
                        f"Key invalid slice {key[i]}. Start >= stop.")
                if start < 0:
                    start = self.shape[i] + start
                if stop < 0:
                    stop = self.shape[i] + stop

                result.append(
                    slice(
                        start,
                        stop,
                        val.step if val.step else 1
                    )
                )
        return tuple(result)

    def __getitem__(self, key) -> np.ndarray:
        slices = self.parse_key_to_slices(key)
        _normalized_shape = [ceil(i/j) for i, j in zip(self.shape, self.chunk_shape)]
        _normalized_ranges = [range(ss.start//c, ss.stop//c + 1, 1) for ss, c in zip(slices, self.chunk_shape)]
        chunks_indexes_gen = product(*_normalized_ranges)
        chunks = [
            (
                (idx := get_chunk_index(i, _normalized_shape)),
                self.get_chunk_slice_by_index(idx)
            )
            for i in chunks_indexes_gen
        ]
        if len(chunks) == 1:
            new_key = compute_key(slices, chunks[0][1])
            return self.get_chunk(chunks[0][0]).__getitem__(new_key)
        sorted_chunks = sort_chunks(chunks)
        datasets = self.initial_merge_of_chunks(sorted_chunks)
        datasets = merge_datasets(datasets)
        while len(datasets) > 1:
            datasets = merge_datasets(datasets)
        new_key = compute_key(slices, datasets[0][1], datasets[0][0].shape)
        return datasets[0][0].__getitem__(new_key)
