import operator
from copy import copy
from functools import reduce
from itertools import product
from math import ceil
from typing import Callable, List, Sequence, Tuple

import numpy as np

from .exceptions import CloudArrayException


def compute_number_of_chunks(shape: Tuple[int], chunk_shape: Tuple[int]) -> int:
    """
    This function computes number of chunks required to fit given shape of array and shape of chunk.
    The shape is shape of whole array and chunk_shape is chunk shape.
    """
    return reduce(operator.mul, map(lambda x: ceil(x[0]/x[1]) or 1, zip(shape, chunk_shape)), 1)


def get_index_of_iter_product(n: int, p: Sequence[Tuple[int]]) -> Tuple[int]:
    """
    This function computes value of product of ranges for given n.
    The p is a sequence of range arguments start, stop, step.
    """
    _p = [ceil((stop-start)/step) for start, stop, step in p]
    result = []
    for i in range(len(_p)-1, 0, -1):
        r = n % _p[i]
        result.append((r+p[i][0])*p[i][2])
        n = (n-r)//_p[i]
    result.append((n+p[0][0])*p[0][2])
    return tuple(result[::-1])


def compute_index_of_slice(slice: Sequence[slice], shape: Tuple[int], chunk_shape: Tuple[int]) -> int:
    m = [1] + [ceil(i/j) for i, j in zip(shape[::-1], chunk_shape[::-1])][:-1]
    m = [i*j for i, j in zip(m, [1]+m[:-1])]
    x = [i.start // j*k for i, j, k in zip(slice[::-1], chunk_shape[::-1], m)]
    return sum(x)


def collect(
    slices: Sequence[slice],
    shape: Sequence[int],
    chunk_shape: Sequence[int],
    get_items: Callable,
    level: int = 0,
) -> np.ndarray:
    p: np.ndarray = None
    if level < len(slices):
        num_of_pieces = ceil(slices[level].stop / chunk_shape[level])

        for i in range(num_of_pieces):
            _slices = copy(slices)
            start = i * chunk_shape[level]

            if i == num_of_pieces - 1:
                stop = shape[level]
            else:
                stop = (i + 1) * chunk_shape[level]

            _slices[level] = slice(start, stop)

            q: np.ndarray = collect(
                slices=_slices,
                level=level + 1,
                shape=shape,
                chunk_shape=chunk_shape,
                get_items=get_items
            )
            if p is None:
                p = q
                continue
            p = np.concatenate((p, q), axis=level)
        return p
    else:
        return get_items(slices)


def chunk2list(chunk: Tuple[slice]) -> List[List[int]]:
    return [[s.start, s.stop, s.step] for s in chunk]


def list2chunk(_list: List[List[int]]) -> Tuple[slice]:
    return tuple([slice(*el) for el in _list])


def generate_chunks_slices(shape: Sequence[int], chunk_shape: Sequence[int]) -> Tuple[slice]:
    _ranges = (
        range(0, a, c)
        for c, a in zip(chunk_shape, shape)
    )
    p = product(*_ranges)
    for i in p:
        yield tuple(
            slice(
                i[j],
                min(shape[j], i[j]+chunk_shape[j])
            )
            for j in range(len(shape))
        )


def get_chunk_slice_by_index(shape: Sequence[int], chunk_shape: Sequence[int], number: int) -> Tuple[slice]:
    p = tuple((0, a, c) for c, a in zip(chunk_shape, shape))
    val = get_index_of_iter_product(number, p)
    return tuple(
        slice(
            val[j],
            min(shape[j], val[j]+chunk_shape[j])
        )
        for j in range(len(shape))
    )


def parse_key_to_slices(shape: Sequence[int], chunk_shape: Sequence[int], key: Tuple[slice]):
    result = []
    for i in range(len(key)):
        val = key[i]
        if isinstance(val, int):
            if val < 0:
                val = shape[i] + val
            result.append(
                slice(val, val+1)
            )
        else:
            start = val.start or 0
            stop = val.stop or shape[i]
            if start > shape[i] or stop > shape[i]:
                raise CloudArrayException(
                    f"Slice {key[i]} does not fit shape: {shape}.")
            if start >= stop:
                raise CloudArrayException(
                    f"Key invalid slice {key[i]}. Start >= stop.")
            if start < 0:
                start = shape[i] + start
            if stop < 0:
                stop = shape[i] + stop

            result.append(
                slice(
                    start,
                    stop,
                    val.step if val.step else 1
                )
            )
    return tuple(result)
