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
    level: int = 0
) -> np.ndarray:
    """
    Recursively collects and concatenates chunks of data based on given slices.

    Args:
        slices (Sequence[slice]): Sequence of slices defining the data to collect.
        shape (Sequence[int]): Shape of the entire data array.
        chunk_shape (Sequence[int]): Shape of each chunk.
        get_items (Callable): Function to retrieve items for a given set of slices.
        level (int): Current recursion level (dimension being processed).

    Returns:
        np.ndarray: Collected and concatenated data.
    """
    if level >= len(slices):
        return get_items(slices)

    current_slice = slices[level]
    num_chunks = ceil(current_slice.stop / chunk_shape[level])

    chunks = []
    for i in range(num_chunks):
        start = i * chunk_shape[level]
        stop = min((i + 1) * chunk_shape[level], shape[level])

        new_slices = list(slices)
        new_slices[level] = slice(start, stop)

        chunk = collect(
            slices=new_slices,
            shape=shape,
            chunk_shape=chunk_shape,
            get_items=get_items,
            level=level + 1
        )
        chunks.append(chunk)

    return np.concatenate(chunks, axis=level)


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
            slice(i[j], min(shape[j], i[j]+chunk_shape[j]))
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


def parse_key_to_slices(shape: Sequence[int], chunk_shape: Sequence[int], key: Tuple[slice]) -> slice:
    """
    Parses and normalizes a key into a tuple of slices based on the given shape.

    This function takes a key (which can be a mix of integers and slices) and
    converts it into a tuple of normalized slices. It handles negative indices
    and ensures that all slices are within the bounds of the given shape.

    Args:
        shape (Sequence[int]): The shape of the array being indexed.
        chunk_shape (Sequence[int]): The shape of each chunk in the array.
            Note: This parameter is not used in the current implementation.
        key (Tuple[slice]): The key used for indexing, can contain integers and slices.

    Returns:
        Tuple[slice]: A tuple of normalized slices corresponding to the input key.

    Raises:
        CloudArrayException: If any slice in the key is invalid or out of bounds.

    Example:
        >>> shape = (10, 10)
        >>> chunk_shape = (5, 5)
        >>> key = (slice(1, 5), 3)
        >>> parse_key_to_slices(shape, chunk_shape, key)
        (slice(1, 5, 1), slice(3, 4, 1))
    """

    def normalize_index(idx: int, dim: int) -> int:
        return idx + dim if idx < 0 else idx

    def validate_slice(s: slice, dim: int) -> slice:
        start, stop = normalize_index(s.start or 0, dim), normalize_index(s.stop or dim, dim)
        if start >= stop or start >= dim or stop > dim:
            raise CloudArrayException(f"Invalid slice {s} for dimension {dim}")
        return slice(start, stop, s.step or 1)

    return tuple(
        slice(normalize_index(k, shape[i]), normalize_index(k, shape[i]) + 1) if isinstance(k, int)
        else validate_slice(k, shape[i])
        for i, k in enumerate(key)
    )
