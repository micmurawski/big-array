import operator
from functools import reduce
from math import ceil
from typing import List, Sequence, Tuple

import numpy as np


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


def get_chunk_index(normalized_chunk, normalized_shape):
    """
    This function computes chunk index based on normalized shape and chunk
    """
    coef = normalized_shape[1:] + [1]
    for i in range(len(coef)):
        coef[i] = reduce(operator.mul, coef[i:])

    return sum(map(lambda x: x[0] * x[1], zip(normalized_chunk, coef)))


def chunk2list(chunk: Tuple[slice]) -> List[List[int]]:
    return [[s.start, s.stop, s.step] for s in chunk]


def list2chunk(_list: List[List[int]]) -> Tuple[slice]:
    return tuple([slice(*el) for el in _list])


def varing_dim(a: List[int], b: List[int]):
    for i, x in enumerate(zip(a, b)):
        if x[0] != x[1]:
            return i


def sort_chunks(chunks) -> List[List]:
    sorting = []
    val = None
    val2 = 1
    for i in range(len(chunks)-1):
        r = varing_dim(
            chunks[i][1],
            chunks[i+1][1]
        )
        if val is None:
            val = r
            sorting.append(
                [
                    [
                        chunks[i][0]
                    ],
                    r,
                    chunks[i][1]
                ]
            )
        elif val == r:
            val2 += 1
            sorting[-1][0].append(
                chunks[i][0]
            )
            sorting[-1][2] = tuple(x[0] if x[0] == x[1] else slice(x[0].start, x[1].stop)
                                   for x in zip(sorting[-1][2], chunks[i+1][1]))
        else:
            sorting[-1][0].append(
                chunks[i][0]
            )
            sorting[-1][2] = tuple(x[0] if x[0] == x[1] else slice(x[0].start, x[1].stop)
                                   for x in zip(sorting[-1][2], chunks[i][1]))
            val = None
            val2 = 0

        if i == len(chunks)-2:
            sorting[-1][0].append(
                chunks[i+1][0]
            )
            sorting[-1][2] = tuple(x[0] if x[0] == x[1] else slice(x[0].start, x[1].stop)
                                   for x in zip(sorting[-1][2], chunks[i+1][1]))
    return sorting


def merge_datasets(datasets) -> List[Tuple[np.ndarray, slice]]:
    dim1 = None
    result = []
    data = None
    ss = None
    for i in range(len(datasets)-1):
        dim2 = varing_dim(
            datasets[i][1],
            datasets[i+1][1]
        )
        if dim1 is None:
            dim1 = dim2
            data = np.concatenate(
                (datasets[i][0], datasets[i+1][0]),
                axis=dim2
            )
            ss = tuple(x[0] if x[0] == x[1] else slice(x[0].start, x[1].stop)
                       for x in zip(datasets[i][1], datasets[i+1][1]))
            result.append([data, ss])
        elif dim1 == dim2:
            result[-1][0] = np.concatenate(
                (result[-1][0], datasets[i+1][0]),
                axis=dim2
            )
            ss = tuple(x[0] if x[0] == x[1] else slice(x[0].start, x[1].stop)
                       for x in zip(ss, datasets[i+1][1]))
            result[-1][1] = ss
        else:
            dim1 = None
    return result


def compute_key(a, b, shape=(0, 0, 0)) -> Tuple[slice]:
    return tuple(
        slice(
            min(i.start - j.start, i.start),
            k-abs(i.stop-j.stop),
            i.step
        ) for i, j, k in zip(a, b, shape)
    )
