from typing import List, Tuple

import numpy as np


def chunk2list(chunk: Tuple[slice]) -> List[List[int]]:
    return [[s.start, s.stop, s.step] for s in chunk]


def list2chunk(_list: List[List[int]]) -> Tuple[slice]:
    return tuple([slice(*l) for l in _list])


def is_in(s, key):
    conditions = []
    for i, j in zip(s, key):
        conditions.append(
            (i.start < j.stop and i.start >= j.start) or (i.stop < j.stop and i.stop >= j.start) or
            (j.start < i.stop and j.start >= i.start) or (
                j.stop < i.stop and j.stop >= i.start)
        )
    return all(conditions)


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
            ss = tuple(x[0] if x[0] == x[1] else slice(x[0].start, x[1].stop)
                       for x in zip(sorting[-1][2], chunks[i+1][1]))
            sorting[-1][2] = ss
        else:
            sorting[-1][0].append(
                chunks[i][0]
            )
            ss = tuple(x[0] if x[0] == x[1] else slice(x[0].start, x[1].stop)
                       for x in zip(sorting[-1][2], chunks[i][1]))
            sorting[-1][2] = ss
            val = None
            val2 = 0

        if i == len(chunks)-2:
            sorting[-1][0].append(
                chunks[i+1][0]
            )
            ss = tuple(x[0] if x[0] == x[1] else slice(x[0].start, x[1].stop)
                       for x in zip(sorting[-1][2], chunks[i+1][1]))
            sorting[-1][2] = ss
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
            i.start,
            k-abs(i.stop-j.stop),
            i.step
        ) for i, j, k in zip(a, b, shape)
    )
