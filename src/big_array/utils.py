from typing import List, Tuple


def chunk2list(chunk: Tuple[slice]) -> List[List[int]]:
    return [[s.start, s.stop, s.step] for s in chunk]


def list2chunk(_list: List[List[int]]) -> Tuple[slice]:
    return tuple([slice(*l) for l in _list])


def is_in(a, b):
    conditions = []
    for i, j in zip(a, b):
        conditions.append(
            i.start >= j.start and i.start < j.stop
        )
    return all(conditions)
