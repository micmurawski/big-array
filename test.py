from itertools import product
import numpy as np

shape = (251, 126, 51)
shape = (256,128,128)

def count_number_of_chunks(shape: int, brick_size):
    x = [round(i/brick_size) for i in shape]
    r = 1
    for i in x:
        if i != 0:
            r *= i
    return r


data = np.array([float(i) for i in range(shape[0]*shape[1] *
                shape[2])], dtype=np.float16).reshape(*shape)

n = 64

print(count_number_of_chunks(shape, n))
def generate_chunks_slices(chunk_shape=(n, n, n), array_shape=shape):
    _ranges = (range(0, a, c) for c, a in zip(chunk_shape, array_shape))
    p = product(*_ranges)
    for i in p:
        _s = []
        for j in range(len(shape)):
            if i[j]+n > shape[j]:
                _s.append(
                    slice(i[j], shape[j])
                )
            else:
                _s.append(
                    slice(i[j], i[j]+n)
                )
        yield _s

print(len(list(generate_chunks_slices())))
for s in generate_chunks_slices():
    print(s)
