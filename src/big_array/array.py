from array import array
from itertools import product


class BigArray(np.array):
    def __init__(self, chunk_shape, url=None, metadata_url=None, *arg, **kwargs):
        self.chunk_shape = chunk_shape
        super().__init__(*arg, **kwargs)

    def generate_chunks_slices(self, chunk_shape=self.chunk_shape, array_shape=self.shape):
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

    def send_to_cloud(url):
        pass
