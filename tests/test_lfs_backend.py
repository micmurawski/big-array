import tempfile

import numpy as np
import pytest

from cloud_array import CloudArray


@pytest.mark.parametrize("key", [
    (slice(10, 32), slice(10, 32), slice(10, 32)),
    (slice(10, 12), slice(10, 12), slice(10, 12))
])
def test_lfs_backend(key):
    with tempfile.TemporaryDirectory() as file:
        shape = (251, 126, 51)
        chunk_shape = (16, 16, 16)
        data = np.random.rand(*shape)
        array = CloudArray(shape=shape, chunk_shape=chunk_shape,
                           url=file, array=data)
        array.save()
        assert data[key].shape == array[key].shape
        assert np.array_equal(
            data[key],
            array[key]
        )
