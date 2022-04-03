import tempfile

import numpy as np
import pytest

from cloud_array import CloudArray


@pytest.fixture(scope="module")
def test_data():
    shape = (251, 126, 51)
    return np.random.rand(*shape)


@pytest.fixture(scope="module")
def lfs_array(test_data):
    with tempfile.TemporaryDirectory() as file:
        chunk_shape = (16, 16, 16)
        array = CloudArray(
            shape=chunk_shape,
            chunk_shape=chunk_shape,
            url=file,
            array=test_data
        )
        array.save()
        yield array
