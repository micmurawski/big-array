import numpy as np
import pytest


@pytest.mark.parametrize("key", [
    (slice(10, 32), slice(10, 32), slice(10, 32)),
    (slice(10, 12), slice(10, 12), slice(10, 12)),
    (slice(None, None), slice(None, None), slice(None, None)),
])
def test_lfs_getitem(key, test_data, s3_array):
    assert test_data[key].shape == s3_array[key].shape
    assert np.array_equal(
        test_data[key],
        s3_array[key]
    )
