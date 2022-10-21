import numpy as np
import pytest


@pytest.mark.parametrize("key", [
    (slice(10, 32), slice(30, 32), slice(30, 32)),
    #(slice(10, 12), slice(10, 12), slice(10, 12)),
    #(slice(None, None), slice(None, None), slice(None, None)),
])
def test_lfs_getitem(key, test_data, lfs_array):
    assert test_data[key].shape == lfs_array[key].shape
    assert np.array_equal(
        test_data[key],
        lfs_array[key]
    )
