# Cloud Array

`cloud-array` is an open-source Python library for storing and streaming large Numpy Arrays on local file systems and major cloud proviers CDNs.
 
 ```python
import numpy as np
from cloud_array import CloudArray

shape = (10000, 100, 100)
chunk_shape = (10, 10, 10)

f = np.memmap(
    'memmapped.dat',
    dtype=np.float32,
    mode='w+',
    shape=shape
)

array = CloudArray(
    shape=shape,
    chunk_shape=chunk_shape,
    url="s3://example_bucket/dataset0"
)
array.save(f)
print(array[:100,:100,:100])

 ```