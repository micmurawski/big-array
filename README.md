# Cloud Array

`cloud-array` is an open-source Python library for storing and streaming large Numpy Arrays on local file systems and major cloud providers CDNs. It automatically chunks a large array of data into arbitrary chunks sizes and uploads them into the targeted direcotry.
 
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
    chunk_shape=chunk_shape,
    array=f,
    url="s3://example_bucket/dataset0"
)
array.save()
print(array[:100,:100,:100])

 ```
 ## Links
* https://pypi.org/project/cloud-array/