import os

import boto3
import pytest
from moto import mock_s3

from cloud_array import CloudArray


@pytest.fixture(scope="session")
def s3_bucket():
    with mock_s3():
        bucket_name = "testlocalstackbucket"
        client = boto3.client(
            "s3",
        )
        client.create_bucket(Bucket=bucket_name)
        yield f"s3://{bucket_name}"


@pytest.fixture(scope="module")
def s3_array(test_data, s3_bucket):
    chunk_shape = (16, 16, 16)
    array = CloudArray(
        chunk_shape=chunk_shape,
        url=os.path.join(s3_bucket, "dataset0"),
        array=test_data
    )
    array.save()
    yield array
