import sys
import boto3
import os


def download_file_from_s3(
    s3_object_key,
    local_file_name,
    aws_access_key_id,
    aws_secret_access_key,
    region_name,
):
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
    )
    s3 = session.client("s3")
    s3.download_file("awss3datasciencepractice", s3_object_key, local_file_name)


if __name__ == "__main__":
    aws_access_key_id = sys.argv[1]
    aws_secret_access_key = sys.argv[2]
    region_name = sys.argv[3]
    s3_object_key = sys.argv[4]
    local_file_name = sys.argv[5]

    download_file_from_s3(
        s3_object_key,
        local_file_name,
        aws_access_key_id,
        aws_secret_access_key,
        region_name,
    )
