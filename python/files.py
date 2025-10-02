import dotenv
import boto3
import os
import tempfile
from contextlib import contextmanager
from typing import Optional, List

dotenv.load_dotenv()

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION_NAME"),
    endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
)


def is_s3(path: str) -> bool:
    """
    Check if a path is an s3 path.
    """
    return path.startswith("s3://")


def parse_s3(path: str) -> tuple[str, str, Optional[str]]:
    """
    Parse an s3 path into a bucket and key.
    """
    if not is_s3(path):
        raise ValueError(f"Path is not an s3 path: {path}")

    without_protocol = path.split("://")[1]
    split_without_protocol = without_protocol.split("/")
    bucket = split_without_protocol[0]
    key = "/".join(split_without_protocol[1:])
    if path.endswith("/"):
        filename = None
    else:
        filename = key.split("/")[-1]

    return bucket, key, filename


def localize_file(path: str) -> str:
    """
    Given a path, download it to local if needed and return the local path.
    """
    if is_s3(path):
        return _fetch_s3_file(path)
    else:
        return path


@contextmanager
def from_localized(path: str):
    """
    Usage:

    with from_localized("s3://alpha-blokus/path/to/file.txt") as filename:
        write_to_file(filename, "Hello, world!")

    When the context manager terminates, the file will be uploaded to the original
    path. If given a path that's already local, then the filename yielded will just
    be that local path.
    """
    if is_s3(path):
        temp_path = temp_directory()
        filename = path.split("/")[-1]
        file_path = os.path.join(temp_path, filename)
        yield file_path

        bucket, key, filename = parse_s3(path)
        s3.upload_file(file_path, bucket, key)
        os.remove(file_path)
    else:
        yield path


def list_files(directory: str, extension: str) -> List[str]:
    """
    Given a directory, return a list of all files with the given extension
    in that directory.
    """
    if is_s3(directory):
        return _list_s3_files(directory, extension)
    else:
        return _list_local_files(directory, extension)


def latest_file(directory: str, extension: str) -> Optional[str]:
    """
    Given a directory, return the latest file with the given extension
    in that directory.
    """
    files = list_files(directory, extension)
    if not files:
        return None
    return max(files)


def _list_s3_files(directory: str, extension: str) -> List[str]:
    """
    Given an S3 directory (e.g., 's3://bucket/path/to/dir/'), return a list of all files with the given extension
    in that directory.
    """
    if not extension.startswith("."):
        raise ValueError("`extension` must start with '.' (e.g., '.parquet').")
    if not directory.endswith("/"):
        raise ValueError("`directory` must end with '/' (e.g., 's3://bucket/path/').")

    bucket, key, filename = parse_s3(directory)

    if filename is not None:
        raise ValueError("`directory` must point to a directory (end with '/').")

    # Do not accept bucket root as a directory target.
    if key in ("", "/"):
        raise ValueError(
            "Bucket root is not supported. Provide a non-empty prefix (e.g., 's3://bucket/path/')."
        )

    prefix = key
    assert prefix.endswith("/")

    params = {
        "Bucket": bucket,
        "Prefix": prefix,
        "Delimiter": "/",  # only immediate children appear in 'Contents'
    }

    token: Optional[str] = None

    keys = []
    while True:
        if token:
            params["ContinuationToken"] = token
        resp = s3.list_objects_v2(**params)

        for obj in resp.get("Contents", []):
            key_full = obj["Key"]
            keys.append(key_full)

        token = resp.get("NextContinuationToken")
        if not token:
            break

    return [f"s3://{bucket}/{key}" for key in keys if not key.endswith("/")]


def _list_local_files(directory: str, extension: str) -> List[str]:
    """
    Given a local directory, return a list of all files with the given extension
    in that directory.
    """
    if not extension.startswith("."):
        raise ValueError("`extension` must start with '.' (e.g., '.pth').")
    return [f for f in os.listdir(directory) if f.endswith(extension)]


def _fetch_s3_file(
    path: str,
) -> str:
    """
    Download a file from s3 to a given local directory.
    """
    bucket, key, filename = parse_s3(path)
    local_path = os.path.join(temp_directory(), filename)
    s3.download_file(bucket, key, local_path)
    return local_path


TEMP_DIRECTORY = None


def temp_directory() -> str:
    """
    Get a temporary directory.
    """
    global TEMP_DIRECTORY
    if TEMP_DIRECTORY is None:
        TEMP_DIRECTORY = tempfile.TemporaryDirectory()
    return TEMP_DIRECTORY.name
