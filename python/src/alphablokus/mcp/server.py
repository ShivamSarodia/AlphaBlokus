"""MCP server for listing objects in the hardcoded alpha-blokus S3 bucket."""

import os
from typing import Any

import boto3
import dotenv
from mcp.server.fastmcp import FastMCP

BUCKET_NAME = "alpha-blokus"
DEFAULT_PAGE_SIZE = 200
MAX_PAGE_SIZE = 1000

dotenv.load_dotenv()

mcp = FastMCP("S3 Client")


def _create_s3_client() -> Any:
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION_NAME"),
        endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
    )


def _normalize_prefix(prefix: str | None) -> str:
    if prefix is None:
        return ""
    normalized = prefix.strip()
    if normalized.startswith("s3://"):
        expected = f"s3://{BUCKET_NAME}/"
        if not normalized.startswith(expected):
            raise ValueError(
                f"Only {expected} is supported. Got bucket path: {normalized}"
            )
        normalized = normalized.removeprefix(expected)
    return normalized.lstrip("/")


def _normalize_key(path_or_key: str, *, field_name: str) -> str:
    normalized = _normalize_prefix(path_or_key)
    if normalized in {"", "/"}:
        raise ValueError(f"`{field_name}` must be a non-empty object key/path.")
    return normalized


def _list_all_keys(prefix: str) -> list[str]:
    s3 = _create_s3_client()
    keys: list[str] = []
    continuation_token: str | None = None

    while True:
        params: dict[str, object] = {
            "Bucket": BUCKET_NAME,
            "Prefix": prefix,
            "MaxKeys": MAX_PAGE_SIZE,
        }
        if continuation_token:
            params["ContinuationToken"] = continuation_token

        response = s3.list_objects_v2(**params)
        keys.extend(
            obj["Key"]
            for obj in response.get("Contents", [])
            if not obj["Key"].endswith("/")
        )

        continuation_token = response.get("NextContinuationToken")
        if not continuation_token:
            break

    return keys


@mcp.tool(name="files.list_s3_files")
def list_s3_files(
    prefix: str = "",
    page_size: int = DEFAULT_PAGE_SIZE,
    page_token: int | None = None,
    direction: str = "asc",
) -> dict[str, object]:
    """
    List object files in s3://alpha-blokus/ with optional prefix filtering.

    Returns:
      - items: list of s3://alpha-blokus/... object paths
      - next_token: continuation token for the next page, or None
      - bucket: the hardcoded bucket name
      - prefix: normalized prefix used for this query
    """
    if page_size <= 0:
        raise ValueError("`page_size` must be a positive integer.")
    if page_size > MAX_PAGE_SIZE:
        raise ValueError(f"`page_size` must be <= {MAX_PAGE_SIZE}.")
    if page_token is not None and page_token < 0:
        raise ValueError("`page_token` must be a non-negative integer.")
    if direction not in {"asc", "desc"}:
        raise ValueError("`direction` must be 'asc' or 'desc'.")

    normalized_prefix = _normalize_prefix(prefix)

    keys = sorted(_list_all_keys(normalized_prefix), reverse=direction == "desc")
    total = len(keys)
    start = page_token or 0
    end = min(start + page_size, total)
    paged_keys = keys[start:end]
    next_token = end if end < total else None

    return {
        "items": [f"s3://{BUCKET_NAME}/{key}" for key in paged_keys],
        "next_token": next_token,
        "bucket": BUCKET_NAME,
        "prefix": normalized_prefix,
        "direction": direction,
        "total": total,
    }


@mcp.tool(name="files.copy_s3_file")
def copy_s3_file(source: str, destination: str) -> dict[str, object]:
    """
    Copy one object inside s3://alpha-blokus/.

    Args:
      - source: full s3 path (s3://alpha-blokus/...) or bucket-relative key
      - destination: full s3 path (s3://alpha-blokus/...) or bucket-relative key
    """
    source_key = _normalize_key(source, field_name="source")
    destination_key = _normalize_key(destination, field_name="destination")

    s3 = _create_s3_client()
    s3.copy_object(
        Bucket=BUCKET_NAME,
        CopySource={"Bucket": BUCKET_NAME, "Key": source_key},
        Key=destination_key,
    )

    return {
        "bucket": BUCKET_NAME,
        "source": f"s3://{BUCKET_NAME}/{source_key}",
        "destination": f"s3://{BUCKET_NAME}/{destination_key}",
        "copied": True,
    }


@mcp.tool(name="files.move_s3_file")
def move_s3_file(source: str, destination: str) -> dict[str, object]:
    """
    Move one object inside s3://alpha-blokus/ (copy, then delete source).

    Args:
      - source: full s3 path (s3://alpha-blokus/...) or bucket-relative key
      - destination: full s3 path (s3://alpha-blokus/...) or bucket-relative key
    """
    source_key = _normalize_key(source, field_name="source")
    destination_key = _normalize_key(destination, field_name="destination")

    s3 = _create_s3_client()
    s3.copy_object(
        Bucket=BUCKET_NAME,
        CopySource={"Bucket": BUCKET_NAME, "Key": source_key},
        Key=destination_key,
    )
    # Safety: never issue delete_object with an empty key.
    if source_key in {"", "/"}:
        raise ValueError("Refusing to delete blank source key.")
    s3.delete_object(Bucket=BUCKET_NAME, Key=source_key)

    return {
        "bucket": BUCKET_NAME,
        "source": f"s3://{BUCKET_NAME}/{source_key}",
        "destination": f"s3://{BUCKET_NAME}/{destination_key}",
        "moved": True,
    }


def run() -> None:
    """Run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    run()
