from typing import List

from alphablokus.files import list_files, parse_num_games_from_filename, parse_s3, s3


S3_PREFIX = "s3://alpha-blokus/full_v2/games/"
SAMPLE_THRESHOLD = 18_001_395
ARCHIVE_PREFIX = "s3://alpha-blokus/full_v2/games_archived_from_18001395/"


def files_after_sample_threshold(
    files: List[str],
    *,
    threshold: int,
) -> List[str]:
    cumulative = 0
    for index, path in enumerate(files):
        cumulative += parse_num_games_from_filename(path)
        if cumulative > threshold:
            return files[index + 1 :]
    return []


def _ensure_s3_prefix(prefix: str) -> None:
    if not prefix.endswith("/"):
        raise ValueError("S3 prefix must end with '/'")


def _move_s3_object(source_path: str, dest_path: str) -> None:
    source_bucket, source_key, _ = parse_s3(source_path)
    dest_bucket, dest_key, _ = parse_s3(dest_path)
    s3.copy(
        {"Bucket": source_bucket, "Key": source_key},
        dest_bucket,
        dest_key,
    )
    s3.delete_object(Bucket=source_bucket, Key=source_key)


def main() -> None:
    _ensure_s3_prefix(S3_PREFIX)
    _ensure_s3_prefix(ARCHIVE_PREFIX)

    files = sorted(list_files(S3_PREFIX))
    remaining = files_after_sample_threshold(files, threshold=SAMPLE_THRESHOLD)
    print("Number of files:", len(remaining))
    if not remaining:
        return

    confirmation = input(
        "Type OK to continue and move these files to the archive prefix: "
    ).strip()
    if confirmation != "OK":
        print("Aborted.")
        return

    _, source_prefix_key, _ = parse_s3(S3_PREFIX)
    for path in remaining:
        _, source_key, _ = parse_s3(path)
        if not source_key.startswith(source_prefix_key):
            raise ValueError(f"Unexpected source key outside prefix: {source_key}")
        relative_key = source_key[len(source_prefix_key) :]
        dest_path = f"{ARCHIVE_PREFIX}{relative_key}"
        _move_s3_object(path, dest_path)


if __name__ == "__main__":
    main()
