"""AlphaBlokus MCP server."""

from mcp.server.fastmcp import FastMCP

from alphablokus.files import cached_list_files, is_s3

mcp = FastMCP("AlphaBlokus")


@mcp.tool(name="files.list_s3_files")
def list_s3_files(
    directory: str,
    page_size: int = 200,
    page_token: int | None = None,
    direction: str = "asc",
) -> dict[str, object]:
    """
    List files in an S3 directory (e.g., 's3://bucket/path/'), sorted alphabetically.
    Set direction to "asc" or "desc".

    Returns a paginated response with:
      - items: list of file paths
      - next_token: int offset for the next page, or None
      - total: total number of files
    """
    if not is_s3(directory):
        raise ValueError(f"Expected an s3:// path, got: {directory}")
    if not directory.endswith("/"):
        raise ValueError("`directory` must end with '/' (e.g., 's3://bucket/path/').")
    if page_size <= 0:
        raise ValueError("`page_size` must be a positive integer.")
    if page_token is not None and page_token < 0:
        raise ValueError("`page_token` must be a non-negative integer.")
    if direction not in {"asc", "desc"}:
        raise ValueError("`direction` must be 'asc' or 'desc'.")

    all_files = sorted(cached_list_files(directory, None), reverse=direction == "desc")

    total = len(all_files)
    start = page_token or 0
    end = min(start + page_size, total)
    items = all_files[start:end]
    next_token = end if end < total else None
    return {"items": items, "next_token": next_token, "total": total}


def run() -> None:
    """Run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    run()
