"""MCP server skeleton with no tools registered."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("AlphaBlokus MCP")


def run() -> None:
    """Run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    run()
