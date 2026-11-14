"""Local protocol fixture. Writes are confined to the pytest-owned directory."""
import json
import os
import sys

import anyio
from mcp.server.fastmcp import FastMCP

port = int(sys.argv[1]) if len(sys.argv) > 1 else 0
mcp = FastMCP("local-test-only", host="127.0.0.1", port=port or 8000)


@mcp.tool()
def echo(text: str) -> dict:
    return {"text": text, "credential": os.environ.get("MCP_FIXTURE_TOKEN", ""),
            "unexpected_environment": bool(os.environ.get("UNRELATED_SECRET"))}


@mcp.tool()
async def save_note(resource: str, idempotency_key: str, text: str, delay: bool = False) -> dict:
    from pathlib import Path
    root = Path(os.environ["MCP_FIXTURE_STORE"]).resolve()
    target = (root / resource).resolve()
    if target.parent != root or not target.name.endswith(".json"):
        raise ValueError("fixture target must stay inside its temporary directory")
    record = {"resource": resource, "idempotency_key": idempotency_key, "text": text}
    if not target.exists():
        target.write_text(json.dumps(record), encoding="utf-8")
    elif json.loads(target.read_text(encoding="utf-8")) != record:
        raise ValueError("idempotency conflict")
    if delay:
        await anyio.sleep(5)
    return {"receipt": resource, "idempotency_key": idempotency_key}


@mcp.tool()
def unlisted() -> dict:
    raise AssertionError("Unlisted tool must never execute")


if __name__ == "__main__":
    mcp.run(transport="streamable-http" if port else "stdio")
