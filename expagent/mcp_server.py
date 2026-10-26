"""Expose the toolbox as an MCP server over stdio (JSON-RPC 2.0, one message per line).

Implements the parts of the Model Context Protocol a tools-only server needs:
initialize, ping, tools/list and tools/call. stdout carries protocol messages
only; logs go to stderr.

    python -m expagent mcp --runs-dir examples/runs
"""

from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from . import __version__
from .tools import Toolbox, ToolError

# Newest first. If the client asks for one of these we answer with it, otherwise with the newest.
PROTOCOL_VERSIONS = ["2025-06-18", "2025-03-26", "2024-11-05"]

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class RPCError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class MCPServer:
    def __init__(self, toolbox: Toolbox, name: str = "exp-agent"):
        self.toolbox = toolbox
        self.name = name

    def handle(self, msg: Any) -> dict | None:
        """Handle one decoded JSON-RPC message; return the response, or None for notifications."""
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
            return _error(msg.get("id") if isinstance(msg, dict) else None, INVALID_REQUEST, "invalid JSON-RPC request")
        is_notification = "id" not in msg
        try:
            result = self._dispatch(msg["method"], msg.get("params") or {})
        except RPCError as e:
            return None if is_notification else _error(msg["id"], e.code, e.message)
        except Exception as e:
            print(f"[mcp] internal error in {msg['method']}: {e!r}", file=sys.stderr)
            return None if is_notification else _error(msg["id"], INTERNAL_ERROR, f"internal error: {e}")
        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": msg["id"], "result": result}

    def _dispatch(self, method: str, params: dict) -> dict:
        if method == "initialize":
            requested = params.get("protocolVersion")
            version = requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
            return {"protocolVersion": version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": self.name, "version": __version__}}
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": self.toolbox.mcp_schemas()}
        if method == "tools/call":
            name = params.get("name")
            if name not in self.toolbox:
                raise RPCError(INVALID_PARAMS, f"unknown tool: {name!r}")
            try:
                result = self.toolbox.call(name, params.get("arguments") or {})
            except ToolError as e:
                # Tool-level failures are results with isError, so the model can read them and retry.
                return {"content": [{"type": "text", "text": str(e)}], "isError": True}
            text = json.dumps(result, ensure_ascii=False, indent=2)
            return {"content": [{"type": "text", "text": text}], "isError": False}
        if method.startswith("notifications/"):
            return {}  # e.g. notifications/initialized; nothing to do
        raise RPCError(METHOD_NOT_FOUND, f"method not found: {method}")

    def serve(self, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
        print(f"[mcp] {self.name} {__version__} ready, runs dir: {self.toolbox.store.root}", file=sys.stderr)
        for line in stdin:
            if not line.strip():
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as e:
                resp = _error(None, PARSE_ERROR, f"parse error: {e.msg}")
            else:
                resp = self.handle(msg)
            if resp is not None:
                stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                stdout.flush()


def _error(id_: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}
