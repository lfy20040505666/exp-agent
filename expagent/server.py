"""A small JSON HTTP API around the toolbox and the agent (standard library only).

    GET  /health              -> {"status": "ok"}
    GET  /tools               -> tool schemas
    GET  /runs                -> same as the list_runs tool
    POST /tools/<name>        -> call one tool; body is its arguments object
    POST /ask                 -> {"question": "..."}; runs the agent, returns answer + trace
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

from .agent import Agent
from .llm import ConfigError, LLMError
from .tools import Toolbox, ToolError

MAX_BODY_BYTES = 1_000_000


class HTTPError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def make_server(host: str, port: int, toolbox: Toolbox,
                agent_factory: Callable[[], Agent], log_requests: bool = True) -> ThreadingHTTPServer:
    """Build (but don't start) the server. agent_factory makes a fresh Agent per /ask request."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "ExpAgent"

        def do_GET(self):
            self._route(self._get)

        def do_POST(self):
            self._route(self._post)

        def _get(self):
            if self.path == "/health":
                return 200, {"status": "ok"}
            if self.path == "/tools":
                return 200, {"tools": toolbox.mcp_schemas()}
            if self.path == "/runs":
                return 200, {"runs": toolbox.call("list_runs", {})}
            raise HTTPError(404, f"no route for GET {self.path}")

        def _post(self):
            body = self._read_json()
            if self.path.startswith("/tools/"):
                name = self.path[len("/tools/"):]
                if name not in toolbox:
                    raise HTTPError(404, f"unknown tool {name!r}")
                return 200, {"tool": name, "result": toolbox.call(name, body)}
            if self.path == "/ask":
                question = body.get("question") if isinstance(body, dict) else None
                if not isinstance(question, str) or not question.strip():
                    raise HTTPError(400, "body must be {\"question\": \"...\"}")
                try:
                    agent = agent_factory()
                except ConfigError as e:
                    raise HTTPError(503, f"agent not configured: {e}")
                try:
                    return 200, agent.run(question).to_dict()
                except LLMError as e:
                    raise HTTPError(502, f"LLM call failed: {e}")
            raise HTTPError(404, f"no route for POST {self.path}")

        def _read_json(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                raise HTTPError(413, "request body too large")
            raw = self.rfile.read(length) if length else b"{}"
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise HTTPError(400, "request body is not valid JSON")

        def _route(self, handler):
            try:
                status, payload = handler()
            except HTTPError as e:
                status, payload = e.status, {"error": e.message}
            except ToolError as e:
                status, payload = 400, {"error": str(e)}
            except Exception as e:
                print(f"[http] internal error: {e!r}", file=sys.stderr)
                status, payload = 500, {"error": "internal server error"}
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt, *args):
            if log_requests:
                print(f"[http] {self.address_string()} {fmt % args}", file=sys.stderr)

    return ThreadingHTTPServer((host, port), Handler)
