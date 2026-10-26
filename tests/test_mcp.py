import json
import subprocess
import sys
import unittest
from pathlib import Path

from expagent.mcp_server import METHOD_NOT_FOUND, INVALID_PARAMS, INVALID_REQUEST, MCPServer
from helpers import RUNS_DIR, toolbox

ROOT = Path(__file__).resolve().parent.parent


def req(method, params=None, id_=1):
    msg = {"jsonrpc": "2.0", "id": id_, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


class MCPHandleTests(unittest.TestCase):
    def setUp(self):
        self.server = MCPServer(toolbox())

    def test_initialize_negotiates_version(self):
        r = self.server.handle(req("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                                                  "clientInfo": {"name": "t", "version": "0"}}))
        self.assertEqual(r["result"]["protocolVersion"], "2024-11-05")
        self.assertIn("tools", r["result"]["capabilities"])
        r = self.server.handle(req("initialize", {"protocolVersion": "1999-01-01"}))
        self.assertEqual(r["result"]["protocolVersion"], "2025-06-18")

    def test_notification_gets_no_response(self):
        self.assertIsNone(self.server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))

    def test_tools_list(self):
        tools = self.server.handle(req("tools/list"))["result"]["tools"]
        self.assertIn("diagnose_run", [t["name"] for t in tools])
        self.assertTrue(all("inputSchema" in t for t in tools))

    def test_tools_call_ok(self):
        r = self.server.handle(req("tools/call", {"name": "get_final_metrics", "arguments": {"run": "room_short"}}))
        self.assertFalse(r["result"]["isError"])
        self.assertEqual(json.loads(r["result"]["content"][0]["text"])["metrics"]["psnr"], 29.81)

    def test_tool_error_is_a_result_not_a_protocol_error(self):
        r = self.server.handle(req("tools/call", {"name": "get_final_metrics", "arguments": {"run": "x"}}))
        self.assertTrue(r["result"]["isError"])
        self.assertIn("unknown run", r["result"]["content"][0]["text"])

    def test_unknown_tool_and_method(self):
        r = self.server.handle(req("tools/call", {"name": "nope", "arguments": {}}))
        self.assertEqual(r["error"]["code"], INVALID_PARAMS)
        r = self.server.handle(req("resources/list"))
        self.assertEqual(r["error"]["code"], METHOD_NOT_FOUND)

    def test_invalid_request(self):
        self.assertEqual(self.server.handle({"id": 3, "method": "ping"})["error"]["code"], INVALID_REQUEST)
        self.assertEqual(self.server.handle([1, 2])["error"]["code"], INVALID_REQUEST)


class MCPStdioTests(unittest.TestCase):
    """Run the real server as a subprocess, the way an MCP client (e.g. Claude Code) starts it."""

    def test_stdio_session(self):
        lines = [
            req("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                               "clientInfo": {"name": "test", "version": "0"}}, 1),
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            req("tools/list", id_=2),
            req("tools/call", {"name": "diagnose_run", "arguments": {"run": "lego_high_lr"}}, 3),
        ]
        stdin = "\n".join(json.dumps(m) for m in lines) + "\nnot json\n"
        proc = subprocess.run([sys.executable, "-m", "expagent", "--runs-dir", str(RUNS_DIR), "mcp"],
                              input=stdin, capture_output=True, text=True, cwd=ROOT, timeout=30)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = [json.loads(line) for line in proc.stdout.splitlines()]
        self.assertEqual([m["id"] for m in out], [1, 2, 3, None])  # notification has no reply
        self.assertEqual(out[0]["result"]["serverInfo"]["name"], "exp-agent")
        self.assertEqual(len(out[1]["result"]["tools"]), 6)
        self.assertIn("diverged", out[2]["result"]["content"][0]["text"])
        self.assertEqual(out[3]["error"]["code"], -32700)
        self.assertIn("ready", proc.stderr)  # logs go to stderr, never stdout


if __name__ == "__main__":
    unittest.main()
