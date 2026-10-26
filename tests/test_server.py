import json
import threading
import unittest
import urllib.error
import urllib.request

from expagent.agent import Agent
from expagent.llm import ConfigError, LLMError, ScriptedLLM
from expagent.server import make_server
from helpers import answer, call, toolbox


class HTTPServerTests(unittest.TestCase):
    def setUp(self):
        self.tb = toolbox()
        self.factory = lambda: Agent(ScriptedLLM([call("list_runs", {}), answer("there are 5 runs")]), self.tb)
        self.server = make_server("127.0.0.1", 0, self.tb, lambda: self.factory(), log_requests=False)
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def request(self, method, path, body=None, raw=None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_health_runs_tools(self):
        self.assertEqual(self.request("GET", "/health"), (200, {"status": "ok"}))
        status, body = self.request("GET", "/runs")
        self.assertEqual((status, len(body["runs"])), (200, 5))
        status, body = self.request("GET", "/tools")
        self.assertEqual(len(body["tools"]), 6)

    def test_call_tool(self):
        status, body = self.request("POST", "/tools/get_final_metrics", {"run": "lego_baseline"})
        self.assertEqual(status, 200)
        self.assertEqual(body["result"]["metrics"]["psnr"], 35.12)

    def test_errors(self):
        self.assertEqual(self.request("GET", "/nope")[0], 404)
        self.assertEqual(self.request("POST", "/tools/nope", {})[0], 404)
        self.assertEqual(self.request("POST", "/tools/get_run_config", {"run": "../x"})[0], 400)
        self.assertEqual(self.request("POST", "/tools/get_run_config", raw=b"{oops")[0], 400)
        self.assertEqual(self.request("POST", "/ask", {"q": "missing field"})[0], 400)

    def test_ask_returns_answer_and_trace(self):
        status, body = self.request("POST", "/ask", {"question": "how many runs?"})
        self.assertEqual(status, 200)
        self.assertEqual(body["answer"], "there are 5 runs")
        self.assertEqual(body["steps"][0]["tool"], "list_runs")

    def test_ask_without_api_key_is_503(self):
        def no_key():
            raise ConfigError("set EXPAGENT_API_KEY")
        self.factory = no_key
        status, body = self.request("POST", "/ask", {"question": "hi"})
        self.assertEqual(status, 503)
        self.assertIn("EXPAGENT_API_KEY", body["error"])

    def test_llm_failure_is_502(self):
        self.factory = lambda: Agent(ScriptedLLM([]), self.tb)  # raises LLMError on first call
        self.assertEqual(self.request("POST", "/ask", {"question": "hi"})[0], 502)


if __name__ == "__main__":
    unittest.main()
