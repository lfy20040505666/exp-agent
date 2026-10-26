import json
import unittest

from expagent.agent import STEP_LIMIT_PROMPT, Agent
from expagent.llm import LLMResponse, ScriptedLLM, ToolCall
from helpers import answer, call, toolbox


class AgentLoopTests(unittest.TestCase):
    def run_agent(self, responses, question="q", **kw):
        llm = ScriptedLLM(responses)
        result = Agent(llm, toolbox(), **kw).run(question)
        return result, llm

    def test_direct_answer_without_tools(self):
        result, llm = self.run_agent([answer("hello")])
        self.assertEqual(result.answer, "hello")
        self.assertEqual(result.steps, [])
        self.assertEqual(result.stopped_reason, "answered")
        self.assertEqual(llm.requests[0]["messages"][0]["role"], "system")

    def test_tool_result_is_fed_back(self):
        result, llm = self.run_agent([call("get_final_metrics", {"run": "lego_baseline"}), answer("35.12 dB")])
        self.assertEqual(result.tools_called, ["get_final_metrics"])
        self.assertTrue(result.steps[0].ok)
        second = llm.requests[1]["messages"]
        self.assertEqual(second[-2]["role"], "assistant")
        self.assertEqual(second[-2]["tool_calls"][0]["function"]["name"], "get_final_metrics")
        self.assertEqual(second[-1]["role"], "tool")
        self.assertEqual(second[-1]["tool_call_id"], "c1")
        self.assertEqual(json.loads(second[-1]["content"])["result"]["metrics"]["psnr"], 35.12)
        self.assertEqual(result.llm_calls, 2)
        self.assertEqual(result.usage, {"prompt_tokens": 10, "completion_tokens": 5})

    def test_parallel_tool_calls_in_one_turn(self):
        both = LLMResponse(None, [ToolCall("a", "get_run_config", '{"run": "room_short"}'),
                                  ToolCall("b", "get_run_config", '{"run": "room_baseline"}')])
        result, llm = self.run_agent([both, answer("done")])
        self.assertEqual(len(result.steps), 2)
        tool_msgs = [m for m in llm.requests[1]["messages"] if m["role"] == "tool"]
        self.assertEqual([m["tool_call_id"] for m in tool_msgs], ["a", "b"])

    def test_invalid_json_arguments_are_reported_not_raised(self):
        result, llm = self.run_agent([call("get_run_config", '{"run": lego'), answer("sorry")])
        self.assertFalse(result.steps[0].ok)
        self.assertIn("not valid JSON", result.steps[0].error)
        obs = json.loads(llm.requests[1]["messages"][-1]["content"])
        self.assertEqual(obs["ok"], False)

    def test_model_recovers_from_tool_error(self):
        result, _ = self.run_agent([call("get_final_metrics", {"run": "lego"}),
                                    call("get_final_metrics", {"run": "lego_baseline"}, "c2"),
                                    answer("ok")])
        self.assertEqual([s.ok for s in result.steps], [False, True])
        self.assertIn("Available runs", result.steps[0].error)

    def test_unknown_tool(self):
        result, _ = self.run_agent([call("delete_everything", {}), answer("no")])
        self.assertIn("unknown tool", result.steps[0].error)

    def test_duplicate_call_is_blocked(self):
        same = {"run": "lego_baseline"}
        result, _ = self.run_agent([call("get_run_config", same), call("get_run_config", same, "c2"), answer("x")])
        self.assertTrue(result.steps[0].ok)
        self.assertIn("duplicate call", result.steps[1].error)

    def test_step_limit_forces_final_answer_without_tools(self):
        responses = [call("get_run_config", {"run": r}, f"c{i}")
                     for i, r in enumerate(["lego_baseline", "room_short", "room_baseline"])]
        result, llm = self.run_agent(responses + [answer("partial")], max_steps=3)
        self.assertEqual(result.stopped_reason, "max_steps")
        self.assertEqual(result.answer, "partial")
        last = llm.requests[-1]
        self.assertEqual(last["tools"], [])
        self.assertEqual(last["messages"][-1]["content"], STEP_LIMIT_PROMPT)

    def test_internal_tool_bug_does_not_crash_agent(self):
        tb = toolbox()
        tb.tools["list_runs"].func = lambda **kw: 1 / 0
        result = Agent(ScriptedLLM([call("list_runs", {}), answer("x")]), tb).run("q")
        self.assertIn("ZeroDivisionError", result.steps[0].error)

    def test_on_step_callback_and_to_dict(self):
        seen = []
        llm = ScriptedLLM([call("list_runs", {}), answer("five runs")])
        result = Agent(llm, toolbox(), on_step=seen.append).run("q")
        self.assertEqual([s.tool for s in seen], ["list_runs"])
        json.dumps(result.to_dict(), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
