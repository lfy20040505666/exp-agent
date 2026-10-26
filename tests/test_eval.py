import unittest

from evals.baseline import KeywordRouterLLM
from evals.run_eval import check_answer, check_tools, load_cases, run_eval, to_markdown
from expagent.agent import Agent
from helpers import toolbox


class ScoringTests(unittest.TestCase):
    def test_check_tools_alternatives(self):
        self.assertEqual(check_tools([["compare_runs", "get_final_metrics"]], ["get_final_metrics"]), (True, []))
        self.assertEqual(check_tools(["diagnose_run"], ["list_runs"]), (False, ["diagnose_run"]))
        self.assertEqual(check_tools([], []), (True, []))

    def test_check_answer(self):
        case = {"must_contain": ["35.12"], "must_contain_any": ["dB", "分贝"], "must_not_contain": ["unknown"]}
        self.assertTrue(check_answer(case, "PSNR is 35.12 DB")[0])
        ok, problems = check_answer(case, "PSNR is 35.1, run unknown")
        self.assertFalse(ok)
        self.assertEqual(len(problems), 3)

    def test_cases_file_is_well_formed(self):
        cases = load_cases()
        self.assertGreaterEqual(len(cases), 10)
        self.assertEqual(len({c["id"] for c in cases}), len(cases))
        known = set(toolbox().tools)
        for c in cases:
            for item in c["expect_tools"]:
                self.assertTrue(set(item if isinstance(item, list) else [item]) <= known, c["id"])


class BaselineEvalTests(unittest.TestCase):
    def test_baseline_runs_end_to_end(self):
        tb = toolbox()
        report = run_eval(lambda: Agent(KeywordRouterLLM(), tb), load_cases())
        s = report["summary"]
        self.assertEqual(s["cases"], len(load_cases()))
        # The one-shot keyword router cannot do multi-step questions like "which lego run is best".
        self.assertLess(s["pass_rate"], 1.0)
        self.assertGreater(s["pass_rate"], 0.5)
        self.assertIn("| case | pass |", to_markdown(report, "baseline"))


if __name__ == "__main__":
    unittest.main()
