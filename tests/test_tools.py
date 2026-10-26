import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from expagent.tools import RunStore, Toolbox, ToolError, clean
from helpers import RUNS_DIR, toolbox


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.tb = toolbox()

    def test_list_runs_and_status(self):
        runs = {r["run"]: r for r in self.tb.call("list_runs", {})}
        self.assertEqual(len(runs), 5)
        self.assertEqual(runs["lego_high_lr"]["status"], "no_eval")
        self.assertEqual(runs["lego_baseline"]["status"], "finished")
        self.assertEqual({r["run"] for r in self.tb.call("list_runs", {"scene": "room"})},
                         {"room_baseline", "room_short"})

    def test_final_metrics(self):
        out = self.tb.call("get_final_metrics", {"run": "lego_baseline"})
        self.assertEqual(out["metrics"]["psnr"], 35.12)
        missing = self.tb.call("get_final_metrics", {"run": "lego_high_lr"})
        self.assertIsNone(missing["metrics"])
        self.assertEqual(missing["status"], "no_eval")

    def test_compare_lpips_lower_is_better_and_skips_crashed(self):
        out = self.tb.call("compare_runs", {"runs": ["lego_baseline", "lego_low_thresh", "lego_high_lr"],
                                            "metric": "lpips"})
        self.assertFalse(out["higher_is_better"])
        self.assertEqual(out["best"], "lego_low_thresh")
        self.assertEqual(out["skipped_no_eval"], ["lego_high_lr"])

    def test_compare_psnr_higher_is_better(self):
        out = self.tb.call("compare_runs", {"runs": ["room_short", "room_baseline"], "metric": "psnr"})
        self.assertEqual([r["run"] for r in out["ranking"]], ["room_baseline", "room_short"])

    def test_diagnose_finds_divergence(self):
        out = self.tb.call("diagnose_run", {"run": "lego_high_lr"})
        self.assertFalse(out["healthy"])
        text = " ".join(out["issues"])
        self.assertIn("nan", text)
        self.assertIn("stopped early", text)
        self.assertIn("spiked", text)

    def test_diagnose_healthy_run(self):
        out = self.tb.call("diagnose_run", {"run": "lego_baseline"})
        self.assertTrue(out["healthy"], out["issues"])

    def test_curve_downsamples_and_keeps_last_point(self):
        out = self.tb.call("get_metric_curve", {"run": "lego_low_thresh", "metric": "num_gaussians", "points": 5})
        self.assertEqual(len(out["curve"]), 5)
        self.assertEqual(out["curve"][-1]["step"], 30000)
        self.assertEqual(out["last"]["num_gaussians"], 424972)

    def test_nan_is_serialisable_as_strict_json(self):
        out = self.tb.call("get_metric_curve", {"run": "lego_high_lr", "metric": "loss", "points": 20})
        json.dumps(out, allow_nan=False)  # must not raise
        self.assertEqual(out["last"]["loss"], "nan")
        self.assertEqual(clean([math.inf, {"x": -math.inf}]), ["inf", {"x": "-inf"}])

    def test_path_traversal_rejected(self):
        for bad in ["../examples", "/etc", "lego_baseline/../../", ""]:
            with self.assertRaises(ToolError):
                self.tb.call("get_run_config", {"run": bad})

    def test_argument_validation(self):
        cases = [
            ("get_run_config", {}, "missing required"),
            ("get_run_config", {"run": "lego_baseline", "foo": 1}, "unknown argument"),
            ("get_run_config", {"run": 3}, "must be of type string"),
            ("compare_runs", {"runs": ["lego_baseline"], "metric": "fid"}, "must be one of"),
            ("get_metric_curve", {"run": "lego_baseline", "points": True}, "must be of type integer"),
            ("get_metric_curve", {"run": "lego_baseline", "metric": "fid"}, "not logged"),
            ("nope", {}, "unknown tool"),
        ]
        for name, args, msg in cases:
            with self.subTest(name=name, args=args):
                with self.assertRaisesRegex(ToolError, msg):
                    self.tb.call(name, args)

    def test_schemas_are_consistent(self):
        for schema in self.tb.openai_schemas():
            params = schema["function"]["parameters"]
            self.assertEqual(params["type"], "object")
            self.assertTrue(set(params["required"]) <= set(params["properties"]))


class BrokenRunTests(unittest.TestCase):
    """A run with corrupt log lines and no metrics file at all."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        shutil.copytree(RUNS_DIR / "lego_baseline", self.tmp / "corrupt")
        with open(self.tmp / "corrupt" / "metrics.jsonl", "a") as f:
            f.write('{"step": 30500, "loss": \n')
        (self.tmp / "empty").mkdir()
        (self.tmp / "empty" / "config.json").write_text('{"scene": "x", "iterations": 100}')
        (self.tmp / "not_a_run").mkdir()  # no config.json, must be ignored
        self.tb = Toolbox(RunStore(self.tmp))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_only_folders_with_config_are_runs(self):
        self.assertEqual([r["run"] for r in self.tb.call("list_runs", {})], ["corrupt", "empty"])

    def test_corrupt_lines_are_reported(self):
        out = self.tb.call("diagnose_run", {"run": "corrupt"})
        self.assertIn("could not be parsed", " ".join(out["issues"]))

    def test_empty_run(self):
        out = self.tb.call("diagnose_run", {"run": "empty"})
        self.assertIn("metrics.jsonl is missing or empty", out["issues"])
        with self.assertRaisesRegex(ToolError, "no logged metrics"):
            self.tb.call("get_metric_curve", {"run": "empty"})

    def test_missing_runs_dir(self):
        with self.assertRaises(FileNotFoundError):
            RunStore(self.tmp / "does_not_exist")


if __name__ == "__main__":
    unittest.main()
