"""Tools the agent can call. They all read training runs from one runs directory.

A run is a folder containing:
    config.json    training config (must exist, it marks the folder as a run)
    metrics.jsonl  one JSON object per logged step
    eval.json      final test metrics; missing if the run crashed or is still going
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Final-eval metrics and whether a higher value is better.
EVAL_METRICS = {"psnr": True, "ssim": True, "lpips": False}

# Thresholds used by diagnose_run.
LOSS_SPIKE_RATIO = 3.0
MAX_REASONABLE_GAUSSIANS = 3_000_000


class ToolError(Exception):
    """An error the model should see and can recover from (bad run name, bad argument...)."""


def clean(value: Any) -> Any:
    """Make values safe for strict JSON: NaN/inf become strings instead of invalid tokens."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)  # 'nan', 'inf', '-inf'
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


class RunStore:
    """Read-only access to the runs directory."""

    def __init__(self, runs_dir: str | Path):
        self.root = Path(runs_dir).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"runs directory not found: {self.root}")

    def run_names(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if (p / "config.json").is_file())

    def run_dir(self, name: Any) -> Path:
        # Only names that are listed are accepted, so '../secret' or '/etc' can never be read.
        names = self.run_names()
        if not isinstance(name, str) or name not in names:
            raise ToolError(f"unknown run {name!r}. Available runs: {', '.join(names)}")
        return self.root / name

    def config(self, name: str) -> dict:
        return json.loads((self.run_dir(name) / "config.json").read_text())

    def final_eval(self, name: str) -> dict | None:
        path = self.run_dir(name) / "eval.json"
        return json.loads(path.read_text()) if path.is_file() else None

    def metrics(self, name: str) -> tuple[list[dict], int]:
        """Return (rows, number of unreadable lines)."""
        path = self.run_dir(name) / "metrics.jsonl"
        rows, bad = [], 0
        if not path.is_file():
            return rows, bad
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))  # json accepts bare NaN, which real loggers often write
            except json.JSONDecodeError:
                bad += 1
        return rows, bad


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema of the arguments object
    func: Callable[..., Any]

    def openai_schema(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}

    def mcp_schema(self) -> dict:
        return {"name": self.name, "description": self.description, "inputSchema": self.parameters}


def _schema(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


RUN_ARG = {"type": "string", "description": "Run name exactly as returned by list_runs, e.g. 'lego_baseline'."}

_JSON_TYPES = {"string": str, "integer": int, "number": (int, float), "array": list, "object": dict, "boolean": bool}


def validate_args(tool: Tool, args: Any) -> None:
    """Small subset of JSON Schema: object, required, unknown keys, top-level types, enums."""
    if not isinstance(args, dict):
        raise ToolError(f"{tool.name}: arguments must be a JSON object, got {type(args).__name__}")
    props = tool.parameters.get("properties", {})
    missing = [k for k in tool.parameters.get("required", []) if k not in args]
    if missing:
        raise ToolError(f"{tool.name}: missing required argument(s): {', '.join(missing)}")
    unknown = [k for k in args if k not in props]
    if unknown:
        raise ToolError(f"{tool.name}: unknown argument(s): {', '.join(unknown)}. Allowed: {', '.join(props)}")
    for key, value in args.items():
        spec = props[key]
        expected = _JSON_TYPES.get(spec.get("type"))
        # bool is a subclass of int in Python; don't let True pass as an integer.
        if expected and (not isinstance(value, expected) or (isinstance(value, bool) and spec["type"] != "boolean")):
            raise ToolError(f"{tool.name}: argument {key!r} must be of type {spec['type']}")
        if "enum" in spec and value not in spec["enum"]:
            raise ToolError(f"{tool.name}: argument {key!r} must be one of {spec['enum']}, got {value!r}")


class Toolbox:
    """The set of tools over one RunStore, plus validated dispatch."""

    def __init__(self, store: RunStore):
        self.store = store
        self.tools: dict[str, Tool] = {t.name: t for t in self._build()}

    def __contains__(self, name: str) -> bool:
        return name in self.tools

    def openai_schemas(self) -> list[dict]:
        return [t.openai_schema() for t in self.tools.values()]

    def mcp_schemas(self) -> list[dict]:
        return [t.mcp_schema() for t in self.tools.values()]

    def call(self, name: str, args: Any) -> Any:
        tool = self.tools.get(name)
        if tool is None:
            raise ToolError(f"unknown tool {name!r}. Available tools: {', '.join(self.tools)}")
        validate_args(tool, args)
        return clean(tool.func(**args))

    # ---- tool implementations -------------------------------------------------

    def list_runs(self, scene: str | None = None) -> list[dict]:
        out = []
        for name in self.store.run_names():
            cfg = self.store.config(name)
            if scene is not None and cfg.get("scene") != scene:
                continue
            out.append({"run": name, "scene": cfg.get("scene"), "iterations": cfg.get("iterations"),
                        "status": "finished" if self.store.final_eval(name) is not None else "no_eval"})
        return out

    def get_run_config(self, run: str) -> dict:
        return {"run": run, "config": self.store.config(run)}

    def get_final_metrics(self, run: str) -> dict:
        final = self.store.final_eval(run)
        if final is None:
            return {"run": run, "status": "no_eval", "metrics": None,
                    "note": "This run has no eval.json, so it has no final metrics. Use diagnose_run to see why."}
        return {"run": run, "status": "finished", "metrics": final}

    def compare_runs(self, runs: list, metric: str) -> dict:
        if not runs:
            raise ToolError("compare_runs: 'runs' must contain at least one run name")
        rows, skipped = [], []
        for run in runs:
            final = self.store.final_eval(run)  # raises ToolError for unknown runs
            if final is None or metric not in final:
                skipped.append(run)
            else:
                rows.append({"run": run, metric: final[metric]})
        higher_is_better = EVAL_METRICS[metric]
        rows.sort(key=lambda r: r[metric], reverse=higher_is_better)
        return {"metric": metric, "higher_is_better": higher_is_better, "ranking": rows,
                "best": rows[0]["run"] if rows else None,
                "skipped_no_eval": skipped}

    def get_metric_curve(self, run: str, metric: str = "loss", points: int = 10) -> dict:
        rows, bad = self.store.metrics(run)
        if not rows:
            raise ToolError(f"run {run!r} has no logged metrics")
        available = sorted({k for r in rows for k in r if k != "step"})
        if metric not in available:
            raise ToolError(f"metric {metric!r} is not logged for {run!r}. Logged metrics: {', '.join(available)}")
        if points < 2:
            raise ToolError("points must be at least 2")
        series = [(r["step"], r.get(metric)) for r in rows if "step" in r]
        if len(series) > points:
            # Evenly spaced samples that always keep the first and last point.
            idx = sorted({round(i * (len(series) - 1) / (points - 1)) for i in range(points)})
            series = [series[i] for i in idx]
        return {"run": run, "metric": metric, "total_logged": len(rows), "unreadable_lines": bad,
                "curve": [{"step": s, metric: v} for s, v in series],
                "last": {"step": rows[-1].get("step"), metric: rows[-1].get(metric)}}

    def diagnose_run(self, run: str) -> dict:
        cfg = self.store.config(run)
        rows, bad = self.store.metrics(run)
        issues = []
        if self.store.final_eval(run) is None:
            issues.append("no final evaluation: eval.json is missing")
        if bad:
            issues.append(f"{bad} line(s) in metrics.jsonl could not be parsed")
        if not rows:
            issues.append("metrics.jsonl is missing or empty")
        else:
            losses = [(r["step"], r["loss"]) for r in rows if isinstance(r.get("loss"), (int, float))]
            nonfinite = next(((s, l) for s, l in losses if not math.isfinite(l)), None)
            if nonfinite:
                issues.append(f"loss became {nonfinite[1]} at step {nonfinite[0]} (training diverged)")
            finite = [(s, l) for s, l in losses if math.isfinite(l)]
            for (_, prev), (step, cur) in zip(finite, finite[1:]):
                if prev > 0 and cur / prev >= LOSS_SPIKE_RATIO:
                    issues.append(f"loss spiked {cur / prev:.1f}x at step {step} ({prev:.4g} -> {cur:.4g})")
                    break
            last_step = rows[-1].get("step")
            planned = cfg.get("iterations")
            if isinstance(last_step, int) and isinstance(planned, int) and last_step < planned:
                issues.append(f"training stopped early: last logged step {last_step} of {planned}")
            peak = max((r.get("num_gaussians", 0) for r in rows), default=0)
            if peak > MAX_REASONABLE_GAUSSIANS:
                issues.append(f"very high Gaussian count: {peak:,}")
        return {"run": run, "healthy": not issues, "issues": issues,
                "last_logged_step": rows[-1].get("step") if rows else None,
                "planned_iterations": cfg.get("iterations")}

    def _build(self) -> list[Tool]:
        return [
            Tool("list_runs",
                 "List all training runs with scene, planned iterations and status "
                 "('finished' has final metrics, 'no_eval' does not). Call this first if you don't know the run names.",
                 _schema({"scene": {"type": "string", "description": "Only list runs of this scene, e.g. 'lego'."}}, []),
                 self.list_runs),
            Tool("get_run_config",
                 "Get the training config of one run (iterations, learning rates, densification settings...).",
                 _schema({"run": RUN_ARG}, ["run"]),
                 self.get_run_config),
            Tool("get_final_metrics",
                 "Get the final test metrics (psnr, ssim, lpips) of one run. Returns status 'no_eval' if the run has none.",
                 _schema({"run": RUN_ARG}, ["run"]),
                 self.get_final_metrics),
            Tool("compare_runs",
                 "Rank several runs by one final metric and say which is best. Knows that lower lpips is better.",
                 _schema({"runs": {"type": "array", "items": {"type": "string"}, "description": "Run names to compare."},
                          "metric": {"type": "string", "enum": list(EVAL_METRICS)}}, ["runs", "metric"]),
                 self.compare_runs),
            Tool("get_metric_curve",
                 "Get a downsampled training curve of one logged metric (e.g. loss, train_psnr, num_gaussians) "
                 "plus its last value.",
                 _schema({"run": RUN_ARG,
                          "metric": {"type": "string", "description": "Logged metric name. Default 'loss'."},
                          "points": {"type": "integer", "description": "Max number of points to return. Default 10."}},
                         ["run"]),
                 self.get_metric_curve),
            Tool("diagnose_run",
                 "Check one run for problems: missing final eval, NaN/inf loss, loss spikes, stopping early, "
                 "too many Gaussians. Use it to explain why a run failed or has no results.",
                 _schema({"run": RUN_ARG}, ["run"]),
                 self.diagnose_run),
        ]
