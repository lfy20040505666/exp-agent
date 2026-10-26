"""A non-LLM baseline that plugs into the same Agent loop.

It picks exactly one tool by keyword, extracts run names with a regex and
returns the raw tool output as its "answer". It shows what the eval harness
measures without needing an API key, and gives a floor that a real LLM agent
should beat.
"""

from __future__ import annotations

import json
import re

from expagent.llm import LLMResponse, ToolCall

RUN_NAME = re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\b")
METRICS = ("psnr", "ssim", "lpips")


def _has(text: str, words) -> bool:
    return any(w in text for w in words)


class KeywordRouterLLM:
    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        last = messages[-1]
        if last["role"] == "tool":
            return LLMResponse(content=last["content"])  # answer = raw tool output
        if not tools:
            return LLMResponse(content="(step limit reached)")
        q = next(m["content"] for m in messages if m["role"] == "user").lower()
        runs = RUN_NAME.findall(q)
        metric = next((m for m in METRICS if m in q), None)

        if runs and _has(q, ["为什么", "why", "诊断", "diagnose", "失败", "nan"]):
            name, args = "diagnose_run", {"run": runs[0]}
        elif metric and len(runs) >= 2:
            name, args = "compare_runs", {"runs": runs, "metric": metric}
        elif runs and _has(q, ["多少步", "iterations", "配置", "config", "学习率"]):
            name, args = "get_run_config", {"run": runs[0]}
        elif runs and _has(q, ["高斯", "gaussian", "曲线", "curve", "loss"]):
            metric_name = "num_gaussians" if _has(q, ["高斯", "gaussian"]) else "loss"
            name, args = "get_metric_curve", {"run": runs[0], "metric": metric_name}
        elif runs and metric:
            name, args = "get_final_metrics", {"run": runs[0]}
        else:
            name, args = "list_runs", {}
        return LLMResponse(content=None, tool_calls=[ToolCall("call_0", name, json.dumps(args))])
