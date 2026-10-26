"""The agent loop: ask the model, run the tools it asks for, feed results back, repeat."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from .llm import LLM, ToolCall
from .tools import Toolbox, ToolError

SYSTEM_PROMPT = """You are ExpAgent, an assistant that answers questions about machine-learning training runs.
Rules:
- Get every number from tools. Never guess or estimate metric values.
- If you don't know the exact run names, call list_runs first.
- If a run has no final evaluation, say so plainly instead of inventing results; use diagnose_run to explain why.
- If a tool returns an error, read it and fix your call (e.g. use a run name from the list) or tell the user.
- Answer in the user's language. Be concise and name the runs and numbers you used."""

STEP_LIMIT_PROMPT = ("You have reached the tool-call limit. Answer now using only the results above, "
                     "and say clearly what you could not find out.")


@dataclass
class Step:
    index: int
    tool: str
    arguments: Any
    ok: bool
    result: Any = None
    error: str | None = None
    seconds: float = 0.0

    def observation(self) -> dict:
        return {"ok": True, "result": self.result} if self.ok else {"ok": False, "error": self.error}


@dataclass
class AgentResult:
    question: str
    answer: str
    stopped_reason: str  # 'answered' or 'max_steps'
    steps: list[Step] = field(default_factory=list)
    llm_calls: int = 0
    usage: dict = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def tools_called(self) -> list[str]:
        return [s.tool for s in self.steps]

    def to_dict(self) -> dict:
        return asdict(self)


class Agent:
    def __init__(self, llm: LLM, toolbox: Toolbox, max_steps: int = 8, system_prompt: str = SYSTEM_PROMPT,
                 on_step: Callable[[Step], None] | None = None):
        self.llm = llm
        self.toolbox = toolbox
        self.max_steps = max_steps  # max model turns that may call tools
        self.system_prompt = system_prompt
        self.on_step = on_step

    def run(self, question: str) -> AgentResult:
        start = time.perf_counter()
        messages: list[dict] = [{"role": "system", "content": self.system_prompt},
                                {"role": "user", "content": question}]
        result = AgentResult(question=question, answer="", stopped_reason="answered")
        seen: dict[tuple[str, str], int] = {}  # (tool, canonical args) -> step index, to catch loops
        tools = self.toolbox.openai_schemas()

        for _ in range(self.max_steps):
            resp = self._chat(result, messages, tools)
            if not resp.tool_calls:
                result.answer = resp.content or ""
                break
            messages.append({"role": "assistant", "content": resp.content or "",
                             "tool_calls": [c.to_message() for c in resp.tool_calls]})
            for call in resp.tool_calls:
                step = self._execute(call, len(result.steps) + 1, seen)
                result.steps.append(step)
                if self.on_step:
                    self.on_step(step)
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": json.dumps(step.observation(), ensure_ascii=False)})
        else:
            # Out of tool turns: force a final answer with tools disabled.
            result.stopped_reason = "max_steps"
            resp = self._chat(result, messages + [{"role": "user", "content": STEP_LIMIT_PROMPT}], [])
            result.answer = resp.content or ""

        result.seconds = round(time.perf_counter() - start, 3)
        return result

    def _chat(self, result: AgentResult, messages: list[dict], tools: list[dict]):
        resp = self.llm.chat(messages, tools)
        result.llm_calls += 1
        for k, v in resp.usage.items():
            if isinstance(v, int):
                result.usage[k] = result.usage.get(k, 0) + v
        return resp

    def _execute(self, call: ToolCall, index: int, seen: dict) -> Step:
        start = time.perf_counter()
        step = Step(index=index, tool=call.name, arguments=call.arguments, ok=False)
        try:
            args = json.loads(call.arguments) if call.arguments.strip() else {}
        except json.JSONDecodeError as e:
            step.error = f"arguments are not valid JSON ({e.msg}); send a JSON object"
            return self._finish(step, start)
        step.arguments = args
        key = (call.name, json.dumps(args, sort_keys=True, ensure_ascii=False))
        if key in seen:
            step.error = (f"duplicate call: {call.name} was already called with these arguments at step "
                          f"{seen[key]}; reuse that result instead of calling again")
            return self._finish(step, start)
        seen[key] = index
        try:
            step.result = self.toolbox.call(call.name, args)
            step.ok = True
        except ToolError as e:
            step.error = str(e)
        except Exception as e:  # a bug in a tool must not kill the whole conversation
            step.error = f"internal tool error: {type(e).__name__}: {e}"
        return self._finish(step, start)

    @staticmethod
    def _finish(step: Step, start: float) -> Step:
        step.seconds = round(time.perf_counter() - start, 4)
        return step
