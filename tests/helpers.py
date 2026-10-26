import json
from pathlib import Path

from expagent.llm import LLMResponse, ToolCall
from expagent.tools import RunStore, Toolbox

RUNS_DIR = Path(__file__).resolve().parent.parent / "examples" / "runs"


def toolbox() -> Toolbox:
    return Toolbox(RunStore(RUNS_DIR))


def call(name: str, args: dict | str, id_: str = "c1") -> LLMResponse:
    raw = args if isinstance(args, str) else json.dumps(args)
    return LLMResponse(content=None, tool_calls=[ToolCall(id_, name, raw)])


def answer(text: str) -> LLMResponse:
    return LLMResponse(content=text, usage={"prompt_tokens": 10, "completion_tokens": 5})
