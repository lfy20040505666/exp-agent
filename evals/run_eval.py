"""Run the eval cases against an agent and report per-case and overall scores.

    python -m evals.run_eval --llm baseline          # no API key needed
    python -m evals.run_eval --llm api               # uses EXPAGENT_API_KEY / _BASE_URL / _MODEL
    python -m evals.run_eval --llm api --out evals/results/deepseek.json

A case passes when
  - tools:  every expected tool was called (an inner list means "any of these"),
            and an empty list means no tool is required;
  - answer: all `must_contain`, at least one of `must_contain_any` and none of
            `must_not_contain` appear in the answer (case-insensitive).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from expagent.agent import Agent, AgentResult
from expagent.cli import DEFAULT_RUNS_DIR
from expagent.llm import LLMError, OpenAICompatibleLLM
from expagent.tools import RunStore, Toolbox

CASES = Path(__file__).parent / "cases.jsonl"


def load_cases(path: Path = CASES) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check_tools(expected: list, called: list[str]) -> tuple[bool, list[str]]:
    missing = []
    for item in expected:
        options = item if isinstance(item, list) else [item]
        if not any(o in called for o in options):
            missing.append("|".join(options))
    return not missing, missing


def check_answer(case: dict, answer: str) -> tuple[bool, list[str]]:
    text = answer.lower()
    problems = [f"missing {s!r}" for s in case.get("must_contain", []) if s.lower() not in text]
    any_of = case.get("must_contain_any")
    if any_of and not any(s.lower() in text for s in any_of):
        problems.append(f"none of {any_of}")
    problems += [f"contains {s!r}" for s in case.get("must_not_contain", []) if s.lower() in text]
    return not problems, problems


def score(case: dict, result: AgentResult) -> dict:
    tools_ok, missing_tools = check_tools(case.get("expect_tools", []), result.tools_called)
    answer_ok, answer_problems = check_answer(case, result.answer)
    return {"id": case["id"], "pass": tools_ok and answer_ok, "tools_ok": tools_ok, "answer_ok": answer_ok,
            "tools_called": result.tools_called, "missing_tools": missing_tools,
            "answer_problems": answer_problems, "tool_errors": sum(not s.ok for s in result.steps),
            "llm_calls": result.llm_calls, "stopped_reason": result.stopped_reason,
            "seconds": result.seconds, "usage": result.usage, "answer": result.answer}


def run_eval(make_agent, cases: list[dict]) -> dict:
    rows = []
    for case in cases:
        try:
            result = make_agent().run(case["question"])
        except LLMError as e:
            rows.append({"id": case["id"], "pass": False, "tools_ok": False, "answer_ok": False,
                         "error": str(e), "tools_called": [], "llm_calls": 0, "seconds": 0, "tool_errors": 0})
            continue
        rows.append(score(case, result))
    n = len(rows)
    summary = {
        "cases": n,
        "pass_rate": round(sum(r["pass"] for r in rows) / n, 3),
        "tool_accuracy": round(sum(r["tools_ok"] for r in rows) / n, 3),
        "answer_accuracy": round(sum(r["answer_ok"] for r in rows) / n, 3),
        "avg_llm_calls": round(sum(r["llm_calls"] for r in rows) / n, 2),
        "total_tool_errors": sum(r["tool_errors"] for r in rows),
        "avg_seconds": round(sum(r["seconds"] for r in rows) / n, 3),
    }
    return {"summary": summary, "cases": rows}


def to_markdown(report: dict, label: str) -> str:
    s = report["summary"]
    lines = [f"## {label}", "",
             f"pass {s['pass_rate']:.0%} | tools {s['tool_accuracy']:.0%} | answers {s['answer_accuracy']:.0%} | "
             f"avg LLM calls {s['avg_llm_calls']} | tool errors {s['total_tool_errors']} | avg {s['avg_seconds']}s",
             "", "| case | pass | tools called | problems |", "|---|---|---|---|"]
    for r in report["cases"]:
        problems = r.get("error") or "; ".join([f"missing tool {t}" for t in r.get("missing_tools", [])]
                                               + r.get("answer_problems", []))
        lines.append(f"| {r['id']} | {'✅' if r['pass'] else '❌'} | {', '.join(r['tools_called']) or '-'} | {problems or ''} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm", choices=["baseline", "api"], default="baseline")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--out", help="write the full JSON report here")
    args = parser.parse_args(argv)

    toolbox = Toolbox(RunStore(args.runs_dir))
    if args.llm == "baseline":
        from .baseline import KeywordRouterLLM
        llm, label = KeywordRouterLLM(), "keyword-router baseline (no LLM)"
    else:
        llm = OpenAICompatibleLLM.from_env()
        label = f"{llm.model} @ {llm.base_url}"

    report = run_eval(lambda: Agent(llm, toolbox, max_steps=args.max_steps), load_cases())
    report["label"] = label
    report["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    print(to_markdown(report, label))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\nfull report: {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
