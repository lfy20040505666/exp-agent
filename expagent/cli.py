"""Command line entry point: python -m expagent <command> ..."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .agent import Agent, Step
from .llm import ConfigError, LLMError, OpenAICompatibleLLM
from .mcp_server import MCPServer
from .server import make_server
from .tools import RunStore, Toolbox, ToolError

DEFAULT_RUNS_DIR = Path(__file__).resolve().parent.parent / "examples" / "runs"


def _print_step(step: Step) -> None:
    status = "ok" if step.ok else f"error: {step.error}"
    args = json.dumps(step.arguments, ensure_ascii=False) if not isinstance(step.arguments, str) else step.arguments
    print(f"  [{step.index}] {step.tool}({args}) -> {status}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="expagent", description="Ask questions about ML training runs.")
    parser.add_argument("--runs-dir", default=os.environ.get("EXPAGENT_RUNS_DIR", str(DEFAULT_RUNS_DIR)),
                        help="folder with one sub-folder per run (default: examples/runs or $EXPAGENT_RUNS_DIR)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("tools", help="list the tools and their argument schemas")
    p_call = sub.add_parser("call", help="call one tool directly, without the LLM")
    p_call.add_argument("tool")
    p_call.add_argument("arguments", nargs="?", default="{}", help="JSON object, e.g. '{\"run\": \"lego_baseline\"}'")
    p_ask = sub.add_parser("ask", help="ask the agent a question (needs EXPAGENT_API_KEY)")
    p_ask.add_argument("question")
    p_ask.add_argument("--max-steps", type=int, default=8)
    p_ask.add_argument("--json", action="store_true", help="print the full trace as JSON")
    p_serve = sub.add_parser("serve", help="start the HTTP API")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    sub.add_parser("mcp", help="run as an MCP server over stdio")

    args = parser.parse_args(argv)
    toolbox = Toolbox(RunStore(args.runs_dir))

    if args.command == "tools":
        print(json.dumps(toolbox.mcp_schemas(), ensure_ascii=False, indent=2))
    elif args.command == "call":
        try:
            result = toolbox.call(args.tool, json.loads(args.arguments))
        except (ToolError, json.JSONDecodeError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "ask":
        try:
            agent = Agent(OpenAICompatibleLLM.from_env(), toolbox, max_steps=args.max_steps,
                          on_step=None if args.json else _print_step)
            result = agent.run(args.question)
        except (ConfigError, LLMError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) if args.json else result.answer)
    elif args.command == "serve":
        server = make_server(args.host, args.port, toolbox,
                             lambda: Agent(OpenAICompatibleLLM.from_env(), toolbox))
        print(f"listening on http://{args.host}:{server.server_port}", file=sys.stderr)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    elif args.command == "mcp":
        MCPServer(toolbox).serve()
    return 0
