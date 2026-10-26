"""LLM clients. The agent only needs `chat(messages, tools) -> LLMResponse`.

OpenAICompatibleLLM talks to any OpenAI-style /chat/completions endpoint
(DeepSeek, OpenAI, Qwen/DashScope compatible mode, vLLM, Ollama...) using only
the standard library. ScriptedLLM replays fixed responses for tests.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol


class LLMError(Exception):
    """The model endpoint failed or returned something we can't use."""


class ConfigError(LLMError):
    """Missing API key or other configuration."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string as produced by the model; may be invalid

    def to_message(self) -> dict:
        return {"id": self.id, "type": "function", "function": {"name": self.name, "arguments": self.arguments}}


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


class LLM(Protocol):
    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


class OpenAICompatibleLLM:
    # Defaults point at DeepSeek's OpenAI-compatible API; override with env vars for other providers.
    DEFAULT_BASE_URL = "https://api.deepseek.com"
    DEFAULT_MODEL = "deepseek-chat"
    RETRY_STATUS = {429, 500, 502, 503, 504}

    def __init__(self, api_key: str, base_url: str | None = None, model: str | None = None,
                 timeout: float = 60, max_retries: int = 2, temperature: float = 0.0):
        if not api_key:
            raise ConfigError("missing API key")
        self.api_key = api_key
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature

    @classmethod
    def from_env(cls) -> "OpenAICompatibleLLM":
        key = os.environ.get("EXPAGENT_API_KEY")
        if not key:
            raise ConfigError("set EXPAGENT_API_KEY (and optionally EXPAGENT_BASE_URL, EXPAGENT_MODEL)")
        return cls(key, os.environ.get("EXPAGENT_BASE_URL"), os.environ.get("EXPAGENT_MODEL"))

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        body = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if tools:
            body["tools"] = tools
        data = self._post("/chat/completions", body)
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected response shape: {str(data)[:300]}") from e
        calls = [ToolCall(c.get("id") or f"call_{i}", c["function"]["name"], c["function"].get("arguments") or "{}")
                 for i, c in enumerate(message.get("tool_calls") or [])]
        return LLMResponse(message.get("content"), calls, data.get("usage") or {})

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + path, data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                if e.code not in self.RETRY_STATUS or attempt == self.max_retries:
                    raise LLMError(f"HTTP {e.code} from {self.base_url}: {detail}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == self.max_retries:
                    raise LLMError(f"cannot reach {self.base_url}: {e}") from e
            time.sleep(2 ** attempt)  # 1s, 2s, ...
        raise AssertionError("unreachable")


class ScriptedLLM:
    """Returns pre-written responses in order and records what it was sent. For tests."""

    def __init__(self, responses: list[LLMResponse]):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        self.requests.append({"messages": json.loads(json.dumps(messages)), "tools": tools})
        if not self.responses:
            raise LLMError("ScriptedLLM ran out of responses")
        return self.responses.pop(0)
