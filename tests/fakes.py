"""Fakes shared by the v2 tests: a scripted OpenAI-compatible client.

The point of scripting the client is to test *our* loop control (budget,
clamping, malformed arguments, missing submit) rather than the provider.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def tool_call(tc_id: str, name: str, arguments: Any) -> SimpleNamespace:
    """Build something shaped like openai's ChatCompletionMessageToolCall."""
    raw = arguments if isinstance(arguments, str) else _dumps(arguments)
    return SimpleNamespace(
        id=tc_id, type="function",
        function=SimpleNamespace(name=name, arguments=raw),
    )


def _dumps(obj: Any) -> str:
    import json
    return json.dumps(obj)


def usage(prompt: int = 10, completion: int = 5) -> SimpleNamespace:
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                           total_tokens=prompt + completion)


class ScriptedClient:
    """Returns a canned response per call; records every request it saw."""

    def __init__(self, responses: list[SimpleNamespace]):
        self._responses = list(responses)
        self.requests: list[dict] = []

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs) -> SimpleNamespace:
        self.requests.append(kwargs)
        if not self._responses:
            raise AssertionError(
                "ScriptedClient ran out of responses — the loop called the "
                "model more times than the test scripted. This usually means "
                "the loop failed to terminate.")
        return self._responses.pop(0)


def response(message: SimpleNamespace | None = None,
             content: str | None = None,
             tool_calls: list | None = None,
             prompt_tokens: int = 10,
             completion_tokens: int = 5) -> SimpleNamespace:
    msg = message or SimpleNamespace(
        role="assistant", content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg)],
        usage=usage(prompt_tokens, completion_tokens),
    )
