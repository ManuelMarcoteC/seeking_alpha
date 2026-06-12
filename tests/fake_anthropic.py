"""Scripted fake of the anthropic client for offline agent tests.

Builds response objects shaped like the SDK's (content blocks with .type/.name/
.input/.id, .stop_reason, .usage) from a queue of scripted turns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any


def usage(in_=1000, out=200, cache_read=0, cache_write=0):
    return SimpleNamespace(
        input_tokens=in_,
        output_tokens=out,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
    )


def tool_use(name: str, input_: dict, block_id: str = "toolu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=input_, id=block_id)


def text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def response(blocks: list, stop_reason: str = "tool_use", **usage_kwargs):
    return SimpleNamespace(
        content=blocks, stop_reason=stop_reason, usage=usage(**usage_kwargs)
    )


def parsed_response(parsed_output: Any, **usage_kwargs):
    return SimpleNamespace(parsed_output=parsed_output, usage=usage(**usage_kwargs))


@dataclass
class FakeMessages:
    create_queue: list = field(default_factory=list)
    parse_queue: list = field(default_factory=list)
    create_calls: list = field(default_factory=list)
    parse_calls: list = field(default_factory=list)

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        if not self.create_queue:
            raise AssertionError("FakeAnthropic.create called more times than scripted")
        item = self.create_queue.pop(0)
        return item(kwargs) if callable(item) else item

    def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        if not self.parse_queue:
            raise AssertionError("FakeAnthropic.parse called more times than scripted")
        item = self.parse_queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item(kwargs) if callable(item) else item


@dataclass
class FakeAnthropic:
    messages: FakeMessages = field(default_factory=FakeMessages)
