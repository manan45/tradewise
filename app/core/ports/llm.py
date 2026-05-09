"""LLM orchestrator port.

The hard rule (final_requirements §6, §10): the LLM never computes numbers;
it only reasons over structured outputs from deterministic Python tools.
Hence Tool/ToolSpec are first-class here — the orchestrator dispatches tool
calls, executes them, and feeds results back into the model loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class Tool:
    spec: ToolSpec
    handler: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class LLMResponse:
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str = "end_turn"


class LLMClient(Protocol):
    async def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        ...
