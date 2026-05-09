"""Tier routing — pick Haiku/Sonnet/Opus based on task and risk."""
from __future__ import annotations

from enum import Enum


class TaskKind(str, Enum):
    TOOL_LOOP = "tool_loop"
    DAILY_BRIEF = "daily_brief"
    POST_TRADE = "post_trade"
    OPERATOR_QA = "operator_qa"


class LLMUnavailable(Exception):
    pass


_MODEL_MAP = {
    TaskKind.TOOL_LOOP: "claude-haiku-4-5",
    TaskKind.DAILY_BRIEF: "claude-sonnet-4-6",
    TaskKind.POST_TRADE: "claude-sonnet-4-6",
    TaskKind.OPERATOR_QA: "claude-opus-4-7",
}


def pick_model(task: TaskKind, no_network: bool = False) -> str:
    if no_network:
        raise LLMUnavailable("NO_NETWORK is set; LLM unavailable.")
    return _MODEL_MAP[task]
