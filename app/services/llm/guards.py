"""Input + output guards."""
from __future__ import annotations

import re
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

_PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),           # SSN
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # email
    re.compile(r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),  # phone
]

_JAILBREAK_PATTERNS = [
    "ignore previous instructions",
    "pretend you are",
    "you are now",
    "disregard your",
    "forget your guidelines",
    "act as",
    "bypass",
]


def redact_pii(text: str) -> str:
    for pattern in _PII_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def looks_like_jailbreak(text: str) -> bool:
    lower = text.lower()
    return any(p in lower for p in _JAILBREAK_PATTERNS)


def validate_output(raw: dict, schema: type[T]) -> T:
    return schema.model_validate(raw)


def enforce_no_math(text: str, allowed_numbers: set[float]) -> bool:
    """Return True iff every number in `text` is in `allowed_numbers`."""
    found = re.findall(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", text)
    for token in found:
        try:
            val = float(token)
        except ValueError:
            continue
        if val not in allowed_numbers:
            return False
    return True
