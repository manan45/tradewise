"""Frozen policy bundle."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Policy:
    name: str
    scenarios: tuple[str, ...]
    sizing: dict[str, Any] = field(default_factory=dict)
    breaker: dict[str, Any] = field(default_factory=dict)
    frictions: dict[str, Any] = field(default_factory=dict)
    feature_set_overrides: dict[str, list[tuple[str, int]]] = field(
        default_factory=dict)
