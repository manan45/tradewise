"""LLM orchestrator.

Hard rule (final_requirements §6, §10): the model never computes numbers.
It picks tools, reads structured tool outputs, and produces structured
plans/explanations. All math lives in deterministic Python.

Layout:
- anthropic_client.py — concrete LLMClient impl using the anthropic SDK
- tools.py — registry of Tool instances available to the model
- schemas.py — pydantic models for every tool input + output (single source of
  truth for both the model's input_schema and our own validation)
- routing.py — picks model tier (Haiku vs Sonnet vs Opus) based on task type
  and risk
- guards.py — input/output guards (PII, jailbreak, price-leak prevention)
- orchestrator.py — top-level loop wiring all of the above
"""
from .orchestrator import LLMOrchestrator

__all__ = ["LLMOrchestrator"]
