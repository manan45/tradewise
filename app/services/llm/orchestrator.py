"""LLM orchestration loop — tool-call dispatch, guard enforcement, retries."""
from __future__ import annotations

from typing import Any

from app.core.ports.llm import LLMClient, Tool
from .guards import enforce_no_math

MAX_TOOL_HOPS = 8


class LLMOrchestrator:
    def __init__(self, client: LLMClient, tools: list[Tool],
                 audit_sink: Any | None = None):
        self.client = client
        self.tools = {t.spec.name: t for t in tools}
        self.audit_sink = audit_sink

    async def run(
        self,
        system: str,
        user_prompt: str,
        response_schema: type | None = None,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]
        tool_specs = [t.spec for t in self.tools.values()]
        allowed_numbers: set[float] = set()

        for _ in range(MAX_TOOL_HOPS):
            response = await self.client.complete(
                system=system,
                messages=messages,
                tools=tool_specs if tool_specs else None,
                max_tokens=max_tokens,
            )

            if not response.tool_calls:
                # Final response
                if response_schema is not None:
                    from .guards import validate_output
                    import json
                    try:
                        parsed = json.loads(response.text)
                        result = validate_output(parsed, response_schema)
                        return result.model_dump()
                    except Exception:
                        pass
                # enforce no-math
                enforce_no_math(response.text, allowed_numbers)
                return {"text": response.text}

            # Process tool calls
            tool_results = []
            for call in response.tool_calls:
                name = call["name"]
                args = call["input"]
                tool = self.tools.get(name)
                if tool is None:
                    result_payload = {"error": f"Unknown tool: {name}"}
                else:
                    result_payload = await tool.handler(args)
                    # Collect numeric values from tool outputs
                    self._extract_numbers(result_payload, allowed_numbers)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": call["id"],
                    "content": str(result_payload),
                })

            messages.append({"role": "assistant", "content": response.tool_calls})
            messages.append({"role": "user", "content": tool_results})

        raise RuntimeError(f"LLMOrchestrator exceeded MAX_TOOL_HOPS ({MAX_TOOL_HOPS})")

    def _extract_numbers(self, obj: Any, numbers: set[float]) -> None:
        if isinstance(obj, (int, float)):
            numbers.add(float(obj))
        elif isinstance(obj, dict):
            for v in obj.values():
                self._extract_numbers(v, numbers)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                self._extract_numbers(v, numbers)
